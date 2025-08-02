/*
 * Copyright © 2023 Collabora Ltd.
 * Copyright © 2024 Valve Software
 *
 * SPDX-License-Identifier: MIT
 */

use anyhow::{anyhow, bail, ensure, Result};
use async_trait::async_trait;
use num_enum::TryFromPrimitive;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::ops::RangeInclusive;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use strum::{Display, EnumString, VariantNames};
use tokio::fs::{self, try_exists, File};
use tokio::io::{AsyncWriteExt, Interest};
use tokio::net::unix::pipe;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::{oneshot, Mutex, Notify, OnceCell};
use tokio::task::JoinSet;
use tracing::{debug, error, warn};
use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue};
use zbus::Connection;

use crate::gpu::AMDGPU_HWMON_NAME;
use crate::hardware::device_config;
use crate::manager::root::RootManagerProxy;
use crate::manager::user::{TdpLimit1, MANAGER_PATH};
use crate::Service;
use crate::{path, write_synced};

#[cfg(not(test))]
const HWMON_PREFIX: &str = "/sys/class/hwmon";
#[cfg(test)]
pub const HWMON_PREFIX: &str = "hwmon";

const CPU_PREFIX: &str = "/sys/devices/system/cpu";
const CPUFREQ_PREFIX: &str = "cpufreq";
const CPUFREQ_BOOST_SUFFIX: &str = "boost";
const INTEL_PSTATE_PREFIX: &str = "intel_pstate";
const INTEL_PSTATE_NO_TURBO_SUFFIX: &str = "no_turbo";

const CPU0_NAME: &str = "policy0";
const CPU_POLICY_NAME: &str = "policy";

const CPU_SCALING_GOVERNOR_SUFFIX: &str = "scaling_governor";
const CPU_SCALING_AVAILABLE_GOVERNORS_SUFFIX: &str = "scaling_available_governors";

const PLATFORM_PROFILE_PREFIX: &str = "/sys/class/platform-profile";

const TDP_LIMIT1: &str = "power1_cap";
const TDP_LIMIT2: &str = "power2_cap";

static SYSFS_WRITER: OnceCell<Arc<SysfsWriterQueue>> = OnceCell::const_new();

#[derive(Display, EnumString, Hash, Eq, PartialEq, Debug, Copy, Clone)]
#[strum(serialize_all = "lowercase")]
pub enum CPUScalingGovernor {
    Conservative,
    OnDemand,
    UserSpace,
    PowerSave,
    Performance,
    SchedUtil,
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum CpuBoostDriver {
    IntelPstate,
    CpuFreq,
}

#[derive(Display, EnumString, PartialEq, Debug, Copy, Clone, TryFromPrimitive)]
#[strum(ascii_case_insensitive)]
#[repr(u32)]
pub enum CPUBoostState {
    #[strum(
        to_string = "disabled",
        serialize = "off",
        serialize = "disable",
        serialize = "0"
    )]
    Disabled = 0,
    #[strum(
        to_string = "enabled",
        serialize = "on",
        serialize = "enable",
        serialize = "1"
    )]
    Enabled = 1,
}

#[derive(Display, EnumString, VariantNames, PartialEq, Debug, Clone)]
#[strum(serialize_all = "snake_case")]
pub enum TdpLimitingMethod {
    AmdgpuHwmon,
    FirmwareAttribute,
    PowerStation,
}

#[derive(Debug)]
pub(crate) struct AmdgpuHwmonTdpLimitManager {}

#[derive(Debug)]
pub(crate) struct FirmwareAttributeLimitManager {
    attribute: String,
    performance_profile: Option<String>,
}

#[derive(Debug)]
pub(crate) struct PowerStationTdpLimitManager {
    // Path to the detected GPU card, cached to avoid repeated lookups
    gpu_card_path: StdMutex<Option<String>>,
}

#[async_trait]
pub(crate) trait TdpLimitManager: Send + Sync {
    async fn get_tdp_limit(&self) -> Result<u32>;
    async fn set_tdp_limit(&self, limit: u32) -> Result<()>;
    async fn get_tdp_limit_range(&self) -> Result<RangeInclusive<u32>>;
    async fn is_active(&self) -> Result<bool> {
        Ok(true)
    }
}

pub(crate) async fn tdp_limit_manager() -> Result<Box<dyn TdpLimitManager>> {
    let config = device_config().await?;
    let config = config
        .as_ref()
        .and_then(|config| config.tdp_limit.as_ref())
        .ok_or(anyhow!("No TDP limit configured"))?;

    Ok(match &config.method {
        TdpLimitingMethod::FirmwareAttribute => {
            let Some(ref firmware_attribute) = config.firmware_attribute else {
                bail!("Firmware attribute TDP limiting method not configured");
            };
            Box::new(FirmwareAttributeLimitManager {
                attribute: firmware_attribute.attribute.clone(),
                performance_profile: firmware_attribute.performance_profile.clone(),
            })
        }
        TdpLimitingMethod::AmdgpuHwmon => Box::new(AmdgpuHwmonTdpLimitManager {}),
        TdpLimitingMethod::PowerStation => Box::new(PowerStationTdpLimitManager {
            gpu_card_path: StdMutex::new(None),
        }),
    })
}

pub(crate) struct TdpManagerService {
    proxy: RootManagerProxy<'static>,
    session: Connection,
    channel: UnboundedReceiver<TdpManagerCommand>,
    download_set: JoinSet<String>,
    download_handles: HashMap<String, u32>,
    download_mode_limit: Option<NonZeroU32>,
    previous_limit: Option<NonZeroU32>,
    manager: Box<dyn TdpLimitManager>,
}

pub(crate) enum TdpManagerCommand {
    SetTdpLimit(u32),
    GetTdpLimit(oneshot::Sender<Result<u32>>),
    GetTdpLimitRange(oneshot::Sender<Result<RangeInclusive<u32>>>),
    IsActive(oneshot::Sender<Result<bool>>),
    UpdateDownloadMode,
    EnterDownloadMode(String, oneshot::Sender<Result<Option<OwnedFd>>>),
    ListDownloadModeHandles(oneshot::Sender<HashMap<String, u32>>),
}

#[derive(Debug)]
pub(crate) enum SysfsWritten {
    Written(Result<()>),
    Superseded,
}

type SysfsQueue = (Vec<u8>, oneshot::Sender<SysfsWritten>);
type SysfsQueueMap = HashMap<PathBuf, SysfsQueue>;

#[derive(Debug)]
struct SysfsWriterQueue {
    values: Mutex<SysfsQueueMap>,
    notify: Notify,
}

impl SysfsWriterQueue {
    fn new() -> SysfsWriterQueue {
        SysfsWriterQueue {
            values: Mutex::new(HashMap::new()),
            notify: Notify::new(),
        }
    }

    async fn send(&self, path: PathBuf, contents: Vec<u8>) -> oneshot::Receiver<SysfsWritten> {
        let (tx, rx) = oneshot::channel();
        if let Some((_, old_tx)) = self.values.lock().await.insert(path, (contents, tx)) {
            let _ = old_tx.send(SysfsWritten::Superseded);
        }
        self.notify.notify_one();
        rx
    }

    async fn recv(&self) -> Option<(PathBuf, Vec<u8>, oneshot::Sender<SysfsWritten>)> {
        // Take an arbitrary file from the map
        self.notify.notified().await;
        let mut values = self.values.lock().await;
        if let Some(path) = values.keys().next().cloned() {
            values
                .remove_entry(&path)
                .map(|(path, (contents, tx))| (path, contents, tx))
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub(crate) struct SysfsWriterService {
    queue: Arc<SysfsWriterQueue>,
}

impl SysfsWriterService {
    pub fn init() -> Result<SysfsWriterService> {
        ensure!(!SYSFS_WRITER.initialized(), "sysfs writer already active");
        let queue = Arc::new(SysfsWriterQueue::new());
        SYSFS_WRITER.set(queue.clone())?;
        Ok(SysfsWriterService { queue })
    }
}

impl Service for SysfsWriterService {
    const NAME: &'static str = "sysfs-writer";

    async fn run(&mut self) -> Result<()> {
        loop {
            let Some((path, contents, tx)) = self.queue.recv().await else {
                continue;
            };
            let res = write_synced(path, &contents)
                .await
                .inspect_err(|message| error!("Error writing to sysfs file: {message}"));
            let _ = tx.send(SysfsWritten::Written(res));
        }
    }
}

async fn read_cpu_sysfs_contents<S: AsRef<Path>>(suffix: S) -> Result<String> {
    let base = path(CPU_PREFIX).join(CPUFREQ_PREFIX).join(CPU0_NAME);
    fs::read_to_string(base.join(suffix.as_ref()))
        .await
        .map_err(|message| anyhow!("Error opening sysfs file for reading {message}"))
}

async fn write_cpu_governor_sysfs_contents(contents: String) -> Result<()> {
    // Iterate over all policyX paths
    let mut dir = fs::read_dir(path(CPU_PREFIX).join(CPUFREQ_PREFIX)).await?;
    let mut wrote_stuff = false;
    loop {
        let Some(entry) = dir.next_entry().await? else {
            ensure!(
                wrote_stuff,
                "No data written, unable to find any policyX sysfs paths"
            );
            return Ok(());
        };
        let file_name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow!("Unable to convert path to string"))?;
        if !file_name.starts_with(CPU_POLICY_NAME) {
            continue;
        }
        let base = entry.path();
        // Write contents to each one
        wrote_stuff = true;
        write_synced(base.join(CPU_SCALING_GOVERNOR_SUFFIX), contents.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to sysfs file: {message}"))?;
    }
}

pub(crate) async fn get_available_cpu_scaling_governors() -> Result<Vec<CPUScalingGovernor>> {
    let contents = read_cpu_sysfs_contents(CPU_SCALING_AVAILABLE_GOVERNORS_SUFFIX).await?;
    // Get the list of supported governors from cpu0
    let mut result = Vec::new();

    let words = contents.split_whitespace();
    for word in words {
        match CPUScalingGovernor::from_str(word) {
            Ok(governor) => result.push(governor),
            Err(message) => warn!("Error parsing governor {message}"),
        }
    }

    Ok(result)
}

pub(crate) async fn get_cpu_scaling_governor() -> Result<CPUScalingGovernor> {
    // get the current governor from cpu0 (assume all others are the same)
    let contents = read_cpu_sysfs_contents(CPU_SCALING_GOVERNOR_SUFFIX).await?;

    let contents = contents.trim();
    CPUScalingGovernor::from_str(contents).map_err(|message| {
        anyhow!(
            "Error converting CPU scaling governor sysfs file contents to enumeration: {message}"
        )
    })
}

pub(crate) async fn set_cpu_scaling_governor(governor: CPUScalingGovernor) -> Result<()> {
    // Set the given governor on all cpus
    let name = governor.to_string();
    write_cpu_governor_sysfs_contents(name).await
}

async fn find_cpu_boost_driver() -> Result<(PathBuf, CpuBoostDriver)> {
    // Try cpufreq path first
    let cpufreq_path = path(CPU_PREFIX)
        .join(CPUFREQ_PREFIX)
        .join(CPUFREQ_BOOST_SUFFIX);
    if try_exists(&cpufreq_path).await? {
        return Ok((cpufreq_path, CpuBoostDriver::CpuFreq));
    }

    // Try intel_pstate path next
    let intel_pstate_path = path(CPU_PREFIX)
        .join(INTEL_PSTATE_PREFIX)
        .join(INTEL_PSTATE_NO_TURBO_SUFFIX);
    if try_exists(&intel_pstate_path).await? {
        return Ok((intel_pstate_path, CpuBoostDriver::IntelPstate));
    }

    bail!("Could not find CPU boost sysfs path");
}

pub(crate) async fn get_cpu_boost_state() -> Result<CPUBoostState> {
    let (path, driver) = find_cpu_boost_driver().await?;
    let contents = fs::read_to_string(&path)
        .await
        .map_err(|message| anyhow!("Error opening CPU boost sysfs file for reading: {message}"))?;
    match driver {
        CpuBoostDriver::CpuFreq => match contents.trim() {
            // cpufreq's boost property is standard
            // 1 means boost is enabled, 0 means boost is disabled
            "1" => Ok(CPUBoostState::Enabled),
            "0" => Ok(CPUBoostState::Disabled),
            _ => Err(anyhow!("Invalid cpufreq boost state: {contents}")),
        },
        CpuBoostDriver::IntelPstate => match contents.trim() {
            // intel_pstate's no_turbo property is inverted
            // 0 means boost is enabled, 1 means boost is disabled
            "0" => Ok(CPUBoostState::Enabled),
            "1" => Ok(CPUBoostState::Disabled),
            _ => Err(anyhow!("Invalid intel_pstate boost state: {contents}")),
        },
    }
}

pub(crate) async fn set_cpu_boost_state(state: CPUBoostState) -> Result<()> {
    let (path, driver) = find_cpu_boost_driver().await?;
    let contents = match (driver, state) {
        (CpuBoostDriver::CpuFreq, CPUBoostState::Enabled) => "1",
        (CpuBoostDriver::CpuFreq, CPUBoostState::Disabled) => "0",
        (CpuBoostDriver::IntelPstate, CPUBoostState::Enabled) => "0",
        (CpuBoostDriver::IntelPstate, CPUBoostState::Disabled) => "1",
    };
    write_synced(path, contents.as_bytes())
        .await
        .inspect_err(|message| error!("Error writing to CPU boost sysfs file: {message}"))
}

async fn find_sysdir(prefix: impl AsRef<Path>, expected: &str) -> Result<PathBuf> {
    let mut dir = fs::read_dir(prefix.as_ref()).await?;
    loop {
        let base = match dir.next_entry().await? {
            Some(entry) => entry.path(),
            None => bail!("prefix not found"),
        };
        let file_name = base.join("name");
        let name = fs::read_to_string(file_name.as_path())
            .await?
            .trim()
            .to_string();
        if name == expected {
            return Ok(base);
        }
    }
}

pub(crate) async fn find_hwmon(hwmon: &str) -> Result<PathBuf> {
    find_sysdir(path(HWMON_PREFIX), hwmon).await
}

async fn find_platform_profile(name: &str) -> Result<PathBuf> {
    find_sysdir(path(PLATFORM_PROFILE_PREFIX), name).await
}

#[async_trait]
impl TdpLimitManager for AmdgpuHwmonTdpLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        let base = find_hwmon(AMDGPU_HWMON_NAME).await?;
        let power1cap = fs::read_to_string(base.join(TDP_LIMIT1)).await?;
        let power1cap: u32 = power1cap.trim_end().parse()?;
        Ok(power1cap / 1_000_000)
    }

    async fn set_tdp_limit(&self, limit: u32) -> Result<()> {
        ensure!(
            self.get_tdp_limit_range().await?.contains(&limit),
            "Invalid limit"
        );

        let data = format!("{limit}000000");

        let base = find_hwmon(AMDGPU_HWMON_NAME).await?;
        write_synced(base.join(TDP_LIMIT1), data.as_bytes())
            .await
            .inspect_err(|message| {
                error!("Error opening sysfs power1_cap file for writing TDP limits {message}");
            })?;

        if let Ok(mut power2file) = File::create(base.join(TDP_LIMIT2)).await {
            power2file
                .write(data.as_bytes())
                .await
                .inspect_err(|message| error!("Error writing to power2_cap file: {message}"))?;
            power2file.flush().await?;
        }
        Ok(())
    }

    async fn get_tdp_limit_range(&self) -> Result<RangeInclusive<u32>> {
        let config = device_config().await?;
        let config = config
            .as_ref()
            .and_then(|config| config.tdp_limit.as_ref())
            .ok_or(anyhow!("No TDP limit configured"))?;

        if let Some(range) = config.range {
            return Ok(range.min..=range.max);
        }
        bail!("No TDP limit range configured");
    }
}

impl FirmwareAttributeLimitManager {
    const PREFIX: &str = "/sys/class/firmware-attributes";
    const SPL_SUFFIX: &str = "ppt_pl1_spl";
    const SPPT_SUFFIX: &str = "ppt_pl2_sppt";
    const FPPT_SUFFIX: &str = "ppt_pl3_fppt";
}

#[async_trait]
impl TdpLimitManager for FirmwareAttributeLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        ensure!(self.is_active().await?, "TDP limiting not active");
        let base = path(Self::PREFIX).join(&self.attribute).join("attributes");

        fs::read_to_string(base.join(Self::SPL_SUFFIX).join("current_value"))
            .await
            .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
            .trim()
            .parse()
            .map_err(|e| anyhow!("Error parsing value: {e}"))
    }

    async fn set_tdp_limit(&self, limit: u32) -> Result<()> {
        ensure!(self.is_active().await?, "TDP limiting not active");
        ensure!(
            self.get_tdp_limit_range().await?.contains(&limit),
            "Invalid limit"
        );

        let limit = limit.to_string();
        let base = path(Self::PREFIX).join(&self.attribute).join("attributes");
        write_synced(
            base.join(Self::SPL_SUFFIX).join("current_value"),
            limit.as_bytes(),
        )
        .await
        .inspect_err(|message| error!("Error writing to sysfs file: {message}"))?;
        write_synced(
            base.join(Self::SPPT_SUFFIX).join("current_value"),
            limit.as_bytes(),
        )
        .await
        .inspect_err(|message| error!("Error writing to sysfs file: {message}"))?;
        write_synced(
            base.join(Self::FPPT_SUFFIX).join("current_value"),
            limit.as_bytes(),
        )
        .await
        .inspect_err(|message| error!("Error writing to sysfs file: {message}"))
    }

    async fn get_tdp_limit_range(&self) -> Result<RangeInclusive<u32>> {
        let base = path(Self::PREFIX)
            .join(&self.attribute)
            .join("attributes")
            .join(Self::SPL_SUFFIX);

        let min: u32 = fs::read_to_string(base.join("min_value"))
            .await
            .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
            .trim()
            .parse()
            .map_err(|e| anyhow!("Error parsing value: {e}"))?;
        let max: u32 = fs::read_to_string(base.join("max_value"))
            .await
            .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
            .trim()
            .parse()
            .map_err(|e| anyhow!("Error parsing value: {e}"))?;
        Ok(min..=max)
    }

    async fn is_active(&self) -> Result<bool> {
        let Some(ref performance_profile) = self.performance_profile else {
            return Ok(true);
        };
        let config = device_config().await?;
        if let Some(config) = config
            .as_ref()
            .and_then(|config| config.performance_profile.as_ref())
        {
            Ok(get_platform_profile(&config.platform_profile_name).await? == *performance_profile)
        } else {
            Ok(true)
        }
    }
}

impl PowerStationTdpLimitManager {
    // PowerStation DBus service information
    const DBUS_SERVICE_NAME: &str = "org.shadowblip.PowerStation";
    const DBUS_BASE_PATH: &str = "/org/shadowblip/Performance/GPU";
    const DBUS_CARD_PREFIX: &str = "card";
    const DBUS_GPU_INTERFACE: &str = "org.shadowblip.GPU";
    const DBUS_GPU_CARD_INTERFACE: &str = "org.shadowblip.GPU.Card";
    const DBUS_TDP_INTERFACE: &str = "org.shadowblip.GPU.Card.TDP";
    const DBUS_INTERFACE_PROPERTIES: &str = "org.freedesktop.DBus.Properties";

    // Find appropriate GPU card path from PowerStation DBus service
    // Uses cached path if available, otherwise performs lookup
    async fn find_gpu_card_path(&self) -> Result<String> {
        // Check if we have a cached path
        if let Some(path) = self.gpu_card_path.lock().unwrap().clone() {
            return Ok(path);
        }

        // Connect to the system DBus
        let connection = Connection::system().await?;

        // Check if PowerStation service is available
        if !self.check_dbus_service(&connection).await? {
            bail!("PowerStation DBus service is not running");
        }

        // Query available GPU cards
        let cards = self.query_gpu_cards(&connection).await?;
        if cards.is_empty() {
            bail!("No GPU cards found in PowerStation");
        }

        // Select the appropriate card (prefer discrete GPU if available)
        let path = self.select_appropriate_card(&connection, cards).await?;

        // Cache the path for future use
        *self.gpu_card_path.lock().unwrap() = Some(path.clone());

        Ok(path)
    }

    // Check if PowerStation DBus service is running
    async fn check_dbus_service(&self, connection: &zbus::Connection) -> Result<bool> {
        use zbus::fdo;
        let proxy = fdo::DBusProxy::new(&connection).await?;
        let names = proxy.list_names().await?;
        Ok(names
            .iter()
            .any(|name| name.as_str() == Self::DBUS_SERVICE_NAME))
    }

    // Query all available GPU cards from PowerStation
    async fn query_gpu_cards(&self, connection: &zbus::Connection) -> Result<Vec<String>> {
        let gpu_path = format!("{}", Self::DBUS_BASE_PATH);

        // Get a proxy to the GPU interface
        let path = ObjectPath::try_from(gpu_path.as_str())?;
        let proxy = zbus::Proxy::new(
            connection,
            Self::DBUS_SERVICE_NAME,
            path,
            Self::DBUS_GPU_INTERFACE,
        )
        .await?;

        // Call EnumerateCards method to get all GPU cards
        let cards: Vec<OwnedObjectPath> = proxy
            .call::<_, _, Vec<OwnedObjectPath>>("EnumerateCards", &())
            .await
            .inspect_err(|message| error!("Error calling EnumerateCards: {message}"))?;

        // Extract card names from paths
        let cards: Vec<String> = cards
            .into_iter()
            .filter_map(|path| {
                let path_str = path.to_string();
                path_str
                    .split('/')
                    .last()
                    .filter(|s| s.starts_with(Self::DBUS_CARD_PREFIX))
                    .map(String::from)
            })
            .collect();

        Ok(cards)
    }

    // Select the appropriate GPU card, preferring discrete GPU over integrated
    async fn select_appropriate_card(
        &self,
        connection: &zbus::Connection,
        cards: Vec<String>,
    ) -> Result<String> {
        for card in &cards {
            let card_path = format!("{}/{}", Self::DBUS_BASE_PATH, card);

            // Get a proxy to the Properties interface
            let path = ObjectPath::try_from(card_path.as_str())?;
            let proxy = zbus::Proxy::new(
                connection,
                Self::DBUS_SERVICE_NAME,
                path,
                Self::DBUS_INTERFACE_PROPERTIES,
            )
            .await?;

            // Get the Class property
            let class = proxy
                .call::<_, _, OwnedValue>("Get", &(Self::DBUS_GPU_CARD_INTERFACE, "Class"))
                .await
                .inspect_err(|message| error!("Error calling Get: {message}"))?;

            let class: String = class.try_into()?;

            // If we find a discrete GPU, use it
            if class == "discrete" {
                return Ok(card_path);
            }
        }

        // If no discrete GPU found, use the first card
        if let Some(card) = cards.first() {
            return Ok(format!("{}/{}", Self::DBUS_BASE_PATH, card));
        }

        bail!("No suitable GPU card found")
    }

    // Set the TDP boost property
    // Helper method to set any TDP-related property on PowerStation
    async fn set_tdp_property(
        &self,
        interface: &str,
        property_name: &str,
        value: u32,
    ) -> Result<()> {
        debug!("Setting PowerStation TDP property {property_name} to {value}");

        // Connect to system DBus
        let connection = Connection::system().await?;

        // Find the GPU card path
        let card_path = self.find_gpu_card_path().await?;

        // Get a proxy to set the property
        let path = ObjectPath::try_from(card_path.as_str())?;
        let proxy = zbus::Proxy::new(
            &connection,
            Self::DBUS_SERVICE_NAME,
            path,
            Self::DBUS_INTERFACE_PROPERTIES,
        )
        .await
        .inspect_err(|message| error!("Error creating proxy: {message}"))?;

        // Set the property (convert u32 to double)
        proxy
            .call::<_, _, ()>(
                "Set",
                &(
                    interface,
                    property_name,
                    zbus::zvariant::Value::from(value as f64),
                ),
            )
            .await
            .inspect_err(|message| error!("Error calling Set: {message}"))?;

        Ok(())
    }

    // Helper method to get property values from D-Bus
    async fn get_tdp_property(&self, property_name: &str, interface: &str) -> Result<f64> {
        // Connect to system DBus
        let connection = Connection::system().await?;

        // Find the GPU card path
        let card_path = self
            .find_gpu_card_path()
            .await
            .inspect_err(|message| error!("Error finding GPU card path: {message}"))?;

        // Get a proxy to query property
        let path = ObjectPath::try_from(card_path.as_str())?;
        let proxy = zbus::Proxy::new(
            &connection,
            Self::DBUS_SERVICE_NAME,
            path,
            Self::DBUS_INTERFACE_PROPERTIES,
        )
        .await
        .inspect_err(|message| error!("Error creating proxy: {message}"))?;

        // Get the property
        let value = proxy
            .call::<_, _, OwnedValue>("Get", &(interface, property_name))
            .await
            .inspect_err(|message| error!("Error calling Get for {property_name}: {message}"))?;

        // Convert to f64
        let value: f64 = value.try_into()?;
        Ok(value)
    }

    async fn get_min_tdp(&self) -> Result<u32> {
        let min_tdp = self
            .get_tdp_property("MinTdp", Self::DBUS_TDP_INTERFACE)
            .await?;
        Ok(min_tdp.round() as u32)
    }

    async fn get_max_tdp(&self) -> Result<u32> {
        let max_tdp = self
            .get_tdp_property("MaxTdp", Self::DBUS_TDP_INTERFACE)
            .await?;
        Ok(max_tdp.round() as u32)
    }

    // async fn get_max_boost(&self) -> Result<u32> {
    //     let max_boost = self
    //         .get_tdp_property("MaxBoost", Self::DBUS_TDP_INTERFACE)
    //         .await?;
    //     Ok(max_boost.round() as u32)
    // }

    // async fn set_tdp_boost(&self, boost: u32) -> Result<()> {
    //     self.set_tdp_property(Self::DBUS_TDP_INTERFACE, "Boost", boost)
    //         .await
    // }
}

#[async_trait]
impl TdpLimitManager for PowerStationTdpLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        debug!("Getting PowerStation TDP limit");
        let tdp_value = self
            .get_tdp_property("TDP", Self::DBUS_TDP_INTERFACE)
            .await?;
        Ok(tdp_value.round() as u32)
    }

    async fn set_tdp_limit(&self, limit: u32) -> Result<()> {
        debug!("Setting PowerStation TDP limit to {limit}");
        // Check if limit is within range
        let range = self.get_tdp_limit_range().await?;
        ensure!(
            range.contains(&limit),
            "TDP limit {} is outside allowed range {:?}",
            limit,
            range
        );

        // Use the common method to set the TDP property
        self.set_tdp_property(Self::DBUS_TDP_INTERFACE, "TDP", limit)
            .await
    }

    // Implementation of the required get_tdp_limit_range method
    async fn get_tdp_limit_range(&self) -> Result<RangeInclusive<u32>> {
        let min_tdp = self
            .get_min_tdp()
            .await
            .map_err(|e| anyhow!("Error getting min TDP: {e}"))?;
        let max_tdp = self
            .get_max_tdp()
            .await
            .map_err(|e| anyhow!("Error getting max TDP: {e}"))?;
        Ok(min_tdp..=max_tdp)
    }

    async fn is_active(&self) -> Result<bool> {
        // Only check if PowerStation DBus service is running
        let connection = Connection::system().await?;
        self.check_dbus_service(&connection).await
    }
}

pub(crate) async fn get_max_charge_level() -> Result<i32> {
    let config = device_config().await?;
    let config = config
        .as_ref()
        .and_then(|config| config.battery_charge_limit.as_ref())
        .ok_or(anyhow!("No battery charge limit configured"))?;
    let base = find_hwmon(config.hwmon_name.as_str()).await?;

    fs::read_to_string(base.join(config.attribute.as_str()))
        .await
        .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
        .trim()
        .parse()
        .map_err(|e| anyhow!("Error parsing value: {e}"))
}

pub(crate) async fn set_max_charge_level(limit: i32) -> Result<oneshot::Receiver<SysfsWritten>> {
    ensure!((0..=100).contains(&limit), "Invalid limit");
    let data = limit.to_string();
    let config = device_config().await?;
    let config = config
        .as_ref()
        .and_then(|config| config.battery_charge_limit.as_ref())
        .ok_or(anyhow!("No battery charge limit configured"))?;
    let base = find_hwmon(config.hwmon_name.as_str()).await?;

    Ok(SYSFS_WRITER
        .get()
        .ok_or(anyhow!("sysfs writer not running"))?
        .send(
            base.join(config.attribute.clone()),
            data.as_bytes().to_owned(),
        )
        .await)
}

pub(crate) async fn get_available_platform_profiles(name: &str) -> Result<Vec<String>> {
    let base = find_platform_profile(name).await?;
    Ok(fs::read_to_string(base.join("choices"))
        .await
        .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
        .trim()
        .split(' ')
        .map(ToString::to_string)
        .collect())
}

pub(crate) async fn get_platform_profile(name: &str) -> Result<String> {
    let base = find_platform_profile(name).await?;
    Ok(fs::read_to_string(base.join("profile"))
        .await
        .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
        .trim()
        .to_string())
}

pub(crate) async fn set_platform_profile(name: &str, profile: &str) -> Result<()> {
    let base = find_platform_profile(name).await?;
    fs::write(base.join("profile"), profile.as_bytes())
        .await
        .map_err(|message| anyhow!("Error writing to sysfs: {message}"))
}

impl TdpManagerService {
    pub async fn new(
        channel: UnboundedReceiver<TdpManagerCommand>,
        system: &Connection,
        session: &Connection,
    ) -> Result<TdpManagerService> {
        let config = device_config().await?;
        let config = config
            .as_ref()
            .and_then(|config| config.tdp_limit.as_ref())
            .ok_or(anyhow!("No TDP limit configured"))?;

        let manager = tdp_limit_manager().await?;
        let proxy = RootManagerProxy::new(system).await?;

        Ok(TdpManagerService {
            proxy,
            session: session.clone(),
            channel,
            download_set: JoinSet::new(),
            download_handles: HashMap::new(),
            previous_limit: None,
            download_mode_limit: config.download_mode_limit,
            manager,
        })
    }

    async fn update_download_mode(&mut self) -> Result<()> {
        if !self.manager.is_active().await? {
            return Ok(());
        }

        let Some(download_mode_limit) = self.download_mode_limit else {
            return Ok(());
        };

        let Some(current_limit) = NonZeroU32::new(self.manager.get_tdp_limit().await?) else {
            // If current_limit is 0 then the interface is broken, likely because TDP limiting
            // isn't possible with the current power profile or system, so we should just ignore
            // it for now.
            return Ok(());
        };

        if self.download_handles.is_empty() {
            if let Some(previous_limit) = self.previous_limit {
                debug!("Leaving download mode, setting TDP to {previous_limit}");
                self.set_tdp_limit(previous_limit.get()).await?;
                self.previous_limit = None;
            }
        } else {
            if self.previous_limit.is_none() {
                debug!("Entering download mode, caching TDP limit of {current_limit}");
                self.previous_limit = Some(current_limit);
            }
            if current_limit != download_mode_limit {
                self.set_tdp_limit(download_mode_limit.get()).await?;
            }
        }

        Ok(())
    }

    async fn get_download_mode_handle(
        &mut self,
        identifier: impl AsRef<str>,
    ) -> Result<Option<OwnedFd>> {
        if self.download_mode_limit.is_none() {
            return Ok(None);
        }
        let (send, recv) = pipe::pipe()?;
        let identifier = identifier.as_ref().to_string();
        self.download_handles
            .entry(identifier.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        self.download_set
            .spawn(TdpManagerService::wait_on_handle(recv, identifier));
        self.update_download_mode().await?;
        Ok(Some(send.into_blocking_fd()?))
    }

    async fn wait_on_handle(recv: pipe::Receiver, identifier: String) -> String {
        loop {
            let mut buf = [0; 1024];
            let read = match recv.ready(Interest::READABLE).await {
                Ok(r) if r.is_read_closed() => break,
                Ok(r) if r.is_readable() => recv.try_read(&mut buf),
                Err(e) => Err(e),
                Ok(e) => {
                    warn!("Download mode handle received unexpected event: {e:?}");
                    break;
                }
            };
            if let Err(e) = read {
                warn!("Download mode handle received unexpected error: {e:?}");
                break;
            }
        }
        identifier
    }

    async fn set_tdp_limit(&self, limit: u32) -> Result<()> {
        self.proxy
            .set_tdp_limit(limit)
            .await
            .inspect_err(|e| error!("Failed to set TDP limit: {e}"))?;

        if let Ok(interface) = self
            .session
            .object_server()
            .interface::<_, TdpLimit1>(MANAGER_PATH)
            .await
        {
            tokio::spawn(async move {
                let ctx = interface.signal_emitter();
                interface.get().await.tdp_limit_changed(ctx).await
            });
        }
        Ok(())
    }

    async fn handle_command(&mut self, command: TdpManagerCommand) -> Result<()> {
        match command {
            TdpManagerCommand::SetTdpLimit(limit) => {
                if self.download_handles.is_empty() {
                    self.set_tdp_limit(limit).await?;
                }
            }
            TdpManagerCommand::GetTdpLimit(reply) => {
                let _ = reply.send(self.manager.get_tdp_limit().await);
            }
            TdpManagerCommand::GetTdpLimitRange(reply) => {
                let _ = reply.send(self.manager.get_tdp_limit_range().await);
            }
            TdpManagerCommand::IsActive(reply) => {
                let _ = reply.send(self.manager.is_active().await);
            }
            TdpManagerCommand::UpdateDownloadMode => {
                self.update_download_mode().await?;
            }
            TdpManagerCommand::EnterDownloadMode(identifier, reply) => {
                let fd = self.get_download_mode_handle(identifier).await;
                let _ = reply.send(fd);
            }
            TdpManagerCommand::ListDownloadModeHandles(reply) => {
                let _ = reply.send(self.download_handles.clone());
            }
        }
        Ok(())
    }
}

impl Service for TdpManagerService {
    const NAME: &'static str = "tdp-manager";

    async fn run(&mut self) -> Result<()> {
        loop {
            if self.download_set.is_empty() {
                let message = match self.channel.recv().await {
                    None => bail!("TDP manager service channel broke"),
                    Some(message) => message,
                };
                let _ = self
                    .handle_command(message)
                    .await
                    .inspect_err(|e| error!("Failed to handle command: {e}"));
            } else {
                tokio::select! {
                    message = self.channel.recv() => {
                        let message = match message {
                            None => bail!("TDP manager service channel broke"),
                            Some(message) => message,
                        };
                        let _ = self.handle_command(message)
                            .await
                            .inspect_err(|e| error!("Failed to handle command: {e}"));
                    },
                    identifier = self.download_set.join_next() => {
                        match identifier {
                            None => (),
                            Some(Ok(identifier)) => {
                                match self.download_handles.entry(identifier) {
                                    Entry::Occupied(e) if e.get() == &1 => {
                                        e.remove();
                                        if self.download_handles.is_empty() {
                                            if let Err(e) = self.update_download_mode().await {
                                                error!("Failed to update download mode: {e}");
                                            }
                                        }
                                    },
                                    Entry::Occupied(mut e) => *e.get_mut() -= 1,
                                    Entry::Vacant(_) => (),
                                }
                            }
                            Some(Err(e)) => warn!("Failed to get closed download mode handle: {e}"),
                        }
                    },
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::error::to_zbus_fdo_error;
    use crate::hardware::{
        BatteryChargeLimitConfig, DeviceConfig, FirmwareAttributeConfig, PerformanceProfileConfig,
        RangeConfig, TdpLimitConfig,
    };
    use crate::{enum_on_off, enum_roundtrip, testing};
    use anyhow::anyhow;
    use std::time::Duration;
    use tokio::fs::{create_dir_all, read_to_string, remove_dir, write};
    use tokio::sync::mpsc::{channel, unbounded_channel, Sender};
    use tokio::time::sleep;
    use zbus::{fdo, interface};

    async fn setup() -> Result<()> {
        // Use hwmon5 just as a test. We needed a subfolder of HWMON_PREFIX
        // and this is as good as any.
        let base = path(HWMON_PREFIX).join("hwmon5");
        let filename = base.join("device");
        // Creates hwmon path, including device subpath
        create_dir_all(filename).await?;
        // Writes name file as addgpu so find_hwmon() will find it.
        write_synced(base.join("name"), AMDGPU_HWMON_NAME.as_bytes()).await?;
        Ok(())
    }

    pub async fn create_nodes() -> Result<()> {
        setup().await?;
        let base = path(CPU_PREFIX);
        let cpufreq_base = base.join(CPUFREQ_PREFIX);
        create_dir_all(&cpufreq_base).await?;
        write(cpufreq_base.join(CPUFREQ_BOOST_SUFFIX), b"1\n").await?;

        let base = find_hwmon(AMDGPU_HWMON_NAME).await?;

        let filename = base.join(TDP_LIMIT1);
        write(filename.as_path(), "15000000\n").await?;

        let base = path(HWMON_PREFIX).join("hwmon6");
        create_dir_all(&base).await?;

        write(base.join("name"), "steamdeck_hwmon\n").await?;

        write(base.join("max_battery_charge_level"), "10\n").await?;

        let base = path(PLATFORM_PROFILE_PREFIX).join("platform-profile0");
        create_dir_all(&base).await?;
        write_synced(base.join("name"), b"power-driver\n").await?;
        write_synced(base.join("choices"), b"a b c\n").await?;

        Ok(())
    }

    #[test]
    fn cpu_governor_roundtrip() {
        enum_roundtrip!(CPUScalingGovernor {
            "conservative": str = Conservative,
            "ondemand": str = OnDemand,
            "userspace": str = UserSpace,
            "powersave": str = PowerSave,
            "performance": str = Performance,
            "schedutil": str = SchedUtil,
        });
        assert!(CPUScalingGovernor::from_str("usersave").is_err());
    }

    #[tokio::test]
    async fn read_cpu_available_governors() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(CPU0_NAME);
        create_dir_all(&base).await.expect("create_dir_all");

        let contents = "conservative ondemand userspace powersave performance schedutil";
        write(base.join(CPU_SCALING_AVAILABLE_GOVERNORS_SUFFIX), contents)
            .await
            .expect("write");

        assert_eq!(
            get_available_cpu_scaling_governors().await.unwrap(),
            vec![
                CPUScalingGovernor::Conservative,
                CPUScalingGovernor::OnDemand,
                CPUScalingGovernor::UserSpace,
                CPUScalingGovernor::PowerSave,
                CPUScalingGovernor::Performance,
                CPUScalingGovernor::SchedUtil
            ]
        );
    }

    #[tokio::test]
    async fn read_invalid_cpu_available_governors() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(CPU0_NAME);
        create_dir_all(&base).await.expect("create_dir_all");

        let contents =
            "conservative ondemand userspace rescascade powersave performance schedutil\n";
        write(base.join(CPU_SCALING_AVAILABLE_GOVERNORS_SUFFIX), contents)
            .await
            .expect("write");

        assert_eq!(
            get_available_cpu_scaling_governors().await.unwrap(),
            vec![
                CPUScalingGovernor::Conservative,
                CPUScalingGovernor::OnDemand,
                CPUScalingGovernor::UserSpace,
                CPUScalingGovernor::PowerSave,
                CPUScalingGovernor::Performance,
                CPUScalingGovernor::SchedUtil
            ]
        );
    }

    #[tokio::test]
    async fn read_cpu_governor() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(CPU0_NAME);
        create_dir_all(&base).await.expect("create_dir_all");

        let contents = "ondemand\n";
        write(base.join(CPU_SCALING_GOVERNOR_SUFFIX), contents)
            .await
            .expect("write");

        assert_eq!(
            get_cpu_scaling_governor().await.unwrap(),
            CPUScalingGovernor::OnDemand
        );
    }

    #[tokio::test]
    async fn read_invalid_cpu_governor() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(CPU0_NAME);
        create_dir_all(&base).await.expect("create_dir_all");

        let contents = "rescascade\n";
        write(base.join(CPU_SCALING_GOVERNOR_SUFFIX), contents)
            .await
            .expect("write");

        assert!(get_cpu_scaling_governor().await.is_err());
    }

    #[tokio::test]
    async fn test_gpu_hwmon_get_tdp_limit() {
        let handle = testing::start();

        let mut config = DeviceConfig::default();
        config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::AmdgpuHwmon,
            range: Some(RangeConfig { min: 3, max: 15 }),
            download_mode_limit: None,
            firmware_attribute: None,
        });
        handle.test.device_config.replace(Some(config));
        let manager = tdp_limit_manager().await.unwrap();

        setup().await.expect("setup");
        let hwmon = path(HWMON_PREFIX);

        assert!(manager.get_tdp_limit().await.is_err());

        write(hwmon.join("hwmon5").join(TDP_LIMIT1), "15000000\n")
            .await
            .expect("write");
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);
    }

    #[tokio::test]
    async fn test_gpu_hwmon_set_tdp_limit() {
        let handle = testing::start();

        let mut config = DeviceConfig::default();
        config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::AmdgpuHwmon,
            range: Some(RangeConfig { min: 3, max: 15 }),
            download_mode_limit: None,
            firmware_attribute: None,
        });
        handle.test.device_config.replace(Some(config));
        let manager = tdp_limit_manager().await.unwrap();

        assert_eq!(
            manager.set_tdp_limit(2).await.unwrap_err().to_string(),
            anyhow!("Invalid limit").to_string()
        );
        assert_eq!(
            manager.set_tdp_limit(20).await.unwrap_err().to_string(),
            anyhow!("Invalid limit").to_string()
        );
        assert!(manager.set_tdp_limit(10).await.is_err());

        let hwmon = path(HWMON_PREFIX);
        assert_eq!(
            manager.set_tdp_limit(10).await.unwrap_err().to_string(),
            anyhow!("No such file or directory (os error 2)").to_string()
        );

        setup().await.expect("setup");
        let hwmon = hwmon.join("hwmon5");
        create_dir_all(hwmon.join(TDP_LIMIT1))
            .await
            .expect("create_dir_all");
        create_dir_all(hwmon.join(TDP_LIMIT2))
            .await
            .expect("create_dir_all");
        assert_eq!(
            manager.set_tdp_limit(10).await.unwrap_err().to_string(),
            anyhow!("Is a directory (os error 21)").to_string()
        );

        remove_dir(hwmon.join(TDP_LIMIT1))
            .await
            .expect("remove_dir");
        write(hwmon.join(TDP_LIMIT1), "0").await.expect("write");
        assert!(manager.set_tdp_limit(10).await.is_ok());
        let power1_cap = read_to_string(hwmon.join(TDP_LIMIT1))
            .await
            .expect("power1_cap");
        assert_eq!(power1_cap, "10000000");

        remove_dir(hwmon.join(TDP_LIMIT2))
            .await
            .expect("remove_dir");
        write(hwmon.join(TDP_LIMIT2), "0").await.expect("write");
        assert!(manager.set_tdp_limit(15).await.is_ok());
        let power1_cap = read_to_string(hwmon.join(TDP_LIMIT1))
            .await
            .expect("power1_cap");
        assert_eq!(power1_cap, "15000000");
        let power2_cap = read_to_string(hwmon.join(TDP_LIMIT2))
            .await
            .expect("power2_cap");
        assert_eq!(power2_cap, "15000000");
    }

    #[test]
    fn cpu_boost_state_roundtrip() {
        enum_roundtrip!(CPUBoostState {
            0: u32 = Disabled,
            1: u32 = Enabled,
            "disabled": str = Disabled,
            "enabled": str = Enabled,
        });
        enum_on_off!(CPUBoostState => (Enabled, Disabled));
        assert!(CPUBoostState::try_from(2).is_err());
        assert!(CPUBoostState::from_str("enabld").is_err());
    }
    #[tokio::test]
    async fn read_cpu_boost_state_cpufreq() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(CPUFREQ_PREFIX);
        let boost_path = base.join(CPUFREQ_BOOST_SUFFIX);
        create_dir_all(boost_path.parent().unwrap())
            .await
            .expect("create_dir_all");

        write(&boost_path, b"1\n").await.expect("write");
        assert_eq!(get_cpu_boost_state().await.unwrap(), CPUBoostState::Enabled);

        write(&boost_path, b"0\n").await.expect("write");
        assert_eq!(
            get_cpu_boost_state().await.unwrap(),
            CPUBoostState::Disabled
        );
    }

    #[tokio::test]
    async fn read_invalid_cpu_boost_state_cpufreq() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(CPUFREQ_PREFIX);
        let boost_path = base.join(CPUFREQ_BOOST_SUFFIX);
        create_dir_all(boost_path.parent().unwrap())
            .await
            .expect("create_dir_all");

        write(&boost_path, b"2\n").await.expect("write");
        assert!(get_cpu_boost_state().await.is_err());

        tokio::fs::remove_file(&boost_path)
            .await
            .expect("remove_file");
        assert!(get_cpu_boost_state().await.is_err());
    }

    #[tokio::test]
    async fn read_cpu_boost_state_intel_pstate() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(INTEL_PSTATE_PREFIX);
        let no_turbo_path = base.join(INTEL_PSTATE_NO_TURBO_SUFFIX);
        create_dir_all(no_turbo_path.parent().unwrap())
            .await
            .expect("create_dir_all");

        write(&no_turbo_path, b"0\n").await.expect("write");
        assert_eq!(get_cpu_boost_state().await.unwrap(), CPUBoostState::Enabled);

        write(&no_turbo_path, b"1\n").await.expect("write");
        assert_eq!(
            get_cpu_boost_state().await.unwrap(),
            CPUBoostState::Disabled
        );
    }

    #[tokio::test]
    async fn read_invalid_cpu_boost_state_intel_pstate() {
        let _h = testing::start();

        let base = path(CPU_PREFIX).join(INTEL_PSTATE_PREFIX);
        let no_turbo_path = base.join(INTEL_PSTATE_NO_TURBO_SUFFIX);
        create_dir_all(no_turbo_path.parent().unwrap())
            .await
            .expect("create_dir_all");

        write(&no_turbo_path, b"2\n").await.expect("write");
        assert!(get_cpu_boost_state().await.is_err());

        tokio::fs::remove_file(&no_turbo_path)
            .await
            .expect("remove_file");
        assert!(get_cpu_boost_state().await.is_err());
    }

    #[tokio::test]
    async fn read_max_charge_level() {
        let handle = testing::start();

        let mut config = DeviceConfig::default();
        config.battery_charge_limit = Some(BatteryChargeLimitConfig {
            suggested_minimum_limit: Some(10),
            hwmon_name: String::from("steamdeck_hwmon"),
            attribute: String::from("max_battery_charge_level"),
        });
        handle.test.device_config.replace(Some(config));

        let base = path(HWMON_PREFIX).join("hwmon6");
        create_dir_all(&base).await.expect("create_dir_all");

        write(base.join("name"), "steamdeck_hwmon\n")
            .await
            .expect("write");

        assert_eq!(
            find_hwmon("steamdeck_hwmon").await.unwrap(),
            path(HWMON_PREFIX).join("hwmon6")
        );

        write(base.join("max_battery_charge_level"), "10\n")
            .await
            .expect("write");

        assert_eq!(get_max_charge_level().await.unwrap(), 10);

        write(base.join("max_battery_charge_level"), "99\n")
            .await
            .expect("write");

        assert_eq!(get_max_charge_level().await.unwrap(), 99);

        assert!(set_max_charge_level(101).await.is_err());
        assert!(set_max_charge_level(-1).await.is_err());
    }

    #[tokio::test]
    async fn read_available_performance_profiles() {
        let _h = testing::start();

        assert!(get_available_platform_profiles("power-driver")
            .await
            .is_err());

        let base = path(PLATFORM_PROFILE_PREFIX).join("platform-profile0");
        create_dir_all(&base).await.unwrap();
        assert!(get_available_platform_profiles("power-driver")
            .await
            .is_err());

        write_synced(base.join("name"), b"power-driver\n")
            .await
            .unwrap();
        assert!(get_available_platform_profiles("power-driver")
            .await
            .is_err());

        write_synced(base.join("choices"), b"a b c\n")
            .await
            .unwrap();
        assert_eq!(
            get_available_platform_profiles("power-driver")
                .await
                .unwrap(),
            &["a", "b", "c"]
        );
    }

    struct MockTdpLimit {
        queue: Sender<()>,
    }

    #[interface(name = "com.steampowered.SteamOSManager1.RootManager")]
    impl MockTdpLimit {
        async fn set_tdp_limit(&mut self, limit: u32) -> fdo::Result<()> {
            let hwmon = path(HWMON_PREFIX);
            write(
                hwmon.join("hwmon5").join(TDP_LIMIT1),
                format!("{limit}000000\n"),
            )
            .await
            .expect("write");
            self.queue.send(()).await.map_err(to_zbus_fdo_error)?;
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_low_power_lock() {
        let mut h = testing::start();
        setup().await.expect("setup");

        let connection = h.new_dbus().await.expect("new_dbus");
        let (tx, rx) = unbounded_channel();
        let (fin_tx, fin_rx) = oneshot::channel();
        let (start_tx, start_rx) = oneshot::channel();
        let (reply_tx, mut reply_rx) = channel(1);

        let iface = MockTdpLimit { queue: reply_tx };

        let mut config = DeviceConfig::default();
        config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::AmdgpuHwmon,
            range: Some(RangeConfig { min: 3, max: 15 }),
            download_mode_limit: NonZeroU32::new(6),
            firmware_attribute: None,
        });
        h.test.device_config.replace(Some(config));
        let manager = tdp_limit_manager().await.unwrap();

        connection
            .request_name("com.steampowered.SteamOSManager1")
            .await
            .expect("reserve_name");
        let object_server = connection.object_server();
        object_server
            .at("/com/steampowered/SteamOSManager1", iface)
            .await
            .expect("at");

        let mut service = TdpManagerService::new(rx, &connection, &connection)
            .await
            .expect("service");
        let task = tokio::spawn(async move {
            start_tx.send(()).unwrap();
            tokio::select! {
                r = service.run() => r,
                _ = fin_rx => Ok(()),
            }
        });
        start_rx.await.expect("start_rx");

        sleep(Duration::from_millis(1)).await;

        tx.send(TdpManagerCommand::SetTdpLimit(15)).unwrap();
        reply_rx.recv().await;
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);

        let (os_tx, os_rx) = oneshot::channel();
        tx.send(TdpManagerCommand::ListDownloadModeHandles(os_tx))
            .unwrap();
        assert!(os_rx.await.unwrap().is_empty());

        let (h_tx, h_rx) = oneshot::channel();
        tx.send(TdpManagerCommand::EnterDownloadMode(
            String::from("test"),
            h_tx,
        ))
        .unwrap();

        {
            let _h = h_rx.await.unwrap().expect("result").expect("handle");
            reply_rx.recv().await;
            assert_eq!(manager.get_tdp_limit().await.unwrap(), 6);

            let (os_tx, os_rx) = oneshot::channel();
            tx.send(TdpManagerCommand::ListDownloadModeHandles(os_tx))
                .unwrap();
            assert_eq!(os_rx.await.unwrap(), [(String::from("test"), 1u32)].into());

            tx.send(TdpManagerCommand::SetTdpLimit(15)).unwrap();
            assert!(tokio::select! {
                _ = reply_rx.recv() => false,
                _ = sleep(Duration::from_millis(2)) => true,
            });
            assert_eq!(manager.get_tdp_limit().await.unwrap(), 6);
        }
        reply_rx.recv().await;
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);

        tx.send(TdpManagerCommand::SetTdpLimit(12)).unwrap();
        reply_rx.recv().await;
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 12);

        let (os_tx, os_rx) = oneshot::channel();
        tx.send(TdpManagerCommand::ListDownloadModeHandles(os_tx))
            .unwrap();
        assert!(os_rx.await.unwrap().is_empty());

        fin_tx.send(()).expect("fin");
        task.await.expect("exit").expect("exit2");
    }

    #[tokio::test]
    async fn test_disabled_low_power_lock() {
        let mut h = testing::start();
        setup().await.expect("setup");

        let connection = h.new_dbus().await.expect("new_dbus");
        let (tx, rx) = unbounded_channel();
        let (fin_tx, fin_rx) = oneshot::channel();
        let (start_tx, start_rx) = oneshot::channel();
        let (reply_tx, mut reply_rx) = channel(1);

        let iface = MockTdpLimit { queue: reply_tx };

        let mut config = DeviceConfig::default();
        config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::AmdgpuHwmon,
            range: Some(RangeConfig { min: 3, max: 15 }),
            download_mode_limit: None,
            firmware_attribute: None,
        });
        h.test.device_config.replace(Some(config));
        let manager = tdp_limit_manager().await.unwrap();

        connection
            .request_name("com.steampowered.SteamOSManager1")
            .await
            .expect("reserve_name");
        let object_server = connection.object_server();
        object_server
            .at("/com/steampowered/SteamOSManager1", iface)
            .await
            .expect("at");

        let mut service = TdpManagerService::new(rx, &connection, &connection)
            .await
            .expect("service");
        let task = tokio::spawn(async move {
            start_tx.send(()).unwrap();
            tokio::select! {
                r = service.run() => r,
                _ = fin_rx => Ok(()),
            }
        });
        start_rx.await.expect("start_rx");

        sleep(Duration::from_millis(1)).await;

        tx.send(TdpManagerCommand::SetTdpLimit(15)).unwrap();
        reply_rx.recv().await;
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);

        let (os_tx, os_rx) = oneshot::channel();
        tx.send(TdpManagerCommand::ListDownloadModeHandles(os_tx))
            .unwrap();
        assert!(os_rx.await.unwrap().is_empty());

        let (h_tx, h_rx) = oneshot::channel();
        tx.send(TdpManagerCommand::EnterDownloadMode(
            String::from("test"),
            h_tx,
        ))
        .unwrap();

        let h = h_rx.await.unwrap().expect("result");
        assert!(h.is_none());

        fin_tx.send(()).expect("fin");
        task.await.expect("exit").expect("exit2");
    }

    #[tokio::test]
    async fn test_firmware_attribute_tdp_limiter() {
        let h = testing::start();
        setup().await.expect("setup");

        let mut config = DeviceConfig::default();
        config.performance_profile = Some(PerformanceProfileConfig {
            platform_profile_name: String::from("platform-profile0"),
            suggested_default: String::from("custom"),
        });
        config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::FirmwareAttribute,
            range: Some(RangeConfig { min: 3, max: 15 }),
            download_mode_limit: None,
            firmware_attribute: Some(FirmwareAttributeConfig {
                attribute: String::from("tdp0"),
                performance_profile: Some(String::from("custom")),
            }),
        });
        h.test.device_config.replace(Some(config));

        let attributes_base = path(FirmwareAttributeLimitManager::PREFIX)
            .join("tdp0")
            .join("attributes");
        let spl_base = attributes_base.join(FirmwareAttributeLimitManager::SPL_SUFFIX);
        let sppt_base = attributes_base.join(FirmwareAttributeLimitManager::SPPT_SUFFIX);
        let fppt_base = attributes_base.join(FirmwareAttributeLimitManager::FPPT_SUFFIX);
        create_dir_all(&spl_base).await.unwrap();
        write_synced(spl_base.join("current_value"), b"10\n")
            .await
            .unwrap();
        create_dir_all(&sppt_base).await.unwrap();
        write_synced(sppt_base.join("current_value"), b"10\n")
            .await
            .unwrap();
        create_dir_all(&fppt_base).await.unwrap();
        write_synced(fppt_base.join("current_value"), b"10\n")
            .await
            .unwrap();

        write_synced(spl_base.join("min_value"), b"6\n")
            .await
            .unwrap();
        write_synced(spl_base.join("max_value"), b"20\n")
            .await
            .unwrap();

        let platform_profile_base = path(PLATFORM_PROFILE_PREFIX).join("platform-profile0");
        create_dir_all(&platform_profile_base).await.unwrap();
        write_synced(platform_profile_base.join("name"), b"platform-profile0\n")
            .await
            .unwrap();
        write_synced(platform_profile_base.join("profile"), b"custom\n")
            .await
            .unwrap();

        let manager = tdp_limit_manager().await.unwrap();

        assert_eq!(manager.is_active().await.unwrap(), true);
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 10);

        manager.set_tdp_limit(15).await.unwrap();
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);
        assert_eq!(
            read_to_string(spl_base.join("current_value"))
                .await
                .unwrap(),
            "15"
        );
        assert_eq!(
            read_to_string(sppt_base.join("current_value"))
                .await
                .unwrap(),
            "15"
        );
        assert_eq!(
            read_to_string(fppt_base.join("current_value"))
                .await
                .unwrap(),
            "15"
        );

        manager.set_tdp_limit(25).await.unwrap_err();
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);

        manager.set_tdp_limit(2).await.unwrap_err();
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);

        write_synced(platform_profile_base.join("profile"), b"balanced\n")
            .await
            .unwrap();

        manager.set_tdp_limit(10).await.unwrap_err();
    }

    #[tokio::test]
    async fn test_firmware_attribute_tdp_limiter_no_profile() {
        let h = testing::start();
        setup().await.expect("setup");

        let mut config = DeviceConfig::default();
        config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::FirmwareAttribute,
            range: Some(RangeConfig { min: 3, max: 15 }),
            download_mode_limit: None,
            firmware_attribute: Some(FirmwareAttributeConfig {
                attribute: String::from("tdp0"),
                performance_profile: None,
            }),
        });
        h.test.device_config.replace(Some(config));

        let attributes_base = path(FirmwareAttributeLimitManager::PREFIX)
            .join("tdp0")
            .join("attributes");
        let spl_base = attributes_base.join(FirmwareAttributeLimitManager::SPL_SUFFIX);
        let sppt_base = attributes_base.join(FirmwareAttributeLimitManager::SPPT_SUFFIX);
        let fppt_base = attributes_base.join(FirmwareAttributeLimitManager::FPPT_SUFFIX);
        create_dir_all(&spl_base).await.unwrap();
        write_synced(spl_base.join("current_value"), b"10\n")
            .await
            .unwrap();
        create_dir_all(&sppt_base).await.unwrap();
        write_synced(sppt_base.join("current_value"), b"10\n")
            .await
            .unwrap();
        create_dir_all(&fppt_base).await.unwrap();
        write_synced(fppt_base.join("current_value"), b"10\n")
            .await
            .unwrap();

        write_synced(spl_base.join("min_value"), b"6\n")
            .await
            .unwrap();
        write_synced(spl_base.join("max_value"), b"20\n")
            .await
            .unwrap();

        let manager = tdp_limit_manager().await.unwrap();

        assert_eq!(manager.is_active().await.unwrap(), true);
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 10);

        manager.set_tdp_limit(15).await.unwrap();
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);
        assert_eq!(
            read_to_string(spl_base.join("current_value"))
                .await
                .unwrap(),
            "15"
        );
        assert_eq!(
            read_to_string(sppt_base.join("current_value"))
                .await
                .unwrap(),
            "15"
        );
        assert_eq!(
            read_to_string(fppt_base.join("current_value"))
                .await
                .unwrap(),
            "15"
        );

        manager.set_tdp_limit(25).await.unwrap_err();
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);

        manager.set_tdp_limit(2).await.unwrap_err();
        assert_eq!(manager.get_tdp_limit().await.unwrap(), 15);
    }
}
