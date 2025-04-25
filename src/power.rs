/*
 * Copyright © 2023 Collabora Ltd.
 * Copyright © 2024 Valve Software
 *
 * SPDX-License-Identifier: MIT
 */

use anyhow::{anyhow, bail, ensure, Result};
use async_trait::async_trait;
use lazy_static::lazy_static;
use num_enum::TryFromPrimitive;
use regex::Regex;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex;
use strum::{Display, EnumString, VariantNames};
use tokio::fs::{self, try_exists, File};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, warn};
use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue};
use zbus::Connection;

use crate::hardware::{device_type, DeviceType};
use crate::platform::platform_config;
use crate::{path, write_synced};

const HWMON_PREFIX: &str = "/sys/class/hwmon";

const GPU_HWMON_NAME: &str = "amdgpu";

const CPU_PREFIX: &str = "/sys/devices/system/cpu/cpufreq";

const CPU0_NAME: &str = "policy0";
const CPU_POLICY_NAME: &str = "policy";

const GPU_POWER_PROFILE_SUFFIX: &str = "device/pp_power_profile_mode";
const GPU_PERFORMANCE_LEVEL_SUFFIX: &str = "device/power_dpm_force_performance_level";
const GPU_CLOCKS_SUFFIX: &str = "device/pp_od_clk_voltage";
const GPU_CLOCK_LEVELS_SUFFIX: &str = "device/pp_dpm_sclk";
const CPU_SCALING_GOVERNOR_SUFFIX: &str = "scaling_governor";
const CPU_SCALING_AVAILABLE_GOVERNORS_SUFFIX: &str = "scaling_available_governors";

const PLATFORM_PROFILE_PREFIX: &str = "/sys/class/platform-profile";

const TDP_LIMIT1: &str = "power1_cap";
const TDP_LIMIT2: &str = "power2_cap";

lazy_static! {
    static ref GPU_POWER_PROFILE_REGEX: Regex =
        Regex::new(r"^\s*(?<value>[0-9]+)\s+(?<name>[0-9A-Za-z_]+)(?<active>\*)?").unwrap();
    static ref GPU_CLOCK_LEVELS_REGEX: Regex =
        Regex::new(r"^\s*(?<index>[0-9]+): (?<value>[0-9]+)Mhz").unwrap();
}

#[derive(Display, EnumString, PartialEq, Debug, Copy, Clone, TryFromPrimitive)]
#[strum(serialize_all = "snake_case")]
#[repr(u32)]
pub enum GPUPowerProfile {
    // Currently firmware exposes these values, though
    // deck doesn't support them yet
    #[strum(serialize = "3d_full_screen")]
    FullScreen = 1,
    Video = 3,
    VR = 4,
    Compute = 5,
    Custom = 6,
    // Currently only capped and uncapped are supported on
    // deck hardware/firmware. Add more later as needed
    Capped = 8,
    Uncapped = 9,
}

#[derive(Display, EnumString, PartialEq, Debug, Copy, Clone)]
#[strum(serialize_all = "snake_case")]
pub enum GPUPerformanceLevel {
    Auto,
    Low,
    High,
    Manual,
    ProfilePeak,
}

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

#[derive(Display, EnumString, VariantNames, PartialEq, Debug, Copy, Clone)]
#[strum(serialize_all = "snake_case")]
pub enum TdpLimitingMethod {
    GpuHwmon,
    AsusArmoury,
    LenovoWmi,
    PowerStation,
}

#[derive(Debug)]
pub(crate) struct GpuHwmonTdpLimitManager {}

#[derive(Debug)]
pub(crate) struct LenovoWmiTdpLimitManager {}

#[derive(Debug)]
pub(crate) struct AsusArmouryTdpLimitManager {}

#[derive(Debug)]
pub(crate) struct PowerStationTdpLimitManager {
    // Path to the detected GPU card, cached to avoid repeated lookups
    gpu_card_path: Mutex<Option<String>>,
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
    let config = platform_config().await?;
    let config = config
        .as_ref()
        .and_then(|config| config.tdp_limit.as_ref())
        .ok_or(anyhow!("No TDP limit configured"))?;

    Ok(match config.method {
        TdpLimitingMethod::LenovoWmi => Box::new(LenovoWmiTdpLimitManager {}),
        TdpLimitingMethod::AsusArmoury => Box::new(AsusArmouryTdpLimitManager {}),
        TdpLimitingMethod::GpuHwmon => Box::new(GpuHwmonTdpLimitManager {}),
        TdpLimitingMethod::PowerStation => Box::new(PowerStationTdpLimitManager {
            gpu_card_path: Mutex::new(None),
        }),
    })
}

async fn read_gpu_sysfs_contents<S: AsRef<Path>>(suffix: S) -> Result<String> {
    // Read a given suffix for the GPU
    let base = find_hwmon(GPU_HWMON_NAME).await?;
    fs::read_to_string(base.join(suffix.as_ref()))
        .await
        .map_err(|message| anyhow!("Error opening sysfs file for reading {message}"))
}

async fn write_gpu_sysfs_contents<S: AsRef<Path>>(suffix: S, data: &[u8]) -> Result<()> {
    let base = find_hwmon(GPU_HWMON_NAME).await?;
    write_synced(base.join(suffix), data)
        .await
        .inspect_err(|message| error!("Error writing to sysfs file: {message}"))
}

async fn read_cpu_sysfs_contents<S: AsRef<Path>>(suffix: S) -> Result<String> {
    let base = path(CPU_PREFIX).join(CPU0_NAME);
    fs::read_to_string(base.join(suffix.as_ref()))
        .await
        .map_err(|message| anyhow!("Error opening sysfs file for reading {message}"))
}

async fn write_cpu_governor_sysfs_contents(contents: String) -> Result<()> {
    // Iterate over all policyX paths
    let mut dir = fs::read_dir(path(CPU_PREFIX)).await?;
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

pub(crate) async fn get_gpu_power_profile() -> Result<GPUPowerProfile> {
    // check which profile is current and return if possible
    let contents = read_gpu_sysfs_contents(GPU_POWER_PROFILE_SUFFIX).await?;

    // NOTE: We don't filter based on deck here because the sysfs
    // firmware support setting the value to no-op values.
    let lines = contents.lines();
    for line in lines {
        let Some(caps) = GPU_POWER_PROFILE_REGEX.captures(line) else {
            continue;
        };

        let name = &caps["name"].to_lowercase();
        if caps.name("active").is_some() {
            match GPUPowerProfile::from_str(name.as_str()) {
                Ok(v) => {
                    return Ok(v);
                }
                Err(e) => bail!("Unable to parse value for GPU power profile: {e}"),
            }
        }
    }
    bail!("Unable to determine current GPU power profile");
}

pub(crate) async fn get_available_gpu_power_profiles() -> Result<Vec<(u32, String)>> {
    let contents = read_gpu_sysfs_contents(GPU_POWER_PROFILE_SUFFIX).await?;
    let deck = device_type().await.unwrap_or_default() == DeviceType::SteamDeck;

    let mut map = Vec::new();
    let lines = contents.lines();
    for line in lines {
        let Some(caps) = GPU_POWER_PROFILE_REGEX.captures(line) else {
            continue;
        };
        let value: u32 = caps["value"]
            .parse()
            .map_err(|message| anyhow!("Unable to parse value for GPU power profile: {message}"))?;
        let name = &caps["name"];
        if deck {
            // Deck is designed to operate in one of the CAPPED or UNCAPPED power profiles,
            // the other profiles aren't correctly tuned for the hardware.
            if value == GPUPowerProfile::Capped as u32 || value == GPUPowerProfile::Uncapped as u32
            {
                map.push((value, name.to_string()));
            } else {
                // Got unsupported value, so don't include it
            }
        } else {
            // Do basic validation to ensure our enum is up to date?
            map.push((value, name.to_string()));
        }
    }
    Ok(map)
}

pub(crate) async fn set_gpu_power_profile(value: GPUPowerProfile) -> Result<()> {
    let profile = (value as u32).to_string();
    write_gpu_sysfs_contents(GPU_POWER_PROFILE_SUFFIX, profile.as_bytes()).await
}

pub(crate) async fn get_available_gpu_performance_levels() -> Result<Vec<GPUPerformanceLevel>> {
    let base = find_hwmon(GPU_HWMON_NAME).await?;
    if try_exists(base.join(GPU_PERFORMANCE_LEVEL_SUFFIX)).await? {
        Ok(vec![
            GPUPerformanceLevel::Auto,
            GPUPerformanceLevel::Low,
            GPUPerformanceLevel::High,
            GPUPerformanceLevel::Manual,
            GPUPerformanceLevel::ProfilePeak,
        ])
    } else {
        Ok(Vec::new())
    }
}

pub(crate) async fn get_gpu_performance_level() -> Result<GPUPerformanceLevel> {
    let level = read_gpu_sysfs_contents(GPU_PERFORMANCE_LEVEL_SUFFIX).await?;
    Ok(GPUPerformanceLevel::from_str(level.trim())?)
}

pub(crate) async fn set_gpu_performance_level(level: GPUPerformanceLevel) -> Result<()> {
    let level: String = level.to_string();
    write_gpu_sysfs_contents(GPU_PERFORMANCE_LEVEL_SUFFIX, level.as_bytes()).await
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

#[async_trait]
trait GpuClockController: Send + Sync {
    async fn get_clocks_range(&self) -> Result<RangeInclusive<u32>>;
    async fn set_clocks(&self, clocks: u32) -> Result<()>;
    async fn get_clocks(&self) -> Result<u32>;
}

// AMD GPU controller implementation using HWMON
struct AmdGpuController {
    hwmon_path: PathBuf,
}

#[async_trait]
impl GpuClockController for AmdGpuController {
    async fn get_clocks_range(&self) -> Result<RangeInclusive<u32>> {
        let contents = fs::read_to_string(self.hwmon_path.join(GPU_CLOCK_LEVELS_SUFFIX))
            .await
            .map_err(|message| anyhow!("Error opening sysfs file for reading {message}"))?;

        let lines = contents.lines();
        let mut min = 1_000_000;
        let mut max = 0;

        for line in lines {
            let Some(caps) = GPU_CLOCK_LEVELS_REGEX.captures(line) else {
                continue;
            };
            let value: u32 = caps["value"].parse().map_err(|message| {
                anyhow!("Unable to parse value for GPU power profile: {message}")
            })?;
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }

        ensure!(min <= max, "Could not read any clocks");
        Ok(min..=max)
    }

    async fn set_clocks(&self, clocks: u32) -> Result<()> {
        let mut myfile = File::create(self.hwmon_path.join(GPU_CLOCKS_SUFFIX))
            .await
            .inspect_err(|message| error!("Error opening sysfs file for writing: {message}"))?;

        let data = format!("s 0 {clocks}\n");
        myfile
            .write(data.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to sysfs file: {message}"))?;
        myfile.flush().await?;

        let data = format!("s 1 {clocks}\n");
        myfile
            .write(data.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to sysfs file: {message}"))?;
        myfile.flush().await?;

        myfile
            .write("c\n".as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to sysfs file: {message}"))?;
        myfile.flush().await?;

        Ok(())
    }

    async fn get_clocks(&self) -> Result<u32> {
        let clocks_file = File::open(self.hwmon_path.join(GPU_CLOCKS_SUFFIX)).await?;
        let mut reader = BufReader::new(clocks_file);
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await? == 0 {
                break;
            }
            if line != "OD_SCLK:\n" {
                continue;
            }

            let mut line = String::new();
            if reader.read_line(&mut line).await? == 0 {
                break;
            }
            let mhz = match line.split_whitespace().nth(1) {
                Some(mhz) if mhz.ends_with("Mhz") => mhz.trim_end_matches("Mhz"),
                _ => break,
            };

            return Ok(mhz.parse()?);
        }
        Ok(0)
    }
}

// Intel GPU controller implementation using i915 driver
struct IntelI915Controller {
    path: PathBuf,
}

#[async_trait]
impl GpuClockController for IntelI915Controller {
    async fn get_clocks_range(&self) -> Result<RangeInclusive<u32>> {
        // For i915, the supported frequency range is in the rps_* files
        let min_path = self.path.join("gt_RPn_freq_mhz");
        let max_path = self.path.join("gt_RP0_freq_mhz");
        
        let min_val = fs::read_to_string(min_path)
            .await
            .map_err(|message| anyhow!("Error reading Intel min freq: {message}"))?
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow!("Error parsing min freq: {e}"))?;
            
        let max_val = fs::read_to_string(max_path)
            .await
            .map_err(|message| anyhow!("Error reading Intel max freq: {message}"))?
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow!("Error parsing max freq: {e}"))?;
            
        Ok(min_val..=max_val)
    }

    async fn set_clocks(&self, clocks: u32) -> Result<()> {
        // For Intel, we set both min and max to the same value to force the frequency
        let min_path = self.path.join("gt_min_freq_mhz");
        let max_path = self.path.join("gt_max_freq_mhz");

        let clocks_str = clocks.to_string();

        write_synced(min_path, clocks_str.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to Intel min freq: {message}"))?;

        write_synced(max_path, clocks_str.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to Intel max freq: {message}"))?;

        Ok(())
    }

    async fn get_clocks(&self) -> Result<u32> {
        let cur_path = self.path.join("gt_act_freq_mhz");

        let cur_val = fs::read_to_string(cur_path)
            .await
            .map_err(|message| anyhow!("Error reading Intel current freq: {message}"))?
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow!("Error parsing current freq: {e}"))?;

        Ok(cur_val)
    }
}

// Intel GPU controller implementation using Xe driver
struct IntelXeController {
    path: PathBuf,
}

#[async_trait]
impl GpuClockController for IntelXeController {
    async fn get_clocks_range(&self) -> Result<RangeInclusive<u32>> {
        // For Xe, directly use hardware limits
        let min_path = self.path.join("device/tile0/gt0/freq0/rpn_freq");
        let max_path = self.path.join("device/tile0/gt0/freq0/rp0_freq");
        
        let min_val = fs::read_to_string(min_path)
            .await
            .map_err(|message| anyhow!("Error reading Intel Xe min freq: {message}"))?
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow!("Error parsing min freq: {e}"))?;
            
        let max_val = fs::read_to_string(max_path)
            .await
            .map_err(|message| anyhow!("Error reading Intel Xe max freq: {message}"))?
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow!("Error parsing max freq: {e}"))?;
            
        Ok(min_val..=max_val)
    }

    async fn set_clocks(&self, clocks: u32) -> Result<()> {
        let min_path = self.path.join("device/tile0/gt0/freq0/min_freq");
        let max_path = self.path.join("device/tile0/gt0/freq0/max_freq");

        let clocks_str = clocks.to_string();

        write_synced(min_path, clocks_str.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to Intel Xe min freq: {message}"))?;

        write_synced(max_path, clocks_str.as_bytes())
            .await
            .inspect_err(|message| error!("Error writing to Intel Xe max freq: {message}"))?;

        Ok(())
    }

    async fn get_clocks(&self) -> Result<u32> {
        let cur_path = self.path.join("device/tile0/gt0/freq0/cur_freq");

        let cur_val = fs::read_to_string(cur_path)
            .await
            .map_err(|message| anyhow!("Error reading Intel Xe current freq: {message}"))?
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow!("Error parsing current freq: {e}"))?;

        Ok(cur_val)
    }
}

// Helper function to check if a path exists
async fn path_exists(path: impl AsRef<Path>) -> bool {
    match try_exists(path).await {
        Ok(exists) => exists,
        Err(_) => false,
    }
}

// Helper function to check if a GPU is enabled
async fn is_gpu_enabled(card_path: impl AsRef<Path>) -> bool {
    let enable_path = card_path.as_ref().join("device/enable");

    if !path_exists(&enable_path).await {
        return true; // Assume enabled if no enable file exists
    }

    match fs::read_to_string(&enable_path).await {
        Ok(content) => content.trim() == "1",
        Err(_) => true, // Assume enabled if can't read the file
    }
}

// Factory function to get the appropriate GPU controller
async fn get_gpu_controller() -> Result<Box<dyn GpuClockController>> {
    debug!("Looking for GPU controller");
    // First try AMD GPU via HWMON
    if let Ok(path) = find_hwmon(GPU_HWMON_NAME).await {
        debug!("Found AMD GPU via HWMON");
        return Ok(Box::new(AmdGpuController { hwmon_path: path }));
    }

    debug!("No AMD GPU found, trying Intel GPUs");

    // If no AMD GPU, try Intel GPUs
    for card_num in 0..4 {
        // Check card0 to card3
        let card_path = PathBuf::from(format!("/sys/class/drm/card{}", card_num));

        // Skip if path doesn't exist
        if !path_exists(&card_path).await {
            debug!("Skipping non-existent GPU at {}", card_path.display());
            continue;
        }

        // Skip if GPU is disabled
        if !is_gpu_enabled(&card_path).await {
            debug!("Skipping disabled GPU at {}", card_path.display());
            continue;
        }

        debug!("Checking Intel i915 GPU at {}", card_path.display());

        // Check for Intel i915
        if path_exists(card_path.join("gt_min_freq_mhz")).await {
            debug!("Found Intel i915 GPU at {}", card_path.display());
            return Ok(Box::new(IntelI915Controller { path: card_path }));
        }

        debug!("No Intel i915 GPU found, trying Intel Xe");

        // Check for Intel Xe
        if path_exists(card_path.join("device/tile0/gt0/freq0/min_freq")).await {
            debug!("Found Intel Xe GPU at {}", card_path.display());
            return Ok(Box::new(IntelXeController { path: card_path }));
        }

        debug!("No Intel Xe GPU found");
    }

    bail!("No supported GPU found")
}

pub(crate) async fn get_gpu_clocks_range() -> Result<RangeInclusive<u32>> {
    // First try to get from config
    if let Some(range) = platform_config()
        .await?
        .as_ref()
        .and_then(|config| config.gpu_clocks)
    {
        return Ok(range.min..=range.max);
    }

    // If no config, use the appropriate controller
    let controller = get_gpu_controller().await?;
    controller.get_clocks_range().await
}

pub(crate) async fn set_gpu_clocks(clocks: u32) -> Result<()> {
    let controller = get_gpu_controller().await?;
    controller.set_clocks(clocks).await
}

pub(crate) async fn get_gpu_clocks() -> Result<u32> {
    let controller = get_gpu_controller().await?;
    controller.get_clocks().await
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

async fn find_hwmon(hwmon: &str) -> Result<PathBuf> {
    find_sysdir(path(HWMON_PREFIX), hwmon).await
}

async fn find_platform_profile(name: &str) -> Result<PathBuf> {
    find_sysdir(path(PLATFORM_PROFILE_PREFIX), name).await
}

#[async_trait]
impl TdpLimitManager for GpuHwmonTdpLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        let base = find_hwmon(GPU_HWMON_NAME).await?;
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

        let base = find_hwmon(GPU_HWMON_NAME).await?;
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
        let config = platform_config().await?;
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

impl LenovoWmiTdpLimitManager {
    const PREFIX: &str = "/sys/class/firmware-attributes/lenovo-wmi-other-0/attributes";
    const SPL_SUFFIX: &str = "ppt_pl1_spl";
    const SPPT_SUFFIX: &str = "ppt_pl2_sppt";
    const FPPT_SUFFIX: &str = "ppt_pl3_fppt";
}

#[async_trait]
impl TdpLimitManager for LenovoWmiTdpLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        let config = platform_config().await?;
        if let Some(config) = config
            .as_ref()
            .and_then(|config| config.performance_profile.as_ref())
        {
            ensure!(
                get_platform_profile(&config.platform_profile_name).await? == "custom",
                "TDP limiting not active"
            );
        }
        let base = path(Self::PREFIX);

        fs::read_to_string(base.join(Self::SPL_SUFFIX).join("current_value"))
            .await
            .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
            .trim()
            .parse()
            .map_err(|e| anyhow!("Error parsing value: {e}"))
    }

    async fn set_tdp_limit(&self, limit: u32) -> Result<()> {
        ensure!(
            self.get_tdp_limit_range().await?.contains(&limit),
            "Invalid limit"
        );

        let limit = limit.to_string();
        let base = path(Self::PREFIX);
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
        let base = path(Self::PREFIX).join(Self::SPL_SUFFIX);

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
        let config = platform_config().await?;
        if let Some(config) = config
            .as_ref()
            .and_then(|config| config.performance_profile.as_ref())
        {
            Ok(get_platform_profile(&config.platform_profile_name).await? == "custom")
        } else {
            Ok(true)
        }
    }
}

impl AsusArmouryTdpLimitManager {
    const PREFIX: &str = "/sys/class/firmware-attributes/asus-armoury/attributes";
    const SPL_SUFFIX: &str = "ppt_pl1_spl";
    const SPPT_SUFFIX: &str = "ppt_pl2_sppt";
    const FPPT_SUFFIX: &str = "ppt_pl3_fppt";
}

#[async_trait]
impl TdpLimitManager for AsusArmouryTdpLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        let base = path(Self::PREFIX);

        fs::read_to_string(base.join(Self::SPL_SUFFIX).join("current_value"))
            .await
            .map_err(|message| anyhow!("Error reading sysfs: {message}"))?
            .trim()
            .parse()
            .map_err(|e| anyhow!("Error parsing value: {e}"))
    }

    async fn set_tdp_limit(&self, limit: u32) -> Result<()> {
        ensure!(
            self.get_tdp_limit_range().await?.contains(&limit),
            "Invalid limit"
        );

        let limit = limit.to_string();
        let base = path(Self::PREFIX);
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
        let base = path(Self::PREFIX).join(Self::SPL_SUFFIX);

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
}

impl PowerStationTdpLimitManager {
    // PowerStation DBus service information
    const DBUS_SERVICE_NAME: &str = "org.shadowblip.PowerStation";
    const DBUS_BASE_PATH: &str = "/org/shadowblip/Performance/GPU";
    const DBUS_CARD_PREFIX: &str = "card";
    const DBUS_GPU_INTERFACE: &str = "org.shadowblip.GPU.Card";
    const DBUS_TDP_INTERFACE: &str = "org.shadowblip.GPU.Card.TDP";

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
            "org.shadowblip.GPU",
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
                "org.freedesktop.DBus.Properties",
            )
            .await?;

            // Get the Class property
            let class = proxy
                .call::<_, _, OwnedValue>("Get", &(Self::DBUS_GPU_INTERFACE, "Class"))
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
    async fn set_tdp_boost(&self, boost: u32) -> Result<()> {
        debug!("Setting PowerStation TDP boost to {boost}");

        // Connect to system DBus
        let connection = Connection::system().await?;

        // Find the GPU card path
        let card_path = self.find_gpu_card_path().await?;

        // Get a proxy to set the TDP boost property
        let path = ObjectPath::try_from(card_path.as_str())?;
        let proxy = zbus::Proxy::new(
            &connection,
            Self::DBUS_SERVICE_NAME,
            path,
            "org.freedesktop.DBus.Properties",
        )
        .await
        .inspect_err(|message| error!("Error creating proxy: {message}"))?;

        // Set the TDP boost property (convert u32 to double)
        proxy
            .call::<_, _, ()>(
                "Set",
                &(
                    Self::DBUS_TDP_INTERFACE,
                    "Boost",
                    zbus::zvariant::Value::from(boost as f64),
                ),
            )
            .await
            .inspect_err(|message| error!("Error calling Set Boost: {message}"))?;

        Ok(())
    }

    // Helper method to get property values from D-Bus
    async fn get_tdp_property_value(&self, property_name: &str, interface: &str) -> Result<f64> {
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
            "org.freedesktop.DBus.Properties",
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
            .get_tdp_property_value("MinTdp", Self::DBUS_TDP_INTERFACE)
            .await?;
        Ok(min_tdp.round() as u32)
    }

    async fn get_max_tdp(&self) -> Result<u32> {
        let max_tdp = self
            .get_tdp_property_value("MaxTdp", Self::DBUS_TDP_INTERFACE)
            .await?;
        Ok(max_tdp.round() as u32)
    }

    async fn get_max_boost(&self) -> Result<u32> {
        let max_boost = self
            .get_tdp_property_value("MaxBoost", Self::DBUS_TDP_INTERFACE)
            .await?;
        Ok(max_boost.round() as u32)
    }
}

#[async_trait]
impl TdpLimitManager for PowerStationTdpLimitManager {
    async fn get_tdp_limit(&self) -> Result<u32> {
        debug!("Getting PowerStation TDP limit");
        let tdp_value = self
            .get_tdp_property_value("TDP", Self::DBUS_TDP_INTERFACE)
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

        // Set the TDP boost to the limit
        let _ = self
            .set_tdp_boost(0)
            .await
            .inspect_err(|e| error!("Error setting TDP boost before setting TDP: {e}"));

        // Connect to system DBus
        let connection = Connection::system().await?;

        // Find the GPU card path
        let card_path = self.find_gpu_card_path().await?;

        // Get a proxy to set the TDP property
        let path = ObjectPath::try_from(card_path.as_str())?;
        let proxy = zbus::Proxy::new(
            &connection,
            Self::DBUS_SERVICE_NAME,
            path,
            "org.freedesktop.DBus.Properties",
        )
        .await
        .inspect_err(|message| error!("Error creating proxy: {message}"))?;

        // Set the TDP property (convert u32 to double)
        proxy
            .call::<_, _, ()>(
                "Set",
                &(
                    Self::DBUS_TDP_INTERFACE,
                    "TDP",
                    zbus::zvariant::Value::from(limit as f64),
                ),
            )
            .await
            .inspect_err(|message| error!("Error calling Set: {message}"))?;

        Ok(())
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
    let config = platform_config().await?;
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

pub(crate) async fn set_max_charge_level(limit: i32) -> Result<()> {
    ensure!((0..=100).contains(&limit), "Invalid limit");
    let data = limit.to_string();
    let config = platform_config().await?;
    let config = config
        .as_ref()
        .and_then(|config| config.battery_charge_limit.as_ref())
        .ok_or(anyhow!("No battery charge limit configured"))?;
    let base = find_hwmon(config.hwmon_name.as_str()).await?;

    write_synced(base.join(config.attribute.as_str()), data.as_bytes())
        .await
        .inspect_err(|message| error!("Error writing to sysfs file: {message}"))
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

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::hardware::test::fake_model;
    use crate::hardware::SteamDeckVariant;
    use crate::platform::{BatteryChargeLimitConfig, PlatformConfig, RangeConfig, TdpLimitConfig};
    use crate::{enum_roundtrip, testing};
    use anyhow::anyhow;
    use tokio::fs::{create_dir_all, read_to_string, remove_dir, write};

    pub async fn setup() -> Result<()> {
        // Use hwmon5 just as a test. We needed a subfolder of HWMON_PREFIX
        // and this is as good as any.
        let base = path(HWMON_PREFIX).join("hwmon5");
        let filename = base.join(GPU_PERFORMANCE_LEVEL_SUFFIX);
        // Creates hwmon path, including device subpath
        create_dir_all(filename.parent().unwrap()).await?;
        // Writes name file as addgpu so find_hwmon() will find it.
        write_synced(base.join("name"), GPU_HWMON_NAME.as_bytes()).await?;
        Ok(())
    }

    pub async fn create_nodes() -> Result<()> {
        setup().await?;
        let base = find_hwmon(GPU_HWMON_NAME).await?;

        let filename = base.join(GPU_PERFORMANCE_LEVEL_SUFFIX);
        write(filename.as_path(), "auto\n").await?;

        let filename = base.join(GPU_POWER_PROFILE_SUFFIX);
        let contents = " 1 3D_FULL_SCREEN
 3          VIDEO*
 4             VR
 5        COMPUTE
 6         CUSTOM
 8         CAPPED
 9       UNCAPPED";
        write(filename.as_path(), contents).await?;

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

    pub async fn write_clocks(mhz: u32) {
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_CLOCKS_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        let contents = format!(
            "OD_SCLK:
0:       {mhz}Mhz
1:       {mhz}Mhz
OD_RANGE:
SCLK:     200Mhz       1600Mhz
CCLK:    1400Mhz       3500Mhz
CCLK_RANGE in Core0:
0:       1400Mhz
1:       3500Mhz\n"
        );

        write(filename.as_path(), contents).await.expect("write");
    }

    pub async fn read_clocks() -> Result<String, std::io::Error> {
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        read_to_string(base.join(GPU_CLOCKS_SUFFIX)).await
    }

    pub fn format_clocks(mhz: u32) -> String {
        format!("s 0 {mhz}\ns 1 {mhz}\nc\n")
    }

    #[tokio::test]
    async fn test_get_gpu_performance_level() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_PERFORMANCE_LEVEL_SUFFIX);
        assert!(get_gpu_performance_level().await.is_err());

        write(filename.as_path(), "auto\n").await.expect("write");
        assert_eq!(
            get_gpu_performance_level().await.unwrap(),
            GPUPerformanceLevel::Auto
        );

        write(filename.as_path(), "low\n").await.expect("write");
        assert_eq!(
            get_gpu_performance_level().await.unwrap(),
            GPUPerformanceLevel::Low
        );

        write(filename.as_path(), "high\n").await.expect("write");
        assert_eq!(
            get_gpu_performance_level().await.unwrap(),
            GPUPerformanceLevel::High
        );

        write(filename.as_path(), "manual\n").await.expect("write");
        assert_eq!(
            get_gpu_performance_level().await.unwrap(),
            GPUPerformanceLevel::Manual
        );

        write(filename.as_path(), "profile_peak\n")
            .await
            .expect("write");
        assert_eq!(
            get_gpu_performance_level().await.unwrap(),
            GPUPerformanceLevel::ProfilePeak
        );

        write(filename.as_path(), "nothing\n").await.expect("write");
        assert!(get_gpu_performance_level().await.is_err());
    }

    #[tokio::test]
    async fn test_set_gpu_performance_level() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_PERFORMANCE_LEVEL_SUFFIX);

        set_gpu_performance_level(GPUPerformanceLevel::Auto)
            .await
            .expect("set");
        assert_eq!(
            read_to_string(filename.as_path()).await.unwrap().trim(),
            "auto"
        );
        set_gpu_performance_level(GPUPerformanceLevel::Low)
            .await
            .expect("set");
        assert_eq!(
            read_to_string(filename.as_path()).await.unwrap().trim(),
            "low"
        );
        set_gpu_performance_level(GPUPerformanceLevel::High)
            .await
            .expect("set");
        assert_eq!(
            read_to_string(filename.as_path()).await.unwrap().trim(),
            "high"
        );
        set_gpu_performance_level(GPUPerformanceLevel::Manual)
            .await
            .expect("set");
        assert_eq!(
            read_to_string(filename.as_path()).await.unwrap().trim(),
            "manual"
        );
        set_gpu_performance_level(GPUPerformanceLevel::ProfilePeak)
            .await
            .expect("set");
        assert_eq!(
            read_to_string(filename.as_path()).await.unwrap().trim(),
            "profile_peak"
        );
    }

    #[tokio::test]
    async fn test_gpu_hwmon_get_tdp_limit() {
        let handle = testing::start();

        let mut platform_config = PlatformConfig::default();
        platform_config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::GpuHwmon,
            range: Some(RangeConfig { min: 3, max: 15 }),
        });
        handle.test.platform_config.replace(Some(platform_config));
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

        let mut platform_config = PlatformConfig::default();
        platform_config.tdp_limit = Some(TdpLimitConfig {
            method: TdpLimitingMethod::GpuHwmon,
            range: Some(RangeConfig { min: 3, max: 15 }),
        });
        handle.test.platform_config.replace(Some(platform_config));
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

    #[tokio::test]
    async fn test_get_gpu_clocks() {
        let _h = testing::start();

        assert!(get_gpu_clocks().await.is_err());
        setup().await.expect("setup");

        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_CLOCKS_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");
        write(filename.as_path(), b"").await.expect("write");

        assert_eq!(get_gpu_clocks().await.unwrap(), 0);
        write_clocks(1600).await;

        assert_eq!(get_gpu_clocks().await.unwrap(), 1600);
    }

    #[tokio::test]
    async fn test_set_gpu_clocks() {
        let _h = testing::start();

        assert!(set_gpu_clocks(1600).await.is_err());
        setup().await.expect("setup");

        assert!(set_gpu_clocks(200).await.is_ok());

        assert_eq!(read_clocks().await.unwrap(), format_clocks(200));

        assert!(set_gpu_clocks(1600).await.is_ok());
        assert_eq!(read_clocks().await.unwrap(), format_clocks(1600));
    }

    #[tokio::test]
    async fn test_get_gpu_clocks_range() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_CLOCK_LEVELS_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        assert!(get_gpu_clocks_range().await.is_err());

        write(filename.as_path(), &[] as &[u8; 0])
            .await
            .expect("write");
        assert!(get_gpu_clocks_range().await.is_err());

        let contents = "0: 200Mhz *
1: 1100Mhz
2: 1600Mhz";
        write(filename.as_path(), contents).await.expect("write");
        assert_eq!(get_gpu_clocks_range().await.unwrap(), 200..=1600);

        let contents = "0: 1600Mhz *
1: 200Mhz
2: 1100Mhz";
        write(filename.as_path(), contents).await.expect("write");
        assert_eq!(get_gpu_clocks_range().await.unwrap(), 200..=1600);
    }

    #[test]
    fn gpu_power_profile_roundtrip() {
        enum_roundtrip!(GPUPowerProfile {
            1: u32 = FullScreen,
            3: u32 = Video,
            4: u32 = VR,
            5: u32 = Compute,
            6: u32 = Custom,
            8: u32 = Capped,
            9: u32 = Uncapped,
            "3d_full_screen": str = FullScreen,
            "video": str = Video,
            "vr": str = VR,
            "compute": str = Compute,
            "custom": str = Custom,
            "capped": str = Capped,
            "uncapped": str = Uncapped,
        });
        assert!(GPUPowerProfile::try_from(0).is_err());
        assert!(GPUPowerProfile::try_from(2).is_err());
        assert!(GPUPowerProfile::try_from(10).is_err());
        assert!(GPUPowerProfile::from_str("fullscreen").is_err());
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

    #[test]
    fn gpu_performance_level_roundtrip() {
        enum_roundtrip!(GPUPerformanceLevel {
            "auto": str = Auto,
            "low": str = Low,
            "high": str = High,
            "manual": str = Manual,
            "profile_peak": str = ProfilePeak,
        });
        assert!(GPUPerformanceLevel::from_str("peak_performance").is_err());
    }

    #[tokio::test]
    async fn read_power_profiles() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_POWER_PROFILE_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        let contents = " 1 3D_FULL_SCREEN
 3          VIDEO*
 4             VR
 5        COMPUTE
 6         CUSTOM
 8         CAPPED
 9       UNCAPPED";

        write(filename.as_path(), contents).await.expect("write");

        fake_model(SteamDeckVariant::Unknown)
            .await
            .expect("fake_model");

        let profiles = get_available_gpu_power_profiles().await.expect("get");
        assert_eq!(
            profiles,
            &[
                (
                    GPUPowerProfile::FullScreen as u32,
                    String::from("3D_FULL_SCREEN")
                ),
                (GPUPowerProfile::Video as u32, String::from("VIDEO")),
                (GPUPowerProfile::VR as u32, String::from("VR")),
                (GPUPowerProfile::Compute as u32, String::from("COMPUTE")),
                (GPUPowerProfile::Custom as u32, String::from("CUSTOM")),
                (GPUPowerProfile::Capped as u32, String::from("CAPPED")),
                (GPUPowerProfile::Uncapped as u32, String::from("UNCAPPED"))
            ]
        );

        fake_model(SteamDeckVariant::Jupiter)
            .await
            .expect("fake_model");

        let profiles = get_available_gpu_power_profiles().await.expect("get");
        assert_eq!(
            profiles,
            &[
                (GPUPowerProfile::Capped as u32, String::from("CAPPED")),
                (GPUPowerProfile::Uncapped as u32, String::from("UNCAPPED"))
            ]
        );
    }

    #[tokio::test]
    async fn read_unknown_power_profiles() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_POWER_PROFILE_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        let contents = " 1 3D_FULL_SCREEN
 2            CGA
 3          VIDEO*
 4             VR
 5        COMPUTE
 6         CUSTOM
 8         CAPPED
 9       UNCAPPED";

        write(filename.as_path(), contents).await.expect("write");

        fake_model(SteamDeckVariant::Unknown)
            .await
            .expect("fake_model");

        let profiles = get_available_gpu_power_profiles().await.expect("get");
        assert_eq!(
            profiles,
            &[
                (
                    GPUPowerProfile::FullScreen as u32,
                    String::from("3D_FULL_SCREEN")
                ),
                (2, String::from("CGA")),
                (GPUPowerProfile::Video as u32, String::from("VIDEO")),
                (GPUPowerProfile::VR as u32, String::from("VR")),
                (GPUPowerProfile::Compute as u32, String::from("COMPUTE")),
                (GPUPowerProfile::Custom as u32, String::from("CUSTOM")),
                (GPUPowerProfile::Capped as u32, String::from("CAPPED")),
                (GPUPowerProfile::Uncapped as u32, String::from("UNCAPPED"))
            ]
        );

        fake_model(SteamDeckVariant::Jupiter)
            .await
            .expect("fake_model");

        let profiles = get_available_gpu_power_profiles().await.expect("get");
        assert_eq!(
            profiles,
            &[
                (GPUPowerProfile::Capped as u32, String::from("CAPPED")),
                (GPUPowerProfile::Uncapped as u32, String::from("UNCAPPED"))
            ]
        );
    }

    #[tokio::test]
    async fn read_power_profile() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_POWER_PROFILE_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        let contents = " 1 3D_FULL_SCREEN
 3          VIDEO*
 4             VR
 5        COMPUTE
 6         CUSTOM
 8         CAPPED
 9       UNCAPPED";

        write(filename.as_path(), contents).await.expect("write");

        fake_model(SteamDeckVariant::Unknown)
            .await
            .expect("fake_model");
        assert_eq!(
            get_gpu_power_profile().await.expect("get"),
            GPUPowerProfile::Video
        );

        fake_model(SteamDeckVariant::Jupiter)
            .await
            .expect("fake_model");
        assert_eq!(
            get_gpu_power_profile().await.expect("get"),
            GPUPowerProfile::Video
        );
    }

    #[tokio::test]
    async fn read_no_power_profile() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_POWER_PROFILE_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        let contents = " 1 3D_FULL_SCREEN
 3          VIDEO
 4             VR
 5        COMPUTE
 6         CUSTOM
 8         CAPPED
 9       UNCAPPED";

        write(filename.as_path(), contents).await.expect("write");

        fake_model(SteamDeckVariant::Unknown)
            .await
            .expect("fake_model");
        assert!(get_gpu_power_profile().await.is_err());

        fake_model(SteamDeckVariant::Jupiter)
            .await
            .expect("fake_model");
        assert!(get_gpu_power_profile().await.is_err());
    }

    #[tokio::test]
    async fn read_unknown_power_profile() {
        let _h = testing::start();

        setup().await.expect("setup");
        let base = find_hwmon(GPU_HWMON_NAME).await.unwrap();
        let filename = base.join(GPU_POWER_PROFILE_SUFFIX);
        create_dir_all(filename.parent().unwrap())
            .await
            .expect("create_dir_all");

        let contents = " 1 3D_FULL_SCREEN
 2            CGA*
 3          VIDEO
 4             VR
 5        COMPUTE
 6         CUSTOM
 8         CAPPED
 9       UNCAPPED";

        write(filename.as_path(), contents).await.expect("write");

        fake_model(SteamDeckVariant::Unknown)
            .await
            .expect("fake_model");
        assert!(get_gpu_power_profile().await.is_err());

        fake_model(SteamDeckVariant::Jupiter)
            .await
            .expect("fake_model");
        assert!(get_gpu_power_profile().await.is_err());
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
    async fn read_max_charge_level() {
        let handle = testing::start();

        let mut platform_config = PlatformConfig::default();
        platform_config.battery_charge_limit = Some(BatteryChargeLimitConfig {
            suggested_minimum_limit: Some(10),
            hwmon_name: String::from("steamdeck_hwmon"),
            attribute: String::from("max_battery_charge_level"),
        });
        handle.test.platform_config.replace(Some(platform_config));

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

        set_max_charge_level(99).await.expect("set");
        assert_eq!(get_max_charge_level().await.unwrap(), 99);

        set_max_charge_level(0).await.expect("set");
        assert_eq!(get_max_charge_level().await.unwrap(), 0);

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
}
