/*
 * Copyright © 2023 Collabora Ltd.
 * Copyright © 2024 Valve Software
 * Copyright © 2024 Igalia S.L.
 *
 * SPDX-License-Identifier: MIT
 */

use proc_macro::TokenStream;
use proc_macro2::{Group, Literal, Span, TokenStream as TokenStream2, TokenTree};
use quote::{ToTokens, format_ident, quote};
use std::collections::{HashMap, HashSet};
use syn::parse::{self, Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    self, Attribute, Error, Fields, FnArg, GenericArgument, Ident, ImplItem, ItemImpl, ItemStruct,
    Meta, Pat, PatType, PathArguments, ReturnType, Token, Type, TypePath, parse_macro_input,
};

#[derive(Debug)]
struct BaseInterface {
    name: String,
    properties: Vec<Property>,
    methods: Vec<Method>,
    register: Option<Ident>,
    unregister: Option<Ident>,
    context: Option<Type>,
}

#[derive(Debug)]
struct Interface {
    name: Ident,
    items: Vec<ImplItem>,
}

#[derive(Debug)]
struct Method {
    name: Ident,
    args: Vec<Type>,
    ret: Option<Type>,
}

#[derive(Debug)]
struct Property {
    name: Ident,
    attr: Attribute,
    emits_changed: bool,
    ty: Type,
    setter: bool,
}

#[derive(Debug)]
struct RemoteInterface {
    name: Ident,
    vars: Vec<Ident>,
    ifaces: Vec<Ident>,
    contexts: HashSet<Ident>,
}

fn clean_return_type(ty: Type) -> Type {
    match ty {
        Type::Path(ref path) => {
            if let Some(tail) = path.path.segments.last() {
                if tail.ident == "Result" {
                    match &tail.arguments {
                        PathArguments::None => ty,
                        PathArguments::AngleBracketed(args) => match args.args.first() {
                            Some(GenericArgument::Type(ty)) => ty.clone(),
                            _ => todo!(),
                        },
                        PathArguments::Parenthesized(_) => todo!("parenthesized return type"),
                    }
                } else {
                    ty
                }
            } else {
                todo!("no tail");
            }
        }
        other => todo!("unimplemented return type {other:?}"),
    }
}

fn parse_kv_pairs(group: Group) -> parse::Result<HashMap<String, Literal>> {
    let mut tokens = group.stream().into_iter();
    let mut kv = HashMap::new();
    loop {
        let prop = match tokens.next() {
            Some(TokenTree::Ident(prop)) => prop,
            Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => continue,
            Some(token) => {
                return Err(syn::Error::new(token.span(), "expected `,` or identifier"));
            }
            None => break,
        };
        let value = {
            match tokens.next() {
                Some(TokenTree::Punct(punct)) if punct.as_char() == '=' => (),
                Some(token) => {
                    return Err(syn::Error::new(token.span(), "expected `=`"));
                }
                None => {
                    return Err(syn::Error::new(group.span_close(), "expected `=`"));
                }
            }
            match tokens.next() {
                Some(TokenTree::Literal(lit)) => lit,
                Some(token) => {
                    return Err(syn::Error::new(token.span(), "expected string"));
                }
                None => {
                    return Err(syn::Error::new(group.span_close(), "expected string"));
                }
            }
        };
        let prop_str = prop.to_string();
        if kv.insert(prop_str, value).is_some() {
            return Err(syn::Error::new(
                prop.span(),
                format!("duplicate key \"{prop}\""),
            ));
        }
    }
    Ok(kv)
}

fn decompose_generic<'a>(ty: &'a Type, expected: &str) -> parse::Result<&'a Type> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return Err(Error::new(ty.span(), "Unexpected type"));
    };
    if path.segments.len() != 1 {
        return Err(Error::new(path.span(), format!("Expected `{expected}`")));
    }
    let ty = &path.segments[0];
    if ty.ident != expected {
        return Err(Error::new(ty.span(), format!("Expected `{expected}`")));
    }
    let PathArguments::AngleBracketed(args) = &ty.arguments else {
        return Err(Error::new(ty.span(), format!("Expected `{expected}`")));
    };
    if args.args.len() != 1 {
        return Err(Error::new(
            args.span(),
            format!("Expected single argument to `{expected}`"),
        ));
    }
    let GenericArgument::Type(ty) = &args.args[0] else {
        return Err(Error::new(
            args.span(),
            format!("Expected single argument to `{expected}`"),
        ));
    };
    Ok(ty)
}

fn decompose_eq(
    mut input: impl Iterator<Item = TokenTree>,
    span: Span,
) -> parse::Result<TokenStream> {
    let Some(eq) = input.next() else {
        return Err(parse::Error::new(span, "Expected `=` following"));
    };
    match &eq {
        TokenTree::Punct(eq) if eq.as_char() == '=' => (),
        _ => return Err(parse::Error::new(eq.span(), "Expected `=`")),
    }
    Ok(TokenStream2::from_iter(input).into())
}

impl Parse for Interface {
    fn parse(input: ParseStream<'_>) -> parse::Result<Interface> {
        let iface_impl: ItemImpl = input.parse()?;
        let Type::Path(path) = *iface_impl.self_ty else {
            return Err(syn::Error::new(input.span(), "Invalid name identifier"));
        };
        let name = path.path.require_ident()?.clone();
        let items = iface_impl.items.clone();

        Ok(Interface { name, items })
    }
}

impl Parse for BaseInterface {
    fn parse(input: ParseStream<'_>) -> parse::Result<BaseInterface> {
        let iface_impl: ItemImpl = input.parse()?;
        let Type::Path(path) = *iface_impl.self_ty else {
            return Err(syn::Error::new(input.span(), "Invalid name identifier"));
        };
        let name = path.path.require_ident()?;
        let mut properties = Vec::new();
        let mut methods = Vec::new();
        for item in iface_impl.items {
            let ImplItem::Fn(fn_item) = item else {
                continue;
            };
            let mut prop_attr = None;
            let mut emits_changed = true;
            for attr in fn_item.attrs {
                let Meta::List(ref list) = attr.meta else {
                    continue;
                };
                if list.path.require_ident()? != "zbus" {
                    continue;
                }
                let mut tokens = list.tokens.clone().into_iter();
                let first = tokens.next();
                match first {
                    Some(TokenTree::Ident(ident)) if ident == "property" => {
                        prop_attr = Some(attr);
                        if let Some(TokenTree::Group(group)) = tokens.next() {
                            let kv = parse_kv_pairs(group)?;
                            match kv.get("emits_changed_signal") {
                                None => emits_changed = true,
                                Some(val) if val.to_string() == "true" => emits_changed = true,
                                _ => emits_changed = false,
                            }
                        }
                    }
                    Some(TokenTree::Ident(ident)) if ident == "signal" => {
                        todo!("signals not implemented")
                    }
                    other => todo!("unknown attribute {other:?}"),
                }
            }
            let sig = fn_item.sig;
            let name = sig.ident;
            let inputs = sig.inputs;
            if !matches!(inputs.first(), Some(FnArg::Receiver(_))) {
                return Err(syn::Error::new(
                    sig.paren_token.span.open(),
                    "expected `self`",
                ));
            }

            if let Some(attr) = prop_attr {
                let setter = name.to_string().starts_with("set_");
                let ty = if setter {
                    let mut ty = None;
                    emits_changed = false;
                    'input: for input in inputs.into_iter().skip(1) {
                        let span = input.span();
                        let FnArg::Typed(fty) = input else {
                            continue;
                        };
                        for attr in &fty.attrs {
                            let Meta::List(ref list) = attr.meta else {
                                continue;
                            };
                            let Some(ident) = list.path.get_ident() else {
                                continue;
                            };
                            if ident == "zbus" {
                                continue 'input;
                            }
                        }
                        if ty.is_some() {
                            return Err(syn::Error::new(span, "unexpected argument type"));
                        }
                        ty = Some(*fty.ty);
                    }
                    ty.unwrap()
                } else {
                    if inputs.len() != 1 {
                        return Err(syn::Error::new(
                            sig.paren_token.span.join(),
                            "expected 1 argument",
                        ));
                    }
                    let ReturnType::Type(_, ret) = sig.output else {
                        return Err(syn::Error::new(sig.fn_token.span, "expected return value"));
                    };
                    clean_return_type(*ret)
                };
                properties.push(Property {
                    name,
                    attr,
                    setter,
                    ty,
                    emits_changed,
                });
            } else {
                let ret = match sig.output {
                    ReturnType::Type(_, ret) => Some(clean_return_type(*ret)),
                    ReturnType::Default => None,
                };
                let args = inputs
                    .into_iter()
                    .skip(1)
                    .map(|arg| {
                        let FnArg::Typed(ty) = arg else {
                            panic!();
                        };
                        *ty.ty
                    })
                    .collect();
                methods.push(Method { name, args, ret });
            }
        }

        Ok(BaseInterface {
            name: name.to_string(),
            methods,
            properties,
            register: None,
            unregister: None,
            context: None,
        })
    }
}

impl Parse for RemoteInterface {
    fn parse(input: ParseStream<'_>) -> parse::Result<RemoteInterface> {
        enum FieldType {
            Normal,
            Remote,
            Context,
        }

        let iface_struct: ItemStruct = input.parse()?;
        let mut ifaces = Vec::new();
        let mut vars = Vec::new();
        let mut contexts = HashSet::new();

        let Fields::Named(fields) = &iface_struct.fields else {
            return Err(Error::new(
                iface_struct.span(),
                "RemoteInterface requires named fields",
            ));
        };

        for field in &fields.named {
            let mut field_type = FieldType::Normal;
            for attr in &field.attrs {
                let Meta::Path(path) = &attr.meta else {
                    continue;
                };
                match path.require_ident() {
                    Ok(x) if x == "remote" => field_type = FieldType::Remote,
                    Ok(x) if x == "context" => field_type = FieldType::Context,
                    _ => continue,
                }
                break;
            }
            match field_type {
                FieldType::Remote => {
                    let Some(ident) = &field.ident else {
                        return Err(Error::new(
                            field.span(),
                            "RemoteInterface requires named fields",
                        ));
                    };
                    let ty = decompose_generic(&field.ty, "Option")?;
                    let Type::Path(TypePath { path, .. }) = &ty else {
                        return Err(Error::new(
                            ty.span(),
                            "RemoteInterface requires an interface Remote",
                        ));
                    };
                    let iface = path.require_ident()?.to_string();
                    let Some(iface) = iface.strip_suffix("RemoteOwner") else {
                        return Err(Error::new(
                            path.span(),
                            "RemoteInterface requires an interface Remote",
                        ));
                    };
                    ifaces.push(format_ident!("{iface}"));
                    vars.push(ident.clone());
                }
                FieldType::Context => {
                    let Some(ident) = &field.ident else {
                        return Err(Error::new(
                            field.span(),
                            "RemoteInterface requires named fields",
                        ));
                    };
                    contexts.insert(ident.clone());
                }
                FieldType::Normal => (),
            }
        }

        let name = iface_struct.ident;
        Ok(RemoteInterface {
            name,
            ifaces,
            vars,
            contexts,
        })
    }
}

impl ToTokens for Interface {
    fn to_tokens(&self, stream: &mut TokenStream2) {
        let name = &self.name;
        let mut items = TokenStream2::new();

        for item in &self.items {
            let ImplItem::Fn(fn_item) = item else {
                item.to_tokens(&mut items);
                continue;
            };

            let mut fn_item = fn_item.clone();
            let sig = &mut fn_item.sig;

            let mut option;
            let header_name;
            let connection_name;
            let mut ns = Ident::new("fdo", Span::call_site());

            fn is_zbus_attr(arg: &PatType, var: &str) -> bool {
                if !arg.attrs.iter().any(|attr| {
                    let Meta::List(attr) = &attr.meta else {
                        return false;
                    };
                    let Ok(path) = attr.path.require_ident() else {
                        return false;
                    };
                    if path != "zbus" {
                        return false;
                    }
                    let Ok(ident) = syn::parse::<Ident>(attr.tokens.clone().into()) else {
                        return false;
                    };
                    ident == var
                }) {
                    return false;
                }
                true
            }

            // Find Header varible, if present
            if let Some(header) = sig.inputs.iter().find_map(|arg| {
                let FnArg::Typed(arg) = arg else {
                    return None;
                };
                if !is_zbus_attr(arg, "header") {
                    return None;
                }
                let Pat::Ident(name) = &*arg.pat else {
                    return None;
                };
                let Type::Path(ty) = &*arg.ty else {
                    return None;
                };
                let option = if let Some(ident) = ty.path.segments.first().map(|seg| &seg.ident) {
                    ident == "Option"
                } else {
                    false
                };
                Some((name.ident.clone(), option))
            }) {
                (header_name, option) = header;
            } else {
                option = false;
                for attr in &fn_item.attrs {
                    let Meta::List(attr) = &attr.meta else {
                        continue;
                    };
                    let Ok(path) = attr.path.require_ident() else {
                        continue;
                    };
                    if path != "zbus" {
                        continue;
                    }
                    let Some(TokenTree::Ident(ident)) = attr.tokens.clone().into_iter().next()
                    else {
                        continue;
                    };
                    if ident != "property" {
                        continue;
                    }
                    option = true;
                    if sig.ident.to_string().starts_with("set_") {
                        ns = Ident::new("zbus", Span::call_site());
                    }
                    break;
                }
                let arg = syn::parse::<FnArg>(
                    if option {
                        quote!(#[zbus(header)] header: Option<Header<'_>>)
                    } else {
                        quote!(#[zbus(header)] header: Header<'_>)
                    }
                    .into(),
                )
                .unwrap();
                sig.inputs.push(arg);
                header_name = Ident::new("header", Span::call_site());
            }

            // Find Connection variable, if present
            if let Some(connection) = sig.inputs.iter().find_map(|arg| {
                let FnArg::Typed(arg) = arg else {
                    return None;
                };
                if !is_zbus_attr(arg, "connection") {
                    return None;
                }
                let Pat::Ident(name) = &*arg.pat else {
                    return None;
                };
                Some(name.ident.clone())
            }) {
                connection_name = connection;
            } else {
                let arg =
                    syn::parse::<FnArg>(quote!(#[zbus(connection)] connection: &Connection).into())
                        .unwrap();
                sig.inputs.push(arg);
                connection_name = Ident::new("connection", Span::call_site());
            }

            let validate_header = if option {
                quote!(if let Some(ref header) = #header_name {
                    validate_header(&header, #connection_name).await?;
                })
            } else {
                quote!(validate_header(&#header_name, #connection_name).await?;)
            };

            let mut block = TokenStream2::new();
            let fn_block = &fn_item.block;
            if let ReturnType::Type(_, ty) = &sig.output {
                if let Type::Path(path) = &**ty {
                    let last = path.path.segments.last();
                    if let Some(last) = last
                        && last.ident == "Result"
                    {
                        quote!({
                            #validate_header
                            #fn_block
                        })
                        .to_tokens(&mut block);
                    } else {
                        let ret = syn::parse::<TypePath>(quote!(#ns::Result<#ty>).into()).unwrap();
                        sig.output =
                            ReturnType::Type(<Token![->]>::default(), Box::new(Type::Path(ret)));
                        quote!({
                            #validate_header
                            Ok(#fn_block)
                        })
                        .to_tokens(&mut block);
                    }
                } else {
                    todo!("unimplemented return");
                };
            } else {
                let ret = syn::parse::<TypePath>(quote!(#ns::Result<()>).into()).unwrap();
                sig.output = ReturnType::Type(<Token![->]>::default(), Box::new(Type::Path(ret)));
                quote!({
                    #validate_header
                    #fn_block
                    Ok(())
                })
                .to_tokens(&mut block);
            }
            fn_item.block = syn::parse(block.into()).unwrap();
            fn_item.to_tokens(&mut items);
        }

        stream.extend(quote! {
            impl #name {
                #items
            }
        });
    }
}

impl ToTokens for BaseInterface {
    fn to_tokens(&self, stream: &mut TokenStream2) {
        let mut substream = TokenStream2::new();
        let mut signals = Vec::new();
        for prop in self.properties.iter() {
            prop.to_tokens(&mut substream);
            if prop.emits_changed {
                signals.push(format_ident!("{}_changed", prop.name.clone()));
            }
        }
        for method in self.methods.iter() {
            method.to_tokens(&mut substream);
        }

        let name = format_ident!("{}", self.name);
        let struct_name: Ident = format_ident!("{}Remote", self.name);
        let proxy_name: Ident = format_ident!("{}Proxy", self.name);
        let owner_name: Ident = format_ident!("{}RemoteOwner", self.name);

        let receivers: Vec<Ident> = signals
            .iter()
            .map(|name| format_ident!("receive_{name}"))
            .collect();

        let signal_task = if receivers.is_empty() {
            quote! {
                async fn start_signal_task(&self) -> Result<()> {
                    Ok(())
                }
            }
        } else {
            quote! {
                async fn start_signal_task(&mut self) -> Result<()> {
                    let signal_task = Self::signal_task(
                        self.destination.clone(),
                        self.path.clone(),
                        self.session.clone(),
                        self.system.clone()
                    )
                    .await?;
                    self.signal_task = Some(signal_task);
                    Ok(())
                }

                async fn signal_task_impl(
                    destination: OwnedBusName,
                    path: OwnedObjectPath,
                    session: Connection,
                    system: Connection,
                    tx1: oneshot::Sender<()>,
                    rx2: oneshot::Receiver<()>,
                ) -> Result<()> {
                    use zbus::proxy::ProxyImpl;

                    let object_server = session.object_server();
                    let proxy = #proxy_name::builder(&system)
                        .path(path)?
                        .destination(destination)?
                        .build()
                        .await?;
                    #(let mut #receivers = proxy.#receivers().await;)*
                    // This should never fail. If it does, something has gone very wrong.
                    tx1.send(()).unwrap();
                    rx2.await?;
                    loop {
                        tokio::select! {
                            #(Some(val) = #receivers.next() => {
                                let Ok(interface) = object_server
                                    .interface::<_, #struct_name>(MANAGER_PATH)
                                    .await else
                                {
                                    warn!("Lost signal");
                                    continue;
                                };

                                let emitter = interface.signal_emitter();
                                if let Err(e) = interface.get().await.#signals(emitter).await {
                                    error!("Error receiving signal: {e}");
                                };
                            },)*
                        }
                    }
                }

                async fn signal_task(
                    destination: OwnedBusName,
                    path: OwnedObjectPath,
                    session: Connection,
                    system: Connection,
                ) -> Result<JoinHandle<Result<()>>> {
                    let (tx1, rx1) = oneshot::channel();
                    let (tx2, rx2) = oneshot::channel();
                    let handle = spawn(Self::signal_task_impl(destination, path, session, system, tx1, rx2));
                    rx1.await?;
                    let _ = tx2.send(());
                    Ok(handle)
                }
            }
        };

        let context = if let Some(context) = &self.context {
            quote!(#context)
        } else {
            quote!(())
        };

        let register = if let Some(register) = &self.register {
            quote! {
                #register(&mut self.context, proxy, object_server).await?;
            }
        } else {
            quote! {
                let interface = #struct_name {
                    proxy,
                    job_manager: self.job_manager.clone()
                };
                if !register::<#name>(object_server, interface).await? {
                    return Ok(false);
                }
            }
        };

        let unregister = self.unregister.as_ref().map(|unregister| {
            quote! {
                async fn unregister(
                    context: Option<&mut <#owner_name as RemoteOwner>::Context>,
                    object_server: &ObjectServer
                ) -> zbus::Result<()> {
                    #unregister(context, object_server).await
                }
            }
        });

        stream.extend(quote! {
            impl #struct_name {
                #substream
            }

            #[derive(Debug)]
            pub(crate) struct #struct_name {
                proxy: #proxy_name<'static>,
                job_manager: UnboundedSender<JobManagerCommand>,
            }

            #[derive(Debug)]
            pub(crate) struct #owner_name {
                session: Connection,
                system: Connection,
                destination: OwnedBusName,
                path: OwnedObjectPath,
                load_task: JoinHandle<anyhow::Result<()>>,
                signal_task: Option<JoinHandle<anyhow::Result<()>>>,
                registered: bool,
                is_transient: bool,
                #[cfg(test)]
                ping_success: bool,
                context: <Self as RemoteOwner>::Context,
                job_manager: UnboundedSender<JobManagerCommand>,
            }

            impl RemoteOwner for #owner_name {
                type Context = #context;

                async fn new(
                    destination: OwnedBusName,
                    path: OwnedObjectPath,
                    session: &Connection,
                    system: &Connection,
                    is_transient: bool,
                    context: <Self as RemoteOwner>::Context,
                    job_manager: UnboundedSender<JobManagerCommand>,
                )
                -> fdo::Result<#owner_name> {
                    let load_task = Self::load_task(
                        destination.clone(),
                        path.clone(),
                        session.clone(),
                        system.clone()
                    )
                    .await
                    .map_err(to_zbus_fdo_error)?;
                    Ok(#owner_name {
                        session: session.clone(),
                        system: system.clone(),
                        path,
                        destination,
                        load_task,
                        signal_task: None,
                        is_transient,
                        registered: false,
                        #[cfg(test)]
                        ping_success: false,
                        context,
                        job_manager,
                    })
                }
            }

            impl #owner_name {
                async fn load_task_impl(
                    destination: OwnedBusName,
                    path: OwnedObjectPath,
                    session: Connection,
                    system: Connection,
                    tx1: oneshot::Sender<()>,
                    rx2: oneshot::Receiver<()>,
                ) -> Result<()> {
                    let object_server = session.object_server();
                    let dbus_proxy = DBusProxy::new(&system).await?;
                    let mut name_changed_receiver = dbus_proxy.receive_name_owner_changed().await?;
                    // This should never fail. If it does, something has gone very wrong.
                    tx1.send(()).unwrap();
                    rx2.await?;
                    while let Some(changed) = name_changed_receiver.next().await {
                        match changed.args() {
                            Ok(args) => {
                                if args.name() != &destination {
                                    continue;
                                }
                                match args.new_owner().as_ref() {
                                    None => {
                                        let manager = object_server
                                            .interface::<_, RemoteInterface1>(MANAGER_PATH)
                                            .await?;
                                        let emitter = manager.signal_emitter();
                                        manager
                                            .get_mut()
                                            .await
                                            .unload(
                                                <#struct_name as Interface>::name().as_str(),
                                                emitter,
                                            )
                                            .await?;
                                    }
                                    Some(owner) => {
                                        let manager = object_server
                                            .interface::<_, RemoteInterface1>(MANAGER_PATH)
                                            .await?;
                                        let emitter = manager.signal_emitter();
                                        manager
                                            .get_mut()
                                            .await
                                            .load(
                                                <#struct_name as Interface>::name().as_str(),
                                                emitter,
                                            )
                                            .await?;
                                    }
                                }
                            },
                            Err(e) => error!("Error receiving signal: {e}"),
                        }
                    }
                    Ok(())
                }

                async fn load_task(
                    destination: OwnedBusName,
                    path: OwnedObjectPath,
                    session: Connection,
                    system: Connection,
                ) -> Result<JoinHandle<Result<()>>> {
                    let (tx1, rx1) = oneshot::channel();
                    let (tx2, rx2) = oneshot::channel();
                    let handle = spawn(Self::load_task_impl(destination, path, session, system, tx1, rx2));
                    rx1.await?;
                    let _ = tx2.send(());
                    Ok(handle)
                }

                #signal_task

                async fn register(&mut self) -> fdo::Result<bool> {
                    use zbus::proxy::ProxyImpl;

                    if !self.is_transient {
                        #[cfg(not(test))]
                        {
                            // Ping the remote to see if the service exists
                            // TODO: Tap into the ObjectManager to find out if the object exists too
                            let peer_proxy = fdo::PeerProxy::new(
                                &self.system,
                                &self.destination,
                                "/",
                            )
                            .await?;
                            peer_proxy.ping().await?;
                        }
                        #[cfg(test)]
                        {
                            if !self.ping_success {
                                return Err(fdo::Error::ServiceUnknown(String::from("No")));
                            }
                        }
                    }

                    let object_server = self.session.object_server();
                    let proxy = #proxy_name::builder(&self.system)
                        .path(self.path.clone())?
                        .destination(self.destination.clone())?
                        .build()
                        .await?;
                    #register

                    self.registered = true;
                    Ok(true)
                }

                fn remote(&self) -> &BusName<'_> {
                    self.destination.inner()
                }
            }

            impl Drop for #owner_name {
                fn drop(&mut self) {
                    self.load_task.abort();
                    if let Some(signal_task) = self.signal_task.take() {
                        signal_task.abort();
                    }
                }
            }

            #[async_trait]
            impl RemoteInterface for #name {
                type Remote = #struct_name;
                type Owner = #owner_name;

                #unregister
            }
        });
    }
}

impl BaseInterface {
    fn split_attrs(&mut self, attr: TokenStream2) -> parse::Result<TokenStream2> {
        let mut new_attrs = TokenStream2::new();
        let mut accum = TokenStream2::new();
        for token in attr.into_iter() {
            match token {
                TokenTree::Punct(x) if x.as_char() == ',' => {
                    if let Some(attr) = self.handle_token_group(accum)? {
                        new_attrs.extend(attr);
                    }
                    accum = TokenStream2::new();
                }
                x => accum.extend([x]),
            }
        }
        if let Some(attr) = self.handle_token_group(accum)? {
            new_attrs.extend(attr);
        }
        Ok(new_attrs)
    }

    fn handle_token_group(&mut self, accum: TokenStream2) -> parse::Result<Option<TokenStream2>> {
        let mut iter = accum.into_iter();
        let Some(token) = iter.next() else {
            return Ok(None);
        };
        match token {
            TokenTree::Ident(ident) if ident == "register" => {
                let tokens = decompose_eq(iter, ident.span())?;
                self.register = Some(syn::parse::<Ident>(tokens)?);
            }
            TokenTree::Ident(ident) if ident == "unregister" => {
                let tokens = decompose_eq(iter, ident.span())?;
                self.unregister = Some(syn::parse::<Ident>(tokens)?);
            }
            TokenTree::Ident(ident) if ident == "context" => {
                let tokens = decompose_eq(iter, ident.span())?;
                self.context = Some(syn::parse::<Type>(tokens)?);
            }
            x => {
                let mut attr = TokenStream2::from(x);
                attr.extend(iter);
                return Ok(Some(attr));
            }
        }
        Ok(None)
    }
}

impl ToTokens for Method {
    fn to_tokens(&self, stream: &mut TokenStream2) {
        let name = &self.name;
        let args = &self.args;
        let ret = &self.ret;
        let arg_names: Vec<Ident> = (0..args.len()).map(|i| format_ident!("arg{i}")).collect();

        if let Some(Type::Path(ty)) = ret
            && ty
                .path
                .get_ident()
                .map(|ident| ident == "OwnedObjectPath")
                .unwrap_or_default()
        {
            stream.extend(quote! {
                async fn #name(&self #(, #arg_names: #args)*) -> fdo::Result<#ret> {
                    let (tx, rx) = oneshot::channel();
                    let path = self.proxy.#name(#(#arg_names),*).await.map_err(zbus_to_zbus_fdo)?;
                    self.job_manager.send(JobManagerCommand::MirrorJob {
                        connection: self.proxy.inner().connection().clone(),
                        path,
                        reply: tx,
                    }).map_err(to_zbus_fdo_error)?;
                    rx.await.map_err(to_zbus_fdo_error)?
                }
            });
        } else {
            stream.extend(quote! {
                async fn #name(&self #(, #arg_names: #args)*) -> fdo::Result<#ret> {
                    self.proxy.#name(#(#arg_names),*).await.map_err(zbus_to_zbus_fdo)
                }
            });
        }
    }
}

impl ToTokens for Property {
    fn to_tokens(&self, stream: &mut TokenStream2) {
        let attr = &self.attr;
        let ty = &self.ty;
        let name = &self.name;
        if self.setter {
            stream.extend(quote! {
                #attr
                async fn #name(&self, arg: #ty) -> zbus::Result<()> {
                    self.proxy.#name(arg).await
                }
            });
        } else {
            stream.extend(quote! {
                #attr
                async fn #name(&self) -> fdo::Result<#ty> {
                    Ok(self.proxy.#name().await?)
                }
            });
        }
    }
}

#[proc_macro_attribute]
pub fn interface(attr: TokenStream, input: TokenStream) -> TokenStream {
    let iface = parse_macro_input!(input as Interface);
    let attr = TokenStream2::from(attr);

    let out = quote! {
        #[zbus::interface(#attr)]
        #iface
    };
    out.into()
}

#[proc_macro_attribute]
pub fn remote(attr: TokenStream, input: TokenStream) -> TokenStream {
    let imp: TokenStream2 = input.clone().into();
    let mut iface = parse_macro_input!(input as BaseInterface);
    let attr = iface
        .split_attrs(attr.into())
        .unwrap_or_else(syn::Error::into_compile_error);

    let out = quote! {
        #[interface(#attr)]
        #iface

        #[interface(#attr)]
        #imp
    };
    out.into()
}

#[proc_macro_derive(RemoteManager, attributes(remote, context))]
pub fn remote_manager(input: TokenStream) -> TokenStream {
    let iface = parse_macro_input!(input as RemoteInterface);

    let name = &iface.name;
    let var = &iface.vars;

    let config_name = format_ident!("{name}Config");
    let stripped_vars: Vec<Ident> = var
        .iter()
        .map(|var| {
            var.to_string()
                .strip_prefix("remote_")
                .map(|var| format_ident!("{var}"))
                .unwrap_or(var.clone())
        })
        .collect();
    let context: Vec<TokenStream2> = stripped_vars
        .iter()
        .map(|var| {
            let context = format_ident!("context_{var}");
            if iface.contexts.contains(&context) {
                quote!(self.#context.clone())
            } else {
                quote!(())
            }
        })
        .collect();

    let iface = &iface.ifaces;
    let iface_str: Vec<String> = iface.iter().map(ToString::to_string).collect();

    let tokens = quote! {
        impl #name {
            async fn register(
                &mut self,
                name: &str,
                object: OwnedObjectPath,
                bus_name: OwnedBusName,
                ctxt: Option<&SignalEmitter<'_>>,
                is_transient: bool,
            ) -> fdo::Result<bool> {
                let object_server = self.session.object_server();
                tracing::debug!("Registering interface {name} at {object} {bus_name}, transient: {is_transient}");

                match name {
                    #(_ if name == <#iface as Interface>::name().as_str() => {
                        if let Some(iface) = self.#var.as_mut() {
                            match iface.register().await {
                                Ok(_) => (),
                                Err(fdo::Error::ServiceUnknown(_) | fdo::Error::NameHasNoOwner(_)) => (),
                                Err(e) => return Err(e.into()),
                            }
                        } else {
                            let mut iface = <#iface as RemoteInterface>::Owner::new(
                                bus_name,
                                object,
                                &self.session,
                                &self.system,
                                is_transient,
                                #context,
                                self.job_manager.clone(),
                            )
                            .await?;
                            match iface.register().await {
                                Ok(_) => (),
                                Err(fdo::Error::ServiceUnknown(_) | fdo::Error::NameHasNoOwner(_)) => (),
                                Err(e) => return Err(e.into()),
                            }
                            self.#var = Some(iface);
                        }
                        if let Some(ctxt) = ctxt {
                            self.remote_interfaces_changed(ctxt).await?;
                        }
                        Ok(true)
                    })*
                    _ => {
                        Err(fdo::Error::InvalidArgs(format!(
                            "Unknown interface {name}"
                        )))
                    }
                }
            }

            async fn load(&mut self, name: &str, ctxt: &SignalEmitter<'_>) -> Result<()> {
                use anyhow::bail;

                tracing::debug!("Interface {name} loading");
                match name {
                    #(_ if name == <#iface as Interface>::name().as_str() => {
                        let Some(iface) = self.#var.as_mut() else {
                            bail!("Interface {name} not registered!");
                        };
                        match iface.register().await {
                            Ok(_) => {
                                iface.start_signal_task().await?;
                            },
                            Err(fdo::Error::ServiceUnknown(_) | fdo::Error::NameHasNoOwner(_)) => (),
                            Err(e) => return Err(e.into()),
                        }
                        self.remote_interfaces_changed(ctxt).await?;
                    })*
                    _ => bail!("Unknown interface {name}"),
                }
                Ok(())
            }

            async fn unregister(
                &mut self,
                name: &str,
                sender: Option<&UniqueName<'_>>,
                ctxt: &SignalEmitter<'_>,
                transient: bool,
            ) -> fdo::Result<bool> {
                let object_server = self.session.object_server();

                match name {
                    #(_ if name == <#iface as Interface>::name().as_str() => {
                        let Some(iface) = self.#var.as_ref() else {
                            return Ok(false);
                        };
                        if let Some(sender) = sender {
                            if iface.remote() != sender {
                                return Err(fdo::Error::AccessDenied(format!(
                                    "Interface {name} is owned by a different remote"
                                )));
                            }
                        }
                        if let Some(owner) = self.#var.as_mut() {
                            #iface::unregister(Some(&mut owner.context), object_server).await?;
                            owner.registered = false;
                            if !transient || owner.is_transient {
                                self.#var = None;
                            }
                        } else {
                            #iface::unregister(None, object_server).await?;
                        }
                        self.remote_interfaces_changed(ctxt).await?;
                        Ok(true)
                    })*
                    _ => {
                        Err(fdo::Error::InvalidArgs(format!(
                            "Unknown interface {name}"
                        )))
                    }
                }
            }

            async fn unload(
                &mut self,
                name: &str,
                ctxt: &SignalEmitter<'_>,
            ) -> Result<()> {
                self.unregister(name, None, ctxt, true).await?;
                Ok(())
            }

            async fn configure(
                &mut self,
                config: &#config_name,
            ) -> Result<()> {
                use zbus::names::WellKnownName;
                #(if let Some(config) = &config.#stripped_vars {
                    self.register(
                        #iface::name().as_str(),
                        config.object_path.clone(),
                        BusName::WellKnown(WellKnownName::from(config.bus_name.clone())).into(),
                        None,
                        false,
                    ).await?;
                })*
                Ok(())
            }
        }

        #[interface(name = "com.steampowered.SteamOSManager1.RemoteInterface1")]
        impl #name {
            #[zbus(property)]
            async fn remote_interfaces(&self) -> fdo::Result<Vec<String>> {
                let mut ifaces = Vec::new();
                #(if let Some(iface) = self.#var.as_ref() {
                    if iface.registered {
                        ifaces.push(#iface::name().to_string());
                    }
                })*
                Ok(ifaces)
            }
        }

        #[derive(Clone, Default, Deserialize, Debug)]
        pub(crate) struct #config_name {
            #(
                #[serde(rename = #iface_str)]
                #stripped_vars: Option<RemoteInterfaceConfig>,
            )*
        }

        impl #config_name {
            pub(crate) async fn load() -> Result<#config_name> {
                use config::builder::AsyncState;
                use config::{ConfigBuilder, FileFormat, FileStoredFormat};
                use crate::read_config_directory;

                let builder = ConfigBuilder::<AsyncState>::default();
                let builder = read_config_directory(
                    builder,
                    path("/usr/share/steamos-manager/remotes.d"),
                    FileFormat::Toml.file_extensions(),
                    FileFormat::Toml,
                )
                .await?;
                let builder = read_config_directory(
                    builder,
                    path("/etc/steamos-manager/remotes.d"),
                    FileFormat::Toml.file_extensions(),
                    FileFormat::Toml,
                )
                .await?;
                let config = builder.build().await?;
                Ok(config.try_deserialize()?)
            }
        }
    };

    tokens.into()
}
