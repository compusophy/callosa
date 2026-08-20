//! WebRTC peer link.
//!
//! Two details here matter and are easy to get wrong:
//!
//! 1. ICE candidates routinely arrive before the remote description is set.
//!    `addIceCandidate` rejects in that window, and a dropped candidate can be
//!    the one that would have completed the connection — so they are buffered
//!    and flushed once the description lands.
//!
//! 2. Sends are checked against `bufferedAmount`. A data channel written faster
//!    than it drains grows without bound and eventually closes.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Event, MessageEvent, RtcConfiguration, RtcDataChannel, RtcDataChannelEvent,
    RtcDataChannelState, RtcDataChannelType, RtcIceCandidate, RtcIceCandidateInit,
    RtcIceGatheringState, RtcPeerConnection, RtcPeerConnectionIceEvent, RtcSdpType,
    RtcSessionDescriptionInit, RtcSignalingState,
};

use super::dom::{js_error_string, sleep, Listener};

const CHANNEL_LABEL: &str = "pipeline";
const BUFFER_HIGH_WATER: u32 = 512 * 1024;

/// Events the app reacts to.
#[derive(Debug, Clone)]
pub enum PeerEvent {
    LocalDescription { kind: String, sdp: String },
    LocalCandidate(String),
    ConnectionState(String),
    ChannelOpen,
    ChannelClosed,
    Frame(Vec<u8>),
    Log(String),
}

type Sink = Rc<dyn Fn(PeerEvent)>;

/// STUN alone only works when at least one side is directly reachable once a
/// hole is punched. Behind symmetric NAT -- common on mobile networks and some
/// corporate wifi -- the initial checks can succeed and then the path dies,
/// which looks exactly like a connection that comes up and drops seconds later.
///
/// TURN is the fallback for that: it relays the flow when no direct path
/// survives. The trade is real and worth stating plainly -- when the selected
/// pair is `relay`, activations pass through the TURN server rather than going
/// peer to peer. The topology panel names the candidate types for exactly this
/// reason, so you can see when it happens.
const STUN_URLS: [&str; 2] = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
];

/// Public TURN, overridable with `?turn=turn:host:port` plus `?turn_user=` and
/// `?turn_pass=` when you would rather not depend on someone else's.
const TURN_URLS: [&str; 3] = [
    "turn:openrelay.metered.ca:80",
    "turn:openrelay.metered.ca:443",
    "turn:openrelay.metered.ca:443?transport=tcp",
];
const TURN_USER: &str = "openrelayproject";
const TURN_PASS: &str = "openrelayproject";

fn ice_server(urls: &str, credentials: Option<(&str, &str)>) -> js_sys::Object {
    let server = js_sys::Object::new();
    let set = |key: &str, value: &str| {
        let _ = js_sys::Reflect::set(&server, &JsValue::from_str(key), &JsValue::from_str(value));
    };
    set("urls", urls);
    if let Some((user, pass)) = credentials {
        set("username", user);
        set("credential", pass);
    }
    server
}

fn ice_servers() -> JsValue {
    let servers = js_sys::Array::new();
    for url in STUN_URLS {
        servers.push(&ice_server(url, None));
    }

    match super::dom::query_param("turn") {
        Some(url) => {
            let user = super::dom::query_param("turn_user").unwrap_or_default();
            let pass = super::dom::query_param("turn_pass").unwrap_or_default();
            servers.push(&ice_server(&url, Some((&user, &pass))));
        }
        None => {
            for url in TURN_URLS {
                servers.push(&ice_server(url, Some((TURN_USER, TURN_PASS))));
            }
        }
    }
    servers.into()
}

pub struct PeerLink {
    connection: RtcPeerConnection,
    channel: RefCell<Option<RtcDataChannel>>,
    pending_candidates: RefCell<Vec<String>>,
    sink: Sink,
    trickle: bool,
    /// Listeners are owned so they detach when the link is dropped.
    listeners: RefCell<Vec<Listener>>,
    _on_ice: Closure<dyn FnMut(RtcPeerConnectionIceEvent)>,
    _on_data_channel: Closure<dyn FnMut(RtcDataChannelEvent)>,
    _on_state: Closure<dyn FnMut(Event)>,
}

impl PeerLink {
    pub fn new(initiator: bool, trickle: bool, sink: Sink) -> Result<Rc<Self>, String> {
        let config = RtcConfiguration::new();
        config.set_ice_servers(&ice_servers());
        let connection = RtcPeerConnection::new_with_configuration(&config).map_err(|e| {
            format!(
                "could not create the peer connection: {}",
                js_error_string(&e)
            )
        })?;

        let on_ice = {
            let sink = Rc::clone(&sink);
            Closure::<dyn FnMut(RtcPeerConnectionIceEvent)>::new(
                move |event: RtcPeerConnectionIceEvent| {
                    // Without trickle the candidates are already folded into the
                    // description shipped once gathering completes.
                    if !trickle {
                        return;
                    }
                    if let Some(candidate) = event.candidate() {
                        if let Ok(text) = js_sys::JSON::stringify(&candidate.to_json()) {
                            if let Some(text) = text.as_string() {
                                sink(PeerEvent::LocalCandidate(text));
                            }
                        }
                    }
                },
            )
        };
        connection.set_onicecandidate(Some(on_ice.as_ref().unchecked_ref()));

        let on_state = {
            let sink = Rc::clone(&sink);
            let pc = connection.clone();
            Closure::<dyn FnMut(Event)>::new(move |_: Event| {
                let state = format!("{:?}", pc.connection_state()).to_lowercase();
                sink(PeerEvent::ConnectionState(state));
            })
        };
        connection.set_onconnectionstatechange(Some(on_state.as_ref().unchecked_ref()));

        // ICE reaches `disconnected` well before the connection does, so this is
        // the earliest honest signal that a path has stopped carrying traffic.
        let on_ice_state = {
            let sink = Rc::clone(&sink);
            let pc = connection.clone();
            Closure::<dyn FnMut(Event)>::new(move |_: Event| {
                let state = format!("{:?}", pc.ice_connection_state()).to_lowercase();
                sink(PeerEvent::Log(format!("ice: {state}")));
            })
        };
        connection.set_oniceconnectionstatechange(Some(on_ice_state.as_ref().unchecked_ref()));
        on_ice_state.forget();

        let link = Rc::new(PeerLink {
            connection,
            channel: RefCell::new(None),
            pending_candidates: RefCell::new(Vec::new()),
            sink: Rc::clone(&sink),
            trickle,
            listeners: RefCell::new(Vec::new()),
            _on_ice: on_ice,
            _on_data_channel: Closure::<dyn FnMut(RtcDataChannelEvent)>::new(|_| {}),
            _on_state: on_state,
        });

        // The answering side receives the channel; the offering side creates it.
        let on_data_channel = {
            let link = Rc::downgrade(&link);
            Closure::<dyn FnMut(RtcDataChannelEvent)>::new(move |event: RtcDataChannelEvent| {
                if let Some(link) = link.upgrade() {
                    link.attach_channel(event.channel());
                }
            })
        };
        link.connection
            .set_ondatachannel(Some(on_data_channel.as_ref().unchecked_ref()));
        // Keep it alive for the lifetime of the link.
        on_data_channel.forget();

        if initiator {
            let channel = link.connection.create_data_channel(CHANNEL_LABEL);
            link.attach_channel(channel);
        }

        Ok(link)
    }

    fn attach_channel(self: &Rc<Self>, channel: RtcDataChannel) {
        channel.set_binary_type(RtcDataChannelType::Arraybuffer);
        channel.set_buffered_amount_low_threshold(64 * 1024);

        let target: web_sys::EventTarget = channel.clone().into();
        let mut listeners = self.listeners.borrow_mut();
        listeners.clear();

        listeners.push(Listener::attach(&target, "open", {
            let sink = Rc::clone(&self.sink);
            move |_| sink(PeerEvent::ChannelOpen)
        }));
        listeners.push(Listener::attach(&target, "close", {
            let sink = Rc::clone(&self.sink);
            move |_| sink(PeerEvent::ChannelClosed)
        }));
        listeners.push(Listener::attach(&target, "error", {
            let sink = Rc::clone(&self.sink);
            move |_| sink(PeerEvent::Log("data channel error".to_string()))
        }));
        listeners.push(Listener::attach(&target, "message", {
            let sink = Rc::clone(&self.sink);
            move |event: Event| {
                let Ok(message) = event.dyn_into::<MessageEvent>() else {
                    return;
                };
                let buffer = message.data();
                if let Some(array) = buffer.dyn_ref::<js_sys::ArrayBuffer>() {
                    let bytes = js_sys::Uint8Array::new(array).to_vec();
                    sink(PeerEvent::Frame(bytes));
                }
            }
        }));
        drop(listeners);

        *self.channel.borrow_mut() = Some(channel);
    }

    pub fn is_open(&self) -> bool {
        self.channel
            .borrow()
            .as_ref()
            .map(|c| c.ready_state() == RtcDataChannelState::Open)
            .unwrap_or(false)
    }

    /// Wait until ICE gathering finishes so the description carries every
    /// candidate. Bounded: a gatherer stuck on an unreachable STUN server would
    /// otherwise hang the handshake, and host candidates usually suffice on a LAN.
    async fn wait_for_gathering(&self) {
        for _ in 0..80 {
            if self.connection.ice_gathering_state() == RtcIceGatheringState::Complete {
                return;
            }
            sleep(50).await;
        }
        (self.sink)(PeerEvent::Log(
            "ice gathering timed out; sending the candidates found so far".to_string(),
        ));
    }

    pub async fn create_offer(&self) -> Result<(), String> {
        let offer = JsFuture::from(self.connection.create_offer())
            .await
            .map_err(|e| format!("createOffer failed: {}", js_error_string(&e)))?;
        let description: RtcSessionDescriptionInit = offer.unchecked_into();
        JsFuture::from(self.connection.set_local_description(&description))
            .await
            .map_err(|e| format!("setLocalDescription failed: {}", js_error_string(&e)))?;
        self.emit_local_description("offer").await
    }

    pub async fn accept_offer(&self, sdp: &str) -> Result<(), String> {
        let remote = RtcSessionDescriptionInit::new(RtcSdpType::Offer);
        remote.set_sdp(sdp);
        JsFuture::from(self.connection.set_remote_description(&remote))
            .await
            .map_err(|e| format!("setRemoteDescription failed: {}", js_error_string(&e)))?;
        self.flush_candidates().await;

        let answer = JsFuture::from(self.connection.create_answer())
            .await
            .map_err(|e| format!("createAnswer failed: {}", js_error_string(&e)))?;
        let description: RtcSessionDescriptionInit = answer.unchecked_into();
        JsFuture::from(self.connection.set_local_description(&description))
            .await
            .map_err(|e| format!("setLocalDescription failed: {}", js_error_string(&e)))?;
        self.emit_local_description("answer").await
    }

    pub async fn accept_answer(&self, sdp: &str) -> Result<(), String> {
        // Ignore an answer we are not expecting; a duplicated relay message
        // would otherwise reject with InvalidStateError.
        if self.connection.signaling_state() != RtcSignalingState::HaveLocalOffer {
            return Ok(());
        }
        let remote = RtcSessionDescriptionInit::new(RtcSdpType::Answer);
        remote.set_sdp(sdp);
        JsFuture::from(self.connection.set_remote_description(&remote))
            .await
            .map_err(|e| format!("setRemoteDescription failed: {}", js_error_string(&e)))?;
        self.flush_candidates().await;
        Ok(())
    }

    async fn emit_local_description(&self, kind: &str) -> Result<(), String> {
        if !self.trickle {
            self.wait_for_gathering().await;
        }
        let sdp = self
            .connection
            .local_description()
            .map(|d| d.sdp())
            .ok_or_else(|| "no local description after setLocalDescription".to_string())?;
        (self.sink)(PeerEvent::LocalDescription {
            kind: kind.to_string(),
            sdp,
        });
        Ok(())
    }

    pub async fn add_candidate(&self, json: &str) {
        if self.connection.remote_description().is_none() {
            self.pending_candidates.borrow_mut().push(json.to_string());
            return;
        }
        self.apply_candidate(json).await;
    }

    async fn apply_candidate(&self, json: &str) {
        let Ok(parsed) = js_sys::JSON::parse(json) else {
            return;
        };
        let get = |key: &str| {
            js_sys::Reflect::get(&parsed, &JsValue::from_str(key))
                .ok()
                .and_then(|v| v.as_string())
        };
        let Some(candidate) = get("candidate") else {
            return;
        };

        let init = RtcIceCandidateInit::new(&candidate);
        if let Some(mid) = get("sdpMid") {
            init.set_sdp_mid(Some(&mid));
        }
        if let Some(index) = js_sys::Reflect::get(&parsed, &JsValue::from_str("sdpMLineIndex"))
            .ok()
            .and_then(|v| v.as_f64())
        {
            init.set_sdp_m_line_index(Some(index as u16));
        }

        let Ok(ice) = RtcIceCandidate::new(&init) else {
            return;
        };
        if let Err(e) = JsFuture::from(
            self.connection
                .add_ice_candidate_with_opt_rtc_ice_candidate(Some(&ice)),
        )
        .await
        {
            (self.sink)(PeerEvent::Log(format!(
                "ignored an ice candidate: {}",
                js_error_string(&e)
            )));
        }
    }

    async fn flush_candidates(&self) {
        let queued: Vec<String> = self.pending_candidates.borrow_mut().drain(..).collect();
        for candidate in queued {
            self.apply_candidate(&candidate).await;
        }
    }

    /// Send a frame, waiting for the channel to drain if it has backed up.
    pub async fn send(&self, frame: &[u8]) -> Result<(), String> {
        let channel = self
            .channel
            .borrow()
            .clone()
            .ok_or_else(|| "no data channel".to_string())?;
        if channel.ready_state() != RtcDataChannelState::Open {
            return Err("data channel is not open".to_string());
        }

        let mut waited = 0;
        while channel.buffered_amount() > BUFFER_HIGH_WATER {
            sleep(4).await;
            waited += 4;
            if waited > 5000 {
                return Err("data channel stayed congested".to_string());
            }
            if channel.ready_state() != RtcDataChannelState::Open {
                return Err("data channel closed while draining".to_string());
            }
        }

        channel
            .send_with_u8_array(frame)
            .map_err(|e| format!("send failed: {}", js_error_string(&e)))
    }

    /// The negotiated candidate pair, for the transport readout: which route
    /// the data channel actually took (`host` on a LAN, `srflx` through NAT,
    /// `relay` via TURN) and the transport-level round trip.
    pub async fn selected_path(&self) -> Option<(String, String, Option<f64>)> {
        let report = JsFuture::from(self.connection.get_stats()).await.ok()?;
        // RTCStatsReport implements the maplike interface (get/keys/size) but is
        // not an `instanceof Map`, so a checked cast would always fail here.
        let map = report.unchecked_into::<js_sys::Map>();

        let field =
            |entry: &JsValue, key: &str| js_sys::Reflect::get(entry, &JsValue::from_str(key)).ok();
        let candidate_type = |id: Option<JsValue>| -> String {
            id.and_then(|id| id.as_string())
                .and_then(|id| {
                    let entry = map.get(&JsValue::from_str(&id));
                    js_sys::Reflect::get(&entry, &JsValue::from_str("candidateType"))
                        .ok()
                        .and_then(|v| v.as_string())
                })
                .unwrap_or_else(|| "?".to_string())
        };

        let keys = map.keys();
        while let Ok(next) = keys.next() {
            if next.done() {
                break;
            }
            let entry = map.get(&next.value());
            let is_pair = field(&entry, "type").and_then(|v| v.as_string()).as_deref()
                == Some("candidate-pair");
            let succeeded = field(&entry, "state")
                .and_then(|v| v.as_string())
                .as_deref()
                == Some("succeeded");
            let nominated = field(&entry, "nominated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_pair && succeeded && nominated {
                return Some((
                    candidate_type(field(&entry, "localCandidateId")),
                    candidate_type(field(&entry, "remoteCandidateId")),
                    field(&entry, "currentRoundTripTime")
                        .and_then(|v| v.as_f64())
                        .map(|seconds| seconds * 1000.0),
                ));
            }
        }
        None
    }

    pub fn close(&self) {
        self.listeners.borrow_mut().clear();
        if let Some(channel) = self.channel.borrow_mut().take() {
            channel.close();
        }
        self.connection.set_onicecandidate(None);
        self.connection.set_ondatachannel(None);
        self.connection.set_onconnectionstatechange(None);
        self.connection.close();
    }
}

impl Drop for PeerLink {
    fn drop(&mut self) {
        self.listeners.borrow_mut().clear();
    }
}
