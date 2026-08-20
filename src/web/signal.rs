//! Server-free pairing.
//!
//! WebRTC needs the two peers introduced before they can talk. Neither of these
//! transports involves a server:
//!
//! * [`BroadcastTransport`] — two tabs on the same origin exchange SDP and ICE
//!   through a `BroadcastChannel`. Zero configuration, works on static hosting.
//! * [`ManualTransport`] — for two genuinely separate devices, the description
//!   is packed into a short text blob the user carries across by hand.
//!
//! [`RelayTransport`] covers the case those two cannot: different browsers, or
//! different machines, without making the user ferry anything. It uses a
//! signaling server, but only to introduce the peers — once the data channel is
//! open the relay carries nothing, and the client disconnects from it.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{BroadcastChannel, MessageEvent, WebSocket};

use super::dom::{now_us, set_interval, IntervalHandle};

pub const HEARTBEAT_MS: i32 = 2000;
pub const PEER_TIMEOUT_US: f64 = 7000.0 * 1000.0;

/// What the app needs to hear from a transport.
#[derive(Debug, Clone)]
pub enum SignalEvent {
    Registered {
        polite: bool,
        room: String,
    },
    PeerJoined,
    PeerLeft,
    /// This peer's role is already occupied in the room. The name is not
    /// carried: the taken role is always the local one, so a listener derives
    /// the free half from its own state rather than trusting the wire.
    RoleTaken,
    Offer(String),
    Answer(String),
    IceCandidate(String),
    /// A blob for the user to carry to the other device (manual pairing only).
    Blob {
        kind: String,
        blob: String,
    },
    /// The pairing channel itself failed, as opposed to the peer connection.
    TransportError(String),
}

/// A signaling payload travelling between peers.
#[derive(Debug, Clone)]
pub enum Signal {
    Offer(String),
    Answer(String),
    Ice(String),
}

impl Signal {
    fn kind(&self) -> &'static str {
        match self {
            Signal::Offer(_) => "offer",
            Signal::Answer(_) => "answer",
            Signal::Ice(_) => "ice",
        }
    }

    fn body(&self) -> &str {
        match self {
            Signal::Offer(s) | Signal::Answer(s) | Signal::Ice(s) => s,
        }
    }

    fn parse(kind: &str, body: String) -> Option<Signal> {
        match kind {
            "offer" => Some(Signal::Offer(body)),
            "answer" => Some(Signal::Answer(body)),
            "ice" => Some(Signal::Ice(body)),
            _ => None,
        }
    }
}

type Sink = Rc<dyn Fn(SignalEvent)>;

/// A retained JS callback; dropping it detaches the listener.
type JsHandler = Closure<dyn FnMut(JsValue)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    /// Two tabs in one browser. Cannot see other browsers or other devices.
    Broadcast,
    /// Any two peers, introduced by a signaling server.
    Relay,
    /// Any two peers, introduced by the user carrying a blob across.
    Manual,
}

impl TransportKind {
    pub fn label(self) -> &'static str {
        match self {
            TransportKind::Broadcast => "same browser",
            TransportKind::Relay => "relay",
            TransportKind::Manual => "copy / paste",
        }
    }

    pub fn parse(id: &str) -> TransportKind {
        match id {
            "broadcast" => TransportKind::Broadcast,
            "manual" => TransportKind::Manual,
            _ => TransportKind::Relay,
        }
    }

    /// Manual pairing cannot carry candidates incrementally, so the peer must
    /// wait for ICE gathering to finish before handing over a description.
    pub fn trickles(self) -> bool {
        !matches!(self, TransportKind::Manual)
    }
}

pub enum Transport {
    Broadcast(BroadcastTransport),
    Relay(RelayTransport),
    Manual(ManualTransport),
}

impl Transport {
    pub fn new(kind: TransportKind, room: &str, role: &str, sink: Sink) -> Transport {
        match kind {
            TransportKind::Broadcast => {
                Transport::Broadcast(BroadcastTransport::new(room, role, sink))
            }
            TransportKind::Relay => Transport::Relay(RelayTransport::new(room, role, sink)),
            TransportKind::Manual => Transport::Manual(ManualTransport::new(room, role, sink)),
        }
    }

    pub fn send(&self, signal: Signal) {
        match self {
            Transport::Broadcast(t) => t.send(signal),
            Transport::Relay(t) => t.send(signal),
            Transport::Manual(t) => t.send(signal),
        }
    }

    /// Release the signaling connection once the peers are talking directly.
    /// The relay's concurrent load then tracks peers that are *pairing*, not
    /// peers that exist.
    pub fn release(&self) {
        if let Transport::Relay(t) = self {
            t.release();
        }
    }

    /// Manual pairing only: feed in a blob pasted from the other device.
    pub fn accept_blob(&self, text: &str) -> Result<String, String> {
        match self {
            Transport::Manual(t) => t.accept_blob(text),
            _ => Err("this pairing mode does not use blobs".to_string()),
        }
    }
}

/// Default signaling endpoint, overridable at runtime with `?relay=wss://...`
/// so a fork can point at its own without rebuilding.
pub const DEFAULT_RELAY: &str = "wss://callosa-relay.up.railway.app/ws";

pub fn relay_endpoint() -> String {
    super::dom::query_param("relay").unwrap_or_else(|| DEFAULT_RELAY.to_string())
}

fn other_role(role: &str) -> &'static str {
    if role == "node0" {
        "node1"
    } else {
        "node0"
    }
}

// ---------------------------------------------------------------------------
// BroadcastChannel
// ---------------------------------------------------------------------------

struct BroadcastState {
    peer_present: bool,
    peer_seen_us: f64,
}

pub struct BroadcastTransport {
    channel: BroadcastChannel,
    role: String,
    room: String,
    /// Random per-tab id. When two tabs claim the same role the higher id
    /// yields, so exactly one of them moves and they never swap in lockstep.
    instance: f64,
    state: Rc<RefCell<BroadcastState>>,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _heartbeat: IntervalHandle,
}

impl BroadcastTransport {
    fn new(room: &str, role: &str, sink: Sink) -> Self {
        let channel = BroadcastChannel::new(&format!("pipeline-signal:{room}"))
            .expect("BroadcastChannel is unavailable");
        let instance = js_sys::Math::random();
        let state = Rc::new(RefCell::new(BroadcastState {
            peer_present: false,
            peer_seen_us: 0.0,
        }));

        let on_message = {
            let sink = Rc::clone(&sink);
            let state = Rc::clone(&state);
            let channel_for_reply = channel.clone();
            let role = role.to_string();
            let room = room.to_string();
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                let data = event.data();
                let field = |key: &str| {
                    js_sys::Reflect::get(&data, &JsValue::from_str(key))
                        .ok()
                        .and_then(|v| v.as_string())
                };
                let (Some(kind), Some(from)) = (field("kind"), field("from")) else {
                    return;
                };
                // BroadcastChannel never echoes to its sender, so anything
                // claiming our own role is a second tab in the same seat. The
                // tab with the higher instance id is the one that gives way.
                if from == role {
                    let theirs = js_sys::Reflect::get(&data, &JsValue::from_str("instance"))
                        .ok()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    if instance > theirs {
                        sink(SignalEvent::RoleTaken);
                    }
                    return;
                }
                if from != other_role(&role) {
                    return;
                }

                let note_peer = || {
                    let mut s = state.borrow_mut();
                    s.peer_seen_us = now_us();
                    if !s.peer_present {
                        s.peer_present = true;
                        drop(s);
                        sink(SignalEvent::PeerJoined);
                    }
                };

                match kind.as_str() {
                    "announce" => {
                        {
                            let mut s = state.borrow_mut();
                            s.peer_seen_us = now_us();
                            s.peer_present = true;
                        }
                        sink(SignalEvent::PeerJoined);
                        let _ = channel_for_reply
                            .post_message(&presence_message("present", &role, &room, instance));
                    }
                    "present" | "heartbeat" => note_peer(),
                    "leave" => {
                        let mut s = state.borrow_mut();
                        if s.peer_present {
                            s.peer_present = false;
                            drop(s);
                            sink(SignalEvent::PeerLeft);
                        }
                    }
                    "signal" => {
                        note_peer();
                        let (Some(signal_kind), Some(body)) = (field("signal"), field("body"))
                        else {
                            return;
                        };
                        match Signal::parse(&signal_kind, body) {
                            Some(Signal::Offer(sdp)) => sink(SignalEvent::Offer(sdp)),
                            Some(Signal::Answer(sdp)) => sink(SignalEvent::Answer(sdp)),
                            Some(Signal::Ice(candidate)) => {
                                sink(SignalEvent::IceCandidate(candidate))
                            }
                            None => {}
                        }
                    }
                    _ => {}
                }
            })
        };
        channel.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // A tab that dies without firing `pagehide` is noticed by the timeout.
        let heartbeat = {
            let channel = channel.clone();
            let state = Rc::clone(&state);
            let sink = Rc::clone(&sink);
            let role = role.to_string();
            let room = room.to_string();
            set_interval(HEARTBEAT_MS, move || {
                let _ =
                    channel.post_message(&presence_message("heartbeat", &role, &room, instance));
                let mut s = state.borrow_mut();
                if s.peer_present && now_us() - s.peer_seen_us > PEER_TIMEOUT_US {
                    s.peer_present = false;
                    drop(s);
                    sink(SignalEvent::PeerLeft);
                }
            })
        };

        // `Drop` never runs on navigation, so a departure has to be announced
        // from an unload event or the surviving tab waits out the heartbeat.
        {
            let channel = channel.clone();
            let role = role.to_string();
            let room = room.to_string();
            let target: web_sys::EventTarget = super::dom::window().into();
            super::dom::on_event(&target, "pagehide", move |_| {
                let _ = channel.post_message(&presence_message("leave", &role, &room, instance));
            });
        }

        let transport = BroadcastTransport {
            channel,
            role: role.to_string(),
            room: room.to_string(),
            instance,
            state,
            _on_message: on_message,
            _heartbeat: heartbeat,
        };

        sink(SignalEvent::Registered {
            // node0 offers, node1 answers. Fixed by role, so no glare.
            polite: role == "node1",
            room: room.to_string(),
        });
        transport.announce();
        transport
    }

    fn announce(&self) {
        let _ = self.channel.post_message(&presence_message(
            "announce",
            &self.role,
            &self.room,
            self.instance,
        ));
    }

    fn send(&self, signal: Signal) {
        let message = js_sys::Object::new();
        let set = |key: &str, value: &str| {
            let _ =
                js_sys::Reflect::set(&message, &JsValue::from_str(key), &JsValue::from_str(value));
        };
        set("kind", "signal");
        set("from", &self.role);
        let _ = js_sys::Reflect::set(
            &message,
            &JsValue::from_str("instance"),
            &JsValue::from_f64(self.instance),
        );
        set("room", &self.room);
        set("signal", signal.kind());
        set("body", signal.body());
        let _ = self.channel.post_message(&message);
    }
}

impl Drop for BroadcastTransport {
    fn drop(&mut self) {
        let _ = self.channel.post_message(&presence_message(
            "leave",
            &self.role,
            &self.room,
            self.instance,
        ));
        self.channel.set_onmessage(None);
        self.channel.close();
        self.state.borrow_mut().peer_present = false;
    }
}

fn presence_message(kind: &str, role: &str, room: &str, instance: f64) -> JsValue {
    let message = js_sys::Object::new();
    for (key, value) in [("kind", kind), ("from", role), ("room", room)] {
        let _ = js_sys::Reflect::set(&message, &JsValue::from_str(key), &JsValue::from_str(value));
    }
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("instance"),
        &JsValue::from_f64(instance),
    );
    message.into()
}

// ---------------------------------------------------------------------------
// Manual copy / paste
// ---------------------------------------------------------------------------

pub struct ManualTransport {
    role: String,
    sink: Sink,
}

impl ManualTransport {
    fn new(room: &str, role: &str, sink: Sink) -> Self {
        sink(SignalEvent::Registered {
            polite: role == "node1",
            room: room.to_string(),
        });
        ManualTransport {
            role: role.to_string(),
            sink,
        }
    }

    fn send(&self, signal: Signal) {
        // Candidates ride inside the description here, so standalone ICE is
        // dropped rather than queued for a channel that does not exist.
        let kind = match signal {
            Signal::Ice(_) => return,
            ref other => other.kind(),
        };
        (self.sink)(SignalEvent::Blob {
            kind: kind.to_string(),
            blob: encode_blob(kind, signal.body()),
        });
    }

    fn accept_blob(&self, text: &str) -> Result<String, String> {
        let (kind, body) = decode_blob(text)?;
        match kind.as_str() {
            "offer" => (self.sink)(SignalEvent::Offer(body)),
            "answer" => (self.sink)(SignalEvent::Answer(body)),
            other => return Err(format!("expected an offer or answer blob, got \"{other}\"")),
        }
        let _ = &self.role;
        Ok(kind)
    }
}

/// Blobs are `kind` + base64url(SDP), which keeps them URL-safe and free of
/// characters that break when pasted through a chat client.
fn encode_blob(kind: &str, body: &str) -> String {
    format!("{kind}.{}", base64url_encode(body.as_bytes()))
}

fn decode_blob(text: &str) -> Result<(String, String), String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("nothing to decode".to_string());
    }
    let (kind, encoded) = trimmed
        .split_once('.')
        .ok_or_else(|| "that does not look like a pairing blob".to_string())?;
    let bytes = base64url_decode(encoded)?;
    let body = String::from_utf8(bytes).map_err(|_| "blob is not valid utf-8".to_string())?;
    Ok((kind.to_string(), body))
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

pub fn base64url_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64[(triple >> 18) as usize & 63] as char);
        out.push(B64[(triple >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(B64[(triple >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(B64[triple as usize & 63] as char);
        }
    }
    out
}

pub fn base64url_decode(input: &str) -> Result<Vec<u8>, String> {
    let mut accumulator: u32 = 0;
    let mut bits = 0u32;
    let mut out = Vec::with_capacity(input.len() * 3 / 4);

    for ch in input.chars().filter(|c| !c.is_whitespace() && *c != '=') {
        let value = B64
            .iter()
            .position(|&b| b as char == ch)
            .ok_or_else(|| format!("invalid character '{ch}' in blob"))? as u32;
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64url_round_trips_every_length_class() {
        for text in [
            "",
            "a",
            "ab",
            "abc",
            "abcd",
            "v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\n",
        ] {
            let encoded = base64url_encode(text.as_bytes());
            assert!(
                !encoded.contains('+') && !encoded.contains('/') && !encoded.contains('='),
                "encoding must stay url-safe and unpadded: {encoded}"
            );
            let decoded = base64url_decode(&encoded).expect("decode");
            assert_eq!(decoded, text.as_bytes(), "round trip failed for {text:?}");
        }
    }

    #[test]
    fn base64url_round_trips_arbitrary_bytes() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let decoded = base64url_decode(&base64url_encode(&bytes)).expect("decode");
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn blob_round_trips_and_rejects_garbage() {
        let sdp = "v=0\r\na=ice-ufrag:abcd\r\n";
        let blob = encode_blob("offer", sdp);
        let (kind, body) = decode_blob(&blob).expect("decode");
        assert_eq!(kind, "offer");
        assert_eq!(body, sdp);

        assert!(decode_blob("").is_err());
        assert!(decode_blob("no-separator-here").is_err());
        assert!(decode_blob("offer.!!!!").is_err());
    }
}

// ---------------------------------------------------------------------------
// Signaling relay
// ---------------------------------------------------------------------------

/// Pairs through a WebSocket signaling server.
///
/// This is the only transport that works between different browsers or
/// different machines without human involvement. The server sees the SDP and
/// nothing else: activations and tokens go peer to peer, and [`release`] hangs
/// up the socket once the data channel is open.
///
/// [`release`]: RelayTransport::release
pub struct RelayTransport {
    socket: RefCell<Option<WebSocket>>,
    role: String,
    room: String,
    sink: Sink,
    /// Retained so the callbacks outlive this scope.
    handlers: RefCell<Vec<JsHandler>>,
}

impl RelayTransport {
    fn new(room: &str, role: &str, sink: Sink) -> Self {
        let transport = RelayTransport {
            socket: RefCell::new(None),
            role: role.to_string(),
            room: room.to_string(),
            sink,
            handlers: RefCell::new(Vec::new()),
        };
        transport.connect();
        transport
    }

    fn connect(&self) {
        let url = format!("{}?room={}&role={}", relay_endpoint(), self.room, self.role);

        let socket = match WebSocket::new(&url) {
            Ok(socket) => socket,
            Err(e) => {
                (self.sink)(SignalEvent::TransportError(format!(
                    "could not reach the relay at {url}: {}",
                    super::dom::js_error_string(&e)
                )));
                return;
            }
        };

        let mut handlers = self.handlers.borrow_mut();

        let on_message = {
            let sink = Rc::clone(&self.sink);
            Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
                let Some(text) = js_sys::Reflect::get(&event, &JsValue::from_str("data"))
                    .ok()
                    .and_then(|d| d.as_string())
                else {
                    return;
                };
                let Ok(value) = js_sys::JSON::parse(&text) else {
                    return;
                };
                let field = |key: &str| {
                    js_sys::Reflect::get(&value, &JsValue::from_str(key))
                        .ok()
                        .and_then(|v| v.as_string())
                };
                let flag = |key: &str| {
                    js_sys::Reflect::get(&value, &JsValue::from_str(key))
                        .ok()
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                };

                match field("type").as_deref() {
                    Some("registered") => {
                        sink(SignalEvent::Registered {
                            polite: flag("polite"),
                            room: field("room").unwrap_or_default(),
                        });
                        if flag("peerPresent") {
                            sink(SignalEvent::PeerJoined);
                        }
                    }
                    Some("peer-joined") => sink(SignalEvent::PeerJoined),
                    Some("peer-left") => sink(SignalEvent::PeerLeft),
                    Some("role-taken") => sink(SignalEvent::RoleTaken),
                    Some("relay-full") => sink(SignalEvent::TransportError(
                        "the relay is at capacity; try again shortly".to_string(),
                    )),
                    // Anything else is a peer's signal, forwarded verbatim.
                    Some(kind) => {
                        if let Some(body) = field("body") {
                            match Signal::parse(kind, body) {
                                Some(Signal::Offer(sdp)) => sink(SignalEvent::Offer(sdp)),
                                Some(Signal::Answer(sdp)) => sink(SignalEvent::Answer(sdp)),
                                Some(Signal::Ice(c)) => sink(SignalEvent::IceCandidate(c)),
                                None => {}
                            }
                        }
                    }
                    None => {}
                }
            })
        };
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        handlers.push(on_message);

        let on_error = {
            let sink = Rc::clone(&self.sink);
            let url = url.clone();
            Closure::<dyn FnMut(JsValue)>::new(move |_: JsValue| {
                sink(SignalEvent::TransportError(format!(
                    "the relay connection failed ({url})"
                )));
            })
        };
        socket.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        handlers.push(on_error);

        drop(handlers);
        *self.socket.borrow_mut() = Some(socket);
    }

    fn send(&self, signal: Signal) {
        let Some(socket) = self.socket.borrow().clone() else {
            return;
        };
        if socket.ready_state() != WebSocket::OPEN {
            // Queue nothing: the peer re-offers if a signal is lost, and a
            // dropped candidate is recoverable.
            return;
        }
        let payload = js_sys::Object::new();
        for (key, value) in [("type", signal.kind()), ("body", signal.body())] {
            let _ =
                js_sys::Reflect::set(&payload, &JsValue::from_str(key), &JsValue::from_str(value));
        }
        if let Ok(text) = js_sys::JSON::stringify(&payload) {
            if let Some(text) = text.as_string() {
                let _ = socket.send_with_str(&text);
            }
        }
    }

    /// Hang up. The peers are connected directly now and need nothing further.
    fn release(&self) {
        if let Some(socket) = self.socket.borrow_mut().take() {
            socket.set_onmessage(None);
            socket.set_onerror(None);
            let _ = socket.close();
        }
        self.handlers.borrow_mut().clear();
    }
}

impl Drop for RelayTransport {
    fn drop(&mut self) {
        self.release();
    }
}
