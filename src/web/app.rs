//! The client, in Rust.
//!
//! Owns the DOM, the pairing transport, the WebRTC link and the inference loop.
//! Node 0 is the coordinator: it embeds a token, runs block 0, ships the hidden
//! state and waits for a token back. Node 1 is the worker: it runs block 1 and
//! the head, samples, and replies.
//!
//! Replies are matched by `request_id`, not by "whatever arrives next". A frame
//! whose id has no pending entry is dropped with a log line instead of resolving
//! somebody else's step.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use futures_channel::oneshot;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::Event;

use crate::config::{self, Role, DIM, EOS_TOKEN, MAX_SEQ, N_LAYERS, VOCAB_SIZE};
use crate::model::{self, PipelineShard};
use crate::protocol::{
    self, Codec, Header, Hello, TokenReply, FLAG_FINAL, HELLO_ANNOUNCE, HELLO_REPLY,
};
use crate::tensor::{self, SamplerConfig};

use super::dom::{document, now_us, on_event, query_param, set_timeout, window, Refs};
use super::peer::{PeerEvent, PeerLink};
use super::signal::{Signal, SignalEvent, Transport, TransportKind};
use super::viz::{self, LatencySample, NODE0_ACCENT, NODE1_ACCENT};

const MAX_LOG_LINES: usize = 400;
const LATENCY_WINDOW: usize = 48;
const REPLY_TIMEOUT_MS: i32 = 8000;

const RELAY_HINT: &str = "works across browsers and devices. open this url on your phone or another machine \u{2014} same room name \u{2014} and the two pair automatically. the relay only introduces them; activations go peer to peer and it disconnects once you are paired.";
const BROADCAST_HINT: &str = "two tabs in THIS browser only \u{2014} it cannot see other browsers or other devices. use relay or copy/paste for those.";
const MANUAL_HINT: &str = "for two separate devices: node 0 creates a blob and sends it across; node 1 pastes it, applies, and sends its own blob back. nothing but the two browsers is involved.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Info,
    Ok,
    Warn,
    Error,
    Token,
}

impl Level {
    fn as_str(self) -> &'static str {
        match self {
            Level::Info => "info",
            Level::Ok => "ok",
            Level::Warn => "warn",
            Level::Error => "error",
            Level::Token => "token",
        }
    }
}

struct AppState {
    role: Role,
    peer: Option<Rc<PeerLink>>,
    transport: Option<Transport>,
    transport_kind: TransportKind,
    room: String,
    /// node 0 offers, node 1 answers.
    initiator: bool,
    pending: HashMap<u32, oneshot::Sender<TokenReply>>,
    next_request_id: u32,
    generating: bool,
    abort: bool,
    latency: Vec<LatencySample>,
    tokens_seen: u32,
    init_seq: u32,
    log_lines: usize,
    worker_queue: VecDeque<Vec<u8>>,
    worker_busy: bool,
    backend_label: String,
    /// Whether the pairing transport currently sees the other role.
    peer_present: bool,
}

pub struct App {
    refs: Refs,
    /// Taken out of the cell for the duration of a forward pass, so a second
    /// caller gets a clean error instead of a double-borrow panic.
    shard: RefCell<Option<PipelineShard>>,
    state: RefCell<AppState>,
}

impl App {
    pub fn boot() {
        let refs = Refs::collect();

        // Without a room in the URL, invent one and write it back. A shared
        // relay makes a fixed default actively wrong: whoever arrived first
        // would be paired with the next stranger to open the page.
        let room = match query_param("room") {
            Some(room) => room,
            None => {
                let generated = super::dom::random_room_id();
                super::dom::set_query_param("room", &generated);
                generated
            }
        };

        let app = Rc::new(App {
            refs,
            shard: RefCell::new(None),
            state: RefCell::new(AppState {
                role: Role::Node0,
                peer: None,
                transport: None,
                transport_kind: TransportKind::Broadcast,
                room: room.clone(),
                initiator: true,
                pending: HashMap::new(),
                next_request_id: 1,
                generating: false,
                abort: false,
                latency: Vec::new(),
                tokens_seen: 0,
                init_seq: 0,
                log_lines: 0,
                worker_queue: VecDeque::new(),
                worker_busy: false,
                backend_label: String::new(),
                peer_present: false,
            }),
        });

        app.refs.set_value("roomInput", &room);
        app.pill("pillWasm", "ok", "rust/wasm: ready");
        app.render_invite();
        app.render_spec_strip();
        app.wire_controls();

        let boot = Rc::clone(&app);
        spawn_local(async move {
            boot.log(
                &format!(
                    "protocol v{} \u{b7} {} params across {} blocks",
                    protocol::PROTOCOL_VERSION,
                    format_count(model::total_param_count()),
                    N_LAYERS
                ),
                Level::Info,
            );
            if !webgpu_available() {
                boot.refs.set_value("backendPref", "cpu");
                boot.banner(
                    "warn",
                    "this browser does not expose webgpu, so the cpu reference kernels run instead. the pipeline still works, just slower.",
                );
            }
            boot.initialise_shard().await;
            // The relay is the only mode that reaches another browser or another
            // device, so it is the default; the picker can switch away from it.
            let kind = TransportKind::parse(&boot.refs.value("pairingMode"));
            boot.attach_transport(kind);
            boot.redraw();
        });
    }

    // -- ui plumbing -------------------------------------------------------

    fn render_spec_strip(&self) {
        let specs = [
            format!("dim {DIM}"),
            format!("{}x{} heads", config::N_HEADS, config::HEAD_DIM),
            format!("ffn {}", config::FFN_HIDDEN),
            format!("vocab {VOCAB_SIZE}"),
            format!("ctx {MAX_SEQ}"),
            format!("{} params", format_count(model::total_param_count())),
            format!("proto v{}", protocol::PROTOCOL_VERSION),
        ];
        let strip = self.refs.get("specStrip");
        strip.set_inner_html("");
        for spec in specs {
            if let Ok(el) = document().create_element("span") {
                el.set_class_name("spec");
                el.set_text_content(Some(&spec));
                let _ = strip.append_child(&el);
            }
        }
    }

    fn log(&self, message: &str, level: Level) {
        let stream = self.refs.get("log");
        let Ok(line) = document().create_element("div") else {
            return;
        };
        line.set_class_name("log-line");
        let _ = line.set_attribute("data-level", level.as_str());

        if let (Ok(time), Ok(body)) = (
            document().create_element("span"),
            document().create_element("span"),
        ) {
            time.set_class_name("log-time");
            time.set_text_content(Some(&clock()));
            body.set_class_name("log-text");
            body.set_text_content(Some(message));
            let _ = line.append_child(&time);
            let _ = line.append_child(&body);
        }
        let _ = stream.append_child(&line);

        // A long run would otherwise grow the DOM without bound and slowly
        // starve the render loop.
        let mut state = self.state.borrow_mut();
        state.log_lines += 1;
        while state.log_lines > MAX_LOG_LINES {
            if let Some(first) = stream.first_child() {
                let _ = stream.remove_child(&first);
            }
            state.log_lines -= 1;
        }
        drop(state);
        stream.set_scroll_top(stream.scroll_height());
    }

    fn pill(&self, name: &str, state: &str, label: &str) {
        self.refs.set_attr(name, "data-state", state);
        if let Ok(Some(inner)) = self.refs.get(name).query_selector("[data-pill-label]") {
            inner.set_text_content(Some(label));
        }
    }

    fn banner(&self, tone: &str, message: &str) {
        self.refs.set_attr("banner", "data-tone", tone);
        self.refs.set_attr("banner", "data-visible", "true");
        self.refs.set_text("banner", message);
    }

    // -- shard -------------------------------------------------------------

    async fn initialise_shard(self: &Rc<Self>) {
        let seq = {
            let mut state = self.state.borrow_mut();
            state.init_seq += 1;
            state.init_seq
        };
        let role = self.state.borrow().role;
        let prefer_gpu = self.refs.value("backendPref") == "gpu";

        self.pill("pillBackend", "busy", "backend: initialising");
        *self.shard.borrow_mut() = None;
        self.set_controls_enabled(false);
        // After a role switch the previous role's "local" tag would otherwise
        // stay on the card that is now the remote one.
        self.refs.set_text("node0Backend", "\u{2014}");
        self.refs.set_text("node1Backend", "\u{2014}");

        let shard = PipelineShard::new(role, prefer_gpu).await;

        // A newer initialisation started while this one awaited the adapter.
        if self.state.borrow().init_seq != seq {
            return;
        }

        let backend = shard.backend_kind().as_str().to_string();
        let device = shard.device_label().to_string();
        let dispatches = shard.dispatch_count();
        let trace = shard.kernel_trace();
        let remaining = shard.remaining_context();
        *self.shard.borrow_mut() = Some(shard);
        self.state.borrow_mut().backend_label = device.clone();
        self.apply_sampler();

        let on_gpu = backend == "webgpu";
        self.pill(
            "pillBackend",
            if on_gpu { "ok" } else { "warn" },
            &format!("backend: {backend}"),
        );
        self.refs.set_text("shardDevice", &device);
        self.refs.set_text(
            "shardDispatches",
            &if on_gpu {
                dispatches.to_string()
            } else {
                format!("{dispatches} (cpu ops)")
            },
        );
        self.refs
            .set_text("shardContext", &format!("{remaining} pos"));
        self.refs
            .set_text("shardNote", &format!("{} \u{b7} {backend}", role.as_str()));
        self.refs.set_text(
            "node0Params",
            &format_count(model::params_for_role(Role::Node0)),
        );
        self.refs.set_text(
            "node1Params",
            &format_count(model::params_for_role(Role::Node1)),
        );

        let local_tag = if role == Role::Node0 {
            "node0Backend"
        } else {
            "node1Backend"
        };
        self.refs
            .set_text(local_tag, &format!("{backend} \u{b7} local"));
        self.refs.set_attr(local_tag, "title", &device);

        self.refs.set_text(
            "kernelTrace",
            &if trace.is_empty() {
                "cpu backend: the same op sequence runs in src/tensor.rs".to_string()
            } else {
                trace
                    .iter()
                    .enumerate()
                    .map(|(i, name)| format!("{:02}  {name}", i + 1))
                    .collect::<Vec<_>>()
                    .join("\n")
            },
        );

        self.log(
            &format!(
                "shard ready \u{2014} {} on {backend} ({device})",
                role.as_str()
            ),
            Level::Ok,
        );
        self.set_controls_enabled(true);
        self.update_generate_button();

        // Re-announce so the peer's topology card stops showing what this node
        // used to be running.
        self.announce_capabilities();
    }

    fn apply_sampler(&self) {
        if let Some(shard) = self.shard.borrow_mut().as_mut() {
            shard.set_sampler(SamplerConfig {
                temperature: self.refs.number("temperature", 0.85).max(0.0) as f32,
                top_k: self.refs.number("topK", 8.0).max(0.0) as u32,
                seed: 0xA5A5_1234,
            });
        }
    }

    // -- pairing -----------------------------------------------------------

    fn attach_transport(self: &Rc<Self>, kind: TransportKind) {
        {
            let mut state = self.state.borrow_mut();
            state.transport = None;
            state.transport_kind = kind;
            state.peer_present = false;
            state.room = {
                let room = self.refs.value("roomInput").trim().to_string();
                if room.is_empty() {
                    "default".to_string()
                } else {
                    room
                }
            };
        }
        self.teardown_link();

        let role = self.state.borrow().role;
        let room = self.state.borrow().room.clone();

        let app = Rc::clone(self);
        let sink: Rc<dyn Fn(SignalEvent)> = Rc::new(move |event| {
            let app = Rc::clone(&app);
            spawn_local(async move { app.on_signal(event).await });
        });

        let manual = kind == TransportKind::Manual;
        self.refs.set_hidden("manualPanel", !manual);
        self.refs.set_value("manualOut", "");
        self.refs.set_value("manualIn", "");
        self.refs.set_text(
            "pairingHint",
            match kind {
                TransportKind::Manual => MANUAL_HINT,
                TransportKind::Relay => RELAY_HINT,
                TransportKind::Broadcast => BROADCAST_HINT,
            },
        );
        self.refs.set_text(
            "btnPair",
            if manual {
                "create my blob"
            } else {
                "pair with peer"
            },
        );

        let transport = Transport::new(kind, &room, role.as_str(), sink);
        self.state.borrow_mut().transport = Some(transport);

        self.render_invite();
        self.pill("pillSignal", "ok", &format!("pairing: {}", kind.label()));
        self.log(&format!("pairing via {}", kind.label()), Level::Info);

        if kind == TransportKind::Broadcast {
            let app = Rc::clone(self);
            spawn_local(async move {
                super::dom::sleep(6000).await;
                let looking = !app.state.borrow().peer_present
                    && app.state.borrow().transport_kind == TransportKind::Broadcast;
                if looking {
                    app.banner(
                        "warn",
                        concat!(
                            "no peer found. \u{201c}same browser\u{201d} pairing only sees ",
                            "other tabs in THIS browser \u{2014} it cannot reach a different ",
                            "browser or another device. switch pairing to ",
                            "\u{201c}relay\u{201d} for those.",
                        ),
                    );
                    app.log(
                        "no peer in this browser; other browsers and devices need relay pairing",
                        Level::Warn,
                    );
                }
            });
        }
    }

    async fn on_signal(self: &Rc<Self>, event: SignalEvent) {
        match event {
            SignalEvent::Registered { polite, room } => {
                self.state.borrow_mut().initiator = !polite;
                let role = self.state.borrow().role;
                self.refs.set_text(
                    "pairNote",
                    &format!("registered as {} in room \"{room}\"", role.as_str()),
                );
                self.refs.set_text("topologyNote", "waiting for a peer");
            }
            SignalEvent::PeerJoined => {
                self.state.borrow_mut().peer_present = true;
                self.log("peer joined", Level::Ok);
                self.refs
                    .set_text("topologyNote", "peer joined \u{2014} negotiating");
                if self.state.borrow().initiator {
                    self.start_pairing().await;
                }
            }
            SignalEvent::PeerLeft => {
                self.state.borrow_mut().peer_present = false;
                self.log("peer left", Level::Warn);
                self.refs.set_text("topologyNote", "peer disconnected");
                self.teardown_link();
            }
            SignalEvent::RoleTaken => {
                // Whatever the message calls it, the role that is taken is ours
                // -- that is what the collision means -- so the free one is
                // simply the other half of the pipeline. Deriving it from our
                // own state means a transport that omits the name still works.
                let taken = self.state.borrow().role;
                let free = taken.other();
                self.log(
                    &format!(
                        "{} is taken in this room; taking {} instead",
                        taken.as_str(),
                        free.as_str()
                    ),
                    Level::Warn,
                );
                self.refs.set_value(
                    if free == Role::Node0 {
                        "roleNode0"
                    } else {
                        "roleNode1"
                    },
                    "on",
                );
                if let Some(input) = self
                    .refs
                    .get(if free == Role::Node0 {
                        "roleNode0"
                    } else {
                        "roleNode1"
                    })
                    .dyn_ref::<web_sys::HtmlInputElement>()
                {
                    input.set_checked(true);
                }
                self.switch_role(free).await;
            }
            SignalEvent::Offer(sdp) => {
                if self.state.borrow().initiator {
                    return; // node 0 never accepts an offer
                }
                self.log("received an sdp offer", Level::Info);
                let link = self.ensure_link();
                if let Err(err) = link.accept_offer(&sdp).await {
                    self.log(&err, Level::Error);
                }
            }
            SignalEvent::Answer(sdp) => {
                self.log("received an sdp answer", Level::Info);
                let link = self.peer_link();
                if let Some(link) = link {
                    if let Err(err) = link.accept_answer(&sdp).await {
                        self.log(&err, Level::Error);
                    }
                }
            }
            SignalEvent::IceCandidate(json) => {
                let link = self.peer_link();
                if let Some(link) = link {
                    link.add_candidate(&json).await;
                }
            }
            SignalEvent::TransportError(message) => {
                self.pill("pillSignal", "warn", "pairing: relay unreachable");
                self.log(&message, Level::Error);

                // A dead relay should not leave the page unable to pair at all.
                // Fall back to same-browser so two tabs still work, and say
                // plainly what that costs.
                if self.state.borrow().transport_kind == TransportKind::Relay {
                    self.banner(
                        "warn",
                        concat!(
                            "the signaling relay is unreachable, so pairing fell back to ",
                            "\u{201c}same browser\u{201d} \u{2014} two tabs here still work. ",
                            "to pair with another device choose \u{201c}copy / paste\u{201d}, ",
                            "or point at a relay with ?relay=wss://your-relay/ws",
                        ),
                    );
                    self.refs.set_value("pairingMode", "broadcast");
                    self.attach_transport(TransportKind::Broadcast);
                }
            }
            SignalEvent::Blob { kind, blob } => {
                self.refs.set_value("manualOut", &blob);
                self.refs.set_text(
                    "manualOutNote",
                    &format!("{kind} \u{b7} {} chars", blob.len()),
                );
                self.refs.set_text(
                    "manualOutLabel",
                    if kind == "offer" {
                        "1 \u{b7} your offer \u{2014} send it to node 1"
                    } else {
                        "2 \u{b7} your answer \u{2014} send it back to node 0"
                    },
                );
                self.log(
                    &format!("{kind} blob ready ({} chars)", blob.len()),
                    Level::Ok,
                );
            }
        }
    }

    /// A failed connection is not recoverable in place: ICE restart still needs
    /// a fresh offer. Rebuild it, but only from the offering side and only while
    /// the peer is still announcing itself.
    fn retry_pairing(self: &Rc<Self>) {
        let (initiator, peer_present, kind) = {
            let state = self.state.borrow();
            (state.initiator, state.peer_present, state.transport_kind)
        };
        if !initiator {
            return;
        }
        // The signaling socket is dropped once paired, so it has to come back
        // before a fresh offer can reach anyone.
        if !peer_present || kind == TransportKind::Relay {
            self.attach_transport(kind);
            return;
        }
        let app = Rc::clone(self);
        spawn_local(async move {
            super::dom::sleep(1200).await;
            // The peer may have gone for good while we waited.
            if app.state.borrow().peer_present {
                app.log("retrying the peer connection", Level::Warn);
                app.start_pairing().await;
            }
        });
    }

    async fn start_pairing(self: &Rc<Self>) {
        self.pill("pillLink", "busy", "peer link: negotiating");
        let link = self.ensure_link();
        if let Err(err) = link.create_offer().await {
            self.log(&err, Level::Error);
        } else {
            self.log("sent an sdp offer", Level::Info);
        }
    }

    /// Build a fresh peer connection, discarding any previous one.
    fn ensure_link(self: &Rc<Self>) -> Rc<PeerLink> {
        self.teardown_link();

        let (initiator, trickle) = {
            let state = self.state.borrow();
            (state.initiator, state.transport_kind.trickles())
        };

        let app = Rc::clone(self);
        let sink: Rc<dyn Fn(PeerEvent)> = Rc::new(move |event| {
            let app = Rc::clone(&app);
            spawn_local(async move { app.on_peer(event).await });
        });

        let link = PeerLink::new(initiator, trickle, sink).expect("peer connection");
        self.state.borrow_mut().peer = Some(Rc::clone(&link));
        link
    }

    fn teardown_link(&self) {
        let previous = self.state.borrow_mut().peer.take();
        if let Some(link) = previous {
            link.close();
        }
        self.refs.set_attr("bridgeWire", "data-live", "false");
        self.pill("pillLink", "idle", "peer link: idle");
        self.reject_pending("peer link closed");
        self.update_generate_button();
    }

    async fn on_peer(self: &Rc<Self>, event: PeerEvent) {
        match event {
            PeerEvent::LocalDescription { kind, sdp } => {
                let signal = if kind == "offer" {
                    Signal::Offer(sdp)
                } else {
                    Signal::Answer(sdp)
                };
                if let Some(transport) = self.state.borrow().transport.as_ref() {
                    transport.send(signal);
                }
            }
            PeerEvent::LocalCandidate(json) => {
                if let Some(transport) = self.state.borrow().transport.as_ref() {
                    transport.send(Signal::Ice(json));
                }
            }
            PeerEvent::ConnectionState(state) => {
                self.log(&format!("peer connection state: {state}"), Level::Info);
                match state.as_str() {
                    "connected" => self.pill("pillLink", "ok", "peer link: connected"),
                    "failed" => {
                        self.pill("pillLink", "error", "peer link: failed");
                        self.retry_pairing();
                    }
                    "disconnected" => self.pill("pillLink", "warn", "peer link: disconnected"),
                    _ => {}
                }
            }
            PeerEvent::ChannelOpen => {
                self.pill("pillLink", "ok", "peer link: channel open");
                self.refs.set_text("topologyNote", "data channel open");
                self.refs.set_attr("bridgeWire", "data-live", "true");
                self.log("data channel open", Level::Ok);
                self.announce_capabilities();
                self.update_generate_button();
                self.poll_transport_path();
                self.release_signaling();
            }
            PeerEvent::ChannelClosed => {
                self.refs.set_attr("bridgeWire", "data-live", "false");
                self.pill("pillLink", "warn", "peer link: closed");
                self.reject_pending("peer link closed");
                self.update_generate_button();
            }
            PeerEvent::Frame(bytes) => self.on_frame(bytes).await,
            PeerEvent::Log(message) => self.log(&message, Level::Warn),
        }
    }

    /// Show the URL that carries this room, plus a QR of it for phones.
    ///
    /// Only meaningful for transports that can reach another device; the
    /// same-browser mode pairs tabs, not machines.
    fn render_invite(&self) {
        let kind = self.state.borrow().transport_kind;
        let room = self.state.borrow().room.clone();
        let show = kind != TransportKind::Broadcast;
        self.refs.set_hidden("invitePanel", !show);
        if !show {
            return;
        }

        let url = super::dom::set_query_param("room", &room);
        self.refs.set_value("inviteLink", &url);

        match viz::draw_qr(&self.refs.canvas("inviteQr"), &url) {
            Ok(()) => self
                .refs
                .set_text("inviteNote", &format!("room \u{201c}{room}\u{201d}")),
            Err(err) => self.refs.set_text("inviteNote", &err),
        }
    }

    /// Hang up the signaling connection a moment after pairing succeeds.
    ///
    /// The delay covers a link that fails immediately. After this the relay
    /// holds no connection for this peer at all, so its concurrent load tracks
    /// how many peers are *pairing* rather than how many exist.
    fn release_signaling(self: &Rc<Self>) {
        let app = Rc::clone(self);
        spawn_local(async move {
            super::dom::sleep(5000).await;
            let still_open = app
                .state
                .borrow()
                .peer
                .as_ref()
                .map(|p| p.is_open())
                .unwrap_or(false);
            if !still_open {
                return;
            }
            if let Some(transport) = app.state.borrow().transport.as_ref() {
                transport.release();
            }
            app.log(
                "paired \u{2014} released the signaling connection; the relay is out of the loop",
                Level::Info,
            );
        });
    }

    /// Report which route the data channel took, refreshed while it stays open.
    fn poll_transport_path(self: &Rc<Self>) {
        let app = Rc::clone(self);
        spawn_local(async move {
            loop {
                let Some(link) = app.peer_link() else { break };
                if !link.is_open() {
                    break;
                }
                if let Some((local, remote, rtt_ms)) = link.selected_path().await {
                    let transport = rtt_ms
                        .map(|ms| format!(" \u{b7} transport {ms:.1} ms"))
                        .unwrap_or_default();
                    app.refs.set_text(
                        "bridgePath",
                        &format!("path: {local} \u{2192} {remote}{transport}"),
                    );
                }
                super::dom::sleep(2000).await;
            }
        });
    }

    // -- protocol ----------------------------------------------------------

    /// Clone the link out of the cell so the borrow never spans an await.
    fn peer_link(&self) -> Option<Rc<PeerLink>> {
        self.state.borrow().peer.clone()
    }

    fn local_hello(&self) -> Hello {
        Hello {
            dim: DIM as u32,
            n_layers: N_LAYERS as u32,
            vocab_size: VOCAB_SIZE as u32,
            max_seq: MAX_SEQ as u32,
            // The device label already names the backend, so it stands alone on
            // the peer's card.
            backend: self.state.borrow().backend_label.clone(),
        }
    }

    fn announce_capabilities(self: &Rc<Self>) {
        self.send_hello(HELLO_ANNOUNCE);
    }

    fn send_hello(self: &Rc<Self>, request_id: u32) {
        let Some(link) = self.peer_link() else {
            return;
        };
        if !link.is_open() {
            return;
        }
        let frame = protocol::encode_hello(&self.local_hello(), request_id);
        let app = Rc::clone(self);
        spawn_local(async move {
            if let Err(err) = link.send(&frame).await {
                app.log(
                    &format!("could not announce capabilities: {err}"),
                    Level::Warn,
                );
            }
        });
    }

    async fn on_frame(self: &Rc<Self>, bytes: Vec<u8>) {
        let header = match Header::parse(&bytes) {
            Ok(header) => header,
            Err(err) => {
                self.log(&format!("unparseable frame dropped: {err}"), Level::Error);
                return;
            }
        };

        match header.opcode {
            protocol::OP_HELLO => self.on_hello(&bytes, header.request_id),
            protocol::OP_TOKEN => self.on_token_frame(&bytes, header.request_id),
            protocol::OP_ERROR => self.on_error_frame(&bytes, header.request_id),
            protocol::OP_RESET => {
                if let Some(shard) = self.shard.borrow_mut().as_mut() {
                    shard.reset();
                }
                self.state.borrow_mut().tokens_seen = 0;
                self.refs.set_text("node1Tokens", "0");
                self.log(
                    "coordinator reset the sequence; kv cache cleared",
                    Level::Info,
                );
            }
            protocol::OP_ACTIVATION => self.enqueue_activation(bytes),
            other => self.log(&format!("ignored opcode 0x{other:02x}"), Level::Warn),
        }
    }

    fn on_hello(self: &Rc<Self>, bytes: &[u8], request_id: u32) {
        match protocol::decode_hello(bytes) {
            Ok((_, peer)) => {
                let local = self.local_hello();
                if peer.shape() != local.shape() {
                    let message = format!(
                        "peer model mismatch: peer is dim={} layers={} vocab={} ctx={}, this build is dim={} layers={} vocab={} ctx={}",
                        peer.dim, peer.n_layers, peer.vocab_size, peer.max_seq,
                        local.dim, local.n_layers, local.vocab_size, local.max_seq
                    );
                    self.banner("error", &message);
                    self.log(&message, Level::Error);
                    return;
                }

                self.log(
                    &format!(
                        "peer model matches: dim {} | {} layers | vocab {} | ctx {} \u{b7} peer backend {}",
                        peer.dim, peer.n_layers, peer.vocab_size, peer.max_seq, peer.backend
                    ),
                    Level::Ok,
                );
                self.refs
                    .set_text("topologyNote", "paired \u{2014} models agree");

                let peer_tag = if self.state.borrow().role == Role::Node0 {
                    "node1Backend"
                } else {
                    "node0Backend"
                };
                self.refs.set_text(peer_tag, &peer.backend);
                self.refs.set_attr(peer_tag, "title", &peer.backend);

                // Answer an announcement exactly once; a reply is never answered.
                if request_id == HELLO_ANNOUNCE {
                    self.send_hello(HELLO_REPLY);
                }
            }
            Err(err) => {
                let message = format!("peer handshake rejected: {err}");
                self.banner("error", &message);
                self.log(&message, Level::Error);
            }
        }
    }

    fn on_token_frame(&self, bytes: &[u8], request_id: u32) {
        match protocol::decode_token(bytes) {
            Ok((_, reply)) => {
                let sender = self.state.borrow_mut().pending.remove(&request_id);
                match sender {
                    Some(sender) => {
                        let _ = sender.send(reply);
                    }
                    None => self.log(
                        &format!("dropped an unmatched token frame (request {request_id})"),
                        Level::Warn,
                    ),
                }
            }
            Err(err) => self.log(&format!("malformed token frame: {err}"), Level::Error),
        }
    }

    fn on_error_frame(&self, bytes: &[u8], request_id: u32) {
        let message = protocol::decode_error(bytes)
            .map(|(_, m)| m)
            .unwrap_or_else(|e| e.to_string());
        // Dropping the sender wakes the waiting step with a cancellation.
        self.state.borrow_mut().pending.remove(&request_id);
        self.log(&format!("worker reported: {message}"), Level::Error);
    }

    fn reject_pending(&self, _reason: &str) {
        self.state.borrow_mut().pending.clear();
    }

    // -- worker (node 1) ---------------------------------------------------

    /// Block 1 is stateful, so frames are processed strictly in order; running
    /// two positions concurrently would corrupt the KV cache.
    fn enqueue_activation(self: &Rc<Self>, bytes: Vec<u8>) {
        {
            let mut state = self.state.borrow_mut();
            state.worker_queue.push_back(bytes);
            if state.worker_busy {
                return;
            }
            state.worker_busy = true;
        }

        let app = Rc::clone(self);
        spawn_local(async move {
            loop {
                let next = app.state.borrow_mut().worker_queue.pop_front();
                match next {
                    Some(frame) => app.process_activation(frame).await,
                    None => break,
                }
            }
            app.state.borrow_mut().worker_busy = false;
        });
    }

    async fn process_activation(self: &Rc<Self>, bytes: Vec<u8>) {
        let request = match protocol::decode_activation(&bytes) {
            Ok(request) => request,
            Err(err) => {
                self.log(&format!("malformed activation frame: {err}"), Level::Error);
                return;
            }
        };
        let seq_pos = request.header.seq_pos;
        let request_id = request.header.request_id;

        let started = now_us();
        let outcome = self.forward(&request.values, seq_pos).await;
        let logits = match outcome {
            Ok(logits) => logits,
            Err(err) => return self.report_worker_error(request_id, seq_pos, &err).await,
        };

        let sampled = self
            .shard
            .borrow()
            .as_ref()
            .map(|shard| shard.sample(&logits, seq_pos))
            .unwrap_or_else(|| Err("shard is not initialised".to_string()));
        let token_id = match sampled {
            Ok(id) => id,
            Err(err) => return self.report_worker_error(request_id, seq_pos, &err).await,
        };
        let compute_us = (now_us() - started).max(0.0) as u32;
        if let Some(shard) = self.shard.borrow_mut().as_mut() {
            shard.record_step(compute_us);
        }

        let header =
            Header::new(protocol::OP_TOKEN, request_id, seq_pos).with_flags(request.header.flags);
        let frame = protocol::encode_token(
            header,
            &TokenReply {
                token_id,
                compute_us,
            },
        );

        let link = self.peer_link();
        if let Some(link) = link {
            if let Err(err) = link.send(&frame).await {
                self.log(&format!("could not reply: {err}"), Level::Error);
                return;
            }
        }

        // Telemetry for the worker's own view.
        let tokens = {
            let mut state = self.state.borrow_mut();
            state.tokens_seen += 1;
            state.tokens_seen
        };
        viz::draw_activations(
            &self.refs.canvas("activationCanvas"),
            &request.values,
            &NODE1_ACCENT,
        );
        let text = config::decode_token_str(token_id);
        self.refs.set_text(
            "activationLabel",
            &format!("hidden state \u{2190} node 0 (pos {seq_pos})"),
        );
        self.refs.set_text(
            "activationStats",
            &format!(
                "norm {:.2} \u{b7} {} received",
                tensor::l2_norm(&request.values),
                format_bytes(bytes.len())
            ),
        );
        self.refs
            .set_text("node1Compute", &format_micros(compute_us));
        self.refs.set_text("node1Tokens", &tokens.to_string());
        self.refs.set_text(
            "tensorNote",
            &format!(
                "codec {} \u{b7} {}",
                if request.header.codec == Codec::Q8 {
                    "int8"
                } else {
                    "f32"
                },
                format_bytes(bytes.len())
            ),
        );
        self.refresh_shard_stats();
        self.log(
            &format!(
                "pos {seq_pos} \u{b7} block 1 + head in {} \u{2192} \"{}\"",
                format_micros(compute_us),
                text.trim()
            ),
            Level::Token,
        );
    }

    /// Report a worker-side failure so the coordinator fails fast instead of
    /// waiting out its timeout.
    async fn report_worker_error(self: &Rc<Self>, request_id: u32, seq_pos: u32, message: &str) {
        self.log(&format!("step failed: {message}"), Level::Error);
        let link = self.peer_link();
        if let Some(link) = link {
            let frame = protocol::encode_error(request_id, seq_pos, message);
            let _ = link.send(&frame).await;
        }
    }

    /// Run the local block. The shard is moved out of its cell for the duration
    /// so a concurrent caller sees `None` rather than panicking on a borrow.
    async fn forward(&self, x: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        let mut shard = self
            .shard
            .borrow_mut()
            .take()
            .ok_or_else(|| "shard is not initialised".to_string())?;
        let result = shard.forward(x, pos).await;
        *self.shard.borrow_mut() = Some(shard);
        result
    }

    // -- coordinator (node 0) ----------------------------------------------

    async fn generate(self: &Rc<Self>) {
        {
            let state = self.state.borrow();
            if state.generating || state.role != Role::Node0 {
                return;
            }
        }
        let link = match self.peer_link() {
            Some(link) if link.is_open() => link,
            _ => {
                self.log("the peer link is not open", Level::Warn);
                return;
            }
        };

        let prompt = {
            let text = self.refs.value("promptInput").trim().to_string();
            if text.is_empty() {
                "distributed gpu compute pipeline".to_string()
            } else {
                text
            }
        };
        let tokens = config::encode_prompt_tokens(&prompt);
        let max_tokens = (self.refs.number("maxTokens", 24.0) as usize).clamp(1, MAX_SEQ);
        let codec = if self.refs.value("codec") == "1" {
            Codec::Q8
        } else {
            Codec::F32
        };

        self.reset_transcript();
        self.refs
            .set_text("transcriptPrompt", &format!("{prompt} "));
        self.set_generating(true);
        self.apply_sampler();
        self.log(
            &format!(
                "starting: {max_tokens} tokens, seed token {}, codec {}",
                tokens[tokens.len() - 1],
                if codec == Codec::Q8 { "int8" } else { "f32" }
            ),
            Level::Info,
        );

        {
            let mut state = self.state.borrow_mut();
            state.generating = true;
            state.abort = false;
        }
        if let Some(shard) = self.shard.borrow_mut().as_mut() {
            shard.reset();
        }

        // Tell the worker to drop its own KV cache so both halves start at the
        // same sequence position.
        let reset_id = self.next_request_id();
        if let Err(err) = link.send(&protocol::encode_reset(reset_id)).await {
            self.log(&format!("could not reset the worker: {err}"), Level::Error);
            self.finish_generation(0, 0.0);
            return;
        }

        let started = now_us();
        let mut token = tokens[tokens.len() - 1];
        let mut produced = 0u32;

        for pos in 0..max_tokens as u32 {
            if self.state.borrow().abort {
                break;
            }
            let request_id = self.next_request_id();
            let is_final = pos as usize == max_tokens - 1;

            let step_started = now_us();
            let embedded = {
                let shard = self.shard.borrow();
                match shard.as_ref().map(|s| s.embed(token)) {
                    Some(Ok(x)) => x,
                    Some(Err(err)) => {
                        drop(shard);
                        self.log(&err, Level::Error);
                        break;
                    }
                    None => break,
                }
            };
            let hidden = match self.forward(&embedded, pos).await {
                Ok(hidden) => hidden,
                Err(err) => {
                    self.log(&format!("stage 0 failed: {err}"), Level::Error);
                    break;
                }
            };
            let local_us = (now_us() - step_started).max(0.0) as u32;
            if let Some(shard) = self.shard.borrow_mut().as_mut() {
                shard.record_step(local_us);
            }

            let header = Header::new(protocol::OP_ACTIVATION, request_id, pos)
                .with_codec(codec)
                .with_flags(if is_final { FLAG_FINAL } else { 0 });
            let frame = protocol::encode_activation(header, &hidden);

            self.render_stage0(&hidden, frame.len(), pos, local_us);

            // Park a sender under this request id; the reply resolves it, and a
            // timeout removes it, which cancels the receiver.
            let (sender, receiver) = oneshot::channel::<TokenReply>();
            self.state.borrow_mut().pending.insert(request_id, sender);
            let timeout = {
                let app = Rc::clone(self);
                set_timeout(REPLY_TIMEOUT_MS, move || {
                    app.state.borrow_mut().pending.remove(&request_id);
                })
            };

            let sent_at = now_us();
            if let Err(err) = link.send(&frame).await {
                self.state.borrow_mut().pending.remove(&request_id);
                self.log(&format!("send failed: {err}"), Level::Error);
                break;
            }

            let reply = match receiver.await {
                Ok(reply) => reply,
                Err(_) => {
                    self.log(
                        &format!("no reply to request {request_id} within {REPLY_TIMEOUT_MS} ms"),
                        Level::Error,
                    );
                    break;
                }
            };
            drop(timeout);

            let round_trip_ms = (now_us() - sent_at) / 1000.0;
            token = reply.token_id;
            produced += 1;

            self.render_token(
                pos,
                token,
                round_trip_ms,
                local_us,
                reply.compute_us,
                frame.len(),
                produced,
                (now_us() - started) / 1000.0,
            );

            if token == EOS_TOKEN {
                self.log("worker sampled <eos>, stopping early", Level::Info);
                break;
            }
            // Yield so the UI can paint between tokens.
            super::dom::sleep(0).await;
        }

        self.finish_generation(produced, (now_us() - started) / 1000.0);
    }

    fn next_request_id(&self) -> u32 {
        let mut state = self.state.borrow_mut();
        let id = state.next_request_id;
        state.next_request_id = state.next_request_id.wrapping_add(1);
        id
    }

    fn render_stage0(&self, hidden: &[f32], frame_len: usize, pos: u32, local_us: u32) {
        viz::draw_activations(&self.refs.canvas("activationCanvas"), hidden, &NODE0_ACCENT);
        self.refs.set_text(
            "activationLabel",
            &format!("hidden state \u{2192} node 1 (pos {pos})"),
        );
        self.refs.set_text(
            "activationStats",
            &format!(
                "norm {:.2} \u{b7} {} on the wire",
                tensor::l2_norm(hidden),
                format_bytes(frame_len)
            ),
        );
        self.refs.set_text("node0Compute", &format_micros(local_us));
        self.refs.set_text("node0Sent", &(pos + 1).to_string());
        self.refs.set_text(
            "tensorNote",
            &format!("{DIM} floats \u{b7} {} per frame", format_bytes(frame_len)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_token(
        &self,
        pos: u32,
        token: u32,
        round_trip_ms: f64,
        local_us: u32,
        remote_us: u32,
        frame_len: usize,
        produced: u32,
        elapsed_ms: f64,
    ) {
        let text = config::decode_token_str(token);
        self.append_token(text);

        self.refs
            .set_text("node1Compute", &format_micros(remote_us));
        self.refs.set_text("node1Tokens", &produced.to_string());
        self.refs.set_text(
            "stepNote",
            &format!("pos {} \u{b7} {produced} tokens", pos + 1),
        );
        self.refs.set_text(
            "throughput",
            &format!(
                "{:.2} tok/s",
                produced as f64 / (elapsed_ms / 1000.0).max(1e-6)
            ),
        );
        self.refs.set_text(
            "bridgeBytes",
            &format!("{} \u{2192} 20 b", format_bytes(frame_len)),
        );
        self.refs
            .set_text("bridgeRtt", &format!("{round_trip_ms:.1} ms"));
        self.refresh_shard_stats();

        let sample = LatencySample {
            total_ms: round_trip_ms + local_us as f64 / 1000.0,
            local_ms: local_us as f64 / 1000.0,
            remote_ms: remote_us as f64 / 1000.0,
        };
        let mut state = self.state.borrow_mut();
        state.latency.push(sample);
        if state.latency.len() > LATENCY_WINDOW {
            state.latency.remove(0);
        }
        let mean =
            state.latency.iter().map(|s| s.total_ms).sum::<f64>() / state.latency.len() as f64;
        let count = state.latency.len();
        drop(state);

        self.refs
            .set_text("latencyStats", &format!("mean {mean:.1} ms over {count}"));
        self.redraw();

        self.log(
            &format!(
                "pos {pos} \u{2192} \"{}\" \u{b7} rtt {round_trip_ms:.1} ms (node0 {:.2} ms, node1 {:.2} ms)",
                text.trim(),
                local_us as f64 / 1000.0,
                remote_us as f64 / 1000.0
            ),
            Level::Token,
        );
    }

    fn append_token(&self, text: &str) {
        let stream = self.refs.get("transcript");
        let caret = self.refs.get("caret");
        let Ok(span) = document().create_element("span") else {
            return;
        };
        span.set_class_name("token token-fresh");
        span.set_text_content(Some(text));
        let _ = stream.insert_before(&span, Some(caret));

        // Drop the highlight shortly after so the transition runs.
        let handle = set_timeout(60, move || span.set_class_name("token"));
        std::mem::forget(handle);

        if let Some(scroll) = stream.parent_element() {
            scroll.set_scroll_top(scroll.scroll_height());
        }
    }

    fn finish_generation(self: &Rc<Self>, produced: u32, elapsed_ms: f64) {
        let aborted = {
            let mut state = self.state.borrow_mut();
            state.generating = false;
            state.pending.clear();
            state.abort
        };
        self.refs.set_text(
            "stepNote",
            &format!(
                "{} \u{2014} {produced} tokens in {:.2} s",
                if aborted { "stopped" } else { "complete" },
                elapsed_ms / 1000.0
            ),
        );
        self.log(
            &format!(
                "{}: {produced} tokens in {:.2} s",
                if aborted {
                    "stopped"
                } else {
                    "generation complete"
                },
                elapsed_ms / 1000.0
            ),
            Level::Ok,
        );
        self.set_generating(false);
    }

    fn refresh_shard_stats(&self) {
        if let Some(shard) = self.shard.borrow().as_ref() {
            self.refs.set_text(
                "shardMean",
                &format!("{:.2} ms", shard.stats().mean_compute_us() / 1000.0),
            );
            self.refs.set_text(
                "shardContext",
                &format!("{} pos", shard.remaining_context()),
            );
        }
    }

    // -- controls ----------------------------------------------------------

    fn set_controls_enabled(&self, enabled: bool) {
        for name in [
            "promptInput",
            "maxTokens",
            "temperature",
            "topK",
            "codec",
            "backendPref",
        ] {
            self.refs.set_disabled(name, !enabled);
        }
    }

    fn update_generate_button(&self) {
        let state = self.state.borrow();
        let has_shard = self.shard.borrow().is_some();
        let open = state.peer.as_ref().map(|p| p.is_open()).unwrap_or(false);
        let ready = has_shard && open && state.role == Role::Node0;
        let generating = state.generating;
        drop(state);

        self.refs.set_disabled("btnGenerate", !ready || generating);
        self.refs.set_text(
            "btnGenerate",
            if !has_shard {
                "initialising shard\u{2026}"
            } else if self.state.borrow().role != Role::Node0 {
                "node 1 answers automatically"
            } else if !open {
                "waiting for the peer link"
            } else {
                "run pipeline"
            },
        );
    }

    fn set_generating(&self, active: bool) {
        self.refs.set_hidden("btnStop", !active);
        self.refs.set_hidden("caret", !active);
        self.update_generate_button();
    }

    fn reset_transcript(&self) {
        let stream = self.refs.get("transcript");
        stream.set_inner_html("");
        let _ = stream.append_child(self.refs.get("transcriptPrompt"));
        let _ = stream.append_child(self.refs.get("caret"));
        self.refs.set_text("transcriptPrompt", "");
        self.refs.set_text("stepNote", "idle");
        self.refs.set_text("throughput", "\u{2014} tok/s");
        self.refs.set_text("node0Sent", "0");
        self.refs.set_text("node1Tokens", "0");

        let mut state = self.state.borrow_mut();
        state.latency.clear();
        state.tokens_seen = 0;
        drop(state);
        self.redraw();
    }

    fn redraw(&self) {
        let state = self.state.borrow();
        viz::draw_latency(&self.refs.canvas("latencyCanvas"), &state.latency);
    }

    async fn switch_role(self: &Rc<Self>, role: Role) {
        if self.state.borrow().role == role {
            return;
        }
        self.state.borrow_mut().role = role;

        self.refs.set_attr(
            "cardNode0",
            "data-active",
            if role == Role::Node0 { "true" } else { "false" },
        );
        self.refs.set_attr(
            "cardNode1",
            "data-active",
            if role == Role::Node1 { "true" } else { "false" },
        );
        self.refs.set_hidden("generatePanel", role != Role::Node0);

        self.teardown_link();
        self.log(&format!("switched role to {}", role.as_str()), Level::Info);
        self.initialise_shard().await;

        let kind = self.state.borrow().transport_kind;
        self.attach_transport(kind);
    }

    fn wire_controls(self: &Rc<Self>) {
        let radio = |name: &str, role: Role| {
            let app = Rc::clone(self);
            let target: web_sys::EventTarget = self.refs.get(name).clone().into();
            on_event(&target, "change", move |_| {
                let app = Rc::clone(&app);
                spawn_local(async move { app.switch_role(role).await });
            });
        };
        radio("roleNode0", Role::Node0);
        radio("roleNode1", Role::Node1);

        let click = |name: &str, handler: Box<dyn Fn(Rc<App>)>| {
            let app = Rc::clone(self);
            let target: web_sys::EventTarget = self.refs.get(name).clone().into();
            on_event(&target, "click", move |_: Event| handler(Rc::clone(&app)));
        };

        click(
            "btnGenerate",
            Box::new(|app| spawn_local(async move { app.generate().await })),
        );
        click(
            "btnStop",
            Box::new(|app| app.state.borrow_mut().abort = true),
        );
        click("btnClear", Box::new(|app| app.reset_transcript()));
        click(
            "btnClearLog",
            Box::new(|app| {
                app.refs.get("log").set_inner_html("");
                app.state.borrow_mut().log_lines = 0;
            }),
        );
        click(
            "btnPair",
            Box::new(|app| {
                spawn_local(async move {
                    if app.state.borrow().initiator {
                        app.start_pairing().await;
                    } else {
                        app.log("node 1 answers; pair from node 0", Level::Info);
                    }
                })
            }),
        );
        click(
            "btnReconnect",
            Box::new(|app| {
                let kind = app.state.borrow().transport_kind;
                app.attach_transport(kind);
            }),
        );
        click(
            "btnApplyBlob",
            Box::new(|app| {
                let text = app.refs.value("manualIn");
                let outcome = app
                    .state
                    .borrow()
                    .transport
                    .as_ref()
                    .map(|t| t.accept_blob(&text))
                    .unwrap_or_else(|| Err("no transport".to_string()));
                match outcome {
                    Ok(kind) => {
                        app.refs
                            .set_text("manualInNote", &format!("applied their {kind}"));
                        app.refs.set_value("manualIn", "");
                        app.log(&format!("applied the peer {kind} blob"), Level::Ok);
                    }
                    Err(err) => {
                        app.refs.set_text("manualInNote", &err);
                        app.log(&format!("could not apply that blob: {err}"), Level::Error);
                    }
                }
            }),
        );
        click(
            "btnCopyInvite",
            Box::new(|app| {
                let text = app.refs.value("inviteLink");
                if text.is_empty() {
                    return;
                }
                let _ = window().navigator().clipboard().write_text(&text);
                app.refs
                    .set_text("inviteNote", "copied \u{2014} open it on the other device");
            }),
        );
        click(
            "btnCopyBlob",
            Box::new(|app| {
                let text = app.refs.value("manualOut");
                if text.is_empty() {
                    return;
                }
                let clipboard = window().navigator().clipboard();
                let _ = clipboard.write_text(&text);
                app.refs
                    .set_text("manualOutNote", "copied to the clipboard");
            }),
        );

        let change = |name: &str, handler: Box<dyn Fn(Rc<App>)>| {
            let app = Rc::clone(self);
            let target: web_sys::EventTarget = self.refs.get(name).clone().into();
            on_event(&target, "change", move |_: Event| handler(Rc::clone(&app)));
        };

        change(
            "backendPref",
            Box::new(|app| spawn_local(async move { app.initialise_shard().await })),
        );
        change(
            "pairingMode",
            Box::new(|app| {
                let kind = TransportKind::parse(&app.refs.value("pairingMode"));
                app.attach_transport(kind);
            }),
        );
        change(
            "roomInput",
            Box::new(|app| {
                let kind = app.state.borrow().transport_kind;
                app.attach_transport(kind);
                app.render_invite();
            }),
        );
        change("temperature", Box::new(|app| app.apply_sampler()));
        change("topK", Box::new(|app| app.apply_sampler()));
        change(
            "codec",
            Box::new(|app| {
                let codec = if app.refs.value("codec") == "1" {
                    Codec::Q8
                } else {
                    Codec::F32
                };
                app.refs.set_text(
                    "tensorNote",
                    &format!(
                        "{DIM} floats \u{b7} {} per frame",
                        format_bytes(protocol::HEADER_LEN + codec.payload_len(DIM))
                    ),
                );
            }),
        );

        let app = Rc::clone(self);
        let target: web_sys::EventTarget = window().into();
        on_event(&target, "resize", move |_| {
            app.redraw();
            app.render_invite();
        });
    }
}

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

fn format_micros(us: u32) -> String {
    format!("{:.2} ms", us as f64 / 1000.0)
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} b")
    } else {
        format!("{:.1} kb", bytes as f64 / 1024.0)
    }
}

fn format_count(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn clock() -> String {
    let date = js_sys::Date::new_0();
    format!(
        "{:02}:{:02}:{:02}",
        date.get_hours(),
        date.get_minutes(),
        date.get_seconds()
    )
}

fn webgpu_available() -> bool {
    js_sys::Reflect::get(
        &js_sys::global(),
        &wasm_bindgen::JsValue::from_str("navigator"),
    )
    .ok()
    .and_then(|nav| js_sys::Reflect::get(&nav, &wasm_bindgen::JsValue::from_str("gpu")).ok())
    .map(|gpu| !gpu.is_undefined() && !gpu.is_null())
    .unwrap_or(false)
}
