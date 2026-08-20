//! Signaling relay for callosa.
//!
//! Its entire job is to let two browsers discover each other's SDP. Once the
//! WebRTC data channel is up the peers talk directly and the relay is out of the
//! loop — activations and tokens never pass through here.
//!
//! That shape is what makes it cheap to scale, and the code is written to defend
//! it:
//!
//! * **Nothing is stored.** Rooms live in memory and disappear when both sides
//!   leave. There is no database and no history.
//! * **Payloads are opaque.** SDP and ICE are forwarded verbatim, never parsed.
//! * **Every dimension is bounded** — rooms, message size, messages per
//!   connection, connection lifetime. A pairing needs a few dozen small frames;
//!   anything beyond that is a bug or an abuser, and gets disconnected.
//! * **Clients hang up once paired.** Concurrent connections track peers that
//!   are *currently pairing*, not peers that exist. A network of any size costs
//!   the relay only its join rate.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tokio::sync::mpsc;

/// A signaling frame is an SDP blob at worst; 64 KiB is already generous.
const MAX_FRAME_BYTES: usize = 64 * 1024;
/// A pairing needs a few dozen frames. Well past that is not a pairing.
const MAX_FRAMES_PER_CONNECTION: u32 = 400;
/// Nobody should still be negotiating after this long.
const MAX_CONNECTION_SECS: u64 = 10 * 60;
/// Backstop against unbounded room growth.
const MAX_ROOMS: usize = 50_000;
const MAX_ROOM_ID_LEN: usize = 64;
const HEARTBEAT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Node0,
    Node1,
}

impl Role {
    fn parse(value: &str) -> Option<Role> {
        match value {
            "node0" => Some(Role::Node0),
            "node1" => Some(Role::Node1),
            _ => None,
        }
    }

    fn other(self) -> Role {
        match self {
            Role::Node0 => Role::Node1,
            Role::Node1 => Role::Node0,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Role::Node0 => "node0",
            Role::Node1 => "node1",
        }
    }
}

type Outbox = mpsc::UnboundedSender<Message>;

/// Outcome of trying to take a role in a room.
enum Claim {
    Joined { peer_present: bool },
    RoleTaken,
    Full,
}

#[derive(Default)]
struct Room {
    node0: Option<Outbox>,
    node1: Option<Outbox>,
}

impl Room {
    fn slot(&mut self, role: Role) -> &mut Option<Outbox> {
        match role {
            Role::Node0 => &mut self.node0,
            Role::Node1 => &mut self.node1,
        }
    }

    fn peer(&self, role: Role) -> Option<&Outbox> {
        match role.other() {
            Role::Node0 => self.node0.as_ref(),
            Role::Node1 => self.node1.as_ref(),
        }
    }

    fn is_empty(&self) -> bool {
        self.node0.is_none() && self.node1.is_none()
    }
}

#[derive(Default)]
struct Metrics {
    connections: AtomicU64,
    pairings: AtomicU64,
    frames: AtomicU64,
    rejected: AtomicU64,
}

struct Relay {
    rooms: Mutex<HashMap<String, Room>>,
    metrics: Metrics,
}

type Shared = Arc<Relay>;

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let relay: Shared = Arc::new(Relay {
        rooms: Mutex::new(HashMap::new()),
        metrics: Metrics::default(),
    });

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/ws", get(upgrade))
        .with_state(Arc::clone(&relay));

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("could not bind {addr}: {e}"));

    println!("[relay] listening on {addr}");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown())
    .await
    .expect("server error");
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
    println!("\n[relay] shutting down");
}

async fn root() -> impl IntoResponse {
    (
        StatusCode::OK,
        "callosa signaling relay\n\n\
         Connect: /ws?room=<id>&role=<node0|node1>\n\
         It forwards SDP and ICE between two peers in a room and nothing else.\n\
         Once their data channel opens, the peers talk directly and this relay\n\
         carries no further traffic.\n",
    )
}

async fn health(State(relay): State<Shared>) -> impl IntoResponse {
    let rooms = relay.rooms.lock().expect("rooms mutex").len();
    let m = &relay.metrics;
    (
        StatusCode::OK,
        format!(
            "{{\"ok\":true,\"rooms\":{},\"open\":{},\"pairings\":{},\"frames\":{},\"rejected\":{}}}",
            rooms,
            m.connections.load(Ordering::Relaxed),
            m.pairings.load(Ordering::Relaxed),
            m.frames.load(Ordering::Relaxed),
            m.rejected.load(Ordering::Relaxed),
        ),
    )
}

#[derive(Debug)]
struct Params {
    room: String,
    role: Role,
}

fn parse_params(raw: &HashMap<String, String>) -> Result<Params, &'static str> {
    let role = raw
        .get("role")
        .and_then(|r| Role::parse(r))
        .ok_or("role must be node0 or node1")?;

    let room = raw.get("room").map(String::as_str).unwrap_or("default");
    if room.is_empty() || room.len() > MAX_ROOM_ID_LEN {
        return Err("room id must be 1..=64 characters");
    }
    // Keep ids to a boring alphabet so they cannot smuggle control characters
    // into logs or collide through unicode tricks.
    if !room
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err("room id must be alphanumeric, dash or underscore");
    }

    Ok(Params {
        room: room.to_string(),
        role,
    })
}

async fn upgrade(
    ws: WebSocketUpgrade,
    Query(raw): Query<HashMap<String, String>>,
    ConnectInfo(_peer): ConnectInfo<SocketAddr>,
    State(relay): State<Shared>,
) -> Response {
    match parse_params(&raw) {
        Ok(params) => ws
            .max_message_size(MAX_FRAME_BYTES)
            .on_upgrade(move |socket| serve(socket, params, relay)),
        Err(message) => {
            relay.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            (StatusCode::BAD_REQUEST, message).into_response()
        }
    }
}

fn control(kind: &str) -> Message {
    Message::Text(Utf8Bytes::from(format!("{{\"type\":\"{kind}\"}}")))
}

async fn serve(socket: WebSocket, params: Params, relay: Shared) {
    let (outbox, mut inbox) = mpsc::unbounded_channel::<Message>();
    let Params { room, role } = params;

    // Claim the role, or bounce if it is already taken. Refusing beats evicting
    // whoever got there first.
    //
    // The decision is made under the lock and acted on after it, so the guard
    // never spans an await -- which would make this future non-Send and pin the
    // whole relay to one task at a time.
    let claim = {
        let mut rooms = relay.rooms.lock().expect("rooms mutex");
        if !rooms.contains_key(&room) && rooms.len() >= MAX_ROOMS {
            Claim::Full
        } else {
            let entry = rooms.entry(room.clone()).or_default();
            if entry.slot(role).is_some() {
                // The slot was occupied, so the room is not empty and needs no
                // cleanup here.
                Claim::RoleTaken
            } else {
                *entry.slot(role) = Some(outbox.clone());
                Claim::Joined {
                    peer_present: entry.peer(role).is_some(),
                }
            }
        }
    };

    let peer_present = match claim {
        Claim::Joined { peer_present } => peer_present,
        Claim::RoleTaken => {
            relay.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            let _ = outbox.send(Message::Text(Utf8Bytes::from(format!(
                "{{\"type\":\"role-taken\",\"role\":\"{}\"}}",
                role.as_str()
            ))));
            return close_with(socket, inbox).await;
        }
        Claim::Full => {
            relay.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            let _ = outbox.send(control("relay-full"));
            return close_with(socket, inbox).await;
        }
    };

    relay.metrics.connections.fetch_add(1, Ordering::Relaxed);

    let _ = outbox.send(Message::Text(Utf8Bytes::from(format!(
        "{{\"type\":\"registered\",\"role\":\"{}\",\"room\":\"{}\",\"polite\":{},\"peerPresent\":{}}}",
        role.as_str(),
        room,
        role == Role::Node1,
        peer_present
    ))));

    if peer_present {
        relay.metrics.pairings.fetch_add(1, Ordering::Relaxed);
        notify_peer(&relay, &room, role, control("peer-joined"));
    }

    let (mut sink, mut stream) = {
        use futures_util::StreamExt;
        socket.split()
    };

    // Pump this connection's outbox to the socket.
    let writer = tokio::spawn(async move {
        use futures_util::SinkExt;
        while let Some(message) = inbox.recv().await {
            if sink.send(message).await.is_err() {
                break;
            }
        }
        let _ = sink
            .send(Message::Close(Some(CloseFrame {
                code: 1000,
                reason: Utf8Bytes::from("bye"),
            })))
            .await;
    });

    let mut frames: u32 = 0;
    let deadline = tokio::time::sleep(Duration::from_secs(MAX_CONNECTION_SECS));
    tokio::pin!(deadline);
    let mut heartbeat = tokio::time::interval(HEARTBEAT);
    heartbeat.tick().await;

    loop {
        use futures_util::StreamExt;
        tokio::select! {
            _ = &mut deadline => break,
            _ = heartbeat.tick() => {
                if outbox.send(Message::Ping(Default::default())).is_err() {
                    break;
                }
            }
            incoming = stream.next() => {
                let Some(Ok(message)) = incoming else { break };
                match message {
                    // Payloads are forwarded byte for byte; the relay never
                    // looks inside an offer, an answer or a candidate.
                    Message::Text(text) => {
                        frames += 1;
                        if frames > MAX_FRAMES_PER_CONNECTION {
                            relay.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                            break;
                        }
                        relay.metrics.frames.fetch_add(1, Ordering::Relaxed);
                        notify_peer(&relay, &room, role, Message::Text(text));
                    }
                    Message::Close(_) => break,
                    // Binary has no meaning here, and pings are answered by axum.
                    _ => {}
                }
            }
        }
    }

    // Release the slot and tell the peer, so it can decide to re-pair.
    {
        let mut rooms = relay.rooms.lock().expect("rooms mutex");
        if let Some(entry) = rooms.get_mut(&room) {
            *entry.slot(role) = None;
            if let Some(peer) = entry.peer(role) {
                let _ = peer.send(control("peer-left"));
            }
            if entry.is_empty() {
                rooms.remove(&room);
            }
        }
    }

    relay.metrics.connections.fetch_sub(1, Ordering::Relaxed);
    drop(outbox);
    let _ = writer.await;
}

fn notify_peer(relay: &Shared, room: &str, role: Role, message: Message) {
    let rooms = relay.rooms.lock().expect("rooms mutex");
    if let Some(peer) = rooms.get(room).and_then(|entry| entry.peer(role)) {
        let _ = peer.send(message);
    }
}

/// Deliver whatever is already queued (a rejection), then hang up.
async fn close_with(socket: WebSocket, mut inbox: mpsc::UnboundedReceiver<Message>) {
    use futures_util::SinkExt;
    let mut socket = socket;
    while let Ok(message) = inbox.try_recv() {
        let _ = socket.send(message).await;
    }
    let _ = socket.close().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn role_is_required_and_validated() {
        assert!(parse_params(&params(&[("room", "a")])).is_err());
        assert!(parse_params(&params(&[("role", "node2"), ("room", "a")])).is_err());
        assert_eq!(
            parse_params(&params(&[("role", "node0")])).unwrap().role,
            Role::Node0
        );
    }

    #[test]
    fn room_defaults_and_rejects_hostile_ids() {
        assert_eq!(
            parse_params(&params(&[("role", "node0")])).unwrap().room,
            "default"
        );
        for bad in ["", "has space", "line\nbreak", "../etc", "emoji-\u{1f600}"] {
            assert!(
                parse_params(&params(&[("role", "node0"), ("room", bad)])).is_err(),
                "room id {bad:?} should be rejected"
            );
        }
        assert!(parse_params(&params(&[
            ("role", "node1"),
            ("room", &"x".repeat(MAX_ROOM_ID_LEN + 1))
        ]))
        .is_err());
        assert!(parse_params(&params(&[("role", "node1"), ("room", "My_Room-2")])).is_ok());
    }

    #[test]
    fn a_room_routes_each_role_to_the_other() {
        let (tx0, mut rx0) = mpsc::unbounded_channel();
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let mut room = Room::default();
        *room.slot(Role::Node0) = Some(tx0);
        *room.slot(Role::Node1) = Some(tx1);

        room.peer(Role::Node0).unwrap().send(control("a")).unwrap();
        room.peer(Role::Node1).unwrap().send(control("b")).unwrap();

        // node0's peer is node1, so node1 receives what node0 sent.
        assert!(matches!(rx1.try_recv(), Ok(Message::Text(_))));
        assert!(matches!(rx0.try_recv(), Ok(Message::Text(_))));
    }

    #[test]
    fn a_room_empties_out_when_both_roles_leave() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut room = Room::default();
        *room.slot(Role::Node0) = Some(tx);
        assert!(!room.is_empty());
        *room.slot(Role::Node0) = None;
        assert!(room.is_empty(), "an empty room must be reclaimable");
    }
}
