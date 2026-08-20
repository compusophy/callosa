# callosa-relay

The signaling server for [callosa](https://github.com/compusophy/callosa).

Its only job is to let two browsers exchange SDP so they can open a WebRTC data
channel. **Once that channel is up, the peers talk directly and this relay
carries nothing** — activations and tokens never pass through it, and the client
closes its signaling socket a few seconds after pairing.

That is what keeps it cheap: concurrent connections track how many peers are
*currently pairing*, not how many exist. A pairing costs about four small
messages.

## Protocol

```
GET /ws?room=<id>&role=<node0|node1>
```

Server → client:

| message | meaning |
|---|---|
| `{"type":"registered","role":…,"room":…,"polite":…,"peerPresent":…}` | you hold the role |
| `{"type":"peer-joined"}` / `{"type":"peer-left"}` | the other half arrived or went |
| `{"type":"role-taken","role":…}` | someone else holds it; the connection then closes |
| `{"type":"relay-full"}` | at room capacity |

Anything else a client sends is forwarded verbatim to its peer. The relay never
parses an offer, an answer or a candidate.

## Limits

Every dimension is bounded, because a pairing is small and anything much larger
is a bug or an abuser:

| | |
|---|---|
| frame size | 64 KiB |
| frames per connection | 400 |
| connection lifetime | 10 minutes |
| rooms | 50,000 |
| room id | 1–64 chars, `[A-Za-z0-9_-]` |

Nothing is persisted. Rooms are in memory and vanish when both sides leave.

## Running

```bash
cargo run --release          # listens on $PORT, default 8080
curl localhost:8080/health   # {"ok":true,"rooms":0,...}
```

Deployed on Railway from `relay/` using the Dockerfile.
