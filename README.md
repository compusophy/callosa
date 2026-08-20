# callosa

Two browser tabs each hold **half a transformer** and run it on their own GPU.
Node 0 embeds a token and runs block 0; the resulting hidden state crosses a
**WebRTC data channel**; node 1 runs block 1 plus the LM head, samples the next
token, and sends it back. Node 0 feeds that token into the next position and the
loop repeats.

*The corpus callosum is the tract carrying signals between the two hemispheres of
a brain. This is the same shape: one model, two halves, and everything that
matters travelling across the gap between them.*

Rust → WebAssembly for the model, `wgpu` → WebGPU for the compute, WebRTC for the
link. **There is no backend at all** — no server, no serverless functions. The
deployment is static files; the peers introduce themselves to each other.

There is also no application JavaScript. The DOM, the canvas rendering, the
WebRTC stack and the pairing transports are all driven from Rust through
`web-sys`. The only script the page loads is the wasm-bindgen glue and a
two-line bootstrap:

```html
<script type="module">
  import init from './pkg/callosa.js';
  init();
</script>
```

```
   ┌────────────────────────────┐                      ┌────────────────────────────┐
   │ node 0 · coordinator       │   OP_ACTIVATION      │ node 1 · worker            │
   │ embed → block 0            │  ── 528 b (f32) ──▶  │ block 1 → norm → lm head   │
   │ 172,288 params on WebGPU   │                      │ 172,416 params on WebGPU   │
   │                            │  ◀── 20 b ────────   │ sample                     │
   └────────────────────────────┘   OP_TOKEN           └────────────────────────────┘
```

Neither side ever sends weights. Both derive identical tensors from a shared seed
(`config::MODEL_SEED`), so pairing costs one 32-byte handshake.

## Running it

```bash
npm run build   # wasm-pack build --target web --out-dir public/pkg
npm start       # http://localhost:3000 (a plain static file server)
```

Open the page in **two tabs**. They negotiate roles between themselves — the
second tab notices node 0 is taken and adopts node 1 — then pair over WebRTC
automatically. Press **run pipeline**.

Add `?room=<name>` to run more than one pair at a time.

`server.js` is a development convenience only. It has no dependencies and does
nothing but serve `public/` with correct MIME types; any static server works.

## Deploying

```bash
npm run deploy   # builds, then: vercel deploy public --prod
```

Static hosting, nothing else. `public/` is the whole artifact.

## Pairing without a server

WebRTC needs the two peers introduced before they can talk, which normally means
a signaling relay. Two ways around that, neither involving a backend:

| mode | how | for |
|---|---|---|
| **same browser** | SDP and ICE cross a `BroadcastChannel` between tabs on the same origin | the two-tab demo; zero configuration |
| **copy / paste** | the description is packed into a base64url blob carried across by hand | two genuinely separate devices |

The WebRTC connection either mode negotiates is entirely real — real offer and
answer, real ICE, real data channel. Only the introduction is local.

Copy/paste pairing is *non-trickle*: candidates cannot be sent incrementally, so
the peer waits for ICE gathering to finish and the candidates ride inside the
description. That wait is bounded, because a gatherer stuck on an unreachable
STUN server would otherwise hang the handshake forever.

## The model

A real (if small) decoder, not a stand-in matmul:

| | |
|---|---|
| residual width | 128 |
| attention | 4 heads × 32, RoPE, KV cache to 128 positions |
| FFN | SwiGLU, inner width 256 |
| norm | RMSNorm with learned gain |
| blocks | 2 — one per node |
| vocabulary | 64 toy tokens |
| parameters | 344,704 total |

Weights are random, so the output is nonsense — the point is the *mechanism*:
every token genuinely runs attention over a real KV cache, split across two
machines, and the transcript changes when you change temperature, top-k, or the
transport codec.

## What runs where

| stage | node 0 | node 1 |
|---|---|---|
| embedding lookup | ✓ | |
| RMSNorm → Q/K/V → RoPE → attention → O-proj | ✓ | ✓ |
| SwiGLU FFN + residuals | ✓ | ✓ |
| final norm + LM head | | ✓ |
| sampling | | ✓ |

Node 1 runs the head locally so the readback is 64 logits rather than a 128-wide
hidden state, and node 0 never needs the head weights.

## Wire protocol

Every frame opens with a fixed 12-byte header, so a receiver can route and
validate before touching the payload:

```
0   version      bump on any layout change; checked first
1   opcode       HELLO | ACTIVATION | TOKEN | RESET | ERROR
2   codec        f32 or int8+scale
3   flags        bit 0 = final frame
4   request_id   u32 — correlates a reply with the step that asked for it
8   seq_pos      u32
```

`request_id` matters more than it looks. Matching replies positionally ("resolve
whatever is pending") means a late reply to a timed-out step silently resolves
the *next* step with stale data. Frames whose id has no pending entry are dropped
with a log line instead.

`OP_ERROR` exists so a worker-side failure reaches the coordinator immediately
rather than as an 8-second timeout, and `OP_HELLO` carries model geometry so two
tabs on mismatched builds fail with a readable message instead of garbage
activations.

### Activation codecs

| codec | frame | note |
|---|---|---|
| `f32` | 528 b | lossless |
| `int8 + scale` | 148 b | 3.6× smaller, quantisation error compounds visibly across a generation |

Switching the codec mid-demo and watching the transcript diverge is the fastest
way to see why activation precision matters in pipeline parallelism.

## GPU design notes

**One dispatch shape, not a general matmul.** Autoregressive decode is batch-1.
A 16×16 tiled matmul spends 15/16 of every workgroup on a phantom M dimension, so
`matvec` uses one thread per output row over `[n_out, n_in]` row-major weights —
every lane busy, every read coalesced along a row.

**One submit per token.** The whole block is recorded into a single command
encoder — two compute passes with the KV-cache writes between them — then
submitted once and read back once. 15 dispatches (17 on node 1), one round trip.
Encoding each matmul as its own submit-and-map costs more in synchronisation than
the arithmetic it schedules.

**Nothing allocates in the loop.** Bind groups, pipelines and uniform buffers are
built once at load. A step writes two buffers, encodes, submits, maps.

**The KV cache write is a buffer copy.** The destination offset is `pos * dim * 4`,
known on the CPU, so it needs no dispatch.

## Backends

`PipelineShard` runs on WebGPU or on the CPU reference kernels in
`src/tensor.rs`, chosen at init and falling back automatically if a device cannot
be acquired. The CPU path is a genuine op-for-op mirror, which is what makes the
GPU testable — and what keeps the demo working in browsers without WebGPU.

Shader compilation is checked explicitly via `get_compilation_info`. WGSL
implementations disagree at the edges (Tint and naga do not accept identical
programs); without that check a rejected shader yields an invalid pipeline that
silently computes **zeros**, which is far worse than refusing to start.

### The CPU backend is faster here, and that is the honest result

At 344k parameters a block is roughly 0.2 ms of arithmetic. One WebGPU step —
even batched into a single submit — costs several milliseconds of queue submit
and buffer-map latency, so measured per-block time runs ~0.2 ms on CPU against
~4 ms on WebGPU. The GPU path is not slow; the model is too small to amortise a
round trip to the device.

That crossover is the real lesson of the demo, and it is visible directly in the
UI: switch the backend selector and watch the per-block number. The pipeline
structure — sharded weights, KV cache per stage, activations on the wire — is
what would keep paying off at a size where the arithmetic actually dominates.

Both backends produce the *same transcript*, which is the property that makes the
comparison meaningful.

## Tests

```bash
cargo test
```

- `src/tensor.rs` — matvec, RMSNorm, RoPE, softmax, attention, sampling
- `src/protocol.rs` — frame round trips, truncation, version and opcode
  mismatches, quantisation error bounds
- `src/model.rs` — full two-stage CPU pipeline, reproducibility, cache reset,
  and a check that attention actually consults history
- `tests/kernels.rs` — **runs the real WGSL on a native adapter and diffs it
  against the CPU reference**, op for op, across a sequence

That last one is the important one: they are the same shader files the browser
loads, so a divergence there is a divergence in the browser. It skips cleanly
with a message when no adapter is available.

## Layout

```
src/
  config.rs        geometry, deterministic weight synthesis, vocabulary
  tensor.rs        CPU reference kernels (also the test oracle)
  model.rs         PipelineShard: backend selection, CPU block, stats
  protocol.rs      frame encode/decode, f32 and int8 codecs
  gpu/mod.rs       device, pipelines, prebuilt bind groups, one-submit forward
  gpu/shaders/     kernels.wgsl (matvec, rmsnorm, add, swiglu, rope), attention.wgsl
  web/app.rs       the client: state, inference loop, every DOM update
  web/peer.rs      webrtc, ICE buffering, send backpressure, candidate stats
  web/signal.rs    BroadcastChannel and copy/paste pairing
  web/viz.rs       activation and latency canvases
  web/dom.rs       element refs, listeners, timers
public/
  index.html       markup + the two-line wasm bootstrap
  style.css
  pkg/             wasm-pack output (built, not checked in)
server.js          dependency-free static file server, for local development
```

## Limits

- 128-position KV cache; the UI shows what is left and a forward past the end is
  rejected rather than wrapping.
- Two stages. The split is `Role`-driven, so more stages means extending that
  enum and chaining the links, not restructuring the protocol.
- `BroadcastChannel` pairing is same-origin and same-browser by construction.
  Across devices, use copy/paste pairing.
- No TURN server, so two peers behind symmetric NATs may fail to connect. STUN
  covers the common cases; the candidate path is shown in the topology panel so
  a failure is diagnosable rather than mysterious.
- Random weights. Swapping in a trained checkpoint means replacing
  `config::synthesize` and shipping the tensors — the execution path does not
  change.

