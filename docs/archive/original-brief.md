> **Archived — superseded. Not a description of this codebase.**
>
> This is the original brief the project was scaffolded from, kept for history.
> The implementation diverged from it substantially; read [the README](../../README.md)
> for what `callosa` actually is. The main differences:
>
> | the brief asked for | what exists |
> |---|---|
> | one 128x128 matmul per node | a 2-block transformer: RMSNorm, 4x32 RoPE attention over a KV cache, SwiGLU FFN, LM head |
> | a 16x16 tiled matmul shader | a mat-vec kernel (batch-1 decode leaves 15/16 of a 16x16 workgroup idle), plus rmsnorm, attention, swiglu and rope kernels |
> | `[u8 opcode, u32 seq_pos, u32 hidden_dim, f32...]` | a 12-byte versioned header with request-id correlation, `HELLO`/`RESET`/`ERROR` opcodes, and an int8 activation codec alongside f32 |
> | a Node.js WebSocket signaling relay | no backend at all — peers pair over `BroadcastChannel` or by copying a blob between devices |
> | `public/main.js` driving the UI | no application JavaScript; the client is Rust via `web-sys` |
> | argmax over synthetic logits | temperature + top-k sampling over a real LM head |
> | wgpu 22 | wgpu 23 |
>
> The sampler in the original implementation also added `sin(v + pos * 3.7)` to
> the logits to manufacture variety, which is why replacing the model was the
> first thing that changed.

---

Please scaffold and implement a complete, working Proof-of-Concept (PoC) for browser-based distributed GPU compute sharing using Rust, WebAssembly, WebGPU, and WebRTC.

### Goal
Build a working 2-node pipeline parallel inference demo where:
1. Tab 1 (Node 0) runs Layer 0 forward pass on WebGPU, extracts intermediate activation tensors, and sends them via WebRTC DataChannel to Tab 2.
2. Tab 2 (Node 1) receives the activation tensor, runs Layer 1 forward pass on WebGPU, samples the next token, and sends the token ID back to Tab 1.
3. Tab 1 receives the token, updates the UI, and loops autoregressively.

### Project Structure to Create:
├── Cargo.toml
├── src/
│ ├── lib.rs # wasm-bindgen exports & pipeline orchestration
│ ├── gpu.rs # wgpu device init, buffer management, compute dispatch
│ ├── protocol.rs # Binary packet serialization/deserialization
│ └── matmul.wgsl # Compute shader for matrix multiplication / linear projection
├── public/
│ ├── index.html # UI: Role toggle (Node 0 vs Node 1), prompt input, token log
│ └── main.js # WebRTC DataChannel connection & Wasm integration
├── server.js # Lightweight Node.js WebSocket signaling relay for WebRTC SDP
└── package.json # Scripts to build Wasm and start the local server
code
Code
### Implementation Details:

1. **`Cargo.toml`**:
   - Crate type `["cdylib"]`.
   - Dependencies: `wgpu = { version = "22.0", default-features = false, features = ["webgpu", "wgsl"] }`, `wasm-bindgen = "0.2"`, `wasm-bindgen-futures = "0.4"`, `bytemuck = { version = "1.16", features = ["derive"] }`, `futures-channel = "0.3"`.

2. **`src/matmul.wgsl`**:
   - Write a WGSL compute shader calculating `output = input * weights` with workgroup size `(16, 16)`.
   - Bindings: `@binding(0) input: array<f32>`, `@binding(1) weights: array<f32>`, `@binding(2) output: array<f32>`, `@binding(3) dims: uniform vec4<u32>` (M, K, N, pad).

3. **`src/protocol.rs`**:
   - Create a binary packet protocol using raw bytes:
     - `OP_ACTIVATION (0x01)`: `[u8 opcode, u32 seq_pos, u32 hidden_dim, [f32] activations]`
     - `OP_TOKEN (0x02)`: `[u8 opcode, u32 token_id]`
   - Provide zero-copy serialization/deserialization helpers for `Uint8Array` / `&[u8]`.

4. **`src/gpu.rs` & `src/lib.rs`**:
   - Implement `GpuNode::new(weights: &[f32], dim_k: u32, dim_n: u32)`.
   - Implement `GpuNode::forward(&self, input: &[f32]) -> js_sys::Promise` which:
     - Copies activation data to GPU storage buffer.
     - Dispatches the compute pass.
     - Maps the staging buffer asynchronously and returns the output tensor as `Float32Array`.
   - Expose initialization functions to JS via `wasm-bindgen`:
     - `init_node_0()`: Initializes Node 0 with synthetic Layer 0 weights (dim: 128 -> 128).
     - `init_node_1()`: Initializes Node 1 with synthetic Layer 1 weights (dim: 128 -> 128) plus an argmax/sampler function returning a `u32` token ID.
     - `node0_step(token_id, pos)`: Runs Layer 0 on GPU and returns the activation packet bytes.
     - `node1_step(activation_bytes)`: Unpacks activations, runs Layer 1 on GPU, runs argmax, and returns the token packet bytes.

5. **`server.js`**:
   - Minimal zero-dependency Node.js WebSocket signaling server (`ws` or raw `http/ws`) that relays SDP offers, answers, and ICE candidates between Node 0 and Node 1.

6. **`public/index.html` & `public/main.js`**:
   - Connects to `server.js` over WebSocket to negotiate WebRTC `RTCPeerConnection` with an `RTCDataChannel` set to `binaryType = "arraybuffer"`.
   - Radio buttons to select `Node 0 (Sender / Coordinator)` or `Node 1 (Worker / Sampler)`.
   - If Node 0: UI has a prompt box, "Generate" button, and token streaming output. On click, it runs `node0_step`, sends the `ArrayBuffer` over WebRTC, waits for the token response from Node 1, prints the token, and repeats for 20 tokens.
   - If Node 1: Listens on the WebRTC data channel, receives activations, executes `node1_step` on WebGPU, and sends the resulting token byte packet back over the data channel.

7. **`package.json`**:
   - Include scripts:
     - `"build:wasm": "wasm-pack build --target web --out-dir ./public/pkg"`
     - `"start": "node server.js"`
   - Dependency: `"ws": "^8.18.0"` (or native Node http/websocket).
