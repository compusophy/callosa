//! Pipeline-parallel transformer inference split across two browsers.
//!
//! Two tabs, two halves of a model. Node 0 embeds a token and runs block 0; the
//! resulting hidden state crosses a WebRTC data channel; node 1 runs block 1 and
//! the LM head, samples, and sends the token back. Both halves derive identical
//! weights from a shared seed, so no checkpoint ever crosses the wire.
//!
//! Everything — model, protocol, GPU kernels and the entire browser client —
//! lives in this crate. The build also targets the host, which is what makes the
//! WGSL kernels testable against the CPU reference in `tests/`.

pub mod config;
pub mod gpu;
pub mod model;
pub mod protocol;
pub mod tensor;

#[cfg(target_arch = "wasm32")]
mod web;
