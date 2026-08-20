//! Backend-agnostic pipeline stage.
//!
//! A `PipelineShard` owns exactly the part of the model its node is responsible
//! for. Node 0 holds the embedding table and block 0; node 1 holds block 1, the
//! final norm and the LM head. The two halves never exchange weights — they are
//! derived from the same seed on both sides — only hidden states and tokens.

use std::rc::Rc;

use crate::config::{
    BlockWeights, EmbeddingWeights, HeadWeights, Role, DIM, FFN_HIDDEN, HEAD_DIM, MAX_SEQ,
    MODEL_SEED, NORM_EPS, N_HEADS, N_LAYERS, ROPE_THETA, VOCAB_SIZE,
};
use crate::gpu::{GpuContext, GpuShard};
use crate::tensor::{self, SamplerConfig};

/// Which compute path a shard ended up on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    WebGpu,
    Cpu,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::WebGpu => "webgpu",
            BackendKind::Cpu => "cpu",
        }
    }
}

/// Reference implementation of a block. Also the fallback when a browser has no
/// WebGPU, which keeps the demo usable everywhere instead of dead-ending.
pub struct CpuShard {
    role: Role,
    block: BlockWeights,
    head: Option<HeadWeights>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    scratch: Scratch,
}

struct Scratch {
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qr: Vec<f32>,
    kr: Vec<f32>,
    att: Vec<f32>,
    attp: Vec<f32>,
    x1: Vec<f32>,
    xb2: Vec<f32>,
    hg: Vec<f32>,
    hu: Vec<f32>,
    hs: Vec<f32>,
    hd: Vec<f32>,
    x2: Vec<f32>,
    xn: Vec<f32>,
    logits: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Scratch {
            xb: vec![0.0; DIM],
            q: vec![0.0; DIM],
            k: vec![0.0; DIM],
            v: vec![0.0; DIM],
            qr: vec![0.0; DIM],
            kr: vec![0.0; DIM],
            att: vec![0.0; DIM],
            attp: vec![0.0; DIM],
            x1: vec![0.0; DIM],
            xb2: vec![0.0; DIM],
            hg: vec![0.0; FFN_HIDDEN],
            hu: vec![0.0; FFN_HIDDEN],
            hs: vec![0.0; FFN_HIDDEN],
            hd: vec![0.0; DIM],
            x2: vec![0.0; DIM],
            xn: vec![0.0; DIM],
            logits: vec![0.0; VOCAB_SIZE],
        }
    }
}

impl CpuShard {
    pub fn new(role: Role, block: BlockWeights, head: Option<HeadWeights>) -> Self {
        CpuShard {
            role,
            block,
            head,
            k_cache: vec![0.0; MAX_SEQ * DIM],
            v_cache: vec![0.0; MAX_SEQ * DIM],
            scratch: Scratch::new(),
        }
    }

    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
    }

    /// Op-for-op mirror of `GpuShard::forward`; the two are checked against each
    /// other in `tests/kernels.rs`.
    pub fn forward(&mut self, x: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        if x.len() != DIM {
            return Err(format!(
                "hidden state width mismatch: expected {DIM}, got {}",
                x.len()
            ));
        }
        if pos as usize >= MAX_SEQ {
            return Err(format!(
                "position {pos} exceeds kv-cache capacity of {MAX_SEQ}"
            ));
        }

        let s = &mut self.scratch;
        let b = &self.block;

        tensor::rmsnorm(&mut s.xb, x, &b.attn_norm, NORM_EPS);
        tensor::matvec(&mut s.q, &s.xb, &b.wq, DIM, DIM);
        tensor::matvec(&mut s.k, &s.xb, &b.wk, DIM, DIM);
        tensor::matvec(&mut s.v, &s.xb, &b.wv, DIM, DIM);

        tensor::rope(&mut s.qr, &s.q, pos, N_HEADS, HEAD_DIM, ROPE_THETA);
        tensor::rope(&mut s.kr, &s.k, pos, N_HEADS, HEAD_DIM, ROPE_THETA);

        let row = pos as usize * DIM;
        self.k_cache[row..row + DIM].copy_from_slice(&s.kr);
        self.v_cache[row..row + DIM].copy_from_slice(&s.v);

        tensor::attention(
            &mut s.att,
            &s.qr,
            &self.k_cache,
            &self.v_cache,
            pos,
            N_HEADS,
            HEAD_DIM,
        );
        tensor::matvec(&mut s.attp, &s.att, &b.wo, DIM, DIM);
        tensor::add_into(&mut s.x1, x, &s.attp);

        tensor::rmsnorm(&mut s.xb2, &s.x1, &b.ffn_norm, NORM_EPS);
        tensor::matvec(&mut s.hg, &s.xb2, &b.w1, DIM, FFN_HIDDEN);
        tensor::matvec(&mut s.hu, &s.xb2, &b.w3, DIM, FFN_HIDDEN);
        tensor::swiglu(&mut s.hs, &s.hg, &s.hu);
        tensor::matvec(&mut s.hd, &s.hs, &b.w2, FFN_HIDDEN, DIM);

        let x1 = s.x1.clone();
        tensor::add_into(&mut s.x2, &x1, &s.hd);

        match &self.head {
            Some(head) => {
                tensor::rmsnorm(&mut s.xn, &s.x2, &head.final_norm, NORM_EPS);
                tensor::matvec(&mut s.logits, &s.xn, &head.lm_head, DIM, VOCAB_SIZE);
                Ok(s.logits.clone())
            }
            None => Ok(s.x2.clone()),
        }
    }

    pub fn role(&self) -> Role {
        self.role
    }
}

/// Both variants are boxed: they differ by an order of magnitude in size, and a
/// `PipelineShard` runs thousands of ops per step, so one pointer hop is free.
enum Backend {
    Gpu(Box<GpuShard>),
    Cpu(Box<CpuShard>),
}

/// Rolling telemetry for the local half of the pipeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShardStats {
    pub steps: u32,
    pub last_compute_us: u32,
    pub total_compute_us: u64,
}

impl ShardStats {
    pub fn mean_compute_us(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_compute_us as f64 / self.steps as f64
        }
    }

    fn record(&mut self, micros: u32) {
        self.steps += 1;
        self.last_compute_us = micros;
        self.total_compute_us += micros as u64;
    }
}

/// One half of the pipeline, wired to whichever backend is available.
pub struct PipelineShard {
    role: Role,
    backend: Backend,
    backend_kind: BackendKind,
    device_label: String,
    embedding: Option<EmbeddingWeights>,
    sampler: SamplerConfig,
    stats: ShardStats,
    /// Next free KV-cache slot, for context-remaining reporting.
    next_position: u32,
}

impl PipelineShard {
    /// Build the shard for `role`. Falls back to the CPU backend when WebGPU is
    /// unavailable or fails to initialise, so the demo degrades instead of dying.
    pub async fn new(role: Role, prefer_gpu: bool) -> Self {
        let layer = role.layer_index();
        let block = BlockWeights::synthesize(MODEL_SEED, layer);
        let head = role.has_head().then(|| HeadWeights::synthesize(MODEL_SEED));
        let embedding = (!role.has_head()).then(|| EmbeddingWeights::synthesize(MODEL_SEED));

        if prefer_gpu {
            match Self::try_gpu(role, &block, head.as_ref()).await {
                Ok((shard, label)) => {
                    return PipelineShard {
                        role,
                        backend: Backend::Gpu(Box::new(shard)),
                        backend_kind: BackendKind::WebGpu,
                        device_label: label,
                        embedding,
                        sampler: SamplerConfig::default(),
                        stats: ShardStats::default(),
                        next_position: 0,
                    };
                }
                Err(reason) => {
                    log_fallback(&reason);
                }
            }
        }

        PipelineShard {
            role,
            backend: Backend::Cpu(Box::new(CpuShard::new(role, block, head))),
            backend_kind: BackendKind::Cpu,
            device_label: "cpu reference kernels".to_string(),
            embedding,
            sampler: SamplerConfig::default(),
            stats: ShardStats::default(),
            next_position: 0,
        }
    }

    async fn try_gpu(
        role: Role,
        block: &BlockWeights,
        head: Option<&HeadWeights>,
    ) -> Result<(GpuShard, String), String> {
        let ctx = Rc::new(GpuContext::new().await?);
        let label = ctx.adapter_label.clone();
        let shard = GpuShard::new(ctx, role, block, head)?;
        Ok((shard, label))
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    pub fn stats(&self) -> ShardStats {
        self.stats
    }

    pub fn sampler(&self) -> SamplerConfig {
        self.sampler
    }

    pub fn set_sampler(&mut self, sampler: SamplerConfig) {
        self.sampler = sampler;
    }

    pub fn dispatch_count(&self) -> usize {
        match &self.backend {
            Backend::Gpu(g) => g.dispatch_count(),
            // The CPU path runs the same op sequence, minus the two cache copies.
            Backend::Cpu(_) => {
                if self.role.has_head() {
                    17
                } else {
                    15
                }
            }
        }
    }

    pub fn kernel_trace(&self) -> Vec<String> {
        match &self.backend {
            Backend::Gpu(g) => g.kernel_trace().into_iter().map(String::from).collect(),
            Backend::Cpu(_) => Vec::new(),
        }
    }

    /// Node 0 only: turn a token id into the hidden state entering block 0.
    pub fn embed(&self, token_id: u32) -> Result<Vec<f32>, String> {
        self.embedding
            .as_ref()
            .map(|e| e.lookup(token_id))
            .ok_or_else(|| "only node 0 owns the embedding table".to_string())
    }

    /// Run this stage for one position.
    pub async fn forward(&mut self, x: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        let out = match &mut self.backend {
            Backend::Gpu(g) => g.forward(x, pos).await?,
            Backend::Cpu(c) => c.forward(x, pos)?,
        };
        self.next_position = self.next_position.max(pos + 1);
        Ok(out)
    }

    /// Record how long the last `forward` took. Timing lives with the host
    /// because only it knows how to read a clock on this platform.
    pub fn record_step(&mut self, micros: u32) {
        self.stats.record(micros);
    }

    /// Node 1 only: pick the next token from this stage's logits.
    pub fn sample(&self, logits: &[f32], step: u32) -> Result<u32, String> {
        if !self.role.has_head() {
            return Err("only node 1 owns the lm head".to_string());
        }
        if logits.len() != VOCAB_SIZE {
            return Err(format!(
                "expected {VOCAB_SIZE} logits, got {}",
                logits.len()
            ));
        }
        Ok(tensor::sample(logits, &self.sampler, step))
    }

    pub fn reset(&mut self) {
        match &mut self.backend {
            Backend::Gpu(g) => g.reset(),
            Backend::Cpu(c) => c.reset(),
        }
        self.next_position = 0;
        self.stats = ShardStats::default();
    }

    /// Positions still free in the KV cache.
    pub fn remaining_context(&self) -> u32 {
        (MAX_SEQ as u32).saturating_sub(self.next_position)
    }

    /// Parameters resident on this node, for the UI's "what am I actually
    /// holding" readout.
    pub fn param_count(&self) -> usize {
        params_for_role(self.role)
    }
}

/// Parameters a given stage owns. Node 0 carries the embedding table, node 1
/// the final norm and the LM head; the transformer block itself dominates both.
pub fn params_for_role(role: Role) -> usize {
    let block = BlockWeights::synthesize(MODEL_SEED, role.layer_index()).param_count();
    let extra = if role.has_head() {
        DIM + VOCAB_SIZE * DIM
    } else {
        VOCAB_SIZE * DIM
    };
    block + extra
}

/// Total parameters across the whole (two-node) model.
pub fn total_param_count() -> usize {
    let per_block = BlockWeights::synthesize(MODEL_SEED, 0).param_count();
    per_block * N_LAYERS + VOCAB_SIZE * DIM * 2 + DIM
}

#[cfg(target_arch = "wasm32")]
fn log_fallback(reason: &str) {
    web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(&format!(
        "[shard] webgpu unavailable, falling back to cpu kernels: {reason}"
    )));
}

#[cfg(not(target_arch = "wasm32"))]
fn log_fallback(reason: &str) {
    eprintln!("[shard] webgpu unavailable, falling back to cpu kernels: {reason}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BOS_TOKEN;

    fn cpu_shard(role: Role) -> CpuShard {
        let block = BlockWeights::synthesize(MODEL_SEED, role.layer_index());
        let head = role.has_head().then(|| HeadWeights::synthesize(MODEL_SEED));
        CpuShard::new(role, block, head)
    }

    /// Drive the full two-stage pipeline entirely in-process, exactly as the two
    /// browser tabs would but without the transport.
    fn generate(steps: u32, sampler: SamplerConfig) -> Vec<u32> {
        let embed = EmbeddingWeights::synthesize(MODEL_SEED);
        let mut node0 = cpu_shard(Role::Node0);
        let mut node1 = cpu_shard(Role::Node1);

        let mut token = BOS_TOKEN;
        let mut out = Vec::new();
        for pos in 0..steps {
            let x = embed.lookup(token);
            let hidden = node0.forward(&x, pos).expect("stage 0");
            assert_eq!(hidden.len(), DIM);
            let logits = node1.forward(&hidden, pos).expect("stage 1");
            assert_eq!(logits.len(), VOCAB_SIZE);
            token = tensor::sample(&logits, &sampler, pos);
            out.push(token);
        }
        out
    }

    #[test]
    fn two_stage_pipeline_produces_finite_activations() {
        let embed = EmbeddingWeights::synthesize(MODEL_SEED);
        let mut node0 = cpu_shard(Role::Node0);
        let mut node1 = cpu_shard(Role::Node1);

        for pos in 0..16 {
            let x = embed.lookup(pos % 64);
            let hidden = node0.forward(&x, pos).expect("stage 0");
            assert!(
                hidden.iter().all(|v| v.is_finite()),
                "stage 0 produced a non-finite activation at pos {pos}"
            );
            let logits = node1.forward(&hidden, pos).expect("stage 1");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "stage 1 produced a non-finite logit at pos {pos}"
            );
        }
    }

    #[test]
    fn generation_is_reproducible() {
        let cfg = SamplerConfig {
            temperature: 0.85,
            top_k: 8,
            seed: 99,
        };
        assert_eq!(generate(20, cfg), generate(20, cfg));
    }

    #[test]
    fn greedy_and_sampled_runs_both_stay_in_vocabulary() {
        for cfg in [
            SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                seed: 1,
            },
            SamplerConfig {
                temperature: 1.2,
                top_k: 16,
                seed: 2,
            },
        ] {
            for token in generate(24, cfg) {
                assert!((token as usize) < VOCAB_SIZE, "token {token} out of range");
            }
        }
    }

    #[test]
    fn attention_actually_uses_history() {
        // Two runs that differ only in their first token must diverge at a later
        // position; if they do not, the KV cache is not being consulted.
        let embed = EmbeddingWeights::synthesize(MODEL_SEED);
        let run = |first: u32| {
            let mut shard = cpu_shard(Role::Node0);
            let mut last = Vec::new();
            for (pos, token) in [first, 5, 9].iter().enumerate() {
                last = shard
                    .forward(&embed.lookup(*token), pos as u32)
                    .expect("forward");
            }
            last
        };

        let a = run(3);
        let b = run(40);
        let delta: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            delta > 1e-3,
            "history had no effect on the final state (delta {delta})"
        );
    }

    #[test]
    fn resetting_clears_history() {
        let embed = EmbeddingWeights::synthesize(MODEL_SEED);
        let mut shard = cpu_shard(Role::Node0);

        let fresh = shard.forward(&embed.lookup(7), 0).expect("forward");
        for pos in 1..5 {
            shard
                .forward(&embed.lookup(pos + 10), pos)
                .expect("forward");
        }
        shard.reset();
        let after_reset = shard.forward(&embed.lookup(7), 0).expect("forward");

        for (a, b) in fresh.iter().zip(&after_reset) {
            assert!((a - b).abs() < 1e-5, "reset did not restore a clean cache");
        }
    }

    #[test]
    fn out_of_range_positions_are_rejected() {
        let mut shard = cpu_shard(Role::Node0);
        let x = vec![0.1f32; DIM];
        assert!(shard.forward(&x, MAX_SEQ as u32).is_err());
        assert!(shard.forward(&vec![0.0; DIM - 1], 0).is_err());
    }
}
