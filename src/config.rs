//! Model geometry and deterministic weight synthesis.
//!
//! Both peers generate the *same* weights from the same seeds, so the demo never
//! has to ship a checkpoint over the wire. Node 0 owns the embedding table and
//! block 0; node 1 owns block 1, the final norm and the LM head.

/// Residual stream width.
pub const DIM: usize = 128;
/// Attention heads per block.
pub const N_HEADS: usize = 4;
/// Width of a single head. `N_HEADS * HEAD_DIM` must equal `DIM`.
pub const HEAD_DIM: usize = DIM / N_HEADS;
/// Inner width of the SwiGLU feed-forward network.
pub const FFN_HIDDEN: usize = 256;
/// Number of tokens in the toy vocabulary.
pub const VOCAB_SIZE: usize = 64;
/// KV-cache capacity. Must match `MAX_SEQ_LEN` in `shaders/attention.wgsl`.
pub const MAX_SEQ: usize = 128;
/// Blocks in the full model; one per pipeline stage.
pub const N_LAYERS: usize = 2;
/// RoPE base frequency.
pub const ROPE_THETA: f32 = 10_000.0;
/// RMSNorm epsilon.
pub const NORM_EPS: f32 = 1e-5;

const _: () = assert!(N_HEADS * HEAD_DIM == DIM, "head geometry must tile DIM");
const _: () = assert!(HEAD_DIM.is_multiple_of(2), "RoPE needs an even head width");

/// Toy vocabulary. Index 0 is BOS, the last index is EOS.
#[rustfmt::skip]
pub const VOCAB: [&str; VOCAB_SIZE] = [
    "<bos>", " distributed", " gpu", " compute", " pipeline", " parallel", " inference", " webgpu",
    " webrtc", " wasm", " cluster", " node0", " node1", " tensor", " activation", " layer",
    " forward", " latency", " fast", " stream", " browser", " network", " speed", " token",
    " throughput", " memory", " buffer", " matrix", " dispatch", " shard", " worker", " state",
    " sync", " zero-copy", " realtime", " benchmark", " verified", " complete", " active", " load",
    " balance", " edge", " client", " server", " socket", " connect", " execute", " success",
    " data", " channel", " float32", " transfer", " return", " sample", " argmax", " cycle",
    " loop", " step", " done", " ok", " ready", " go", " run", "<eos>",
];

pub const BOS_TOKEN: u32 = 0;
pub const EOS_TOKEN: u32 = (VOCAB_SIZE - 1) as u32;

/// Which half of the pipeline a shard is responsible for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Embedding lookup + block 0. Emits a hidden state.
    Node0,
    /// Block 1 + final norm + LM head. Emits logits.
    Node1,
}

impl Role {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "node0" | "0" => Ok(Role::Node0),
            "node1" | "1" => Ok(Role::Node1),
            other => Err(format!("unknown role '{other}' (expected node0 or node1)")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Node0 => "node0",
            Role::Node1 => "node1",
        }
    }

    /// Index of the transformer block this shard executes.
    pub fn layer_index(self) -> usize {
        match self {
            Role::Node0 => 0,
            Role::Node1 => 1,
        }
    }

    /// Node 1 additionally runs the output head, so its readback is `VOCAB_SIZE`
    /// floats rather than `DIM`.
    pub fn has_head(self) -> bool {
        matches!(self, Role::Node1)
    }

    pub fn output_len(self) -> usize {
        if self.has_head() {
            VOCAB_SIZE
        } else {
            DIM
        }
    }
}

/// Small, fast, fully deterministic PRNG (PCG-XSH-RR 64/32).
///
/// Deterministic weight synthesis is what lets two independent browser tabs agree
/// on a model without a checkpoint transfer.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed ^ 0x9E37_79B9_7F4A_7C15);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `[-1, 1)`.
    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Uniform init scaled so a `fan_in`-wide dot product has ~unit variance:
/// `Var(U(-a, a)) = a^2 / 3`, so `a = sqrt(3 / fan_in)`.
fn init_matrix(seed: u64, n_out: usize, n_in: usize) -> Vec<f32> {
    let mut rng = Pcg32::new(seed);
    let scale = (3.0 / n_in as f32).sqrt();
    (0..n_out * n_in)
        .map(|_| rng.next_signed() * scale)
        .collect()
}

/// Norm gains start near 1 with a little jitter so the two blocks differ.
fn init_gain(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Pcg32::new(seed);
    (0..n).map(|_| 1.0 + rng.next_signed() * 0.05).collect()
}

/// Weights for one transformer block. Projection matrices are `[n_out, n_in]`
/// row-major, matching `nn.Linear`, so each mat-vec thread reads one contiguous row.
#[derive(Clone)]
pub struct BlockWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// SwiGLU gate projection, `[FFN_HIDDEN, DIM]`.
    pub w1: Vec<f32>,
    /// SwiGLU down projection, `[DIM, FFN_HIDDEN]`.
    pub w2: Vec<f32>,
    /// SwiGLU up projection, `[FFN_HIDDEN, DIM]`.
    pub w3: Vec<f32>,
}

impl BlockWeights {
    /// Layer `index` derives every tensor from a distinct sub-seed of `base_seed`.
    pub fn synthesize(base_seed: u64, index: usize) -> Self {
        let s = |k: u64| base_seed ^ ((index as u64 + 1).wrapping_mul(0x1000_0000_0000_00FF)) ^ k;
        BlockWeights {
            attn_norm: init_gain(s(1), DIM),
            wq: init_matrix(s(2), DIM, DIM),
            wk: init_matrix(s(3), DIM, DIM),
            wv: init_matrix(s(4), DIM, DIM),
            wo: init_matrix(s(5), DIM, DIM),
            ffn_norm: init_gain(s(6), DIM),
            w1: init_matrix(s(7), FFN_HIDDEN, DIM),
            w2: init_matrix(s(8), DIM, FFN_HIDDEN),
            w3: init_matrix(s(9), FFN_HIDDEN, DIM),
        }
    }

    pub fn param_count(&self) -> usize {
        self.attn_norm.len()
            + self.wq.len()
            + self.wk.len()
            + self.wv.len()
            + self.wo.len()
            + self.ffn_norm.len()
            + self.w1.len()
            + self.w2.len()
            + self.w3.len()
    }
}

/// Embedding table owned by node 0.
#[derive(Clone)]
pub struct EmbeddingWeights {
    /// `[VOCAB_SIZE, DIM]` row-major.
    pub table: Vec<f32>,
}

impl EmbeddingWeights {
    pub fn synthesize(base_seed: u64) -> Self {
        EmbeddingWeights {
            table: init_matrix(base_seed ^ 0x000E_B0DD, VOCAB_SIZE, DIM),
        }
    }

    pub fn lookup(&self, token_id: u32) -> Vec<f32> {
        let idx = (token_id as usize) % VOCAB_SIZE;
        self.table[idx * DIM..(idx + 1) * DIM].to_vec()
    }
}

/// Final norm + LM head, owned by node 1.
#[derive(Clone)]
pub struct HeadWeights {
    pub final_norm: Vec<f32>,
    /// `[VOCAB_SIZE, DIM]` row-major.
    pub lm_head: Vec<f32>,
}

impl HeadWeights {
    pub fn synthesize(base_seed: u64) -> Self {
        HeadWeights {
            final_norm: init_gain(base_seed ^ 0x000F_10A1, DIM),
            lm_head: init_matrix(base_seed ^ 0x0001_4EAD, VOCAB_SIZE, DIM),
        }
    }
}

/// The seed both peers agree on. Changing it changes the "model" everywhere.
pub const MODEL_SEED: u64 = 0x5EED_C0DE_1234_5678;

pub fn decode_token_str(token_id: u32) -> &'static str {
    VOCAB[(token_id as usize) % VOCAB_SIZE]
}

/// Map free text onto the toy vocabulary. Exact matches win; anything else is
/// hashed into the non-special range so the demo always has a seed token.
pub fn encode_prompt_tokens(prompt: &str) -> Vec<u32> {
    let lowered = prompt.to_lowercase();
    let mut tokens: Vec<u32> = lowered
        .split_whitespace()
        .map(|word| {
            VOCAB
                .iter()
                .position(|v| v.trim() == word)
                .map(|i| i as u32)
                .unwrap_or_else(|| {
                    let hash = word.bytes().fold(2_166_136_261u32, |acc, b| {
                        (acc ^ b as u32).wrapping_mul(16_777_619)
                    });
                    1 + (hash % (VOCAB_SIZE as u32 - 2))
                })
        })
        .collect();
    if tokens.is_empty() {
        tokens.push(BOS_TOKEN);
    }
    tokens
}
