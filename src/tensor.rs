//! CPU reference kernels.
//!
//! Two jobs: they are the fallback backend when a browser has no WebGPU, and
//! they are the oracle the WGSL kernels are tested against in `tests/kernels.rs`.
//! Every function here has a one-to-one counterpart in `src/gpu/shaders/`.

use crate::config::Pcg32;

/// `dst[n] = sum_k src[k] * w[n * n_in + k]`, weights `[n_out, n_in]` row-major.
pub fn matvec(dst: &mut [f32], src: &[f32], w: &[f32], n_in: usize, n_out: usize) {
    debug_assert_eq!(src.len(), n_in);
    debug_assert_eq!(dst.len(), n_out);
    debug_assert_eq!(w.len(), n_in * n_out);

    for (n, out) in dst.iter_mut().enumerate() {
        let row = &w[n * n_in..(n + 1) * n_in];
        // Four accumulators mirror the unrolled WGSL loop and keep the two
        // implementations bit-comparable to within normal float reassociation.
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let chunks = n_in / 4;
        for c in 0..chunks {
            let k = c * 4;
            a0 += src[k] * row[k];
            a1 += src[k + 1] * row[k + 1];
            a2 += src[k + 2] * row[k + 2];
            a3 += src[k + 3] * row[k + 3];
        }
        let mut acc = (a0 + a1) + (a2 + a3);
        for k in chunks * 4..n_in {
            acc += src[k] * row[k];
        }
        *out = acc;
    }
}

/// Root-mean-square layer norm with a learned per-channel gain.
pub fn rmsnorm(dst: &mut [f32], src: &[f32], gain: &[f32], eps: f32) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), gain.len());

    let sum_sq: f32 = src.iter().map(|v| v * v).sum();
    let scale = (sum_sq / src.len() as f32 + eps).sqrt().recip();
    for i in 0..src.len() {
        dst[i] = src[i] * scale * gain[i];
    }
}

/// Rotary position embedding, "rotate-half" convention: within each head the
/// first half of the channels pairs with the second half.
pub fn rope(dst: &mut [f32], src: &[f32], pos: u32, n_heads: usize, head_dim: usize, theta: f32) {
    debug_assert_eq!(src.len(), n_heads * head_dim);
    debug_assert_eq!(dst.len(), src.len());

    let half = head_dim / 2;
    for h in 0..n_heads {
        for j in 0..half {
            let freq = 1.0 / theta.powf((2 * j) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = (angle.sin(), angle.cos());
            let i0 = h * head_dim + j;
            let i1 = i0 + half;
            let (v0, v1) = (src[i0], src[i1]);
            dst[i0] = v0 * cos - v1 * sin;
            dst[i1] = v0 * sin + v1 * cos;
        }
    }
}

/// Numerically stable in-place softmax.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU activation: `dst = silu(gate) * up`.
pub fn swiglu(dst: &mut [f32], gate: &[f32], up: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = silu(gate[i]) * up[i];
    }
}

pub fn add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = a[i] + b[i];
    }
}

/// Single-query multi-head attention over a KV cache holding positions `0..=pos`.
///
/// `k_cache` / `v_cache` are `[MAX_SEQ, n_heads * head_dim]` row-major.
pub fn attention(
    dst: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    pos: u32,
    n_heads: usize,
    head_dim: usize,
) {
    let dim = n_heads * head_dim;
    let n_ctx = pos as usize + 1;
    let scale = (head_dim as f32).sqrt().recip();
    let mut scores = vec![0.0f32; n_ctx];

    for h in 0..n_heads {
        let base = h * head_dim;
        for (t, score) in scores.iter_mut().enumerate() {
            let k_row = &k_cache[t * dim + base..t * dim + base + head_dim];
            let dot: f32 = (0..head_dim).map(|d| q[base + d] * k_row[d]).sum();
            *score = dot * scale;
        }
        softmax(&mut scores);
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (t, &p) in scores.iter().enumerate() {
                acc += p * v_cache[t * dim + base + d];
            }
            dst[base + d] = acc;
        }
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

/// How the next token is picked from the logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// `0.0` collapses to greedy argmax.
    pub temperature: f32,
    /// `0` disables the top-k cut.
    pub top_k: u32,
    /// Stream position is mixed into this so a run is reproducible end to end.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.85,
            top_k: 8,
            seed: 0xA5A5_1234,
        }
    }
}

/// Temperature + top-k sampling. Deterministic given `(logits, cfg, step)`, which
/// keeps a full generation reproducible across reloads and across backends.
pub fn sample(logits: &[f32], cfg: &SamplerConfig, step: u32) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if cfg.temperature <= 1e-6 {
        return argmax(logits);
    }

    let mut ranked: Vec<(usize, f32)> = logits
        .iter()
        .map(|&v| v / cfg.temperature)
        .enumerate()
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = if cfg.top_k == 0 {
        ranked.len()
    } else {
        (cfg.top_k as usize).min(ranked.len())
    };
    ranked.truncate(k);

    let mut probs: Vec<f32> = ranked.iter().map(|&(_, v)| v).collect();
    softmax(&mut probs);

    let mut rng = Pcg32::new(cfg.seed ^ ((step as u64 + 1).wrapping_mul(0x9E37_79B9)));
    let roll = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if roll < cumulative {
            return ranked[i].0 as u32;
        }
    }
    ranked[k - 1].0 as u32
}

/// L2 norm, used for activation telemetry in the UI.
pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_naive_reference() {
        let n_in = 6;
        let n_out = 3;
        let src: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.5 - 1.0).collect();
        let w: Vec<f32> = (0..n_in * n_out).map(|i| (i as f32 * 0.25).sin()).collect();
        let mut dst = vec![0.0; n_out];
        matvec(&mut dst, &src, &w, n_in, n_out);

        for n in 0..n_out {
            let expected: f32 = (0..n_in).map(|k| src[k] * w[n * n_in + k]).sum();
            assert!(
                (dst[n] - expected).abs() < 1e-5,
                "row {n}: {} vs {expected}",
                dst[n]
            );
        }
    }

    #[test]
    fn rmsnorm_produces_unit_rms_with_unit_gain() {
        let src = vec![3.0, -4.0, 0.0, 5.0];
        let gain = vec![1.0; 4];
        let mut dst = vec![0.0; 4];
        rmsnorm(&mut dst, &src, &gain, 0.0);
        let rms = (dst.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5, "rms was {rms}");
    }

    #[test]
    fn rope_preserves_pair_magnitude_and_is_identity_at_zero() {
        let src: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let mut dst = vec![0.0; 8];

        rope(&mut dst, &src, 0, 1, 8, 10_000.0);
        for i in 0..8 {
            assert!((dst[i] - src[i]).abs() < 1e-6, "pos 0 must be identity");
        }

        rope(&mut dst, &src, 7, 1, 8, 10_000.0);
        for j in 0..4 {
            let before = src[j].hypot(src[j + 4]);
            let after = dst[j].hypot(dst[j + 4]);
            assert!(
                (before - after).abs() < 1e-5,
                "rotation must preserve magnitude"
            );
        }
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = vec![1000.0, 1001.0, 999.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
        assert!(x.iter().all(|v| v.is_finite()), "overflow leaked through");
        assert!(x[1] > x[0] && x[0] > x[2]);
    }

    #[test]
    fn attention_at_position_zero_returns_the_only_value_row() {
        let (n_heads, head_dim) = (2, 4);
        let dim = n_heads * head_dim;
        let q: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut k_cache = vec![0.0; 4 * dim];
        let mut v_cache = vec![0.0; 4 * dim];
        for i in 0..dim {
            k_cache[i] = 0.1 * i as f32;
            v_cache[i] = 2.0 + i as f32;
        }
        let mut dst = vec![0.0; dim];
        attention(&mut dst, &q, &k_cache, &v_cache, 0, n_heads, head_dim);
        for i in 0..dim {
            assert!((dst[i] - v_cache[i]).abs() < 1e-5, "index {i}");
        }
    }

    #[test]
    fn sampling_is_deterministic_and_respects_top_k() {
        let logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = SamplerConfig {
            temperature: 0.9,
            top_k: 3,
            seed: 7,
        };
        let a = sample(&logits, &cfg, 4);
        let b = sample(&logits, &cfg, 4);
        assert_eq!(a, b, "same inputs must give the same token");
        assert!(
            a >= 13,
            "top-3 of a monotonic ramp must be the last three ids, got {a}"
        );

        let greedy = SamplerConfig {
            temperature: 0.0,
            ..cfg
        };
        assert_eq!(sample(&logits, &greedy, 4), 15);
    }
}
