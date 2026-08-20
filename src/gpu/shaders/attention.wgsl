// Single-query multi-head attention over a KV cache.
//
// One workgroup per head. Scores for the whole context live in workgroup memory,
// so the softmax reduction never round-trips through global memory and the whole
// head completes in one dispatch.
//
//   binding 0  q        read-only storage   query for this position, [n_heads * head_dim]
//   binding 1  k_cache  read-only storage   [MAX_SEQ_LEN, n_heads * head_dim]
//   binding 2  out      read-write storage  [n_heads * head_dim]
//   binding 3  v_cache  read-only storage   [MAX_SEQ_LEN, n_heads * head_dim]
//   binding 4  dims     uniform vec4<u32>   .x = n_heads, .y = head_dim
//   binding 5  step     uniform vec4<u32>   .x = current position

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>;
@group(0) @binding(5) var<uniform> step_info: vec4<u32>;

const WORKGROUP: u32 = 64u;
// Must stay in sync with `config::MAX_SEQ`; asserted on the Rust side at build.
const MAX_SEQ_LEN: u32 = 128u;

var<workgroup> scores: array<f32, MAX_SEQ_LEN>;
var<workgroup> reduce_buf: array<f32, WORKGROUP>;

@compute @workgroup_size(WORKGROUP)
fn attention(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wid.x;
    let tid = lid.x;
    let head_dim = dims.y;
    let dim = dims.x * head_dim;
    let n_ctx = min(step_info.x + 1u, MAX_SEQ_LEN);
    let base = head * head_dim;
    let scale = inverseSqrt(f32(head_dim));

    // 1. Scaled dot-product scores against every cached key.
    for (var t = tid; t < n_ctx; t = t + WORKGROUP) {
        let k_base = t * dim + base;
        var acc = 0.0;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            acc = fma(q[base + d], k_cache[k_base + d], acc);
        }
        scores[t] = acc * scale;
    }
    workgroupBarrier();

    // 2. Max reduction for a numerically stable softmax.
    //
    // Seeded from a real score rather than a negative-infinity sentinel: WGSL
    // has no f32::MIN constant, and a hand-written -3.4028235e38 literal is
    // accepted by naga but rejected by Tint as out of range. Position 0 is
    // always present (n_ctx >= 1) and always written above.
    var local_max = scores[0];
    for (var t = tid; t < n_ctx; t = t + WORKGROUP) {
        local_max = max(local_max, scores[t]);
    }
    reduce_buf[tid] = local_max;
    workgroupBarrier();
    for (var s = WORKGROUP / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + s]);
        }
        workgroupBarrier();
    }
    let max_score = reduce_buf[0];
    // Every lane has read reduce_buf[0]; only now is it safe to overwrite.
    workgroupBarrier();

    // 3. Exponentiate in place and sum.
    var local_sum = 0.0;
    for (var t = tid; t < n_ctx; t = t + WORKGROUP) {
        let e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum = local_sum + e;
    }
    reduce_buf[tid] = local_sum;
    workgroupBarrier();
    for (var s = WORKGROUP / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + s];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / reduce_buf[0];

    // 4. Probability-weighted sum of the cached values, one thread per channel.
    for (var d = tid; d < head_dim; d = d + WORKGROUP) {
        var acc = 0.0;
        for (var t = 0u; t < n_ctx; t = t + 1u) {
            acc = fma(scores[t], v_cache[t * dim + base + d], acc);
        }
        out[base + d] = acc * inv_sum;
    }
}
