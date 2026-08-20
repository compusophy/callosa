// Decode-path kernels for one transformer block.
//
// All five entry points share one bind group layout, so a shard is built from a
// single layout and every per-step dispatch reuses a bind group created at load
// time. Nothing is allocated inside the inference loop.
//
//   binding 0  src   read-only storage    primary input
//   binding 1  aux   read-only storage    weights / second operand
//   binding 2  dst   read-write storage   output
//   binding 3  dims  uniform vec4<u32>    per-op constants, baked once
//   binding 4  step  uniform vec4<u32>    per-step constants, .x = position

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;
@group(0) @binding(4) var<uniform> step_info: vec4<u32>;

const WORKGROUP: u32 = 64u;

// ---------------------------------------------------------------------------
// mat-vec: dst[n] = dot(src, aux[n])
//
// Autoregressive decode is batch-1, so the classic 16x16 tiled matmul wastes
// 15/16 of every workgroup on a phantom M dimension. One thread per output row
// with a contiguous [n_out, n_in] weight layout keeps every lane busy and every
// read coalesced along the row.
// dims: .x = n_in, .y = n_out
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP)
fn matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    let n_in = dims.x;
    if (n >= dims.y) {
        return;
    }

    let base = n * n_in;
    // Four accumulators to give the scheduler independent FMA chains.
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;

    let tail = n_in & 3u;
    let body = n_in - tail;
    var k = 0u;
    for (; k < body; k = k + 4u) {
        a0 = fma(src[k], aux[base + k], a0);
        a1 = fma(src[k + 1u], aux[base + k + 1u], a1);
        a2 = fma(src[k + 2u], aux[base + k + 2u], a2);
        a3 = fma(src[k + 3u], aux[base + k + 3u], a3);
    }
    var acc = (a0 + a1) + (a2 + a3);
    for (; k < n_in; k = k + 1u) {
        acc = fma(src[k], aux[base + k], acc);
    }
    dst[n] = acc;
}

// ---------------------------------------------------------------------------
// RMSNorm with a learned per-channel gain in `aux`.
// dims: .x = n, .y = bitcast<u32>(eps)
// Launched as a single workgroup; the reduction lives in workgroup memory.
// ---------------------------------------------------------------------------
var<workgroup> norm_partial: array<f32, WORKGROUP>;

@compute @workgroup_size(WORKGROUP)
fn rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = dims.x;
    let eps = bitcast<f32>(dims.y);
    let tid = lid.x;

    var acc = 0.0;
    for (var i = tid; i < n; i = i + WORKGROUP) {
        let v = src[i];
        acc = fma(v, v, acc);
    }
    norm_partial[tid] = acc;
    workgroupBarrier();

    for (var s = WORKGROUP / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            norm_partial[tid] = norm_partial[tid] + norm_partial[tid + s];
        }
        workgroupBarrier();
    }

    let scale = inverseSqrt(norm_partial[0] / f32(n) + eps);
    for (var i = tid; i < n; i = i + WORKGROUP) {
        dst[i] = src[i] * scale * aux[i];
    }
}

// ---------------------------------------------------------------------------
// Residual add: dst = src + aux
// dims: .x = n
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) {
        return;
    }
    dst[i] = src[i] + aux[i];
}

// ---------------------------------------------------------------------------
// SwiGLU: dst = silu(src) * aux, with src the gate and aux the up projection.
// dims: .x = n
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) {
        return;
    }
    let g = src[i];
    dst[i] = (g / (1.0 + exp(-g))) * aux[i];
}

// ---------------------------------------------------------------------------
// RoPE, rotate-half convention: channel j of a head pairs with j + head_dim/2.
// dims: .x = n_heads, .y = head_dim, .z = bitcast<u32>(theta)
// step: .x = position
// One thread per (head, pair).
// ---------------------------------------------------------------------------
@compute @workgroup_size(WORKGROUP)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_dim = dims.y;
    let half = head_dim / 2u;
    let total = dims.x * half;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let h = idx / half;
    let j = idx % half;
    let theta = bitcast<f32>(dims.z);
    let freq = 1.0 / pow(theta, f32(2u * j) / f32(head_dim));
    let angle = f32(step_info.x) * freq;
    let c = cos(angle);
    let s = sin(angle);

    let i0 = h * head_dim + j;
    let i1 = i0 + half;
    let v0 = src[i0];
    let v1 = src[i1];
    dst[i0] = v0 * c - v1 * s;
    dst[i1] = v0 * s + v1 * c;
}
