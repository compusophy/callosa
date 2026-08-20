//! Runs the real WGSL kernels on a native GPU and checks them against the CPU
//! reference in `src/tensor.rs`.
//!
//! These are the same shader files the browser loads, so a mismatch here is a
//! mismatch in the browser. If no GPU adapter is available (headless CI, no
//! drivers) the tests report that and pass rather than failing on the
//! environment — the CPU-side unit tests still cover the maths.

use std::rc::Rc;

use callosa::config::{
    BlockWeights, EmbeddingWeights, HeadWeights, Role, DIM, MODEL_SEED, VOCAB_SIZE,
};
use callosa::gpu::{GpuContext, GpuShard};
use callosa::model::CpuShard;

fn context() -> Option<Rc<GpuContext>> {
    match pollster::block_on(GpuContext::new()) {
        Ok(ctx) => {
            eprintln!("[gpu tests] adapter: {}", ctx.adapter_label);
            Some(Rc::new(ctx))
        }
        Err(reason) => {
            eprintln!("[gpu tests] skipped, no adapter: {reason}");
            None
        }
    }
}

fn shards(ctx: Rc<GpuContext>, role: Role) -> (GpuShard, CpuShard) {
    let block = BlockWeights::synthesize(MODEL_SEED, role.layer_index());
    let head = role.has_head().then(|| HeadWeights::synthesize(MODEL_SEED));
    let gpu = GpuShard::new(ctx, role, &block, head.as_ref()).expect("build gpu shard");
    let cpu = CpuShard::new(role, block, head);
    (gpu, cpu)
}

/// Relative tolerance. The GPU accumulates in a different order than the CPU
/// (and uses `fma`), so exact equality is not the bar; agreeing to ~1e-4
/// relative is.
fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], pos: u32) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    let scale = cpu.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            g.is_finite(),
            "{label}: gpu produced {g} at index {i}, pos {pos}"
        );
        let delta = (g - c).abs();
        if delta > worst {
            worst = delta;
            worst_at = i;
        }
    }
    assert!(
        worst / scale < 1e-3,
        "{label} at pos {pos}: worst absolute delta {worst} at index {worst_at} \
         (gpu {}, cpu {}), scale {scale}",
        gpu[worst_at],
        cpu[worst_at]
    );
}

#[test]
fn gpu_block_matches_cpu_reference_across_a_sequence() {
    let Some(ctx) = context() else { return };
    let embed = EmbeddingWeights::synthesize(MODEL_SEED);
    let (gpu, mut cpu) = shards(ctx, Role::Node0);

    // Walk far enough that the KV cache is genuinely multi-row and the softmax
    // has something to normalise over.
    for pos in 0..12u32 {
        let x = embed.lookup(pos * 5 + 1);
        let g = pollster::block_on(gpu.forward(&x, pos)).expect("gpu forward");
        let c = cpu.forward(&x, pos).expect("cpu forward");
        assert_close("block 0 hidden state", &g, &c, pos);
    }
}

#[test]
fn gpu_head_matches_cpu_reference() {
    let Some(ctx) = context() else { return };
    let embed = EmbeddingWeights::synthesize(MODEL_SEED);
    let (gpu, mut cpu) = shards(ctx, Role::Node1);

    for pos in 0..8u32 {
        // Stage 1 consumes a hidden state, not an embedding; any deterministic
        // DIM-wide vector exercises the same code path.
        let x: Vec<f32> = embed
            .lookup(pos + 3)
            .iter()
            .enumerate()
            .map(|(i, v)| v + (i as f32 * 0.01).sin())
            .collect();
        let g = pollster::block_on(gpu.forward(&x, pos)).expect("gpu forward");
        let c = cpu.forward(&x, pos).expect("cpu forward");
        assert_eq!(g.len(), VOCAB_SIZE);
        assert_close("logits", &g, &c, pos);
    }
}

#[test]
fn gpu_and_cpu_agree_on_the_full_two_stage_pipeline() {
    let Some(ctx) = context() else { return };
    let embed = EmbeddingWeights::synthesize(MODEL_SEED);
    let (gpu0, mut cpu0) = shards(Rc::clone(&ctx), Role::Node0);
    let (gpu1, mut cpu1) = shards(ctx, Role::Node1);

    let mut token = 1u32;
    for pos in 0..10u32 {
        let x = embed.lookup(token);

        let hidden_gpu = pollster::block_on(gpu0.forward(&x, pos)).expect("gpu stage 0");
        let hidden_cpu = cpu0.forward(&x, pos).expect("cpu stage 0");
        assert_close("stage 0", &hidden_gpu, &hidden_cpu, pos);

        // Feed each backend its own stage-0 output so divergence compounds the
        // way it would in a real split deployment.
        let logits_gpu = pollster::block_on(gpu1.forward(&hidden_gpu, pos)).expect("gpu stage 1");
        let logits_cpu = cpu1.forward(&hidden_cpu, pos).expect("cpu stage 1");
        assert_close("stage 1", &logits_gpu, &logits_cpu, pos);

        // Greedy choice must be identical, which is the property that actually
        // matters: both backends must generate the same text.
        let pick = |v: &[f32]| callosa::tensor::argmax(v);
        assert_eq!(
            pick(&logits_gpu),
            pick(&logits_cpu),
            "backends disagreed on the argmax token at pos {pos}"
        );
        token = pick(&logits_gpu);
    }
}

#[test]
fn gpu_shard_rejects_bad_input() {
    let Some(ctx) = context() else { return };
    let (gpu, _) = shards(ctx, Role::Node0);

    assert!(pollster::block_on(gpu.forward(&vec![0.0; DIM - 1], 0)).is_err());
    assert!(pollster::block_on(gpu.forward(&vec![0.0; DIM], 10_000)).is_err());
}

#[test]
fn shard_construction_enforces_head_ownership() {
    let Some(ctx) = context() else { return };
    let block = BlockWeights::synthesize(MODEL_SEED, 0);
    let head = HeadWeights::synthesize(MODEL_SEED);

    assert!(
        GpuShard::new(Rc::clone(&ctx), Role::Node1, &block, None).is_err(),
        "node 1 must be given head weights"
    );
    assert!(
        GpuShard::new(ctx, Role::Node0, &block, Some(&head)).is_err(),
        "node 0 must not be given head weights"
    );
}
