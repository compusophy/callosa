//! WebGPU backend: one device, one bind group layout family, zero allocation
//! inside the inference loop.
//!
//! The whole transformer block is recorded into a single command encoder as two
//! compute passes with the KV-cache writes between them, submitted once, and
//! read back once. Encoding each matmul as its own submit-plus-map round trip
//! costs more in synchronisation than the arithmetic it is scheduling, which on
//! a batch-1 decode workload dominates the step time.

use std::collections::HashMap;
use std::rc::Rc;

use wgpu::util::DeviceExt;

use crate::config::{
    BlockWeights, EmbeddingWeights, HeadWeights, Role, DIM, FFN_HIDDEN, HEAD_DIM, MAX_SEQ,
    NORM_EPS, N_HEADS, ROPE_THETA, VOCAB_SIZE,
};

/// Keep the shader's compile-time cache bound honest.
const _: () = assert!(
    MAX_SEQ == 128,
    "MAX_SEQ_LEN in shaders/attention.wgsl is hard-coded to 128"
);

type Buf = Rc<wgpu::Buffer>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kernel {
    MatVec,
    RmsNorm,
    ResidualAdd,
    SwiGlu,
    Rope,
    Attention,
}

/// Device, queue and every compiled pipeline. Created once per tab and shared by
/// whatever shard the tab is currently hosting.
pub struct GpuContext {
    /// Held for the lifetime of the context: the device outlives the instance
    /// that produced it otherwise, which some native drivers dislike at teardown.
    _instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub std_layout: wgpu::BindGroupLayout,
    pub attn_layout: wgpu::BindGroupLayout,
    pub adapter_label: String,
    pipelines: HashMap<Kernel, wgpu::ComputePipeline>,
}

impl GpuContext {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                "no webgpu adapter available (is webgpu enabled in this browser?)".to_string()
            })?;

        let info = adapter.get_info();
        // Browsers deliberately hide the adapter name, so fall back to something
        // more useful than the Debug spelling of the backend enum.
        let adapter_label = if !info.name.is_empty() {
            format!("{} ({:?})", info.name, info.backend)
        } else if info.backend == wgpu::Backend::BrowserWebGpu {
            "browser webgpu (adapter hidden)".to_string()
        } else {
            format!("{:?}", info.backend)
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("pipeline-shard-device"),
                    required_features: wgpu::Features::empty(),
                    // downlevel_defaults allows 4 storage buffers per stage, which
                    // is exactly what the attention layout needs.
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("failed to acquire a webgpu device: {e:?}"))?;

        // Surface validation and out-of-memory errors instead of letting the
        // device quietly die halfway through a generation.
        device.on_uncaptured_error(Box::new(|err| {
            report_device_error(&err.to_string());
        }));

        let std_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("std-kernel-layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
                uniform_entry(4),
            ],
        });

        let attn_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention-layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                uniform_entry(4),
                uniform_entry(5),
            ],
        });

        let kernels_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernels.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kernels.wgsl").into()),
        });
        let attention_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/attention.wgsl").into()),
        });

        // WGSL implementations disagree at the edges -- Tint and naga do not
        // accept exactly the same programs. Without this check a rejected shader
        // yields an invalid pipeline that silently computes zeros, which is a far
        // worse failure than refusing to start.
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::Maintain::Wait);
        check_shader(&kernels_module, "kernels.wgsl").await?;
        check_shader(&attention_module, "attention.wgsl").await?;

        let std_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("std-pipeline-layout"),
            bind_group_layouts: &[&std_layout],
            push_constant_ranges: &[],
        });
        let attn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attention-pipeline-layout"),
            bind_group_layouts: &[&attn_layout],
            push_constant_ranges: &[],
        });

        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut pipelines = HashMap::new();
        for (kernel, entry) in [
            (Kernel::MatVec, "matvec"),
            (Kernel::RmsNorm, "rmsnorm"),
            (Kernel::ResidualAdd, "residual_add"),
            (Kernel::SwiGlu, "swiglu"),
            (Kernel::Rope, "rope"),
        ] {
            pipelines.insert(
                kernel,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&std_pipeline_layout),
                    module: &kernels_module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        pipelines.insert(
            Kernel::Attention,
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("attention"),
                layout: Some(&attn_pipeline_layout),
                module: &attention_module,
                entry_point: Some("attention"),
                compilation_options: Default::default(),
                cache: None,
            }),
        );

        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::Maintain::Wait);
        if let Some(error) = device.pop_error_scope().await {
            return Err(format!("compute pipeline creation failed: {error}"));
        }

        Ok(GpuContext {
            _instance: instance,
            device,
            queue,
            std_layout,
            attn_layout,
            adapter_label,
            pipelines,
        })
    }

    fn pipeline(&self, kernel: Kernel) -> &wgpu::ComputePipeline {
        // Every variant is inserted in `new`, so this cannot miss.
        &self.pipelines[&kernel]
    }

    /// Block until the queue drains. A no-op in the browser, where the JS event
    /// loop drives buffer mapping instead.
    fn drain(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::Maintain::Wait);
    }
}

/// Fail loudly on a shader the driver would not compile.
async fn check_shader(module: &wgpu::ShaderModule, name: &str) -> Result<(), String> {
    let info = module.get_compilation_info().await;
    let errors: Vec<String> = info
        .messages
        .iter()
        .filter(|m| m.message_type == wgpu::CompilationMessageType::Error)
        .map(|m| m.message.trim().to_string())
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("{name} failed to compile: {}", errors.join(" | ")))
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(target_arch = "wasm32")]
fn report_device_error(msg: &str) {
    web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&format!(
        "[webgpu] uncaptured device error: {msg}"
    )));
}

#[cfg(not(target_arch = "wasm32"))]
fn report_device_error(msg: &str) {
    eprintln!("[webgpu] uncaptured device error: {msg}");
}

fn scratch_buffer(device: &wgpu::Device, label: &str, floats: usize) -> Buf {
    Rc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (floats * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
}

fn weight_buffer(device: &wgpu::Device, label: &str, data: &[f32]) -> Buf {
    Rc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        }),
    )
}

/// One recorded dispatch: which pipeline, which prebuilt bind group, how wide.
struct Op {
    kernel: Kernel,
    bind_group: wgpu::BindGroup,
    workgroups: u32,
    label: &'static str,
}

/// A transformer block resident on the GPU, plus (for node 1) the output head.
pub struct GpuShard {
    ctx: Rc<GpuContext>,
    role: Role,
    output_len: usize,

    x: Buf,
    k_cache: Buf,
    v_cache: Buf,
    kr: Buf,
    v: Buf,
    result: Buf,
    staging: wgpu::Buffer,
    step_uniform: Buf,

    /// Projections + RoPE. Must finish before the KV cache writes.
    pre_cache: Vec<Op>,
    /// Attention, output projection, FFN, and the head when present.
    post_cache: Vec<Op>,

    /// Keeps weights and scratch alive for as long as the shard exists.
    _retained: Vec<Buf>,
}

impl GpuShard {
    pub fn new(
        ctx: Rc<GpuContext>,
        role: Role,
        block: &BlockWeights,
        head: Option<&HeadWeights>,
    ) -> Result<Self, String> {
        if role.has_head() != head.is_some() {
            return Err(format!(
                "role {} {} head weights",
                role.as_str(),
                if role.has_head() {
                    "requires"
                } else {
                    "must not be given"
                }
            ));
        }

        let device = &ctx.device;
        let mut retained: Vec<Buf> = Vec::new();
        let mut uniform_cache: HashMap<[u32; 4], Buf> = HashMap::new();

        // Per-op constants never change, so identical (n_in, n_out) pairs share a
        // single uniform buffer.
        let mut dims_uniform = |dims: [u32; 4]| -> Buf {
            uniform_cache
                .entry(dims)
                .or_insert_with(|| {
                    Rc::new(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("op-dims"),
                            contents: bytemuck::cast_slice(&dims),
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                    )
                })
                .clone()
        };

        let step_uniform: Buf = Rc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // --- activations -----------------------------------------------------
        let x = scratch_buffer(device, "x", DIM);
        let xb = scratch_buffer(device, "xb", DIM);
        let q = scratch_buffer(device, "q", DIM);
        let k = scratch_buffer(device, "k", DIM);
        let v = scratch_buffer(device, "v", DIM);
        let qr = scratch_buffer(device, "q-roped", DIM);
        let kr = scratch_buffer(device, "k-roped", DIM);
        let att = scratch_buffer(device, "attn-out", DIM);
        let attp = scratch_buffer(device, "attn-proj", DIM);
        let x1 = scratch_buffer(device, "residual-1", DIM);
        let xb2 = scratch_buffer(device, "xb2", DIM);
        let hg = scratch_buffer(device, "ffn-gate", FFN_HIDDEN);
        let hu = scratch_buffer(device, "ffn-up", FFN_HIDDEN);
        let hs = scratch_buffer(device, "ffn-act", FFN_HIDDEN);
        let hd = scratch_buffer(device, "ffn-down", DIM);
        let x2 = scratch_buffer(device, "residual-2", DIM);
        let k_cache = scratch_buffer(device, "k-cache", MAX_SEQ * DIM);
        let v_cache = scratch_buffer(device, "v-cache", MAX_SEQ * DIM);

        // --- weights ---------------------------------------------------------
        let w_attn_norm = weight_buffer(device, "attn-norm", &block.attn_norm);
        let w_q = weight_buffer(device, "wq", &block.wq);
        let w_k = weight_buffer(device, "wk", &block.wk);
        let w_v = weight_buffer(device, "wv", &block.wv);
        let w_o = weight_buffer(device, "wo", &block.wo);
        let w_ffn_norm = weight_buffer(device, "ffn-norm", &block.ffn_norm);
        let w1 = weight_buffer(device, "w1", &block.w1);
        let w2 = weight_buffer(device, "w2", &block.w2);
        let w3 = weight_buffer(device, "w3", &block.w3);

        // RMSNorm and RoPE need float constants; ferry them through the u32
        // uniform by bit pattern rather than introducing a second uniform type.
        let eps_bits = NORM_EPS.to_bits();
        let theta_bits = ROPE_THETA.to_bits();
        let dim = DIM as u32;
        let ffn = FFN_HIDDEN as u32;

        let std_bg = |label: &str, src: &Buf, aux: &Buf, dst: &Buf, dims: &Buf| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &ctx.std_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: aux.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: step_uniform.as_entire_binding(),
                    },
                ],
            })
        };

        let u_norm = dims_uniform([dim, eps_bits, 0, 0]);
        let u_dim_dim = dims_uniform([dim, dim, 0, 0]);
        let u_dim_ffn = dims_uniform([dim, ffn, 0, 0]);
        let u_ffn_dim = dims_uniform([ffn, dim, 0, 0]);
        let u_elem_dim = dims_uniform([dim, 0, 0, 0]);
        let u_elem_ffn = dims_uniform([ffn, 0, 0, 0]);
        let u_rope = dims_uniform([N_HEADS as u32, HEAD_DIM as u32, theta_bits, 0]);
        let u_attn = dims_uniform([N_HEADS as u32, HEAD_DIM as u32, MAX_SEQ as u32, 0]);

        let groups_dim = dim.div_ceil(64);
        let groups_ffn = ffn.div_ceil(64);
        let groups_rope = ((N_HEADS * HEAD_DIM / 2) as u32).div_ceil(64);

        // `rope` ignores binding 1; any read-only storage buffer satisfies the
        // shared layout, so the norm gain doubles as filler.
        let unused = &w_attn_norm;

        #[rustfmt::skip]
        let pre_cache = vec![
            Op { kernel: Kernel::RmsNorm, bind_group: std_bg("attn-norm", &x, &w_attn_norm, &xb, &u_norm), workgroups: 1, label: "rmsnorm(attn)" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("q-proj", &xb, &w_q, &q, &u_dim_dim), workgroups: groups_dim, label: "matvec(wq)" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("k-proj", &xb, &w_k, &k, &u_dim_dim), workgroups: groups_dim, label: "matvec(wk)" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("v-proj", &xb, &w_v, &v, &u_dim_dim), workgroups: groups_dim, label: "matvec(wv)" },
            Op { kernel: Kernel::Rope, bind_group: std_bg("rope-q", &q, unused, &qr, &u_rope), workgroups: groups_rope, label: "rope(q)" },
            Op { kernel: Kernel::Rope, bind_group: std_bg("rope-k", &k, unused, &kr, &u_rope), workgroups: groups_rope, label: "rope(k)" },
        ];

        let attn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attention"),
            layout: &ctx.attn_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: qr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_cache.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: att.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_cache.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: u_attn.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: step_uniform.as_entire_binding(),
                },
            ],
        });

        #[rustfmt::skip]
        let mut post_cache = vec![
            Op { kernel: Kernel::Attention, bind_group: attn_bg, workgroups: N_HEADS as u32, label: "attention" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("o-proj", &att, &w_o, &attp, &u_dim_dim), workgroups: groups_dim, label: "matvec(wo)" },
            Op { kernel: Kernel::ResidualAdd, bind_group: std_bg("residual-1", &x, &attp, &x1, &u_elem_dim), workgroups: groups_dim, label: "residual(attn)" },
            Op { kernel: Kernel::RmsNorm, bind_group: std_bg("ffn-norm", &x1, &w_ffn_norm, &xb2, &u_norm), workgroups: 1, label: "rmsnorm(ffn)" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("ffn-gate", &xb2, &w1, &hg, &u_dim_ffn), workgroups: groups_ffn, label: "matvec(w1)" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("ffn-up", &xb2, &w3, &hu, &u_dim_ffn), workgroups: groups_ffn, label: "matvec(w3)" },
            Op { kernel: Kernel::SwiGlu, bind_group: std_bg("swiglu", &hg, &hu, &hs, &u_elem_ffn), workgroups: groups_ffn, label: "swiglu" },
            Op { kernel: Kernel::MatVec, bind_group: std_bg("ffn-down", &hs, &w2, &hd, &u_ffn_dim), workgroups: groups_dim, label: "matvec(w2)" },
            Op { kernel: Kernel::ResidualAdd, bind_group: std_bg("residual-2", &x1, &hd, &x2, &u_elem_dim), workgroups: groups_dim, label: "residual(ffn)" },
        ];

        // Node 1 finishes the model: final norm then the LM head. Running the
        // head on the worker means the readback is VOCAB_SIZE floats instead of
        // the full hidden state, and node 0 never needs the head weights.
        let result: Buf = match head {
            Some(head) => {
                let w_final_norm = weight_buffer(device, "final-norm", &head.final_norm);
                let w_lm_head = weight_buffer(device, "lm-head", &head.lm_head);
                let xn = scratch_buffer(device, "final-normed", DIM);
                let logits = scratch_buffer(device, "logits", VOCAB_SIZE);
                let u_dim_vocab = dims_uniform([dim, VOCAB_SIZE as u32, 0, 0]);

                post_cache.push(Op {
                    kernel: Kernel::RmsNorm,
                    bind_group: std_bg("final-norm", &x2, &w_final_norm, &xn, &u_norm),
                    workgroups: 1,
                    label: "rmsnorm(final)",
                });
                post_cache.push(Op {
                    kernel: Kernel::MatVec,
                    bind_group: std_bg("lm-head", &xn, &w_lm_head, &logits, &u_dim_vocab),
                    workgroups: (VOCAB_SIZE as u32).div_ceil(64),
                    label: "matvec(lm_head)",
                });

                retained.extend([w_final_norm, w_lm_head, xn]);
                logits
            }
            None => Rc::clone(&x2),
        };

        let output_len = role.output_len();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-staging"),
            size: (output_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        retained.extend([
            xb,
            q,
            k,
            qr,
            att,
            attp,
            x1,
            xb2,
            hg,
            hu,
            hs,
            hd,
            x2,
            w_attn_norm,
            w_q,
            w_k,
            w_v,
            w_o,
            w_ffn_norm,
            w1,
            w2,
            w3,
        ]);
        retained.extend(uniform_cache.into_values());

        Ok(GpuShard {
            ctx,
            role,
            output_len,
            x,
            k_cache,
            v_cache,
            kr,
            v,
            result,
            staging,
            step_uniform,
            pre_cache,
            post_cache,
            _retained: retained,
        })
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn adapter_label(&self) -> &str {
        &self.ctx.adapter_label
    }

    /// Resetting a sequence needs no cache clear: attention only ever reads
    /// positions `0..=pos`, so anything left from a previous run is unreachable
    /// once `pos` restarts at zero.
    pub fn reset(&mut self) {}

    /// Run the block for one token position.
    ///
    /// `input` is the hidden state entering this stage; the result is the hidden
    /// state leaving it, or the logits when this shard owns the head.
    pub async fn forward(&self, input: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        if input.len() != DIM {
            return Err(format!(
                "hidden state width mismatch: expected {DIM}, got {}",
                input.len()
            ));
        }
        if pos as usize >= MAX_SEQ {
            return Err(format!(
                "position {pos} exceeds kv-cache capacity of {MAX_SEQ}"
            ));
        }

        let ctx = &self.ctx;
        ctx.queue
            .write_buffer(&self.x, 0, bytemuck::cast_slice(input));
        ctx.queue.write_buffer(
            &self.step_uniform,
            0,
            bytemuck::cast_slice(&[pos, 0u32, 0, 0]),
        );

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("block-forward"),
            });

        record_pass(&mut encoder, ctx, "pre-cache", &self.pre_cache);

        // Append this position's rotated key and its value to the cache. The
        // destination offset is known on the CPU, so this is a plain buffer copy
        // rather than another dispatch.
        let row_bytes = (DIM * 4) as u64;
        let offset = pos as u64 * row_bytes;
        encoder.copy_buffer_to_buffer(&self.kr, 0, &self.k_cache, offset, row_bytes);
        encoder.copy_buffer_to_buffer(&self.v, 0, &self.v_cache, offset, row_bytes);

        record_pass(&mut encoder, ctx, "post-cache", &self.post_cache);

        encoder.copy_buffer_to_buffer(
            &self.result,
            0,
            &self.staging,
            0,
            (self.output_len * 4) as u64,
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        ctx.drain();

        rx.await
            .map_err(|_| "gpu readback channel dropped before the map completed".to_string())?
            .map_err(|e| format!("failed to map the readback buffer: {e:?}"))?;

        let view = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.staging.unmap();

        Ok(out)
    }

    /// Number of dispatches issued per token, so the cost of a step is visible
    /// in the UI rather than implied.
    pub fn dispatch_count(&self) -> usize {
        self.pre_cache.len() + self.post_cache.len()
    }

    /// Ordered kernel labels for one step, for the UI and for debugging.
    pub fn kernel_trace(&self) -> Vec<&'static str> {
        self.pre_cache
            .iter()
            .map(|op| op.label)
            .chain(["copy(k->cache)", "copy(v->cache)"])
            .chain(self.post_cache.iter().map(|op| op.label))
            .collect()
    }
}

fn record_pass(encoder: &mut wgpu::CommandEncoder, ctx: &GpuContext, label: &str, ops: &[Op]) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    for op in ops {
        pass.set_pipeline(ctx.pipeline(op.kernel));
        pass.set_bind_group(0, &op.bind_group, &[]);
        pass.dispatch_workgroups(op.workgroups, 1, 1);
    }
}

/// Embedding lookup stays on the CPU: it is a single row copy, and shipping the
/// table to the GPU to feed one gather would cost more than it saves.
pub fn embed(weights: &EmbeddingWeights, token_id: u32) -> Vec<f32> {
    weights.lookup(token_id)
}
