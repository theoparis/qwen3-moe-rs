//! Fake-quantization helpers for dense Qwen3.5/Qwen3.6 linears.
//!
//! This intentionally round-trips weights through a host codec and writes the dequantized values back
//! into ordinary Burn `Linear` modules. Forward execution stays on the model's normal path; the only
//! measured effect is PTQ weight reconstruction error.

use burn::{
    module::{Param, ParamId},
    nn::Linear,
    tensor::{DType, Tensor, TensorData},
};

use crate::{
    nvfp4::{
        Nvfp4HadamardConfig, Nvfp4HadamardSite, dequant_nvfp4, quantize_nvfp4, quantize_nvfp4_clip,
        quantize_nvfp4_mse, rotate_matrix_k, rotate_matrix_k_inverse,
    },
    qwen3_5::{
        Qwen3_5DecoderLayer, Qwen3_5FullAttnLayer, Qwen3_5FusedExperts, Qwen3_5GdnLayer,
        Qwen3_5MoeForCausalLM, Qwen3_5SharedMoeBlock,
    },
};
// ExpertFp8/ExpertQuantSidecar are only constructed in the cuda-gated real-fp8 expert quant pass.
#[cfg(feature = "cuda")]
use crate::qwen3_5::Qwen3_5DenseQuantBackend;
#[cfg(feature = "cuda")]
use crate::qwen3_5::{ExpertFp8, ExpertQuantSidecar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantPrecision {
    Bf16,
    Nvfp4Amax,
    Nvfp4Mse,
    Nvfp4Hadamard,
    Fp8,
}

#[derive(Clone, Copy, Debug)]
pub struct FakeQuantLinearStats {
    pub k: usize,
    pub n: usize,
    pub cosine: f32,
}

#[derive(Clone, Debug)]
pub struct FakeQuantExpertStats {
    pub role: &'static str,
    pub experts: usize,
    pub out: usize,
    pub input: usize,
    pub sample_cosine: f32,
}

#[derive(Clone, Debug)]
pub struct QuantCoverage {
    pub intended: usize,
    pub quantized: usize,
    pub skipped: Vec<String>,
    pub targets: Vec<String>,
}

pub const DEFAULT_DENSE_SKIP: &[&str] = &[
    "L*.moe.gate",
    "L*.moe.shared_gate",
    "mtp.layers.*.moe.gate",
    "mtp.layers.*.moe.shared_gate",
    "lm_head",
];

#[derive(Clone, Copy, Debug)]
struct HadamardContext {
    layer_idx: usize,
    site: Nvfp4HadamardSite,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine input lengths differ");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Round-trip one normal Burn `Linear` weight through the requested codec, then store the dequantized
/// weight back into the same `Linear`.
pub fn fake_quant_linear(lin: &mut Linear, prec: QuantPrecision) -> Option<FakeQuantLinearStats> {
    if prec == QuantPrecision::Nvfp4Hadamard {
        println!(
            "fake_quant_linear: WARNING Nvfp4Hadamard requires a layer/site context; keeping BF16"
        );
        return None;
    }
    fake_quant_linear_inner(lin, prec, None)
}

fn fake_quant_linear_inner(
    lin: &mut Linear,
    prec: QuantPrecision,
    hadamard: Option<HadamardContext>,
) -> Option<FakeQuantLinearStats> {
    if prec == QuantPrecision::Bf16 {
        return None;
    }

    let w = lin.weight.val();
    let [k, n] = w.dims();
    if is_nvfp4(prec) && k % 16 != 0 {
        println!(
            "fake_quant_linear: WARNING prec={prec:?} K={k} is not a multiple of 16; keeping BF16 weights"
        );
        return None;
    }

    let dtype = w.dtype();
    let device = w.device();
    let wv = w
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("fake_quant_linear: read Linear weight as f32");

    let deq = dequant_matrix(&wv, k, n, prec, hadamard);

    let cos = cosine(&wv, &deq);
    let t = Tensor::<2>::from_data(TensorData::new(deq, [k, n]), &device).cast(dtype);
    lin.weight = Param::initialized(ParamId::new(), t);
    println!("fake_quant_linear: {prec:?} K={k} N={n} roundtrip_cos={cos:.9}");
    Some(FakeQuantLinearStats { k, n, cosine: cos })
}

fn dequant_matrix(
    w: &[f32],
    k: usize,
    n: usize,
    prec: QuantPrecision,
    hadamard: Option<HadamardContext>,
) -> Vec<f32> {
    match prec {
        QuantPrecision::Bf16 => w.to_vec(),
        QuantPrecision::Nvfp4Amax => {
            let (qw, bs, g) = quantize_nvfp4(w, k, n);
            dequant_nvfp4(&qw, &bs, g, k, n)
        }
        QuantPrecision::Nvfp4Mse => {
            let (qw, bs, g) = quantize_nvfp4_mse(w, k, n);
            dequant_nvfp4(&qw, &bs, g, k, n)
        }
        QuantPrecision::Nvfp4Hadamard => {
            let ctx = hadamard.expect("Nvfp4Hadamard requires a layer/site context");
            let cfg = Nvfp4HadamardConfig::from_env();
            let seed = cfg.seed_for(ctx.layer_idx, ctx.site);
            let mut rotated = w.to_vec();
            rotate_matrix_k(&mut rotated, k, n, cfg.group_size, seed);
            // NVFP4_HADAMARD_MSE=1: use the SAME MSE scale search as the plain-NVFP4 baseline so
            // the rotation effect is measured like-for-like (R4 review, Gemini demand-probe).
            let (qw, bs, gscale) = if std::env::var("NVFP4_HADAMARD_MSE").as_deref() == Ok("1") {
                quantize_nvfp4_mse(&rotated, k, n)
            } else {
                quantize_nvfp4_clip(&rotated, k, n, cfg.clip_c)
            };
            let mut deq = dequant_nvfp4(&qw, &bs, gscale, k, n);
            rotate_matrix_k_inverse(&mut deq, k, n, cfg.group_size, seed);
            deq
        }
        QuantPrecision::Fp8 => {
            #[cfg(feature = "cuda")]
            {
                let (q, s) = crate::w8a16::quantize_e4m3_per_channel(w, k, n);
                crate::w8a16::dequant_e4m3(&q, &s, k, n)
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("fake_quant_experts: Fp8 requires the cuda feature");
            }
        }
    }
}

fn is_nvfp4(prec: QuantPrecision) -> bool {
    matches!(
        prec,
        QuantPrecision::Nvfp4Amax | QuantPrecision::Nvfp4Mse | QuantPrecision::Nvfp4Hadamard
    )
}

fn expert_hadamard_context(
    layer_idx: usize,
    role: &str,
    prec: QuantPrecision,
) -> Option<HadamardContext> {
    if prec != QuantPrecision::Nvfp4Hadamard {
        return None;
    }
    let site = match role {
        "gate_up_proj" => Nvfp4HadamardSite::MoeIn,
        "down_proj" => Nvfp4HadamardSite::MoeDownIn,
        _ => panic!("expert_hadamard_context: unknown expert role {role}"),
    };
    Some(HadamardContext { layer_idx, site })
}

fn dense_hadamard_context(role: &str, prec: QuantPrecision) -> Option<HadamardContext> {
    if prec != QuantPrecision::Nvfp4Hadamard {
        return None;
    }

    let rest = role.strip_prefix('L')?;
    let (idx, tail) = rest.split_once('.')?;
    let layer_idx = idx.parse::<usize>().ok()?;
    let site = match tail {
        "attn.q_proj" | "attn.k_proj" | "attn.v_proj" => Nvfp4HadamardSite::AttnIn,
        "attn.o_proj" => Nvfp4HadamardSite::AttnOutIn,
        "gdn.in_proj_qkv" | "gdn.in_proj_a" | "gdn.in_proj_b" | "gdn.in_proj_z" => {
            Nvfp4HadamardSite::GdnIn
        }
        "gdn.out_proj" => Nvfp4HadamardSite::GdnOutIn,
        "moe.gate" | "moe.shared_gate" | "moe.shared.gate_proj" | "moe.shared.up_proj" => {
            Nvfp4HadamardSite::MoeIn
        }
        "moe.shared.down_proj" => Nvfp4HadamardSite::MoeDownIn,
        _ => return None,
    };
    Some(HadamardContext { layer_idx, site })
}

fn fake_quant_expert_stack(
    t: &mut Param<Tensor<3>>,
    role: &'static str,
    prec: QuantPrecision,
    layer_idx: usize,
) -> Option<FakeQuantExpertStats> {
    if prec == QuantPrecision::Bf16 {
        return None;
    }

    let w = t.val();
    let [experts, out_dim, in_dim] = w.dims();
    if is_nvfp4(prec) && in_dim % 16 != 0 {
        println!(
            "fake_quant_experts: WARNING role={role} prec={prec:?} input_dim={in_dim} is not a multiple of 16; keeping BF16 weights"
        );
        return None;
    }
    let hadamard = expert_hadamard_context(layer_idx, role, prec);

    let dtype = w.dtype();
    let device = w.device();
    let src = w
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("fake_quant_expert_stack: read expert tensor as f32");
    let mut dst = vec![0.0f32; src.len()];

    let sample_cosine = if experts == 0 {
        1.0
    } else {
        let mut transposed = vec![0.0f32; in_dim * out_dim];
        for out_idx in 0..out_dim {
            for in_idx in 0..in_dim {
                transposed[in_idx * out_dim + out_idx] = src[out_idx * in_dim + in_idx];
            }
        }
        let deq = dequant_matrix(&transposed, in_dim, out_dim, prec, hadamard);
        cosine(&transposed, &deq)
    };

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(experts.max(1));
    if experts > 0 {
        let per = experts.div_ceil(n_threads);
        let src_ref = &src;
        std::thread::scope(|scope| {
            for (chunk_idx, dst_chunk) in dst.chunks_mut(per * out_dim * in_dim).enumerate() {
                let start = chunk_idx * per;
                let count = dst_chunk.len() / (out_dim * in_dim);
                scope.spawn(move || {
                    let mut transposed = vec![0.0f32; in_dim * out_dim];
                    for local in 0..count {
                        let expert = start + local;
                        let src_base = expert * out_dim * in_dim;
                        for out_idx in 0..out_dim {
                            for in_idx in 0..in_dim {
                                transposed[in_idx * out_dim + out_idx] =
                                    src_ref[src_base + out_idx * in_dim + in_idx];
                            }
                        }

                        let deq = dequant_matrix(&transposed, in_dim, out_dim, prec, hadamard);
                        let dst_base = local * out_dim * in_dim;
                        for in_idx in 0..in_dim {
                            for out_idx in 0..out_dim {
                                dst_chunk[dst_base + out_idx * in_dim + in_idx] =
                                    deq[in_idx * out_dim + out_idx];
                            }
                        }
                    }
                });
            }
        });
    }
    if is_nvfp4(prec) {
        println!(
            "fake_quant_expert_stack: role={role} prec={prec:?} experts={experts} threads={n_threads}"
        );
    }

    let quantized =
        Tensor::<3>::from_data(TensorData::new(dst, [experts, out_dim, in_dim]), &device)
            .cast(dtype);
    *t = Param::initialized(ParamId::new(), quantized);

    Some(FakeQuantExpertStats {
        role,
        experts,
        out: out_dim,
        input: in_dim,
        sample_cosine,
    })
}

fn fake_quant_experts_inner(
    experts: &mut Qwen3_5FusedExperts,
    prec: QuantPrecision,
    layer_idx: usize,
) -> Vec<FakeQuantExpertStats> {
    let mut stats = Vec::new();
    if let Some(stat) =
        fake_quant_expert_stack(&mut experts.gate_up_proj, "gate_up_proj", prec, layer_idx)
    {
        stats.push(stat);
    }
    if let Some(stat) =
        fake_quant_expert_stack(&mut experts.down_proj, "down_proj", prec, layer_idx)
    {
        stats.push(stat);
    }
    stats
}

fn summarize_expert_stats(label: &str, prec: QuantPrecision, stats: &[FakeQuantExpertStats]) {
    let stacks = stats.len();
    let expert_matrices = stats.iter().map(|stat| stat.experts).sum::<usize>();
    let sample = stats
        .first()
        .map(|stat| stat.sample_cosine)
        .unwrap_or(f32::NAN);
    println!(
        "{label}: prec={prec:?} stacks={stacks} expert_matrices={expert_matrices} sample_roundtrip_cos={sample:.9}"
    );
}

/// Parallel `&[T] -> Vec<f32>` conversion (dep-free std threads across cores). The bf16->f32 host
/// conversion is ~536M elems/layer over 40 layers in the fp8 expert quant — single-thread it dominates
/// the load-time quant. Each thread converts a disjoint chunk into its own slice; no locking.
#[cfg(feature = "cuda")]
fn par_to_f32<T: Sync>(src: &[T], f: impl Fn(&T) -> f32 + Sync) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    if src.is_empty() {
        return out;
    }
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(src.len());
    let per = src.len().div_ceil(n_threads);
    let fr = &f;
    std::thread::scope(|scope| {
        for (dst_chunk, src_chunk) in out.chunks_mut(per).zip(src.chunks(per)) {
            scope.spawn(move || {
                for (d, v) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                    *d = fr(v);
                }
            });
        }
    });
    out
}

#[cfg(feature = "cuda")]
fn tensor3_host_f32(t: Tensor<3>, label: &str) -> Vec<f32> {
    let dtype = t.dtype();
    let data = t.into_data();
    match dtype {
        DType::F32 => data
            .to_vec::<f32>()
            .unwrap_or_else(|e| panic!("{label}: read f32 tensor data: {e:?}")),
        DType::BF16 => par_to_f32(
            data.as_slice::<burn::tensor::bf16>()
                .unwrap_or_else(|e| panic!("{label}: read bf16 tensor data: {e:?}")),
            |v| v.to_f32(),
        ),
        DType::F16 => par_to_f32(
            data.as_slice::<burn::tensor::f16>()
                .unwrap_or_else(|e| panic!("{label}: read f16 tensor data: {e:?}")),
            |v| v.to_f32(),
        ),
        other => panic!("{label}: unsupported expert weight dtype {other:?}"),
    }
}

pub fn fake_quant_experts(experts: &mut Qwen3_5FusedExperts, prec: QuantPrecision) {
    let stats = fake_quant_experts_inner(experts, prec, 0);
    summarize_expert_stats("fake_quant_experts", prec, &stats);
}

pub fn fake_quant_all_experts(m: &mut Qwen3_5MoeForCausalLM, prec: QuantPrecision) {
    let mut all_stats = Vec::new();
    for (layer_idx, layer) in m.model.layers.iter_mut().enumerate() {
        let layer_stats = match layer {
            Qwen3_5DecoderLayer::Linear(layer) => {
                fake_quant_experts_inner(&mut layer.mlp.experts, prec, layer_idx)
            }
            Qwen3_5DecoderLayer::Full(layer) => {
                fake_quant_experts_inner(&mut layer.mlp.experts, prec, layer_idx)
            }
        };
        all_stats.extend(layer_stats);
    }
    summarize_expert_stats("fake_quant_all_experts", prec, &all_stats);
}

#[cfg(feature = "cuda")]
fn quantize_expert_stack_fp8(
    experts: &mut Qwen3_5FusedExperts,
    role: &'static str,
) -> (Vec<i8>, Vec<f32>, [usize; 3]) {
    let w = match role {
        "gate_up_proj" => experts.gate_up_proj.val(),
        "down_proj" => experts.down_proj.val(),
        _ => panic!("quantize_expert_stack_fp8: unknown role {role}"),
    };
    let [experts_n, out_dim, in_dim] = w.dims();
    let src = tensor3_host_f32(w, role);
    let mut q_all = vec![0i8; experts_n * in_dim * out_dim];
    let mut s_all = vec![0.0f32; experts_n * out_dim];

    // Parallelize the per-expert transpose+quantize across cores (dep-free std threads). The per-expert
    // work is independent; the cache-hostile [out,in]->[in,out] transpose over 20480 matrices is
    // single-thread ~70min otherwise. Each thread owns its `transposed` scratch, reads the shared
    // immutable `src`, and writes DISJOINT output chunks (chunked by expert), so no locking needed.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(experts_n.max(1));
    let per = experts_n.div_ceil(n_threads);
    let src_ref = &src;
    std::thread::scope(|scope| {
        for (chunk_idx, (q_chunk, s_chunk)) in q_all
            .chunks_mut(per * in_dim * out_dim)
            .zip(s_all.chunks_mut(per * out_dim))
            .enumerate()
        {
            let start = chunk_idx * per;
            let count = q_chunk.len() / (in_dim * out_dim);
            scope.spawn(move || {
                let mut transposed = vec![0.0f32; in_dim * out_dim];
                for local in 0..count {
                    let expert = start + local;
                    let src_base = expert * out_dim * in_dim;
                    for out_idx in 0..out_dim {
                        for in_idx in 0..in_dim {
                            transposed[in_idx * out_dim + out_idx] =
                                src_ref[src_base + out_idx * in_dim + in_idx];
                        }
                    }
                    let (q, scale) =
                        crate::w8a16::quantize_e4m3_per_channel(&transposed, in_dim, out_dim);
                    let qb = local * in_dim * out_dim;
                    for (dst, byte) in q_chunk[qb..qb + q.len()].iter_mut().zip(q.iter()) {
                        *dst = *byte as i8;
                    }
                    let sb = local * out_dim;
                    s_chunk[sb..sb + out_dim].copy_from_slice(&scale);
                }
            });
        }
    });

    (q_all, s_all, [experts_n, in_dim, out_dim])
}

/// Quantize routed Qwen3.5/Qwen3.6 experts into real additive FP8 sidecars.
///
/// This is an inference-only terminal load step. The BF16 expert `Param`s are intentionally kept
/// intact in this brick so dimensions and records continue to behave like the loaded model.
#[cfg(feature = "cuda")]
fn quantize_expert_block_fp8(experts: &mut Qwen3_5FusedExperts, role: &str) {
    let device = experts.gate_up_proj.val().device();
    let gu_dims = experts.gate_up_proj.val().dims();
    let dn_dims = experts.down_proj.val().dims();
    let [e, two_i, h] = gu_dims;
    let [e_dn, h_dn, i] = dn_dims;
    assert_eq!(
        e, e_dn,
        "quantize_experts_fp8: expert count mismatch in {role}"
    );
    assert_eq!(h, h_dn, "quantize_experts_fp8: hidden mismatch in {role}");
    assert_eq!(
        two_i,
        i * 2,
        "quantize_experts_fp8: gate_up out dim {two_i} != 2*down inner {} in {role}",
        i
    );

    let (q_gu_i8, s_gu, gu_layout) = quantize_expert_stack_fp8(experts, "gate_up_proj");
    let (q_dn_i8, s_dn, dn_layout) = quantize_expert_stack_fp8(experts, "down_proj");
    assert_eq!(
        gu_layout,
        [e, h, two_i],
        "quantize_experts_fp8: bad gate_up layout"
    );
    assert_eq!(
        dn_layout,
        [e, i, h],
        "quantize_experts_fp8: bad down layout"
    );
    assert!(
        s_gu.iter().all(|v| v.is_finite()) && s_dn.iter().all(|v| v.is_finite()),
        "quantize_experts_fp8: non-finite expert fp8 scale in {role}"
    );

    let q_gu = Tensor::<3, Int>::from_data(
        TensorData::new(q_gu_i8, ([e, h, two_i])),
        &device,
        DType::I8,
    );
    let s_gu = Tensor::<2>::from_data(TensorData::new(s_gu, [e, two_i]), &device);
    let q_dn =
        Tensor::<3, Int>::from_data(TensorData::new(q_dn_i8, ([e, i, h])), &device, DType::I8);
    let s_dn = Tensor::<2>::from_data(TensorData::new(s_dn, [e, h]), &device);

    experts.fp8 = ExpertQuantSidecar(Some(ExpertFp8 {
        q_gu,
        s_gu,
        q_dn,
        s_dn,
        e,
        h,
        i,
    }));
    assert!(
        experts.fp8.0.is_some(),
        "quantize_experts_fp8: {role} did not get an FP8 sidecar"
    );

    // OOM GUARD (3-voice memory warning + user flag): holding bf16(~71GB) + fp8(~30GB) additively is
    // ~101GB, which thrashes the CUDA allocator near the ~119GB ceiling (pathological slowdown) and
    // risks true OOM. Now that the fp8 sidecar holds this layer's weights, FREE the bf16 expert
    // Params (replace with tiny [1,1,1] placeholders). The fp8 branch of `expert_forward` reads dims
    // from the SIDECAR (never these placeholders), so this is safe. Peak stays ~72GB and DECREASES as
    // the bf16 model shrinks layer-by-layer. (Full memory win; also fixes the near-ceiling thrash.)
    let placeholder = Tensor::<3>::zeros([1, 1, 1], &device);
    experts.gate_up_proj = Param::initialized(ParamId::new(), placeholder.clone());
    experts.down_proj = Param::initialized(ParamId::new(), placeholder);
}

#[cfg(feature = "cuda")]
pub fn quantize_experts_fp8(m: &mut Qwen3_5MoeForCausalLM, skip_extra: &[&str]) -> QuantCoverage {
    let mut skipped = Vec::new();
    let mut targets = Vec::new();
    let mut quantized = 0usize;

    for (layer_idx, layer) in m.model.layers.iter_mut().enumerate() {
        let role = format!("L{layer_idx}.moe.experts");
        if should_skip(&role, skip_extra) {
            skipped.push(role);
            continue;
        }
        targets.push(role.clone());

        let experts = match layer {
            Qwen3_5DecoderLayer::Linear(layer) => &mut layer.mlp.experts,
            Qwen3_5DecoderLayer::Full(layer) => &mut layer.mlp.experts,
        };

        quantize_expert_block_fp8(experts, &role);
        quantized += 1;

        if quantized == 1 || quantized % 8 == 0 {
            println!(
                "  quantize_experts_fp8: {quantized} expert layers quantized+freed (through L{layer_idx})"
            );
        }
    }

    for (layer_idx, layer) in m.mtp.layers.iter_mut().enumerate() {
        let role = format!("mtp.layers.{layer_idx}.moe.experts");
        if should_skip(&role, skip_extra) {
            skipped.push(role);
            continue;
        }
        targets.push(role.clone());
        quantize_expert_block_fp8(&mut layer.mlp.experts, &role);
        quantized += 1;
        println!("  quantize_experts_fp8: {role} quantized+freed");
    }

    assert_eq!(
        quantized,
        targets.len(),
        "quantize_experts_fp8: not every targeted expert layer got an FP8 sidecar"
    );
    println!(
        "quantize_experts_fp8: quantized={} intended={} skipped={:?}",
        quantized,
        targets.len(),
        skipped
    );
    QuantCoverage {
        intended: targets.len(),
        quantized,
        skipped,
        targets,
    }
}

fn wildcard_match(pattern: &str, role: &str) -> bool {
    if let Some((prefix, suffix)) = pattern.split_once('*') {
        return role.starts_with(prefix) && role.ends_with(suffix);
    }
    role == pattern || role.contains(pattern)
}

fn should_skip(role: &str, skip: &[&str]) -> bool {
    skip.iter().any(|pattern| wildcard_match(pattern, role))
}

fn push_mlp_roles(out: &mut Vec<String>, prefix: &str) {
    out.push(format!("{prefix}.moe.gate"));
    out.push(format!("{prefix}.moe.shared_gate"));
    out.push(format!("{prefix}.moe.shared.gate_proj"));
    out.push(format!("{prefix}.moe.shared.up_proj"));
    out.push(format!("{prefix}.moe.shared.down_proj"));
}

/// Stable role names for every dense `Linear` touched by this fake-quant gate.
pub fn dense_linear_roles(m: &Qwen3_5MoeForCausalLM) -> Vec<String> {
    let mut out = Vec::new();
    for (i, layer) in m.model.layers.iter().enumerate() {
        match layer {
            Qwen3_5DecoderLayer::Linear(_) => {
                out.push(format!("L{i}.gdn.in_proj_qkv"));
                out.push(format!("L{i}.gdn.in_proj_a"));
                out.push(format!("L{i}.gdn.in_proj_b"));
                out.push(format!("L{i}.gdn.in_proj_z"));
                out.push(format!("L{i}.gdn.out_proj"));
            }
            Qwen3_5DecoderLayer::Full(_) => {
                out.push(format!("L{i}.attn.q_proj"));
                out.push(format!("L{i}.attn.k_proj"));
                out.push(format!("L{i}.attn.v_proj"));
                out.push(format!("L{i}.attn.o_proj"));
            }
        }
        push_mlp_roles(&mut out, &format!("L{i}"));
    }
    out.push("mtp.fc".to_string());
    for (i, _) in m.mtp.layers.iter().enumerate() {
        let prefix = format!("mtp.layers.{i}");
        out.push(format!("{prefix}.attn.q_proj"));
        out.push(format!("{prefix}.attn.k_proj"));
        out.push(format!("{prefix}.attn.v_proj"));
        out.push(format!("{prefix}.attn.o_proj"));
        push_mlp_roles(&mut out, &prefix);
    }
    out.push("lm_head".to_string());
    out
}

pub fn linear_by_role_mut<'a>(
    m: &'a mut Qwen3_5MoeForCausalLM,
    role: &str,
) -> Option<&'a mut Linear> {
    if role == "lm_head" {
        return Some(&mut m.lm_head);
    }
    if role == "mtp.fc" {
        return Some(&mut m.mtp.fc);
    }
    if let Some((idx, tail)) = parse_mtp_layer_role(role) {
        let layer = m.mtp.layers.get_mut(idx)?;
        return linear_full_by_tail_mut(layer, tail);
    }

    let rest = role.strip_prefix('L')?;
    let (idx, tail) = rest.split_once('.')?;
    let idx = idx.parse::<usize>().ok()?;
    let layer = m.model.layers.get_mut(idx)?;
    match layer {
        Qwen3_5DecoderLayer::Linear(layer) => linear_gdn_by_tail_mut(layer, tail),
        Qwen3_5DecoderLayer::Full(layer) => linear_full_by_tail_mut(layer, tail),
    }
}

fn parse_mtp_layer_role(role: &str) -> Option<(usize, &str)> {
    let rest = role.strip_prefix("mtp.layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    Some((idx.parse::<usize>().ok()?, tail))
}

fn linear_gdn_by_tail_mut<'a>(
    layer: &'a mut Qwen3_5GdnLayer,
    tail: &str,
) -> Option<&'a mut Linear> {
    match tail {
        "gdn.in_proj_qkv" => Some(&mut layer.linear_attn.in_proj_qkv),
        "gdn.in_proj_a" => Some(&mut layer.linear_attn.in_proj_a),
        "gdn.in_proj_b" => Some(&mut layer.linear_attn.in_proj_b),
        "gdn.in_proj_z" => Some(&mut layer.linear_attn.in_proj_z),
        "gdn.out_proj" => Some(&mut layer.linear_attn.out_proj),
        _ => linear_mlp_by_tail_mut(&mut layer.mlp, tail),
    }
}

fn linear_full_by_tail_mut<'a>(
    layer: &'a mut Qwen3_5FullAttnLayer,
    tail: &str,
) -> Option<&'a mut Linear> {
    match tail {
        "attn.q_proj" => Some(&mut layer.self_attn.q_proj),
        "attn.k_proj" => Some(&mut layer.self_attn.k_proj),
        "attn.v_proj" => Some(&mut layer.self_attn.v_proj),
        "attn.o_proj" => Some(&mut layer.self_attn.o_proj),
        _ => linear_mlp_by_tail_mut(&mut layer.mlp, tail),
    }
}

fn linear_mlp_by_tail_mut<'a>(
    mlp: &'a mut Qwen3_5SharedMoeBlock,
    tail: &str,
) -> Option<&'a mut Linear> {
    match tail {
        "moe.gate" => Some(&mut mlp.gate),
        "moe.shared_gate" => Some(&mut mlp.shared_expert_gate),
        "moe.shared.gate_proj" => Some(&mut mlp.shared_expert.gate_proj),
        "moe.shared.up_proj" => Some(&mut mlp.shared_expert.up_proj),
        "moe.shared.down_proj" => Some(&mut mlp.shared_expert.down_proj),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
pub fn sidecar_by_role_mut<'a>(
    m: &'a mut Qwen3_5MoeForCausalLM,
    role: &str,
) -> Option<&'a mut Option<crate::nvfp4_linear::QuantLinear>> {
    if role == "lm_head" {
        return Some(&mut m.lm_head_quant.0);
    }
    if role == "mtp.fc" {
        return Some(&mut m.mtp.fc_fp8.0);
    }
    if let Some((idx, tail)) = parse_mtp_layer_role(role) {
        let layer = m.mtp.layers.get_mut(idx)?;
        return sidecar_full_by_tail_mut(layer, tail);
    }

    let rest = role.strip_prefix('L')?;
    let (idx, tail) = rest.split_once('.')?;
    let idx = idx.parse::<usize>().ok()?;
    let layer = m.model.layers.get_mut(idx)?;
    match layer {
        Qwen3_5DecoderLayer::Linear(layer) => sidecar_gdn_by_tail_mut(layer, tail),
        Qwen3_5DecoderLayer::Full(layer) => sidecar_full_by_tail_mut(layer, tail),
    }
}

#[cfg(feature = "cuda")]
fn sidecar_gdn_by_tail_mut<'a>(
    layer: &'a mut Qwen3_5GdnLayer,
    tail: &str,
) -> Option<&'a mut Option<crate::nvfp4_linear::QuantLinear>> {
    match tail {
        "gdn.in_proj_qkv" => Some(&mut layer.linear_attn.in_proj_qkv_fp8.0),
        "gdn.in_proj_a" => Some(&mut layer.linear_attn.in_proj_a_fp8.0),
        "gdn.in_proj_b" => Some(&mut layer.linear_attn.in_proj_b_fp8.0),
        "gdn.in_proj_z" => Some(&mut layer.linear_attn.in_proj_z_fp8.0),
        "gdn.out_proj" => Some(&mut layer.linear_attn.out_proj_fp8.0),
        _ => sidecar_mlp_by_tail_mut(&mut layer.mlp, tail),
    }
}

#[cfg(feature = "cuda")]
fn sidecar_full_by_tail_mut<'a>(
    layer: &'a mut Qwen3_5FullAttnLayer,
    tail: &str,
) -> Option<&'a mut Option<crate::nvfp4_linear::QuantLinear>> {
    match tail {
        "attn.q_proj" => Some(&mut layer.self_attn.q_proj_fp8.0),
        "attn.k_proj" => Some(&mut layer.self_attn.k_proj_fp8.0),
        "attn.v_proj" => Some(&mut layer.self_attn.v_proj_fp8.0),
        "attn.o_proj" => Some(&mut layer.self_attn.o_proj_fp8.0),
        _ => sidecar_mlp_by_tail_mut(&mut layer.mlp, tail),
    }
}

#[cfg(feature = "cuda")]
fn sidecar_mlp_by_tail_mut<'a>(
    mlp: &'a mut Qwen3_5SharedMoeBlock,
    tail: &str,
) -> Option<&'a mut Option<crate::nvfp4_linear::QuantLinear>> {
    match tail {
        "moe.shared.gate_proj" => Some(&mut mlp.shared_expert.gate_proj_fp8.0),
        "moe.shared.up_proj" => Some(&mut mlp.shared_expert.up_proj_fp8.0),
        "moe.shared.down_proj" => Some(&mut mlp.shared_expert.down_proj_fp8.0),
        _ => None,
    }
}

/// Quantize dense Qwen3.5/Qwen3.6 linears into real sidecar FP8 kernels.
///
/// This must be the terminal load step: call after `load_weights_sharded` and after final device
/// placement, immediately before decode. The sidecars are intentionally not parameters/records and
/// are not moved by `to_device`/`fork`; reloading weights also invalidates them.
#[cfg(feature = "cuda")]
pub fn quantize_dense_fp8(m: &mut Qwen3_5MoeForCausalLM, skip_extra: &[&str]) -> QuantCoverage {
    let roles = dense_linear_roles(m);
    let mut skip = DEFAULT_DENSE_SKIP.to_vec();
    skip.extend_from_slice(skip_extra);
    let mut skipped = Vec::new();
    let mut targets = Vec::new();

    for role in roles {
        if should_skip(&role, &skip) {
            skipped.push(role);
        } else {
            targets.push(role);
        }
    }

    for role in &targets {
        let ql = {
            let lin = linear_by_role_mut(m, role)
                .unwrap_or_else(|| panic!("quantize_dense_fp8: missing Linear for role {role}"));
            crate::nvfp4_linear::QuantLinear::Fp8(crate::w8a16_linear::W8A16Linear::from_linear(
                lin,
            ))
        };
        let sidecar = sidecar_by_role_mut(m, role).unwrap_or_else(|| {
            panic!("quantize_dense_fp8: missing sidecar for target role {role}")
        });
        *sidecar = Some(ql);
    }

    let mut quantized = 0usize;
    for role in &targets {
        let sidecar = sidecar_by_role_mut(m, role).unwrap_or_else(|| {
            panic!("quantize_dense_fp8: missing sidecar while verifying {role}")
        });
        assert!(
            sidecar.is_some(),
            "quantize_dense_fp8: target role {role} did not get an FP8 sidecar"
        );
        quantized += 1;
    }

    println!(
        "quantize_dense_fp8: quantized={} intended={} skipped={:?}",
        quantized,
        targets.len(),
        skipped
    );
    QuantCoverage {
        intended: targets.len(),
        quantized,
        skipped,
        targets,
    }
}

/// Quantize exactly one dense `Linear` identified by its stable role name.
pub fn fake_quant_one(
    m: &mut Qwen3_5MoeForCausalLM,
    role: &str,
    prec: QuantPrecision,
) -> Option<f32> {
    let hadamard = dense_hadamard_context(role, prec);
    if prec == QuantPrecision::Nvfp4Hadamard && hadamard.is_none() {
        println!("fake_quant_one: WARNING role={role} has no Hadamard input site; keeping BF16");
        return None;
    }
    let lin = linear_by_role_mut(m, role)?;
    let stats = fake_quant_linear_inner(lin, prec, hadamard)?;
    println!(
        "fake_quant_one: role={role} K={} N={} roundtrip_cos={:.9}",
        stats.k, stats.n, stats.cosine
    );
    Some(stats.cosine)
}

/// Fake-quantize all dense linears except skipped roles. Rank-3 MoE expert tensors are not touched.
pub fn fake_quant_all_dense(m: &mut Qwen3_5MoeForCausalLM, prec: QuantPrecision, skip: &[&str]) {
    let roles = dense_linear_roles(m);
    let mut skipped = Vec::new();
    let mut cosines = Vec::new();

    for role in roles {
        if should_skip(&role, skip) {
            skipped.push(role);
            continue;
        }
        if let Some(cos) = fake_quant_one(m, &role, prec) {
            cosines.push(cos);
        }
    }

    let count = cosines.len();
    let mean = if count == 0 {
        f32::NAN
    } else {
        cosines.iter().sum::<f32>() / count as f32
    };
    let worst = cosines.iter().copied().fold(1.0f32, f32::min);
    println!(
        "fake_quant_all_dense: prec={prec:?} quantized={count} worst_cos={worst:.9} mean_cos={mean:.9} skipped={skipped:?}"
    );
}

#[cfg(test)]
mod role_tests {
    use super::*;
    use crate::qwen3_5::{Qwen3_5LayerType, Qwen3_5MoeConfig};
    use burn::prelude::Device;

    #[test]
    fn dense_linear_roles_enumerates_mtp_block_roles() {
        let device = Device::flex();
        let cfg = Qwen3_5MoeConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            mrope_section: [1, 1, 1],
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            moe_intermediate_size: 8,
            shared_expert_intermediate_size: 8,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            mtp_num_hidden_layers: 1,
        };
        let model = cfg.init_causal_lm(&device);

        let roles = dense_linear_roles(&model);
        assert_eq!(
            roles.len(),
            30,
            "2 main layers (19) + one MTP block (10) + lm_head"
        );
        for role in [
            "mtp.fc",
            "mtp.layers.0.attn.q_proj",
            "mtp.layers.0.attn.k_proj",
            "mtp.layers.0.attn.v_proj",
            "mtp.layers.0.attn.o_proj",
            "mtp.layers.0.moe.gate",
            "mtp.layers.0.moe.shared_gate",
            "mtp.layers.0.moe.shared.gate_proj",
            "mtp.layers.0.moe.shared.up_proj",
            "mtp.layers.0.moe.shared.down_proj",
        ] {
            assert!(roles.iter().any(|r| r == role), "missing dense role {role}");
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::qwen3_5::ExpertNvfp4Sidecar;
    use burn::{
        module::{Param, ParamId},
        nn::LinearConfig,
        prelude::Device,
        tensor::{Distribution, Tensor},
    };

    #[test]
    fn cuda_fake_quant_linear_nvfp4_mse_tracks_original_output() {
        let dev = Device::cuda(0);
        let (m, k, n) = (4usize, 128usize, 96usize);
        let mut lin = LinearConfig::new(k, n).with_bias(false).init::<Cuda>(&dev);
        let x = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

        let reference = lin
            .forward(x.clone())
            .into_data()
            .to_vec::<f32>()
            .expect("reference output");
        let stats = fake_quant_linear(&mut lin, QuantPrecision::Nvfp4Mse).expect("stats");
        let got = lin
            .forward(x)
            .into_data()
            .to_vec::<f32>()
            .expect("fake-quant output");
        let out_cos = cosine(&reference, &got);
        println!(
            "quant_gate cuda test: weight_roundtrip_cos={:.9} output_cos={out_cos:.9}",
            stats.cosine
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "fake-quant output contains NaN/Inf"
        );
        assert!(out_cos > 0.99, "output cosine {out_cos:.9} <= 0.99");
    }

    #[test]
    fn cuda_fake_quant_experts_fp8_tracks_original_weights() {
        let dev = Device::cuda(0);
        let (experts_n, hidden, inner) = (2usize, 4usize, 4usize);
        let mut experts = Qwen3_5FusedExperts {
            gate_up_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::random(
                    [experts_n, inner * 2, hidden],
                    Distribution::Normal(0.0, 1.0),
                    &dev,
                ),
            ),
            down_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::random(
                    [experts_n, hidden, inner],
                    Distribution::Normal(0.0, 1.0),
                    &dev,
                ),
            ),
            fp8: ExpertQuantSidecar(None),
            nvfp4: ExpertNvfp4Sidecar(None),
        };

        let original = experts
            .gate_up_proj
            .val()
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("original gate_up");
        fake_quant_experts(&mut experts, QuantPrecision::Fp8);

        let got = experts
            .gate_up_proj
            .val()
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("fake-quant gate_up");
        let down = experts
            .down_proj
            .val()
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("fake-quant down");

        let slice_len = inner * 2 * hidden;
        let cos = cosine(&original[..slice_len], &got[..slice_len]);
        println!("quant_gate cuda expert test: gate_up_e0_weight_cos={cos:.9}");
        assert!(
            got.iter().chain(down.iter()).all(|v| v.is_finite()),
            "fake-quant expert tensor contains NaN/Inf"
        );
        assert!(cos > 0.99, "expert weight cosine {cos:.9} <= 0.99");
    }
}
