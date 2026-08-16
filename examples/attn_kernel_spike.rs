//! VLLM_KERNELS.md §1 — a custom CubeCL FlashAttention-style kernel (tiled online-softmax,
//! f32 accumulation), validated on the real GB10 GPU against an INDEPENDENT NdArray (CPU f32)
//! oracle — the cross-backend law of `docs/VLLM_KERNELS.md` §0.
//!
//! What this spike does, per shape:
//!   1. Builds Q/K/V (post-RoPE, GQA-UN-expanded K/V) from the SAME host f32 data on every backend.
//!   2. Runs a pure-Burn **NdArray (CPU f32)** reference attention — the trusted oracle.
//!   3. Runs the hand-written CubeCL `flash_attn` kernel on the GB10 (via the typed
//!      `CubeCustomOp` Fusion bridge) and reports cosine + max_abs_diff vs the oracle.
//!   4. ALSO runs Burn's EXISTING CUDA reference-SDPA (`attention_fallback`) vs the SAME oracle —
//!      exposing whether the fused CUDA SDPA is itself silently corrupted on these shapes.
//!
//! Algorithm (FA-2 online softmax, the canonical recurrence, f32 state). Per query row i, scanning
//! the causally-visible keys [0 .. q_global]:
//!   m_new = max(m, s_k);  alpha = exp(m - m_new);  p_k = exp(s_k - m_new);
//!   l = alpha*l + p_k;    acc = alpha*acc + p_k * V_k;   (the acc-rescale by alpha is load-bearing)
//! final once: O_i = acc / l.
//! Guards honored: (a) only visible keys are scanned, so the diagonal key is ALWAYS present and the
//! all-masked-tile `exp(-inf - -inf)=NaN` can never arise (a finite -1e30 sentinel for m, never -inf
//! arithmetic); (b) l<=0 -> O=0; (c) q_idx is GLOBAL (q_offset = kv_len - q_len, KV-cache aware);
//! (d) GQA kv_head = h / n_rep, K/V read un-expanded; (e) offsets via the tensor's native usize
//! strides (so a 64-bit-usize backend gets 64-bit offsets for free; the validated shapes fit u32).
//!
//! SCOPE: this is the CORRECTNESS-first pass. One cube per (q-head, batch, query-row); a single
//! thread per cube scans the KV with the online-softmax above. q_len=1 is the decode case; q_len=S
//! is prefill (one query row per cube). Split-K / Flash-Decode (KV partition + cross-split LSE merge)
//! and tensor-core MMA tiling are the PERF follow-ons — not on the critical path for correctness.
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example attn_kernel_spike 2>&1 | tail -40

use burn::prelude::Device;
use burn::prelude::Device;
use burn::prelude::Device;
use burn::tensor::{
    Bool, Device, Tensor, TensorData, activation::softmax, module::attention_fallback,
    ops::AttentionModuleOptions,
};

// The validated kernel + its host dispatch live in the library (CUDA-gated), so the model can reuse
// them; this example only exercises + validates them against the CPU oracle.
use qwen3_burn::flash_attn::flash_attention;

// -------------------------------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------------------------------

/// A tiny deterministic LCG so every backend sees byte-identical inputs (no per-backend RNG drift).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// Uniform f32 in [-amp, amp].
    fn next(&mut self, amp: f32) -> f32 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 33) as u32) as f32 / (u32::MAX as f32); // [0,1]
        (u * 2.0 - 1.0) * amp
    }
}

fn make_data(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next(amp)).collect()
}

/// Expand GQA K/V from Hkv heads to Hq heads by repeat-interleave (head h -> kv head h / n_rep).
fn expand_kv(t: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return t;
    }
    let [b, hkv, s, d] = t.dims();
    t.unsqueeze_dim::<5>(2) // [B, Hkv, 1, S, D]
        .repeat(&[1, 1, n_rep, 1, 1]) // [B, Hkv, n_rep, S, D]
        .reshape([b, hkv * n_rep, s, d]) // [B, Hq, S, D]
}

/// Causal mask [B, Hq, Sq, Sk], `true` = mask out (future). Key j masked iff j > q_offset + i,
/// q_offset = Sk - Sq (KV-cache global offset). Expanded to full shape (no reliance on broadcast).
fn causal_mask(b: usize, hq: usize, sq: usize, sk: usize, device: &Device) -> Tensor<4, Bool> {
    let q_offset = sk - sq;
    let rows: Vec<f32> = (0..sq).map(|i| (q_offset + i) as f32).collect();
    let cols: Vec<f32> = (0..sk).map(|j| j as f32).collect();
    let r = Tensor::<1>::from_floats(rows.as_slice(), device)
        .unsqueeze_dim::<2>(1)
        .repeat(&[1, sk]); // [Sq, Sk]
    let c = Tensor::<1>::from_floats(cols.as_slice(), device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[sq, 1]); // [Sq, Sk]
    // true where row < col  <=>  (q_offset+i) < j  <=>  j > q_offset+i  (future -> mask).
    r.lower(c)
        .unsqueeze_dims::<4>(&[0, 1]) // [1, 1, Sq, Sk]
        .repeat(&[b, hq, 1, 1]) // [B, Hq, Sq, Sk]
}

/// Pure-Burn reference attention (the trusted math; run on NdArray for the oracle). f32 throughout.
/// Q:[B,Hq,Sq,D], k_kv/v_kv:[B,Hkv,Sk,D].
fn reference_attention(
    q: Tensor<4>,
    k_kv: Tensor<4>,
    v_kv: Tensor<4>,
    scale: f32,
    n_rep: usize,
    device: &Device,
) -> Tensor<4> {
    let [b, hq, sq, _d] = q.dims();
    let sk = k_kv.dims()[2];
    let k = expand_kv(k_kv, n_rep); // [B,Hq,Sk,D]
    let v = expand_kv(v_kv, n_rep); // [B,Hq,Sk,D]

    let scores = q.matmul(k.swap_dims(2, 3)) * scale; // [B,Hq,Sq,Sk]
    let mask = causal_mask::<B>(b, hq, sq, sk, device);
    let scores = scores.mask_fill(mask, f32::NEG_INFINITY);
    let probs = softmax(scores, 3); // stable softmax (max-sub-exp-div)
    probs.matmul(v) // [B,Hq,Sq,D]
}

/// Burn's EXISTING fused-attention reference path (`attention_fallback`) on a given backend, with an
/// explicit GQA expansion + explicit causal mask (same scale = 1/sqrt(D)). This is the path the
/// model actually uses; running it on CUDA vs the NdArray oracle is the corruption check.
fn sdpa_fallback(
    q: Tensor<4>,
    k_kv: Tensor<4>,
    v_kv: Tensor<4>,
    n_rep: usize,
    device: &Device,
) -> Tensor<4> {
    let [b, hq, sq, _d] = q.dims();
    let sk = k_kv.dims()[2];
    let k = expand_kv(k_kv, n_rep);
    let v = expand_kv(v_kv, n_rep);
    let mask = causal_mask::<B>(b, hq, sq, sk, device);
    // options.scale = None -> defaults to 1/sqrt(head_dim), matching our kernel + oracle.
    attention_fallback(q, k, v, Some(mask), None, AttentionModuleOptions::default())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return f32::NAN;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn has_nan(a: &[f32]) -> bool {
    a.iter().any(|x| x.is_nan())
}

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    b: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
}

struct Row {
    label: String,
    flash_cos: f32,
    flash_mad: f32,
    sdpa_cos: f32,
    sdpa_mad: f32,
    flash_ok: bool,
}

fn run_shape(s: Shape, cuda_dev: &Device, nd_dev: &Device) -> Row {
    let n_rep = s.hq / s.hkv;
    let scale = 1.0f32 / (s.d as f32).sqrt();

    // Same host data on every backend (interleave-distinct seeds per tensor).
    let q_data = make_data(s.b * s.hq * s.sq * s.d, 0x1234 ^ (s.sk as u64), 2.0);
    let k_data = make_data(s.b * s.hkv * s.sk * s.d, 0x5678 ^ (s.sk as u64), 2.0);
    let v_data = make_data(s.b * s.hkv * s.sk * s.d, 0x9abc ^ (s.sk as u64), 2.0);

    let q_shape = [s.b, s.hq, s.sq, s.d];
    let kv_shape = [s.b, s.hkv, s.sk, s.d];

    // --- NdArray (CPU f32) oracle ---
    let q_nd = Tensor::<4>::from_data(TensorData::new(q_data.clone(), q_shape), nd_dev);
    let k_nd = Tensor::<4>::from_data(TensorData::new(k_data.clone(), kv_shape), nd_dev);
    let v_nd = Tensor::<4>::from_data(TensorData::new(v_data.clone(), kv_shape), nd_dev);
    let oracle = reference_attention(q_nd, k_nd, v_nd, scale, n_rep, nd_dev)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // --- Custom CUDA flash kernel ---
    let q_cu = Tensor::<4>::from_data(TensorData::new(q_data.clone(), q_shape), cuda_dev);
    let k_cu = Tensor::<4>::from_data(TensorData::new(k_data.clone(), kv_shape), cuda_dev);
    let v_cu = Tensor::<4>::from_data(TensorData::new(v_data.clone(), kv_shape), cuda_dev);
    let flash = flash_attention(q_cu, k_cu, v_cu, scale)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // --- Existing CUDA reference-SDPA (attention_fallback) — the corruption check ---
    let q_cu2 = Tensor::<4>::from_data(TensorData::new(q_data, q_shape), cuda_dev);
    let k_cu2 = Tensor::<4>::from_data(TensorData::new(k_data, kv_shape), cuda_dev);
    let v_cu2 = Tensor::<4>::from_data(TensorData::new(v_data, kv_shape), cuda_dev);
    let sdpa = sdpa_fallback(q_cu2, k_cu2, v_cu2, n_rep, cuda_dev)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let flash_cos = cosine(&flash, &oracle);
    let flash_mad = max_abs_diff(&flash, &oracle);
    let sdpa_cos = cosine(&sdpa, &oracle);
    let sdpa_mad = max_abs_diff(&sdpa, &oracle);
    let flash_ok = !has_nan(&flash) && flash_cos > 0.9999 && flash.len() == oracle.len();

    let label = format!(
        "{:7} b{} hq{} hkv{} Sq{} Sk{} d{}",
        s.label, s.b, s.hq, s.hkv, s.sq, s.sk, s.d
    );
    println!(
        "{label}\n    flash vs oracle: cos={flash_cos:.6} max_abs_diff={flash_mad:.3e}{}\n    \
         sdpa  vs oracle: cos={sdpa_cos:.6} max_abs_diff={sdpa_mad:.3e}",
        if flash_ok { "  [PASS]" } else { "  [FAIL]" },
    );

    Row {
        label,
        flash_cos,
        flash_mad,
        sdpa_cos,
        sdpa_mad,
        flash_ok,
    }
}

fn main() {
    let cuda_dev = Device::cuda(0);
    let nd_dev = Device::flex();
    println!("device: {cuda_dev:?} | oracle: NdArray (CPU f32) | kernel: custom CubeCL flash_attn");
    println!("cross-backend law: oracle is an INDEPENDENT CPU backend (docs/VLLM_KERNELS.md §0)\n");

    // Decode (Sq=1): the rollout-dominant case; one query attends the whole KV. GQA ratio 4, batch 2.
    // Prefill (Sq=Sk): one cube per query row. Short (64, 512) AND long (2048) context; head_dim 64+128.
    let shapes = [
        Shape {
            label: "decode",
            b: 2,
            hq: 8,
            hkv: 2,
            sq: 1,
            sk: 64,
            d: 128,
        },
        Shape {
            label: "decode",
            b: 2,
            hq: 8,
            hkv: 2,
            sq: 1,
            sk: 512,
            d: 128,
        },
        Shape {
            label: "decode",
            b: 2,
            hq: 8,
            hkv: 2,
            sq: 1,
            sk: 2048,
            d: 128,
        },
        Shape {
            label: "prefill",
            b: 2,
            hq: 8,
            hkv: 2,
            sq: 64,
            sk: 64,
            d: 128,
        },
        Shape {
            label: "prefill",
            b: 2,
            hq: 8,
            hkv: 2,
            sq: 512,
            sk: 512,
            d: 128,
        },
        Shape {
            label: "prefill",
            b: 1,
            hq: 8,
            hkv: 2,
            sq: 2048,
            sk: 2048,
            d: 128,
        },
        Shape {
            label: "prefill",
            b: 2,
            hq: 4,
            hkv: 2,
            sq: 128,
            sk: 128,
            d: 64,
        },
    ];

    let mut rows = Vec::new();
    for s in shapes {
        rows.push(run_shape(s, &cuda_dev, &nd_dev));
    }

    println!("\n================ SUMMARY (vs NdArray CPU oracle) ================");
    println!(
        "{:38}  {:>9} {:>11}   {:>9} {:>11}",
        "shape", "flash_cos", "flash_mad", "sdpa_cos", "sdpa_mad"
    );
    let mut all_flash_ok = true;
    let mut sdpa_agrees = true;
    for r in &rows {
        println!(
            "{:38}  {:>9.6} {:>11.3e}   {:>9.6} {:>11.3e}",
            r.label, r.flash_cos, r.flash_mad, r.sdpa_cos, r.sdpa_mad
        );
        all_flash_ok &= r.flash_ok;
        sdpa_agrees &= r.sdpa_cos > 0.9999;
    }

    println!("\n--- VERDICT ---");
    if all_flash_ok {
        println!(
            "FLASH KERNEL: VALIDATED — custom CubeCL online-softmax attention matches the NdArray \
             CPU oracle (cosine > 0.9999) on ALL shapes: decode (Sq=1) + prefill (Sq=S), short \
             (64/512) AND long (2048) context, GQA ratio 4, batch>1, head_dim 64 & 128."
        );
    } else {
        println!(
            "FLASH KERNEL: PARTIAL/FAIL — at least one shape did not reach cosine > 0.9999 vs the \
             oracle (see [FAIL] rows above)."
        );
    }
    if sdpa_agrees {
        println!(
            "CUDA REFERENCE-SDPA (attention_fallback): AGREES with the NdArray oracle on these \
             shapes (cosine > 0.9999) — no silent corruption detected on this path/these shapes."
        );
    } else {
        println!(
            "CUDA REFERENCE-SDPA (attention_fallback): DISAGREES with the NdArray oracle on some \
             shapes (cosine <= 0.9999) — the fused CUDA SDPA is silently corrupted there; the \
             custom kernel (validated against the CPU oracle) is the trustworthy path."
        );
    }

    assert!(
        all_flash_ok,
        "custom flash kernel failed validation vs the NdArray CPU oracle"
    );
}
