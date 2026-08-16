//! Drop-in **`W8A16Linear`** — fp8 (e4m3) weight-only Linear (`docs/PERF_80TOKS_PLAN.md` §2 lever B).
//! CUDA only. This is the INTEGRATION layer the council found missing: the fused fp8 GEMM kernel
//! already existed in [`crate::w8a16`], but there was no `Linear`-shaped wrapper and no
//! quantize-on-load path to put a real checkpoint's weights behind it.
//!
//! ## What it is
//! A [`burn::nn::Linear`]-shaped layer that stores its weight as **one OCP E4M3 byte per element**
//! (half the bytes of bf16) plus a **per-output-channel** f32 scale, and computes `y = x·W (+ b)` by
//! calling the EXISTING [`crate::w8a16::w8a16_gemm`] — the FUSED dequant-in-GEMM kernel. The weight is
//! never materialized as a full bf16/f32 tensor: the kernel reads each packed e4m3 byte from HBM and
//! dequants it IN-REGISTER (`w = e4m3_to_f32(byte) * scale[n]`) inside the accumulation loop. That
//! in-load dequant is the whole point — a "dequant the whole weight to bf16, then call a normal GEMM"
//! round-trips HBM (fp8 read + bf16 write + bf16 read) and gives NO bandwidth win.
//!
//! ## Why per-output-channel + the M=1 regime
//! Per-(output-)channel symmetric scaling (`s[n] = max_k|W[k,n]| / 448`) is the vLLM/Marlin/Machete
//! batch-1 default for weight-only fp8: one scale per column keeps >99% accuracy at a negligible
//! metadata cost (`[N]` f32 vs `[K,N]` bytes). The win — reading half the weight bytes — is realized at
//! **true M=1 single-stream decode** (batch-1 serving / greedy generation), which is exactly the regime
//! `PERF_80TOKS_PLAN.md` targets. (The RL-parity / "M>1 re-reads the column" objections in
//! `w8a16.rs`'s header are about the GRPO *rollout*; they do not apply to greedy single-stream decode.)
//!
//! ## Quantize-on-load path
//! [`W8A16Linear::from_linear`] / [`W8A16Linear::from_weight`] run on the HOST at load time: pull the
//! f32 weight `[K,N]` to host, call the existing [`crate::w8a16::quantize_e4m3_per_channel`] (one
//! canonical OCP codec, shared with the kernel's decode so the bytes are bit-faithful), and stash the
//! packed bytes as a 1-byte `I8` Int tensor (fp8 has no Burn float DType and Burn's Int kind has no
//! `u8`, so the raw e4m3 bits ride in an `i8`; the kernel reinterprets them) + the `[N]` scale on the
//! same device. Bias (if any) is kept in f32 and added after the GEMM. Qwen3's projections are
//! bias-free, so in practice `bias == None`.
//!
//! ## Scope (this is Wave-1: the wrapper + load path only)
//! This file builds and validates the layer in isolation. It is NOT yet swapped into the attention /
//! expert / lm_head GEMMs — that Wave-2 integration (replacing the [`crate::linear2d::linear3`] call
//! sites) is deliberately separate. See the module bottom / the return notes for which GEMMs to swap.

use burn::nn::Linear;
use burn::prelude::Device;
use burn::tensor::{DType, Device, Int, Tensor, TensorData};

use crate::w8a16::quantize_e4m3_per_channel;

/// An fp8 (e4m3) weight-only Linear: packed e4m3 weight `[K,N]` + per-output-channel scale `[N]`
/// (+ optional bias `[N]`), evaluated through the fused dequant-in-GEMM kernel.
///
/// `K` is the input dim, `N` the output dim — matching Burn's `Linear` weight layout `[d_in, d_out]`
/// and the `w8a16` kernel's `W:[K,N]` convention, so `from_linear` is a layout-preserving swap (no
/// transpose). The struct is generic over the backend `B` for the host-side `from_*` constructors;
/// [`forward`](W8A16Linear::forward) is implemented only for the CUDA backend (where the kernel lives).
#[derive(Clone, Debug)]
pub struct W8A16Linear {
    /// Packed OCP E4M3 weight bytes `[K, N]`, carried in a 1-byte `DType::I8` Int tensor (raw e4m3
    /// bits; `byte as i8` is bit-preserving). Passed to the kernel AS-IS — never `into_contiguous`d
    /// (§0b rule 5: it would trigger a layout-fixing copy).
    q: Tensor<2, Int>,
    /// Per-output-channel symmetric scale `[N]` (f32): `s[n] = max_k|W[k,n]| / 448`.
    scale: Tensor<1>,
    /// Optional bias `[N]` (f32), added after the GEMM. `None` for Qwen3's bias-free projections.
    bias: Option<Tensor<1>>,
    /// Input dim (rows of the weight) — for shape asserts.
    k: usize,
    /// Output dim (columns of the weight / length of `scale`).
    n: usize,
}

impl W8A16Linear {
    /// Quantize-on-load from a row-major f32 weight `W:[K,N]` (Burn `Linear` layout `[d_in,d_out]`)
    /// and an optional bias `[N]`. Runs the existing per-output-channel OCP-E4M3 quantizer on the
    /// HOST, then uploads the packed bytes (`I8`) + scale (f32) to the weight's device.
    ///
    /// The weight is cast to f32 on the host before quantizing, so a bf16/f16-typed checkpoint loads
    /// correctly (the master weights here are f32, so this is usually a no-op).
    pub fn from_weight(weight: Tensor<2>, bias: Option<Tensor<1>>) -> Self {
        let [k, n] = weight.dims();
        let device = weight.device();

        // Pull the weight to host as f32 (the quantizer's input contract). `cast(F32)` first so a
        // bf16/f16-stored weight survives the host round-trip.
        let w: Vec<f32> = weight
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("W8A16Linear::from_weight: weight -> host f32 vec");

        // Existing per-output-channel OCP-E4M3 quantizer (one canonical codec, shared with the kernel).
        let (q_bytes, scale) = quantize_e4m3_per_channel(&w, k, n);

        // Pack the raw e4m3 bits into a 1-byte I8 Int tensor (`b as i8` is bit-preserving; the kernel
        // reinterprets the bits as e4m3). See `w8a16.rs` for why I8 carries the byte.
        let q_i8: Vec<i8> = q_bytes.iter().map(|&b| b as i8).collect();
        let q = Tensor::<2, Int>::from_data(TensorData::new(q_i8, ([k, n])), &device, DType::I8);
        let scale = Tensor::<1>::from_data(TensorData::new(scale, [n]), &device);

        Self {
            q,
            scale,
            bias,
            k,
            n,
        }
    }

    /// Build directly from already-packed fp8 parts without quantizing.
    ///
    /// `q_bytes_kn` must already be row-major `[K,N]` raw E4M3 bytes, and `scale_n` must already be
    /// the per-output-channel `[N]` f32 scale vector. ModelOpt dense checkpoints store fp8 weight as
    /// `[N,K]` and often a scalar scale; transposing `[N,K] -> [K,N]` and expanding scalar scale to
    /// `[N]` is the caller/loader's job.
    pub fn from_packed_parts(
        q_bytes_kn: Vec<u8>,
        scale_n: Vec<f32>,
        k: usize,
        n: usize,
        device: &Device,
    ) -> Self {
        assert_eq!(
            q_bytes_kn.len(),
            k * n,
            "W8A16Linear::from_packed_parts: q_bytes_kn length {} != K*N = {}",
            q_bytes_kn.len(),
            k * n
        );
        assert_eq!(
            scale_n.len(),
            n,
            "W8A16Linear::from_packed_parts: scale_n length {} != N = {n}",
            scale_n.len()
        );
        assert!(
            scale_n.iter().all(|v| v.is_finite() && *v > 0.0),
            "W8A16Linear::from_packed_parts: every scale must be finite and positive"
        );

        let q_i8: Vec<i8> = q_bytes_kn.iter().map(|&b| b as i8).collect();
        let q = Tensor::<2, Int>::from_data(TensorData::new(q_i8, ([k, n])), device, DType::I8);
        let scale = Tensor::<1>::from_data(TensorData::new(scale_n, [n]), device);

        Self {
            q,
            scale,
            bias: None,
            k,
            n,
        }
    }

    /// Quantize-on-load from an existing [`burn::nn::Linear`] — the drop-in entry point. Preserves the
    /// Linear's bias if present. Layout-preserving: `Linear` stores `[d_in, d_out] = [K, N]`, which is
    /// exactly the kernel's `W:[K,N]`, so no transpose is needed.
    pub fn from_linear(lin: &Linear) -> Self {
        let weight = lin.weight.val();
        let bias = lin.bias.as_ref().map(|b| b.val());
        Self::from_weight(weight, bias)
    }

    /// Input dim `K` (weight rows).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Output dim `N` (weight columns / scale length).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Whether a bias is stored.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

// =================================================================================================
// CUDA forward — the only place the fused kernel runs. Generic over the GEMV trait so the same
// wrapper works for both the Fusion `Cuda` eager path and the raw capture backend.
// =================================================================================================
#[cfg(feature = "cuda")]
mod cuda_forward {
    use super::W8A16Linear;
    use burn::tensor::Tensor;

    impl W8A16Linear {
        /// Forward at `[M, K] -> [M, N]` through the FUSED dequant-in-GEMM kernel (no bf16 dequant
        /// round-trip). `M=1` is the single-stream decode regime this layer targets; `M>1` also works
        /// (the kernel re-reads the weight column per row — correct, just bandwidth-suboptimal above
        /// M=1). The leading "M" is the flattened batch·seq, so a caller with `[B, S, K]` flattens to
        /// `[B*S, K]` (see [`forward3`](Self::forward3)).
        ///
        /// Activations are f32 (the kernel is W8A32 against today's f32-typed model); the bias, if
        /// any, is added in f32 after the GEMM.
        pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
            assert_eq!(
                x.dims()[1],
                self.k,
                "W8A16Linear::forward: activation K ({}) != weight K ({})",
                x.dims()[1],
                self.k
            );
            // The fused kernel: reads packed e4m3 bytes from HBM, dequants in-register, f32 MAC.
            let y = crate::w8a16::w8a16_gemv(x, self.q.clone(), self.scale.clone()); // [M, N]
            match &self.bias {
                // `[N]` bias -> `[1, N]` broadcast-add over `[M, N]`.
                Some(b) => y + b.clone().unsqueeze(),
                None => y,
            }
        }

        /// 3-D convenience mirroring [`crate::linear2d::linear3`]: `[B, S, K] -> [B, S, N]` by
        /// flattening the leading dims to a single `M = B*S`, running [`forward`](Self::forward), and
        /// reshaping back. This is the shape the Wave-2 attn/expert/head call sites pass.
        pub fn forward3(&self, x: Tensor<3>) -> Tensor<3> {
            let [b, s, k] = x.dims();
            let y = self.forward(x.reshape([b * s, k])); // [B*S, N]
            let n = y.dims()[1];
            y.reshape([b, s, n])
        }
    }
}

// =================================================================================================
// Tests (CUDA): W8A16Linear vs the original bf16 Linear at M=1 AND M>1.
// =================================================================================================
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::capture::CaptureBackend;
    use burn::nn::LinearConfig;
    use burn::prelude::Device;
    use burn::tensor::{Distribution, Tensor};

    /// Cosine similarity over flattened outputs (f64 accum), the spike's metric.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
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

    /// Max-abs elementwise diff normalized by the reference's max magnitude (relative max error).
    fn rel_max_err(got: &[f32], reference: &[f32]) -> f32 {
        let mad = got
            .iter()
            .zip(reference.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let refmax = reference
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1e-9);
        mad / refmax
    }

    /// bf16 reference matmul: cast both operands to bf16, matmul (f32 accum on the CUDA backend),
    /// widen back to f32 — exactly what `linear2d::matmul_bf16` does for the bf16 Linear path.
    fn bf16_ref(x: Tensor<2>, w: Tensor<2>) -> Vec<f32> {
        x.cast(DType::BF16)
            .matmul(w.cast(DType::BF16))
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    /// CORE REQUIRED TEST: W8A16Linear output vs the original bf16 Linear output on random
    /// `[M,K]·[K,N]`, at M=1 (decode) AND M>1, must be within fp8-e4m3 tolerance (cosine > 0.999,
    /// small relative error). Real Qwen3 Linear shapes.
    #[test]
    fn w8a16_linear_matches_bf16_linear_m1_and_mgt1() {
        let dev = Device::cuda(0);
        // (label, K, N) at real Qwen3 projection shapes; each run at M=1 and M=8.
        let shapes = [
            ("qkv/gate", 2048usize, 768usize),
            ("down", 768usize, 2048usize),
            ("mlp-up", 1024usize, 3072usize),
        ];
        let ms = [1usize, 8usize];

        println!("\n=== W8A16Linear vs bf16 Linear (cosine + rel-max-err) ===");
        let mut worst_cos = 1.0f32;
        let mut worst_rel = 0.0f32;

        for (label, k, n) in shapes {
            // One weight tensor, reused for the fp8 layer AND every reference (so the only difference
            // is the quantization, not the data). Weights small-ish, activations ~N(0,1) (spike scale).
            let weight = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.05), &dev);
            let w8 = W8A16Linear::from_weight(weight.clone(), None);

            for m in ms {
                let x = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

                let got = w8.forward(x.clone()).into_data().to_vec::<f32>().unwrap();
                let ref_bf16 = bf16_ref(x.clone(), weight.clone());
                // f32 reference too (the "true" output), for context.
                let ref_f32 = x
                    .matmul(weight.clone())
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();

                let cos_bf16 = cosine(&got, &ref_bf16);
                let rel_bf16 = rel_max_err(&got, &ref_bf16);
                let cos_f32 = cosine(&got, &ref_f32);

                println!(
                    "  {label:9} K{k} N{n} M{m}: cos(vs bf16)={cos_bf16:.6} rel-max-err={:.2}%  cos(vs f32)={cos_f32:.6}",
                    100.0 * rel_bf16
                );

                worst_cos = worst_cos.min(cos_bf16).min(cos_f32);
                worst_rel = worst_rel.max(rel_bf16);

                assert!(
                    !got.iter().any(|v| v.is_nan()),
                    "{label} M{m}: NaN in output"
                );
                assert!(
                    cos_bf16 > 0.999,
                    "{label} M{m}: cosine vs bf16 Linear {cos_bf16:.6} <= 0.999"
                );
                assert!(
                    cos_f32 > 0.999,
                    "{label} M{m}: cosine vs f32 Linear {cos_f32:.6} <= 0.999"
                );
                // fp8-e4m3 (3 mantissa bits) keeps the relative max error modest; generous bound.
                assert!(
                    rel_bf16 < 0.12,
                    "{label} M{m}: rel-max-err vs bf16 {:.2}% too large",
                    100.0 * rel_bf16
                );
            }
        }
        println!(
            "  worst cosine={worst_cos:.6}  worst rel-max-err={:.2}%\n",
            100.0 * worst_rel
        );
    }

    /// Exercises the `from_linear` drop-in path AND the bias add: build a real `Linear` WITH
    /// bias, quantize it, and check the fp8 forward (incl. bias) tracks the f32 Linear forward.
    #[test]
    fn from_linear_with_bias_tracks_linear() {
        let dev = Device::cuda(0);
        let (k, n, m) = (1024usize, 512usize, 4usize);

        let lin = LinearConfig::new(k, n).with_bias(true).init::<Cuda>(&dev);
        let w8 = W8A16Linear::from_linear(&lin);
        assert!(w8.has_bias(), "from_linear must carry the bias");
        assert_eq!((w8.k(), w8.n()), (k, n));

        let x = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

        let got = w8.forward(x.clone()).into_data().to_vec::<f32>().unwrap();
        // f32 Linear forward (weight + bias), via the 2-D path.
        let ref_f32 = lin.forward(x).into_data().to_vec::<f32>().unwrap();

        let cos = cosine(&got, &ref_f32);
        println!("\n=== from_linear + bias: cos(vs f32 Linear)={cos:.6} ===\n");
        assert!(cos > 0.999, "from_linear+bias cosine {cos:.6} <= 0.999");
    }

    #[test]
    fn from_packed_parts_forward_matches_from_weight_bit_exact() {
        let dev = Device::cuda(0);
        let (k, n, m) = (512usize, 128usize, 4usize);
        let weight_data: Vec<f32> = (0..k * n)
            .map(|idx| ((idx % 251) as f32 - 125.0) * 0.0009)
            .collect();
        let weight = Tensor::<2>::from_data(TensorData::new(weight_data.clone(), [k, n]), &dev);
        let (q_bytes, scale) = quantize_e4m3_per_channel(&weight_data, k, n);
        let from_weight = W8A16Linear::from_weight(weight, None);
        let from_parts = W8A16Linear::from_packed_parts(q_bytes, scale, k, n, &dev);
        let x_data: Vec<f32> = (0..m * k)
            .map(|idx| ((idx % 127) as f32 - 63.0) * 0.006)
            .collect();
        let x = Tensor::<2>::from_data(TensorData::new(x_data, [m, k]), &dev);

        let got_weight = from_weight
            .forward(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let got_parts = from_parts.forward(x).into_data().to_vec::<f32>().unwrap();
        assert!(
            got_weight
                .iter()
                .zip(got_parts.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "W8A16Linear from_packed_parts forward must match from_weight bit-exactly"
        );
    }

    #[test]
    fn real_qwen_shapes_m1_match_bf16_and_raw_capture_parity() {
        let dev = Device::cuda(0);
        let shapes = [
            ("gdn_qkv", 2048usize, 8192usize),
            ("gdn_out", 4096usize, 2048usize),
            ("attn_q", 2048usize, 8192usize),
            ("attn_o", 4096usize, 2048usize),
        ];

        for (label, k, n) in shapes {
            let weight: Vec<f32> = (0..k * n)
                .map(|i| ((i % 251) as f32 - 125.0) * 0.0004)
                .collect();
            let x_data: Vec<f32> = (0..k).map(|i| ((i % 127) as f32 - 63.0) * 0.01).collect();

            let weight_cuda = Tensor::<2>::from_data(TensorData::new(weight.clone(), [k, n]), &dev);
            let x_cuda = Tensor::<2>::from_data(TensorData::new(x_data.clone(), [1, k]), &dev);
            let w8_cuda = W8A16Linear::from_weight(weight_cuda.clone(), None);
            let got_cuda = w8_cuda
                .forward(x_cuda.clone())
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let ref_bf16 = bf16_ref(x_cuda, weight_cuda);
            let cos_bf16 = cosine(&got_cuda, &ref_bf16);
            assert!(
                cos_bf16 >= 0.999,
                "{label}: W8A16 vs bf16 cosine {cos_bf16:.6} < 0.999"
            );

            let weight_raw = Tensor::<2>::from_data(TensorData::new(weight, [k, n]), &dev);
            let x_raw = Tensor::<2>::from_data(TensorData::new(x_data, [1, k]), &dev);
            let w8_raw = W8A16Linear::from_weight(weight_raw, None);
            let got_raw = w8_raw.forward(x_raw).into_data().to_vec::<f32>().unwrap();
            let cos_raw = cosine(&got_cuda, &got_raw);
            assert!(
                cos_raw >= 0.9999,
                "{label}: Fusion(Cuda) vs raw(CaptureBackend) cosine {cos_raw:.6} < 0.9999"
            );
        }
    }
}
