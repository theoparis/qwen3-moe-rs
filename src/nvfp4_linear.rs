//! Drop-in **`Nvfp4Linear`** — NVFP4 (E2M1 weight + E4M3 block scale) weight-only Linear.
//!
//! This mirrors [`crate::w8a16_linear::W8A16Linear`], but stores the weight in the NVFP4 codec's
//! native column-major layout:
//! - `qw:[N,K/2]` raw E2M1x2 bytes, carried in an `I8` tensor.
//! - `bs:[N,K/16]` raw E4M3 block-scale bytes, carried in an `I8` tensor.
//! - `gscale:[1]` f32 global scale.
//!
//! Quantization is host-side at load time via [`crate::nvfp4::quantize_nvfp4`]. That codec already
//! transposes Burn's row-major Linear weight `[K,N]` into the kernel layout, so this wrapper uploads
//! the returned byte buffers directly.

use burn::nn::Linear;
use burn::prelude::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};

use crate::nvfp4::quantize_nvfp4;

/// An NVFP4 weight-only Linear: packed weight `[N,K/2]` + per-16-K block scales `[N,K/16]` + global
/// f32 scale `[1]` (+ optional bias `[N]`), evaluated through the fused NVFP4 GEMV backend.
///
/// `K` is the input dim and `N` is the output dim, matching Burn's `Linear` weight layout `[K,N]`.
/// The stored NVFP4 tensors are column-major because [`quantize_nvfp4`] performs that transpose.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Nvfp4Linear<B: Backend> {
    /// Packed E2M1x2 weight bytes `[N, K/2]`, carried as `DType::I8` raw bytes.
    qw: Tensor<B, 2, Int>,
    /// E4M3 block-scale bytes `[N, K/16]`, carried as `DType::I8` raw bytes.
    bs: Tensor<B, 2, Int>,
    /// Global f32 scale `[1]`.
    gscale: Tensor<B, 1>,
    /// Optional bias `[N]`, added after the GEMV.
    bias: Option<Tensor<B, 1>>,
    /// Input dim (columns of the activation / rows of the Burn weight).
    k: usize,
    /// Output dim (columns of the Burn weight).
    n: usize,
    /// Fixed decode batch bound used by the kernel's comptime register arrays.
    m_max: usize,
}

impl<B: Backend> Nvfp4Linear<B> {
    /// Quantize-on-load from a Burn Linear weight `W:[K,N]` and optional bias `[N]`.
    ///
    /// The NVFP4 codec returns the already-transposed kernel layout (`qw:[N,K/2]`, `bs:[N,K/16]`),
    /// so no additional transpose is performed here.
    pub fn from_weight(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        let [k, n] = weight.dims();
        let device = weight.device();

        let w: Vec<f32> = weight
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("Nvfp4Linear::from_weight: weight -> host f32 vec");

        let (qw_bytes, bs_bytes, gscale) = quantize_nvfp4(&w, k, n);

        let qw_i8: Vec<i8> = qw_bytes.iter().map(|&b| b as i8).collect();
        let bs_i8: Vec<i8> = bs_bytes.iter().map(|&b| b as i8).collect();
        let qw = Tensor::<B, 2, Int>::from_data_dtype(
            TensorData::new(qw_i8, [n, k / 2]),
            &device,
            DType::I8,
        );
        let bs = Tensor::<B, 2, Int>::from_data_dtype(
            TensorData::new(bs_i8, [n, k / 16]),
            &device,
            DType::I8,
        );
        let gscale = Tensor::<B, 1>::from_data(TensorData::new(vec![gscale], [1]), &device);

        Self {
            qw,
            bs,
            gscale,
            bias,
            k,
            n,
            m_max: 1,
        }
    }

    /// Build directly from already-packed NVFP4 parts without quantizing.
    ///
    /// `qw` must be the codec/kernel layout `[N,K/2]`, `block_scales` must be `[N,K/16]`, and
    /// `gscale` is the tensor-wide second-level scale. This constructor is for checkpoint loaders and
    /// synthetic tests that already own byte-faithful NVFP4 parts.
    pub fn from_packed_parts(
        qw: Vec<u8>,
        block_scales: Vec<u8>,
        gscale: f32,
        k: usize,
        n: usize,
        device: &B::Device,
    ) -> Self {
        assert_eq!(
            k % 16,
            0,
            "Nvfp4Linear::from_packed_parts requires K to be a multiple of 16, got {k}"
        );
        assert_eq!(
            qw.len(),
            n * (k / 2),
            "Nvfp4Linear::from_packed_parts: qw length {} != N*(K/2) = {}",
            qw.len(),
            n * (k / 2)
        );
        assert_eq!(
            block_scales.len(),
            n * (k / 16),
            "Nvfp4Linear::from_packed_parts: block_scales length {} != N*(K/16) = {}",
            block_scales.len(),
            n * (k / 16)
        );
        assert!(
            gscale.is_finite() && gscale > 0.0,
            "Nvfp4Linear::from_packed_parts: gscale must be finite and positive"
        );

        let qw_i8: Vec<i8> = qw.iter().map(|&b| b as i8).collect();
        let bs_i8: Vec<i8> = block_scales.iter().map(|&b| b as i8).collect();
        let qw = Tensor::<B, 2, Int>::from_data_dtype(
            TensorData::new(qw_i8, [n, k / 2]),
            device,
            DType::I8,
        );
        let bs = Tensor::<B, 2, Int>::from_data_dtype(
            TensorData::new(bs_i8, [n, k / 16]),
            device,
            DType::I8,
        );
        let gscale = Tensor::<B, 1>::from_data(TensorData::new(vec![gscale], [1]), device);

        Self {
            qw,
            bs,
            gscale,
            bias: None,
            k,
            n,
            m_max: 1,
        }
    }

    /// Quantize-on-load from an existing [`burn::nn::Linear`], preserving its bias if present.
    pub fn from_linear(lin: &Linear<B>) -> Self {
        let weight = lin.weight.val();
        let bias = lin.bias.as_ref().map(|b| b.val());
        Self::from_weight(weight, bias)
    }

    /// Set the fixed decode batch bound. The NVFP4 kernel currently supports `1..=8`.
    pub fn with_m_max(mut self, m_max: usize) -> Self {
        assert!(
            (1..=8).contains(&m_max),
            "Nvfp4Linear::with_m_max: m_max must be in 1..=8, got {m_max}"
        );
        self.m_max = m_max;
        self
    }

    /// Input dim `K`.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Output dim `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Whether a bias is stored.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;
    use burn::backend::NdArray;

    use crate::nvfp4::{dequant_nvfp4, quantize_nvfp4};

    type B = NdArray;

    fn synth(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let mut z = seed ^ (idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = ((z ^ (z >> 31)) >> 40) as f32 / 16_777_216.0;
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn int_bytes<const D: usize>(tensor: Tensor<B, D, Int>) -> Vec<u8> {
        tensor
            .into_data()
            .to_vec::<i8>()
            .expect("read i8 bytes")
            .into_iter()
            .map(|b| b as u8)
            .collect()
    }

    fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += x[mm * k + kk] * w[kk * n + nn];
                }
                y[mm * n + nn] = acc;
            }
        }
        y
    }

    #[test]
    fn from_packed_parts_matches_from_weight_host_reference() {
        let device = <B as Backend>::Device::default();
        let (m, k, n) = (3usize, 64usize, 32usize);
        let weight = synth(0x4E56_4650_344C, k * n, 0.03);
        let x = synth(0xAC71_0001, m * k, 0.2);
        let weight_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(weight.clone(), [k, n]), &device);

        let from_weight = Nvfp4Linear::<B>::from_weight(weight_tensor, None);
        let (qw, bs, gscale) = quantize_nvfp4(&weight, k, n);
        let from_parts =
            Nvfp4Linear::<B>::from_packed_parts(qw.clone(), bs.clone(), gscale, k, n, &device);

        let from_weight_qw = int_bytes(from_weight.qw.clone());
        let from_weight_bs = int_bytes(from_weight.bs.clone());
        let from_parts_qw = int_bytes(from_parts.qw.clone());
        let from_parts_bs = int_bytes(from_parts.bs.clone());
        assert_eq!(from_weight_qw, from_parts_qw);
        assert_eq!(from_weight_bs, from_parts_bs);
        let gw = from_weight
            .gscale
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("read from_weight gscale");
        let gp = from_parts
            .gscale
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("read from_parts gscale");
        assert_eq!(gw[0].to_bits(), gp[0].to_bits());

        let w_deq_weight = dequant_nvfp4(&from_weight_qw, &from_weight_bs, gw[0], k, n);
        let w_deq_parts = dequant_nvfp4(&from_parts_qw, &from_parts_bs, gp[0], k, n);
        let y_weight = matmul(&x, &w_deq_weight, m, k, n);
        let y_parts = matmul(&x, &w_deq_parts, m, k, n);
        assert!(
            y_weight
                .iter()
                .zip(y_parts.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

// =================================================================================================
// CUDA forward — backend-generic over the NVFP4 GEMV trait, so the same wrapper works for both the
// Fusion `Cuda` eager path and the raw capture `CubeBackend<CudaRuntime,f32,i32,u8>` path.
// =================================================================================================
#[cfg(feature = "cuda")]
mod cuda_forward {
    use super::Nvfp4Linear;
    use burn::tensor::Tensor;

    impl<B: crate::nvfp4::Nvfp4GemvBackend> Nvfp4Linear<B> {
        /// Forward at `[M,K] -> [M,N]` through the fused NVFP4 decode-GEMV kernel.
        ///
        /// The fused NVFP4 kernel is a *decode* GEMV: its comptime register arrays fix the batch
        /// bound at `self.m_max` (`1..=8`), so a single launch can only cover `M <= m_max` rows.
        /// Decode (`M == 1`) always takes the single-launch fast path with zero overhead.
        ///
        /// When a *prefill or diagnostic* caller presents `M > m_max` (e.g. the shared expert's
        /// weight-only `Nvfp4Linear`, loaded at the default `m_max = 1`, running at prefill `T = 5`
        /// through `ql3`), we chunk the input rows into `ceil(M / m_max)` slices of at most `m_max`
        /// rows each, launch the GEMV per chunk, and `cat` the outputs on the row dim. This restores
        /// correctness for those callers while guaranteeing the kernel's `m <= m_max` contract holds
        /// on every launch; it only affects these perf-insensitive paths (decode is untouched).
        pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            assert_eq!(
                x.dims()[1],
                self.k,
                "Nvfp4Linear::forward: activation K ({}) != weight K ({})",
                x.dims()[1],
                self.k
            );
            let m = x.dims()[0];
            let y = if m <= self.m_max {
                // Fast path: one launch covers all rows (decode M=1 hits this — zero overhead).
                B::nvfp4_gemv(
                    x,
                    self.qw.clone(),
                    self.bs.clone(),
                    self.gscale.clone(),
                    self.k,
                    self.n,
                    self.m_max,
                )
            } else {
                // Row-chunked path: split [M,K] into ceil(M/m_max) slices of <= m_max rows.
                let mut chunks = Vec::with_capacity(m.div_ceil(self.m_max));
                let mut i = 0;
                while i < m {
                    let rows = self.m_max.min(m - i);
                    let xi = x.clone().slice([i..i + rows, 0..self.k]);
                    chunks.push(B::nvfp4_gemv(
                        xi,
                        self.qw.clone(),
                        self.bs.clone(),
                        self.gscale.clone(),
                        self.k,
                        self.n,
                        self.m_max,
                    ));
                    i += rows;
                }
                Tensor::cat(chunks, 0)
            };
            match &self.bias {
                Some(b) => y + b.clone().unsqueeze(),
                None => y,
            }
        }

        /// 3-D convenience: `[B,S,K] -> [B,S,N]` by flattening to `[B*S,K]`, running
        /// [`forward`](Self::forward), and reshaping back.
        pub fn forward3(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
            let [b, s, k] = x.dims();
            let y = self.forward(x.reshape([b * s, k]));
            let n = y.dims()[1];
            y.reshape([b, s, n])
        }
    }
}

/// Quantized Linear dispatch enum for inference-time Linear call sites.
#[derive(Clone, Debug)]
pub enum QuantLinear<B: Backend> {
    Nvfp4(Nvfp4Linear<B>),
    #[cfg(feature = "cuda")]
    Fp8(crate::w8a16_linear::W8A16Linear<B>),
    Bf16(burn::nn::Linear<B>),
}

#[cfg(feature = "cuda")]
impl<B: crate::w8a16::W8A16GemvBackend + crate::nvfp4::Nvfp4GemvBackend> QuantLinear<B> {
    /// Dispatch `[B,S,K] -> [B,S,N]` on any backend with the dense quant GEMV traits.
    pub fn forward3(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Nvfp4(n) => n.forward3(x),
            Self::Fp8(f) => f.forward3(x),
            Self::Bf16(lin) => crate::linear2d::linear3(lin, x, crate::linear2d::Precision::Bf16),
        }
    }
}

// =================================================================================================
// Tests (CUDA): Nvfp4Linear vs bf16 Linear at M=1 AND M>1, plus QuantLinear dispatch.
// =================================================================================================
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};
    use burn::nn::LinearConfig;
    use burn::tensor::{Distribution, Tensor};

    /// Cosine similarity over flattened outputs (f64 accum).
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

    /// Max-abs elementwise diff normalized by the reference's max magnitude.
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

    /// bf16 reference matmul: cast both operands to bf16, matmul, widen back to f32.
    fn bf16_ref(x: Tensor<Cuda, 2>, w: Tensor<Cuda, 2>) -> Vec<f32> {
        x.cast(DType::BF16)
            .matmul(w.cast(DType::BF16))
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn nvfp4_linear_matches_bf16_linear_m1_and_mgt1() {
        let dev = CudaDevice::default();
        let shapes = [
            ("qkv/gate", 2048usize, 768usize),
            ("down", 768usize, 2048usize),
            ("mlp-up", 1024usize, 3072usize),
        ];
        let ms = [1usize, 8usize];

        println!("\n=== Nvfp4Linear vs bf16 Linear (cosine + rel-max-err) ===");
        let mut worst_cos = 1.0f32;
        let mut worst_rel = 0.0f32;

        for (label, k, n) in shapes {
            let weight = Tensor::<Cuda, 2>::random([k, n], Distribution::Normal(0.0, 0.05), &dev);
            let nvfp4 = Nvfp4Linear::from_weight(weight.clone(), None);

            for m in ms {
                let q4 = nvfp4.clone().with_m_max(m);
                let x = Tensor::<Cuda, 2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

                let got = q4.forward(x.clone()).into_data().to_vec::<f32>().unwrap();
                let ref_bf16 = bf16_ref(x, weight.clone());

                let cos_bf16 = cosine(&got, &ref_bf16);
                let rel_bf16 = rel_max_err(&got, &ref_bf16);

                println!(
                    "  {label:9} K{k} N{n} M{m}: cos(vs bf16)={cos_bf16:.6} rel-max-err={:.2}%",
                    100.0 * rel_bf16
                );

                worst_cos = worst_cos.min(cos_bf16);
                worst_rel = worst_rel.max(rel_bf16);

                assert!(
                    !got.iter().any(|v| v.is_nan()),
                    "{label} M{m}: NaN in output"
                );
                assert!(
                    cos_bf16 > 0.99,
                    "{label} M{m}: cosine vs bf16 Linear {cos_bf16:.6} <= 0.99"
                );
                assert!(
                    rel_bf16 < 0.20,
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

    #[test]
    fn from_linear_with_bias_tracks_linear() {
        let dev = CudaDevice::default();
        let (k, n, m) = (1024usize, 512usize, 4usize);

        let lin = LinearConfig::new(k, n).with_bias(true).init::<Cuda>(&dev);
        let q4 = Nvfp4Linear::from_linear(&lin).with_m_max(m);
        assert!(q4.has_bias(), "from_linear must carry the bias");
        assert_eq!((q4.k(), q4.n()), (k, n));

        let x = Tensor::<Cuda, 2>::random([m, k], Distribution::Normal(0.0, 1.0), &dev);

        let got = q4.forward(x.clone()).into_data().to_vec::<f32>().unwrap();
        let reference = lin.forward(x).into_data().to_vec::<f32>().unwrap();

        let cos = cosine(&got, &reference);
        println!("\n=== from_linear + bias: cos(vs Linear)={cos:.6} ===\n");
        assert!(
            !got.iter().any(|v| v.is_nan()),
            "from_linear+bias: NaN in output"
        );
        assert!(cos > 0.99, "from_linear+bias cosine {cos:.6} <= 0.99");
    }

    #[test]
    fn from_packed_parts_forward_matches_from_weight_bit_exact() {
        let dev = CudaDevice::default();
        let (k, n, m) = (512usize, 128usize, 4usize);
        let weight_data: Vec<f32> = (0..k * n)
            .map(|idx| ((idx % 257) as f32 - 128.0) * 0.0007)
            .collect();
        let weight =
            Tensor::<Cuda, 2>::from_data(TensorData::new(weight_data.clone(), [k, n]), &dev);
        let (qw, bs, gscale) = crate::nvfp4::quantize_nvfp4(&weight_data, k, n);
        let from_weight = Nvfp4Linear::from_weight(weight, None).with_m_max(m);
        let from_parts = Nvfp4Linear::from_packed_parts(qw, bs, gscale, k, n, &dev).with_m_max(m);
        let x_data: Vec<f32> = (0..m * k)
            .map(|idx| ((idx % 131) as f32 - 65.0) * 0.004)
            .collect();
        let x = Tensor::<Cuda, 2>::from_data(TensorData::new(x_data, [m, k]), &dev);

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
            "Nvfp4Linear from_packed_parts forward must match from_weight bit-exactly"
        );
    }

    /// Row-chunking equivalence: `forward` on `M > m_max` (which chunks internally) must be
    /// bit-exact with the same rows fed through the single-launch fast path (`M <= m_max`) in
    /// manual chunks and concatenated. Covers the shared-expert prefill bug (M=5 through m_max=1)
    /// and a multi-chunk-plus-remainder case (M=17 through m_max=8 -> 8+8+1).
    #[test]
    fn forward_row_chunking_matches_unchunked_fast_path() {
        let dev = CudaDevice::default();
        let (k, n) = (512usize, 128usize);
        let weight_data: Vec<f32> = (0..k * n)
            .map(|idx| ((idx % 257) as f32 - 128.0) * 0.0007)
            .collect();
        let weight = Tensor::<Cuda, 2>::from_data(TensorData::new(weight_data, [k, n]), &dev);

        // (M, m_max): M=5 through m_max=1 (5 single-row launches); M=17 through m_max=8 (8+8+1).
        for (m, m_max) in [(5usize, 1usize), (17usize, 8usize)] {
            let q4 = Nvfp4Linear::from_weight(weight.clone(), None).with_m_max(m_max);
            let x_data: Vec<f32> = (0..m * k)
                .map(|idx| ((idx % 131) as f32 - 65.0) * 0.004)
                .collect();
            let x = Tensor::<Cuda, 2>::from_data(TensorData::new(x_data, [m, k]), &dev);

            // Chunked: single call with M > m_max hits the internal row-chunking path.
            let got = q4.forward(x.clone()).into_data().to_vec::<f32>().unwrap();

            // Reference: manually slice into <= m_max-row pieces (each takes the single-launch
            // fast path) and cat on the row dim.
            let mut pieces = Vec::new();
            let mut i = 0;
            while i < m {
                let rows = m_max.min(m - i);
                pieces.push(q4.forward(x.clone().slice([i..i + rows, 0..k])));
                i += rows;
            }
            let reference = Tensor::cat(pieces, 0).into_data().to_vec::<f32>().unwrap();

            assert_eq!(got.len(), m * n, "M{m} m_max{m_max}: output shape");
            assert!(
                got.iter()
                    .zip(reference.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "M{m} m_max{m_max}: chunked forward must be bit-exact with unchunked fast path"
            );
        }
    }

    #[test]
    fn quant_linear_dispatch() {
        let dev = CudaDevice::default();
        let (b, s, k, n) = (2usize, 4usize, 1024usize, 768usize);

        let weight = Tensor::<Cuda, 2>::random([k, n], Distribution::Normal(0.0, 0.05), &dev);
        let q4 = Nvfp4Linear::from_weight(weight, None).with_m_max(b * s);
        let x = Tensor::<Cuda, 3>::random([b, s, k], Distribution::Normal(0.0, 1.0), &dev);

        let direct = q4
            .clone()
            .forward3(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let dispatched = QuantLinear::Nvfp4(q4)
            .forward3(x)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        let cos = cosine(&dispatched, &direct);
        assert!(
            !dispatched.iter().any(|v| v.is_nan()),
            "QuantLinear::Nvfp4: NaN in output"
        );
        assert!(
            cos > 0.999_999,
            "QuantLinear::Nvfp4 cosine {cos:.6} <= 0.999999"
        );
    }
}
