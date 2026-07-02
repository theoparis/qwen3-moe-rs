//! Host NVFP4 codec.
//!
//! The stored layout is transposed once at quantization time from Burn's row-major
//! weight `[K,N]` into column-major NVFP4:
//! - `packed_qw: [N, K/2]`, two E2M1 values per byte, low nibble = even K.
//! - `block_scales_e4m3: [N, K/16]`, one E4M3 block scale per 16 K-values.
//! - `gscale: [1]`, global f32 second-level scale.

#[cfg(feature = "cuda")]
use burn::backend::cuda::Cuda;
#[cfg(feature = "cuda")]
use burn::tensor::backend::Backend;
#[cfg(feature = "cuda")]
use burn::tensor::{DType, Int, Tensor, TensorPrimitive};
#[cfg(feature = "cuda")]
use burn_cubecl::kernel::into_contiguous;
#[cfg(feature = "cuda")]
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use cubecl::prelude::*;
#[cfg(feature = "cuda")]
use cubecl::{CubeCount, CubeDim, e4m3};

#[cfg(feature = "cuda")]
use crate::capture::CaptureBackend;
#[cfg(feature = "cuda")]
use crate::cube_custom_op::CubeCustomOp;

pub const E2M1_MAX: f32 = 6.0;
pub const E4M3_MAX: f32 = 448.0;
pub const E4M3_MIN_NORMAL: f32 = 0.015625;

const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[cfg(feature = "cuda")]
const NVFP4_COLS_PER_CTA: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4HadamardSite {
    AttnIn,
    AttnOutIn,
    GdnIn,
    GdnOutIn,
    MoeIn,
    MoeDownIn,
    MtpReserved,
}

impl Nvfp4HadamardSite {
    #[inline]
    fn id(self) -> u64 {
        match self {
            Self::AttnIn => 0,
            Self::AttnOutIn => 1,
            Self::GdnIn => 2,
            Self::GdnOutIn => 3,
            Self::MoeIn => 4,
            Self::MoeDownIn => 5,
            Self::MtpReserved => 6,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Nvfp4HadamardConfig {
    pub group_size: usize,
    pub clip_c: f32,
    pub base_seed: u64,
}

impl Nvfp4HadamardConfig {
    pub fn from_env() -> Self {
        let group_size = std::env::var("NVFP4_HADAMARD_G")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(128);
        let clip_c = std::env::var("NVFP4_CLIP_C")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(3.5);
        let base_seed = std::env::var("NVFP4_HADAMARD_SEED")
            .ok()
            .and_then(|value| parse_u64_seed(&value))
            .unwrap_or(0x4d42_2b2b_9f6c_5a31);
        Self {
            group_size,
            clip_c,
            base_seed,
        }
    }

    pub fn seed_for(self, layer_idx: usize, site: Nvfp4HadamardSite) -> u64 {
        nvfp4_hadamard_seed(self.base_seed, layer_idx, site)
    }
}

fn parse_u64_seed(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub fn nvfp4_hadamard_seed(base_seed: u64, layer_idx: usize, site: Nvfp4HadamardSite) -> u64 {
    let z = base_seed
        ^ (layer_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ site.id().wrapping_mul(0xd1b5_4a32_d192_ed03);
    splitmix64(z)
}

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn hadamard_signs(k: usize, g: usize, seed: u64) -> Vec<f32> {
    assert_eq!(
        g % 16,
        0,
        "NVFP4 Hadamard group size must be a multiple of 16, got {g}"
    );
    assert!(
        g.is_power_of_two(),
        "NVFP4 Hadamard group size must be a power of two, got {g}"
    );
    assert_eq!(
        k % g,
        0,
        "NVFP4 Hadamard requires K ({k}) to be divisible by group size {g}"
    );

    let mut signs = vec![1.0f32; k];
    for group in 0..(k / g) {
        let mut rng = Lcg(splitmix64(
            seed ^ (group as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        ));
        for offset in 0..g {
            signs[group * g + offset] = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        }
    }
    signs
}

#[cfg(feature = "cuda")]
pub mod gpu {
    use cubecl::e4m3;
    use cubecl::prelude::*;

    /// Decode one E2M1 (fp4) 4-bit code to f32, IN-KERNEL — avoids `Line::<f32>::cast_from(e2m1x2)`
    /// which mis-codegens on this cubecl rev (undefined __half_16; docs/L2C-gemv-cubecl-blocker.md).
    /// Layout: bit 3 = sign, bits 2..1 = exp (0..3), bit 0 = mantissa (0/1). Value set
    /// {0,±.5,±1,±1.5,±2,±3,±4,±6} — mirrors the host `e2m1_bits_to_f32`. Branch-only (no runtime
    /// shift / no runtime-indexed table), so it lowers to plain SIMT selects.
    #[cube]
    pub fn e2m1_decode(code: u32) -> f32 {
        let mag = code & 7u32;
        let m = f32::cast_from(mag & 1u32); // mantissa 0.0 / 1.0
        let e = (mag >> 1u32) & 3u32; // exponent field 0..3
        let val_mag = if e == 0u32 {
            f32::new(0.5) * m // subnormal: 0 or 0.5
        } else {
            // normal: 2^(e-1) * (1 + 0.5*m)
            let pe = if e == 1u32 {
                f32::new(1.0)
            } else if e == 2u32 {
                f32::new(2.0)
            } else {
                f32::new(4.0)
            };
            pe * (f32::new(1.0) + f32::new(0.5) * m)
        };
        if (code >> 3u32) & 1u32 == 1u32 {
            -val_mag
        } else {
            val_mag
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_decode_gemv(
        x: &Tensor<f32>,           // [M, K] activations f32
        qw: &Tensor<u8>, // [N, K/2] packed e2m1x2 bytes (2 fp4/byte: low=even K, high=odd K)
        bs: &Tensor<e4m3>, // [N, K/16] E4M3 block scales
        gscale: &Tensor<f32>, // [1] persistent FP32 global scale
        out: &mut Tensor<f32>, // [M, N]
        m_dim: u32,      // runtime batch, guards x/out only
        #[comptime] k: usize, // reduction dim
        #[comptime] blocks: usize, // K/16
        #[comptime] m_max: usize, // register-array bound
    ) {
        let col = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) as usize;
        let lane = UNIT_POS_X;
        let n = out.shape(1);

        if col < n {
            let g = gscale[0];
            let bytes_per_col = blocks * 8usize; // K/2
            let mut acc = Array::<f32>::new(m_max);
            let mut val = Array::<f32>::new(16usize); // this block's 16 dequantized fp4 (reused per block)

            #[unroll]
            for m in 0..m_max {
                acc[m] = f32::new(0.0);
            }

            let mut blk = lane as usize;
            while blk < blocks {
                // manual nibble unpack: 8 bytes -> 16 fp4 -> f32 (comptime-indexed register array)
                let byte_base = col * bytes_per_col + blk * 8usize;
                #[unroll]
                for j in 0..8usize {
                    let byte = u32::cast_from(qw[byte_base + j]);
                    val[j * 2usize] = e2m1_decode(byte & 15u32); // low nibble = even K
                    val[j * 2usize + 1usize] = e2m1_decode((byte >> 4u32) & 15u32); // high nibble = odd K
                }
                let s = f32::cast_from(bs[col * blocks + blk]);
                let k0 = blk * 16usize;

                #[unroll]
                for m in 0..m_max {
                    let mut part = f32::new(0.0);
                    #[unroll]
                    for p in 0..16usize {
                        if (m as u32) < m_dim {
                            part += val[p] * x[m * k + k0 + p];
                        }
                    }
                    acc[m] += s * part;
                }

                blk += 32;
            }

            #[unroll]
            for m in 0..m_max {
                let full = plane_sum(acc[m]);
                if lane == 0 && (m as u32) < m_dim {
                    out[m * n + col] = g * full;
                }
            }
        }
    }
}

/// Quantize a row-major Burn Linear weight `W:[K,N]` to NVFP4 host storage.
///
/// The output is column-major, with packed E2M1 weights in `[N,K/2]` and E4M3
/// block scales in `[N,K/16]`. Non-finite weights are rejected up front.
pub fn quantize_nvfp4(w: &[f32], k: usize, n: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert_eq!(
        w.len(),
        k * n,
        "weight length {} != K*N = {}",
        w.len(),
        k * n
    );
    assert_eq!(
        k % 16,
        0,
        "NVFP4 requires K to be a multiple of 16, got {k}"
    );
    assert!(
        w.iter().all(|x| x.is_finite()),
        "quantize_nvfp4: weight contains non-finite values (NaN/Inf)"
    );

    let mut amax = 0.0f32;
    for &value in w {
        amax = amax.max(value.abs());
    }

    let gscale = (amax / (E2M1_MAX * E4M3_MAX)).max(f32::MIN_POSITIVE);
    let blocks_per_col = k / 16;
    let packed_per_col = k / 2;
    let mut packed_qw = vec![0u8; n * packed_per_col];
    let mut block_scales_e4m3 = vec![0u8; n * blocks_per_col];

    for nn in 0..n {
        for block in 0..blocks_per_col {
            let k0 = block * 16;
            let mut bamax = 0.0f32;
            for offset in 0..16 {
                bamax = bamax.max(w[(k0 + offset) * n + nn].abs());
            }

            let scale_idx = nn * blocks_per_col + block;
            let packed_block = nn * packed_per_col + block * 8;
            if bamax == 0.0 {
                block_scales_e4m3[scale_idx] = f32_to_e4m3(E4M3_MIN_NORMAL);
                for pair in 0..8 {
                    packed_qw[packed_block + pair] = 0;
                }
                continue;
            }

            let sb_ideal = bamax / (E2M1_MAX * gscale);
            let sb_byte = f32_to_e4m3(sb_ideal.max(E4M3_MIN_NORMAL));
            block_scales_e4m3[scale_idx] = sb_byte;

            let block_scale = e4m3_to_f32(sb_byte) * gscale;
            debug_assert!(block_scale > 0.0);

            for pair in 0..8 {
                let k_even = k0 + pair * 2;
                let q0 = f32_to_e2m1_bits(w[k_even * n + nn] / block_scale);
                let q1 = f32_to_e2m1_bits(w[(k_even + 1) * n + nn] / block_scale);
                packed_qw[packed_block + pair] = q0 | (q1 << 4);
            }
        }
    }

    (packed_qw, block_scales_e4m3, gscale)
}

/// In-place orthonormal Walsh-Hadamard transform. The input length must be a power of two.
pub fn fwht_inplace(v: &mut [f32]) {
    let len = v.len();
    assert!(
        len.is_power_of_two(),
        "FWHT length must be a non-zero power of two, got {len}"
    );

    let mut step = 1usize;
    while step < len {
        let span = step * 2;
        for base in (0..len).step_by(span) {
            for offset in 0..step {
                let a = v[base + offset];
                let b = v[base + offset + step];
                v[base + offset] = a + b;
                v[base + offset + step] = a - b;
            }
        }
        step = span;
    }

    let norm = (len as f32).sqrt().recip();
    for value in v {
        *value *= norm;
    }
}

/// Apply the randomized rotation `R = H * D_s` to each contiguous K-group of every column.
pub fn rotate_matrix_k(w: &mut [f32], k: usize, n: usize, g: usize, seed: u64) {
    assert_eq!(
        w.len(),
        k * n,
        "matrix length {} != K*N = {}",
        w.len(),
        k * n
    );
    let signs = hadamard_signs(k, g, seed);
    let mut scratch = vec![0.0f32; g];

    for nn in 0..n {
        for k0 in (0..k).step_by(g) {
            for offset in 0..g {
                scratch[offset] = w[(k0 + offset) * n + nn] * signs[k0 + offset];
            }
            fwht_inplace(&mut scratch);
            for offset in 0..g {
                w[(k0 + offset) * n + nn] = scratch[offset];
            }
        }
    }
}

/// Apply the inverse randomized rotation `R^-1 = D_s * H` to each K-group of every column.
pub fn rotate_matrix_k_inverse(w: &mut [f32], k: usize, n: usize, g: usize, seed: u64) {
    assert_eq!(
        w.len(),
        k * n,
        "matrix length {} != K*N = {}",
        w.len(),
        k * n
    );
    let signs = hadamard_signs(k, g, seed);
    let mut scratch = vec![0.0f32; g];

    for nn in 0..n {
        for k0 in (0..k).step_by(g) {
            for offset in 0..g {
                scratch[offset] = w[(k0 + offset) * n + nn];
            }
            fwht_inplace(&mut scratch);
            for offset in 0..g {
                w[(k0 + offset) * n + nn] = scratch[offset] * signs[k0 + offset];
            }
        }
    }
}

/// Quantize a row-major Burn Linear weight `W:[K,N]` to NVFP4 host storage, clipping each per-16
/// block scale anchor to `min(block_amax, clip_c * block_rms)`.
///
/// `clip_c == 0.0` delegates to [`quantize_nvfp4`] and is bit-identical to the amax path.
pub fn quantize_nvfp4_clip(w: &[f32], k: usize, n: usize, clip_c: f32) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(
        clip_c.is_finite() && clip_c >= 0.0,
        "quantize_nvfp4_clip: clip_c must be finite and non-negative, got {clip_c}"
    );
    if clip_c == 0.0 {
        return quantize_nvfp4(w, k, n);
    }
    assert_eq!(
        w.len(),
        k * n,
        "weight length {} != K*N = {}",
        w.len(),
        k * n
    );
    assert_eq!(
        k % 16,
        0,
        "NVFP4 requires K to be a multiple of 16, got {k}"
    );
    assert!(
        w.iter().all(|x| x.is_finite()),
        "quantize_nvfp4_clip: weight contains non-finite values (NaN/Inf)"
    );

    let mut amax = 0.0f32;
    for &value in w {
        amax = amax.max(value.abs());
    }

    let gscale = (amax / (E2M1_MAX * E4M3_MAX)).max(f32::MIN_POSITIVE);
    let blocks_per_col = k / 16;
    let packed_per_col = k / 2;
    let mut packed_qw = vec![0u8; n * packed_per_col];
    let mut block_scales_e4m3 = vec![0u8; n * blocks_per_col];

    for nn in 0..n {
        for block in 0..blocks_per_col {
            let k0 = block * 16;
            let mut bamax = 0.0f32;
            let mut sum_sq = 0.0f32;
            for offset in 0..16 {
                let value = w[(k0 + offset) * n + nn];
                bamax = bamax.max(value.abs());
                sum_sq += value * value;
            }

            let scale_idx = nn * blocks_per_col + block;
            let packed_block = nn * packed_per_col + block * 8;
            if bamax == 0.0 {
                block_scales_e4m3[scale_idx] = f32_to_e4m3(E4M3_MIN_NORMAL);
                for pair in 0..8 {
                    packed_qw[packed_block + pair] = 0;
                }
                continue;
            }

            let block_rms = (sum_sq / 16.0).sqrt();
            let scale_anchor = bamax.min(clip_c * block_rms);
            let sb_ideal = scale_anchor / (E2M1_MAX * gscale);
            let sb_byte = f32_to_e4m3(sb_ideal.max(E4M3_MIN_NORMAL));
            block_scales_e4m3[scale_idx] = sb_byte;

            let block_scale = e4m3_to_f32(sb_byte) * gscale;
            debug_assert!(block_scale > 0.0);

            for pair in 0..8 {
                let k_even = k0 + pair * 2;
                let q0 = f32_to_e2m1_bits(w[k_even * n + nn] / block_scale);
                let q1 = f32_to_e2m1_bits(w[(k_even + 1) * n + nn] / block_scale);
                packed_qw[packed_block + pair] = q0 | (q1 << 4);
            }
        }
    }

    (packed_qw, block_scales_e4m3, gscale)
}

/// Quantize a row-major Burn Linear weight `W:[K,N]` to NVFP4 host storage, selecting each
/// per-16 block scale from a small clipping grid by minimum reconstruction MSE.
///
/// The output layout and global scale match [`quantize_nvfp4`]. Only the per-block E4M3 scale
/// selection differs.
pub fn quantize_nvfp4_mse(w: &[f32], k: usize, n: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert_eq!(
        w.len(),
        k * n,
        "weight length {} != K*N = {}",
        w.len(),
        k * n
    );
    assert_eq!(
        k % 16,
        0,
        "NVFP4 requires K to be a multiple of 16, got {k}"
    );
    assert!(
        w.iter().all(|x| x.is_finite()),
        "quantize_nvfp4_mse: weight contains non-finite values (NaN/Inf)"
    );

    let mut amax = 0.0f32;
    for &value in w {
        amax = amax.max(value.abs());
    }

    let gscale = (amax / (E2M1_MAX * E4M3_MAX)).max(f32::MIN_POSITIVE);
    let blocks_per_col = k / 16;
    let packed_per_col = k / 2;
    let mut packed_qw = vec![0u8; n * packed_per_col];
    let mut block_scales_e4m3 = vec![0u8; n * blocks_per_col];
    let candidates = [1.0f32, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60];

    for nn in 0..n {
        for block in 0..blocks_per_col {
            let k0 = block * 16;
            let mut bamax = 0.0f32;
            for offset in 0..16 {
                bamax = bamax.max(w[(k0 + offset) * n + nn].abs());
            }

            let scale_idx = nn * blocks_per_col + block;
            let packed_block = nn * packed_per_col + block * 8;
            if bamax == 0.0 {
                block_scales_e4m3[scale_idx] = f32_to_e4m3(E4M3_MIN_NORMAL);
                for pair in 0..8 {
                    packed_qw[packed_block + pair] = 0;
                }
                continue;
            }

            let mut best_sb_byte = f32_to_e4m3((bamax / (E2M1_MAX * gscale)).max(E4M3_MIN_NORMAL));
            let mut best_mse = f64::INFINITY;
            for &clip in &candidates {
                let sb_ideal = (bamax * clip) / (E2M1_MAX * gscale);
                let sb_byte = f32_to_e4m3(sb_ideal.max(E4M3_MIN_NORMAL));
                let block_scale = e4m3_to_f32(sb_byte) * gscale;
                if block_scale <= 0.0 {
                    continue;
                }

                let mut mse = 0.0f64;
                for offset in 0..16 {
                    let wv = w[(k0 + offset) * n + nn];
                    let code = e2m1_bits_to_f32(f32_to_e2m1_bits(wv / block_scale));
                    let recon = code * block_scale;
                    mse += ((recon - wv) as f64).powi(2);
                }

                if mse < best_mse {
                    best_sb_byte = sb_byte;
                    best_mse = mse;
                }
            }

            block_scales_e4m3[scale_idx] = best_sb_byte;

            let block_scale = e4m3_to_f32(best_sb_byte) * gscale;
            debug_assert!(block_scale > 0.0);

            for pair in 0..8 {
                let k_even = k0 + pair * 2;
                let q0 = f32_to_e2m1_bits(w[k_even * n + nn] / block_scale);
                let q1 = f32_to_e2m1_bits(w[(k_even + 1) * n + nn] / block_scale);
                packed_qw[packed_block + pair] = q0 | (q1 << 4);
            }
        }
    }

    (packed_qw, block_scales_e4m3, gscale)
}

/// Dequantize NVFP4 host storage back to row-major f32 `[K,N]`.
pub fn dequant_nvfp4(
    packed_qw: &[u8],
    block_scales_e4m3: &[u8],
    gscale: f32,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(
        k % 16,
        0,
        "NVFP4 requires K to be a multiple of 16, got {k}"
    );
    assert!(
        gscale.is_finite() && gscale > 0.0,
        "dequant_nvfp4: gscale must be finite and positive"
    );

    let blocks_per_col = k / 16;
    let packed_per_col = k / 2;
    assert_eq!(
        packed_qw.len(),
        n * packed_per_col,
        "packed_qw length {} != N*(K/2) = {}",
        packed_qw.len(),
        n * packed_per_col
    );
    assert_eq!(
        block_scales_e4m3.len(),
        n * blocks_per_col,
        "block_scales_e4m3 length {} != N*(K/16) = {}",
        block_scales_e4m3.len(),
        n * blocks_per_col
    );

    let mut w = vec![0.0f32; k * n];
    for nn in 0..n {
        for block in 0..blocks_per_col {
            let scale_idx = nn * blocks_per_col + block;
            let block_scale = e4m3_to_f32(block_scales_e4m3[scale_idx]) * gscale;
            let packed_block = nn * packed_per_col + block * 8;
            let k0 = block * 16;

            for pair in 0..8 {
                let packed = packed_qw[packed_block + pair];
                let k_even = k0 + pair * 2;
                w[k_even * n + nn] = e2m1_bits_to_f32(packed & 0x0f) * block_scale;
                w[(k_even + 1) * n + nn] = e2m1_bits_to_f32((packed >> 4) & 0x0f) * block_scale;
            }
        }
    }

    w
}

/// Repack the codec's K-major NVFP4 bytes `[N, K/2]` into output-major bytes `[K, N/2]`.
///
/// Source layout is the native output of [`quantize_nvfp4`]:
///
/// ```text
/// packed_kmajor[n, k/2] = { high: W[k+1, n], low: W[k, n] }   // low nibble = even K
/// ```
///
/// Target layout is row-major `[K, N/2]`. At a fixed reduction index `k`, byte `j` carries the two
/// adjacent output channels `2j` and `2j+1`, so a vector line of `V` consecutive bytes covers
/// `2V` consecutive output channels:
///
/// ```text
/// row k:
///   outmajor[k, 0] = { high: W[k, 1], low: W[k, 0] }
///   outmajor[k, 1] = { high: W[k, 3], low: W[k, 2] }
///   ...
///
/// memory:
///   k=0: [n0|n1] [n2|n3] [n4|n5] ...
///   k=1: [n0|n1] [n2|n3] [n4|n5] ...
/// ```
///
/// The ModelOpt checkpoint adapter will first convert its `[out, in/2]` tensors into our codec's
/// source convention, then call this for the C2-style output-vectorized kernels. The dequant math is
/// unchanged: `e2m1 * f32(e4m3_block_scale) * weight_scale_2`; our host codec's `gscale` already uses
/// ModelOpt's `amax / (6 * 448)` convention, so no adapter factor is applied. B5.2 can decode an
/// E2M1 nibble without a LUT by mapping it to an E4M3 bit pattern, but host code intentionally uses
/// the straightforward LUT decode.
pub fn repack_kmajor_to_outmajor(packed_kmajor: &[u8], k: usize, n: usize) -> Vec<u8> {
    assert_eq!(
        k % 16,
        0,
        "NVFP4 repack requires K to be a multiple of 16, got {k}"
    );
    assert_eq!(
        n % 2,
        0,
        "NVFP4 output-major repack requires even N, got {n}"
    );
    assert_eq!(
        packed_kmajor.len(),
        n * (k / 2),
        "packed_kmajor length {} != N*(K/2) = {}",
        packed_kmajor.len(),
        n * (k / 2)
    );

    let mut outmajor = vec![0u8; k * (n / 2)];
    let packed_per_col = k / 2;
    for kk in 0..k {
        for out_pair in 0..(n / 2) {
            let n0 = out_pair * 2;
            let n1 = n0 + 1;
            let src0 = packed_kmajor[n0 * packed_per_col + kk / 2];
            let src1 = packed_kmajor[n1 * packed_per_col + kk / 2];
            let q0 = if kk & 1 == 0 { src0 & 0x0f } else { src0 >> 4 };
            let q1 = if kk & 1 == 0 { src1 & 0x0f } else { src1 >> 4 };
            outmajor[kk * (n / 2) + out_pair] = q0 | (q1 << 4);
        }
    }
    outmajor
}

/// Inverse of [`repack_kmajor_to_outmajor`], returning the codec's `[N, K/2]` packed layout.
pub fn repack_outmajor_to_kmajor(packed_outmajor: &[u8], k: usize, n: usize) -> Vec<u8> {
    assert_eq!(
        k % 16,
        0,
        "NVFP4 inverse repack requires K to be a multiple of 16, got {k}"
    );
    assert_eq!(
        n % 2,
        0,
        "NVFP4 inverse output-major repack requires even N, got {n}"
    );
    assert_eq!(
        packed_outmajor.len(),
        k * (n / 2),
        "packed_outmajor length {} != K*(N/2) = {}",
        packed_outmajor.len(),
        k * (n / 2)
    );

    let mut kmajor = vec![0u8; n * (k / 2)];
    let packed_per_col = k / 2;
    for kk in 0..k {
        for out_pair in 0..(n / 2) {
            let packed = packed_outmajor[kk * (n / 2) + out_pair];
            let n0 = out_pair * 2;
            let n1 = n0 + 1;
            let dst0 = n0 * packed_per_col + kk / 2;
            let dst1 = n1 * packed_per_col + kk / 2;
            if kk & 1 == 0 {
                kmajor[dst0] = (kmajor[dst0] & 0xf0) | (packed & 0x0f);
                kmajor[dst1] = (kmajor[dst1] & 0xf0) | (packed >> 4);
            } else {
                kmajor[dst0] = (kmajor[dst0] & 0x0f) | ((packed & 0x0f) << 4);
                kmajor[dst1] = (kmajor[dst1] & 0x0f) | (packed & 0xf0);
            }
        }
    }
    kmajor
}

/// Dequantize output-major NVFP4 storage `[K, N/2]` back to row-major f32 `[K, N]`.
///
/// `block_scales_e4m3` stays in `[N, K/16]`, matching the native codec and ModelOpt per-output block
/// scale layout. `gscale` may be either `[1]` for one tensor-wide scale or `[N]` for fused projections
/// where different output ranges have different ModelOpt `weight_scale_2` scalars.
pub fn dequant_nvfp4_outmajor(
    packed_outmajor: &[u8],
    block_scales_e4m3: &[u8],
    gscale: &[f32],
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(
        k % 16,
        0,
        "NVFP4 requires K to be a multiple of 16, got {k}"
    );
    assert_eq!(
        n % 2,
        0,
        "NVFP4 output-major dequant requires even N, got {n}"
    );
    assert!(
        gscale.len() == 1 || gscale.len() == n,
        "dequant_nvfp4_outmajor: gscale length {} must be 1 or N ({n})",
        gscale.len()
    );
    assert!(
        gscale.iter().all(|v| v.is_finite() && *v > 0.0),
        "dequant_nvfp4_outmajor: every gscale must be finite and positive"
    );

    let blocks_per_col = k / 16;
    assert_eq!(
        packed_outmajor.len(),
        k * (n / 2),
        "packed_outmajor length {} != K*(N/2) = {}",
        packed_outmajor.len(),
        k * (n / 2)
    );
    assert_eq!(
        block_scales_e4m3.len(),
        n * blocks_per_col,
        "block_scales_e4m3 length {} != N*(K/16) = {}",
        block_scales_e4m3.len(),
        n * blocks_per_col
    );

    let mut w = vec![0.0f32; k * n];
    for kk in 0..k {
        let block = kk / 16;
        for out_pair in 0..(n / 2) {
            let packed = packed_outmajor[kk * (n / 2) + out_pair];
            let n0 = out_pair * 2;
            let n1 = n0 + 1;
            let q0 = packed & 0x0f;
            let q1 = packed >> 4;
            let s0 = e4m3_to_f32(block_scales_e4m3[n0 * blocks_per_col + block])
                * gscale[if gscale.len() == 1 { 0 } else { n0 }];
            let s1 = e4m3_to_f32(block_scales_e4m3[n1 * blocks_per_col + block])
                * gscale[if gscale.len() == 1 { 0 } else { n1 }];
            w[kk * n + n0] = e2m1_bits_to_f32(q0) * s0;
            w[kk * n + n1] = e2m1_bits_to_f32(q1) * s1;
        }
    }
    w
}

#[cfg(feature = "cuda")]
fn alloc_f32(like: &CubeTensor<CudaRuntime>, shape: &[usize]) -> CubeTensor<CudaRuntime> {
    let n: usize = shape.iter().product();
    let buffer = like.client.empty(n * DType::F32.size());
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.to_vec().into(),
        buffer,
        DType::F32,
    )
}

#[cfg(feature = "cuda")]
fn assert_nvfp4_gemv_shapes(
    m: usize,
    k: usize,
    n: usize,
    packed_len: usize,
    scale_len: usize,
    m_max: usize,
) {
    assert!(m > 0, "nvfp4_gemv: M must be non-zero");
    assert!(n > 0, "nvfp4_gemv: N must be non-zero");
    assert_eq!(
        k % 16,
        0,
        "nvfp4_gemv requires K to be a multiple of 16, got {k}"
    );
    assert!(
        (1..=8).contains(&m_max),
        "nvfp4_gemv: m_max must be in 1..=8 for decode, got {m_max}"
    );
    assert!(
        m <= m_max,
        "nvfp4_gemv: runtime M ({m}) exceeds fixed decode m_max ({m_max})"
    );
    assert_eq!(
        packed_len,
        n * (k / 2),
        "packed_qw length {packed_len} != N*(K/2) = {}",
        n * (k / 2)
    );
    assert_eq!(
        scale_len,
        n * (k / 16),
        "block_scales length {scale_len} != N*(K/16) = {}",
        n * (k / 16)
    );
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn launch_nvfp4_gemv_handles(
    x: &CubeTensor<CudaRuntime>,
    qw_handle: &cubecl::server::Handle,
    bs_handle: &cubecl::server::Handle,
    gscale_handle: &cubecl::server::Handle,
    k: usize,
    n: usize,
    m: usize,
    m_max: usize,
) -> CubeTensor<CudaRuntime> {
    let blocks = k / 16;
    let out = alloc_f32(x, &[m, n]);

    unsafe {
        gpu::nvfp4_decode_gemv::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static((n as u32).div_ceil(NVFP4_COLS_PER_CTA), 1, 1),
            CubeDim {
                x: 32,
                y: NVFP4_COLS_PER_CTA,
                z: 1,
            },
            x.as_tensor_arg(1),
            // packed weight as RAW u8 bytes [N, K/2] (line_size 1) — the kernel manually unpacks the
            // nibbles, avoiding the e2m1x2->f32 Line cast that mis-codegens on this cubecl rev.
            TensorArg::from_raw_parts::<u8>(qw_handle, &[k / 2, 1], &[n, k / 2], 1),
            TensorArg::from_raw_parts::<e4m3>(bs_handle, &[blocks, 1], &[n, blocks], 1),
            TensorArg::from_raw_parts::<f32>(gscale_handle, &[1], &[1], 1),
            out.as_tensor_arg(1),
            ScalarArg::new(m as u32),
            k,
            blocks,
            m_max,
        )
        .expect("nvfp4_decode_gemv launch failed");
    }

    out
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_nvfp4_gemv_tensors(
    x: CubeTensor<CudaRuntime>,
    qw: CubeTensor<CudaRuntime>,
    bs: CubeTensor<CudaRuntime>,
    gscale: CubeTensor<CudaRuntime>,
    k: usize,
    n: usize,
    m_max: usize,
) -> CubeTensor<CudaRuntime> {
    let [m, xk] = x.meta.shape().dims::<2>();
    assert_eq!(xk, k, "nvfp4_gemv: x K ({xk}) != requested K ({k})");
    assert_eq!(
        x.dtype,
        DType::F32,
        "nvfp4_gemv: x must be f32, got {:?}",
        x.dtype
    );
    assert_eq!(
        qw.dtype,
        DType::I8,
        "nvfp4_gemv: qw must be I8 bytes, got {:?}",
        qw.dtype
    );
    assert_eq!(
        bs.dtype,
        DType::I8,
        "nvfp4_gemv: bs must be I8 bytes, got {:?}",
        bs.dtype
    );
    assert_eq!(
        gscale.dtype,
        DType::F32,
        "nvfp4_gemv: gscale must be f32, got {:?}",
        gscale.dtype
    );
    assert_nvfp4_gemv_shapes(
        m,
        k,
        n,
        qw.meta.shape().num_elements(),
        bs.meta.shape().num_elements(),
        m_max,
    );
    assert_eq!(
        qw.meta.shape().dims::<2>(),
        [n, k / 2],
        "nvfp4_gemv: qw must be [N,K/2]"
    );
    assert_eq!(
        bs.meta.shape().dims::<2>(),
        [n, k / 16],
        "nvfp4_gemv: bs must be [N,K/16]"
    );
    assert_eq!(
        gscale.meta.shape().dims::<1>(),
        [1],
        "nvfp4_gemv: gscale must be [1]"
    );

    launch_nvfp4_gemv_handles(&x, &qw.handle, &bs.handle, &gscale.handle, k, n, m, m_max)
}

/// Decode-packed NVFP4 GEMV on the raw CUDA `CubeBackend` below Fusion.
///
/// `x` is `[M,K]` f32, `packed_qw` is `[N,K/2]` raw E2M1x2 bytes, `block_scales` is `[N,K/16]`
/// raw E4M3 bytes, and the returned tensor is `[M,N]` f32. `n_splits_or_m_max` is the fixed decode
/// batch bound used as the kernel's comptime `m_max`; use `1` for greedy single-stream decode.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemv_raw(
    x: Tensor<CaptureBackend, 2>,
    packed_qw: &[u8],
    block_scales: &[u8],
    gscale: f32,
    k: usize,
    n: usize,
    n_splits_or_m_max: usize,
) -> Tensor<CaptureBackend, 2> {
    assert!(
        gscale.is_finite() && gscale > 0.0,
        "nvfp4_gemv_raw: gscale must be finite and positive"
    );
    let [m, xk] = x.dims();
    assert_eq!(xk, k, "nvfp4_gemv_raw: x K ({xk}) != requested K ({k})");
    assert_eq!(x.dtype(), DType::F32, "nvfp4_gemv_raw: x must be f32");
    assert_nvfp4_gemv_shapes(
        m,
        k,
        n,
        packed_qw.len(),
        block_scales.len(),
        n_splits_or_m_max,
    );

    let x_ct = into_contiguous(x.into_primitive().tensor());
    let client = x_ct.client.clone();
    let qw_handle = client.create_from_slice(packed_qw);
    let bs_handle = client.create_from_slice(block_scales);
    let gscale_words = [gscale];
    let gscale_handle = client.create_from_slice(f32::as_bytes(&gscale_words));

    let out = launch_nvfp4_gemv_handles(
        &x_ct,
        &qw_handle,
        &bs_handle,
        &gscale_handle,
        k,
        n,
        m,
        n_splits_or_m_max,
    );
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(feature = "cuda")]
pub trait Nvfp4GemvBackend: Backend {
    /// Eager/Fusion path for persistent packed NVFP4 tensors.
    ///
    /// `qw:[N,K/2]` and `bs:[N,K/16]` must be `DType::I8` tensors carrying the raw byte encodings.
    fn nvfp4_gemv(
        x: Tensor<Self, 2>,
        qw: Tensor<Self, 2, Int>,
        bs: Tensor<Self, 2, Int>,
        gscale: Tensor<Self, 1>,
        k: usize,
        n: usize,
        m_max: usize,
    ) -> Tensor<Self, 2>;
}

#[cfg(feature = "cuda")]
impl Nvfp4GemvBackend for Cuda {
    fn nvfp4_gemv(
        x: Tensor<Cuda, 2>,
        qw: Tensor<Cuda, 2, Int>,
        bs: Tensor<Cuda, 2, Int>,
        gscale: Tensor<Cuda, 1>,
        k: usize,
        n: usize,
        m_max: usize,
    ) -> Tensor<Cuda, 2> {
        let [m, xk] = x.dims();
        assert_eq!(xk, k, "nvfp4_gemv: x K ({xk}) != requested K ({k})");
        assert_eq!(x.dtype(), DType::F32, "nvfp4_gemv: x must be f32");
        assert_eq!(qw.dtype(), DType::I8, "nvfp4_gemv: qw must be I8");
        assert_eq!(bs.dtype(), DType::I8, "nvfp4_gemv: bs must be I8");
        assert_eq!(gscale.dtype(), DType::F32, "nvfp4_gemv: gscale must be f32");
        assert_eq!(qw.dims(), [n, k / 2], "nvfp4_gemv: qw must be [N,K/2]");
        assert_eq!(bs.dims(), [n, k / 16], "nvfp4_gemv: bs must be [N,K/16]");
        assert_eq!(gscale.dims(), [1], "nvfp4_gemv: gscale must be [1]");
        assert_nvfp4_gemv_shapes(m, k, n, n * (k / 2), n * (k / 16), m_max);

        let x_prim = x.into_primitive().tensor();
        let qw_prim = qw.into_primitive();
        let bs_prim = bs.into_primitive();
        let gscale_prim = gscale.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("nvfp4_gemv")
            .float_input(x_prim)
            .int_input(qw_prim)
            .int_input(bs_prim)
            .float_input(gscale_prim)
            .float_output([m, n], DType::F32)
            .launch(move |inputs| {
                // The nvfp4 kernel FLAT-indexes every input as a dense row-major buffer
                // (`gscale[0]`, `bs[col*blocks+blk]`, `qw[col*bytes_per_col+..]`) and
                // `launch_nvfp4_gemv_handles` hardcodes contiguous strides in `from_raw_parts`. The
                // fusion HandleContainer can hand back a non-contiguous / view-offset pool handle, so
                // every input must be made contiguous first — else the flat indexing reads wrong
                // memory (NaN, shape-dependent on the pool offset). qw/bs are plain row-major [N,K/2]/
                // [N,K/16] byte tensors (NOT swizzled), so into_contiguous is layout-preserving here.
                // Mirrors w8a16_gemm, which into_contiguous's its scale input.
                let x = into_contiguous(inputs[0].clone());
                let qw = into_contiguous(inputs[1].clone());
                let bs = into_contiguous(inputs[2].clone());
                let gscale = into_contiguous(inputs[3].clone());
                vec![run_nvfp4_gemv_tensors(x, qw, bs, gscale, k, n, m_max)]
            });

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }
}

#[cfg(feature = "cuda")]
impl Nvfp4GemvBackend for CubeBackend<CudaRuntime, f32, i32, u8> {
    fn nvfp4_gemv(
        x: Tensor<Self, 2>,
        qw: Tensor<Self, 2, Int>,
        bs: Tensor<Self, 2, Int>,
        gscale: Tensor<Self, 1>,
        k: usize,
        n: usize,
        m_max: usize,
    ) -> Tensor<Self, 2> {
        let x = into_contiguous(x.into_primitive().tensor());
        let qw = qw.into_primitive();
        let bs = bs.into_primitive();
        let gscale = gscale.into_primitive().tensor();
        let out = run_nvfp4_gemv_tensors(x, qw, bs, gscale, k, n, m_max);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}

/// Decode one OCP E4M3 byte to f32.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;

    if exp == 0 {
        if mant == 0 {
            return sign * 0.0;
        }
        return sign * (mant as f32) * 2.0f32.powi(-9);
    }

    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }

    let exponent = exp as i32 - 7;
    sign * (1.0 + (mant as f32) / 8.0) * 2.0f32.powi(exponent)
}

/// Encode an f32 to an OCP E4M3 byte with round-to-nearest-even and finite saturation.
#[inline]
pub fn f32_to_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }

    let sign_bit = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let abs = value.abs();
    if abs == 0.0 {
        return sign_bit;
    }
    if !abs.is_finite() || abs >= E4M3_MAX {
        return sign_bit | 0x7e;
    }

    if abs < E4M3_MIN_NORMAL {
        let q = round_ties_to_even(abs / 2.0f32.powi(-9));
        if q >= 8 {
            return sign_bit | 0x08;
        }
        return sign_bit | q.clamp(0, 7) as u8;
    }

    let mut exponent = abs.log2().floor() as i32;
    let mut q = round_ties_to_even(abs / 2.0f32.powi(exponent - 3));

    if q >= 16 {
        exponent += 1;
        q = 8;
    }

    if exponent >= 8 {
        q = q.min(14);
        return sign_bit | 0x78 | ((q as u8 - 8) & 0x07);
    }

    let exp_bits = (exponent + 7).clamp(1, 14) as u8;
    let mant_bits = (q as u8).saturating_sub(8) & 0x07;
    sign_bit | (exp_bits << 3) | mant_bits
}

#[inline]
pub fn f32_to_e2m1_bits(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7;
    }

    let sign_bit = if value.is_sign_negative() { 0x8 } else { 0x0 };
    let abs = value.abs();
    if abs == 0.0 {
        return sign_bit;
    }
    if !abs.is_finite() || abs >= E2M1_MAX {
        return sign_bit | 0x7;
    }

    let mut best_idx = 0usize;
    let mut best_err = f32::INFINITY;
    for (idx, &candidate) in E2M1_VALUES.iter().enumerate() {
        let err = (abs - candidate).abs();
        if err < best_err || (err == best_err && idx % 2 == 0 && best_idx % 2 != 0) {
            best_idx = idx;
            best_err = err;
        }
    }

    sign_bit | best_idx as u8
}

#[inline]
pub fn e2m1_bits_to_f32(bits: u8) -> f32 {
    let magnitude = E2M1_VALUES[(bits & 0x07) as usize];
    if bits & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[inline]
fn round_ties_to_even(value: f32) -> i32 {
    let floor = value.floor();
    let frac = value - floor;
    let base = floor as i32;

    if frac < 0.5 {
        base
    } else if frac > 0.5 {
        base + 1
    } else if base & 1 == 0 {
        base
    } else {
        base + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        if na == 0.0 || nb == 0.0 {
            return f32::NAN;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        let mut sum = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += ((x - y) as f64).powi(2);
        }
        sum / a.len() as f64
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn matmul_row(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[kk] * w[kk * n + nn];
            }
            out[nn] = acc;
        }
        out
    }

    fn next_u32(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    }

    fn next_unit_f32(state: &mut u64) -> f32 {
        let bits = 0x3f80_0000 | (next_u32(state) >> 9);
        f32::from_bits(bits) - 1.0
    }

    fn next_normal(state: &mut u64) -> f32 {
        let u1 = next_unit_f32(state).max(f32::MIN_POSITIVE);
        let u2 = next_unit_f32(state);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    fn heavy_tailed_weights(k: usize, n: usize) -> Vec<f32> {
        let mut rng = 0x44f7_3b2d_190c_51a9u64;
        let mut w = vec![0.0f32; k * n];
        for kk in 0..k {
            for nn in 0..n {
                w[kk * n + nn] = 0.02 * next_normal(&mut rng);
            }
        }

        for nn in 0..n {
            for outlier in 0..8 {
                let kk = (nn * 131 + outlier * 251 + 7) % k;
                let sign = if next_u32(&mut rng) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                w[kk * n + nn] = sign * (0.16 + 0.08 * next_unit_f32(&mut rng));
            }
        }

        w
    }

    #[test]
    fn fwht_is_orthonormal_and_matches_goldens() {
        let mut h4 = [1.0f32, 2.0, 3.0, 4.0];
        fwht_inplace(&mut h4);
        assert!(max_abs_diff(&h4, &[5.0, -1.0, -2.0, 0.0]) < 1e-6);

        let inv4_before = h4;
        fwht_inplace(&mut h4);
        assert!(max_abs_diff(&h4, &[1.0, 2.0, 3.0, 4.0]) < 1e-6);
        assert!(max_abs_diff(&inv4_before, &[5.0, -1.0, -2.0, 0.0]) < 1e-6);

        let mut h8 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        fwht_inplace(&mut h8);
        let s8 = 8.0f32.sqrt();
        let expected = [
            36.0 / s8,
            -4.0 / s8,
            -8.0 / s8,
            0.0,
            -16.0 / s8,
            0.0,
            0.0,
            0.0,
        ];
        assert!(max_abs_diff(&h8, &expected) < 1e-6);

        let mut rng = 0x6a09_e667_f3bc_c909u64;
        let mut v = (0..128)
            .map(|_| next_normal(&mut rng) * 0.1)
            .collect::<Vec<_>>();
        let original = v.clone();
        fwht_inplace(&mut v);
        fwht_inplace(&mut v);
        assert!(max_abs_diff(&v, &original) < 1e-6);
    }

    #[test]
    fn rotate_inverse_is_identity_for_supported_groups() {
        let (k, n) = (256usize, 8usize);
        let mut rng = 0x510e_527f_ade6_82d1u64;
        let w = (0..k * n)
            .map(|_| next_normal(&mut rng) * 0.05)
            .collect::<Vec<_>>();

        for &g in &[64usize, 128usize] {
            let mut got = w.clone();
            rotate_matrix_k(&mut got, k, n, g, 0x1234_5678_9abc_def0);
            rotate_matrix_k_inverse(&mut got, k, n, g, 0x1234_5678_9abc_def0);
            assert!(
                max_abs_diff(&got, &w) < 1e-6,
                "rotate inverse failed for g={g}"
            );
        }
    }

    #[test]
    fn hadamard_fake_quant_matches_rotated_basis_runtime_math() {
        let (k, n, g) = (256usize, 8usize, 128usize);
        let seed = 0x0f1e_2d3c_4b5a_6978;
        let mut rng = 0xbb67_ae85_84ca_a73bu64;
        let w = (0..k * n)
            .map(|_| next_normal(&mut rng) * 0.02)
            .collect::<Vec<_>>();
        let x = (0..k)
            .map(|_| next_normal(&mut rng) * 0.02)
            .collect::<Vec<_>>();

        let mut w_rot = w.clone();
        rotate_matrix_k(&mut w_rot, k, n, g, seed);
        let (qw, bs, gscale) = quantize_nvfp4_clip(&w_rot, k, n, 3.5);
        let wq_rot = dequant_nvfp4(&qw, &bs, gscale, k, n);

        let mut w_fake = wq_rot.clone();
        rotate_matrix_k_inverse(&mut w_fake, k, n, g, seed);
        let fake_out = matmul_row(&x, &w_fake, k, n);

        let mut x_rot = x.clone();
        rotate_matrix_k(&mut x_rot, k, 1, g, seed);
        let deployed_out = matmul_row(&x_rot, &wq_rot, k, n);

        assert!(max_abs_diff(&fake_out, &deployed_out) < 1e-5);
    }

    #[test]
    fn clip_scale_saturates_outliers_and_zero_clip_is_amax() {
        let (k, n) = (16usize, 1usize);
        let mut w = vec![0.1f32; k * n];
        w[0] = 100.0;

        let (qw_amax, bs_amax, gscale_amax) = quantize_nvfp4(&w, k, n);
        let (qw_zero, bs_zero, gscale_zero) = quantize_nvfp4_clip(&w, k, n, 0.0);
        assert_eq!(qw_zero, qw_amax);
        assert_eq!(bs_zero, bs_amax);
        assert_eq!(gscale_zero.to_bits(), gscale_amax.to_bits());

        let (qw_clip, bs_clip, gscale_clip) = quantize_nvfp4_clip(&w, k, n, 3.5);
        let amax_scale = e4m3_to_f32(bs_amax[0]) * gscale_amax;
        let clip_scale = e4m3_to_f32(bs_clip[0]) * gscale_clip;
        assert!(
            clip_scale < amax_scale,
            "clip scale {clip_scale} must be smaller than amax scale {amax_scale}"
        );
        assert_eq!(
            qw_clip[0] & 0x0f,
            0x07,
            "positive outlier must saturate to +E2M1 max"
        );
        assert_eq!(f32_to_e2m1_bits(f32::INFINITY), 0x07);
        assert_eq!(f32_to_e2m1_bits(f32::NEG_INFINITY), 0x0f);
    }

    #[test]
    fn hadamard_quantization_is_deterministic_by_layer_and_site() {
        let (k, n, g) = (256usize, 8usize, 128usize);
        let mut rng = 0x3c6e_f372_fe94_f82bu64;
        let w = (0..k * n)
            .map(|_| next_normal(&mut rng) * 0.03)
            .collect::<Vec<_>>();
        let cfg = Nvfp4HadamardConfig {
            group_size: g,
            clip_c: 3.5,
            base_seed: 0x1357_9bdf_2468_ace0,
        };
        let seed_a = cfg.seed_for(7, Nvfp4HadamardSite::MoeIn);
        let seed_b = cfg.seed_for(7, Nvfp4HadamardSite::MoeDownIn);

        let mut rot_a0 = w.clone();
        rotate_matrix_k(&mut rot_a0, k, n, g, seed_a);
        let bytes_a0 = quantize_nvfp4_clip(&rot_a0, k, n, cfg.clip_c);

        let mut rot_a1 = w.clone();
        rotate_matrix_k(&mut rot_a1, k, n, g, seed_a);
        let bytes_a1 = quantize_nvfp4_clip(&rot_a1, k, n, cfg.clip_c);

        let mut rot_b = w.clone();
        rotate_matrix_k(&mut rot_b, k, n, g, seed_b);
        let bytes_b = quantize_nvfp4_clip(&rot_b, k, n, cfg.clip_c);

        assert_eq!(bytes_a0.0, bytes_a1.0);
        assert_eq!(bytes_a0.1, bytes_a1.1);
        assert_eq!(bytes_a0.2.to_bits(), bytes_a1.2.to_bits());
        assert_ne!(bytes_a0.0, bytes_b.0);
    }

    #[test]
    fn e2m1_pack_bits_match_cubecl_order() {
        let low = f32_to_e2m1_bits(1.0);
        let high = f32_to_e2m1_bits(-2.0);
        let packed = low | (high << 4);
        assert_eq!(e2m1_bits_to_f32(packed & 0x0f), 1.0);
        assert_eq!(e2m1_bits_to_f32((packed >> 4) & 0x0f), -2.0);
    }

    fn e2m1_marlin_decode_host(code: u8) -> f32 {
        let top = (code & 0x0f) << 4;
        let fp8_bits = (top & 0x80) | ((top & 0x70) >> 2);
        e4m3_to_f32(fp8_bits) * 64.0
    }

    #[test]
    fn e2m1_marlin_bit_trick_matches_lut_for_all_packed_bytes() {
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let low = byte & 0x0f;
            let high = byte >> 4;
            let low_lut = e2m1_bits_to_f32(low);
            let high_lut = e2m1_bits_to_f32(high);
            let low_trick = e2m1_marlin_decode_host(low);
            let high_trick = e2m1_marlin_decode_host(high);
            assert!(
                low_lut.to_bits() == low_trick.to_bits(),
                "low nibble mismatch for byte 0x{byte:02x}: LUT={low_lut:?} trick={low_trick:?}"
            );
            assert!(
                high_lut.to_bits() == high_trick.to_bits(),
                "high nibble mismatch for byte 0x{byte:02x}: LUT={high_lut:?} trick={high_trick:?}"
            );
        }
    }

    #[test]
    fn zero_blocks_dequant_to_exact_zero() {
        let w = vec![0.0f32; 32 * 3];
        let (qw, bs, gscale) = quantize_nvfp4(&w, 32, 3);
        let decoded = dequant_nvfp4(&qw, &bs, gscale, 32, 3);
        assert!(decoded.iter().all(|&v| v == 0.0));
        assert!(bs.iter().all(|&b| b == f32_to_e4m3(E4M3_MIN_NORMAL)));
    }

    #[test]
    fn output_major_repack_roundtrip_is_bit_exact() {
        for &k in &[32usize, 512, 2048] {
            for &n in &[8usize, 32, 512] {
                let mut rng = 0x8f3d_2c1b_a987_6543u64 ^ ((k as u64) << 32) ^ n as u64;
                let w = (0..k * n)
                    .map(|_| next_normal(&mut rng) * 0.04)
                    .collect::<Vec<_>>();
                let (qw, bs, gscale) = quantize_nvfp4(&w, k, n);
                let outmajor = repack_kmajor_to_outmajor(&qw, k, n);
                let roundtrip = repack_outmajor_to_kmajor(&outmajor, k, n);
                assert_eq!(
                    roundtrip, qw,
                    "packed byte roundtrip failed for K={k}, N={n}"
                );

                let decoded = dequant_nvfp4(&qw, &bs, gscale, k, n);
                let decoded_roundtrip = dequant_nvfp4(&roundtrip, &bs, gscale, k, n);
                assert!(
                    decoded
                        .iter()
                        .zip(decoded_roundtrip.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "dequant after inverse repack changed bits for K={k}, N={n}"
                );
            }
        }
    }

    #[test]
    fn output_major_dequant_matches_kmajor_dequant() {
        for &k in &[32usize, 512, 2048] {
            for &n in &[8usize, 32, 512] {
                let mut rng = 0x1020_3040_5060_7080u64 ^ ((k as u64) << 24) ^ n as u64;
                let w = (0..k * n)
                    .map(|idx| {
                        let tail = if idx % 97 == 0 { 4.0 } else { 1.0 };
                        next_normal(&mut rng) * 0.02 * tail
                    })
                    .collect::<Vec<_>>();
                let (qw, bs, gscale) = quantize_nvfp4(&w, k, n);
                let outmajor = repack_kmajor_to_outmajor(&qw, k, n);
                let kmajor = dequant_nvfp4(&qw, &bs, gscale, k, n);
                let output_major = dequant_nvfp4_outmajor(&outmajor, &bs, &[gscale], k, n);
                assert!(
                    kmajor
                        .iter()
                        .zip(output_major.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "out-major dequant differed from k-major for K={k}, N={n}"
                );
            }
        }
    }

    #[test]
    fn round_trip_cosine_covers_amax_and_mse() {
        let (k, n) = (256usize, 64usize);
        let w = heavy_tailed_weights(k, n);

        let (qw_amax, bs_amax, gscale_amax) = quantize_nvfp4(&w, k, n);
        let decoded_amax = dequant_nvfp4(&qw_amax, &bs_amax, gscale_amax, k, n);
        let cos_amax = cosine(&w, &decoded_amax);

        let (qw_mse, bs_mse, gscale_mse) = quantize_nvfp4_mse(&w, k, n);
        let decoded_mse = dequant_nvfp4(&qw_mse, &bs_mse, gscale_mse, k, n);
        let cos_mse = cosine(&w, &decoded_mse);

        println!("round-trip cosine: amax={cos_amax:.6}, mse={cos_mse:.6}");
        assert!(
            cos_amax > 0.98,
            "amax round-trip cosine {cos_amax:.6} <= 0.98"
        );
        assert!(cos_mse > 0.98, "mse round-trip cosine {cos_mse:.6} <= 0.98");
        assert!(decoded_mse.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mse_beats_amax_cosine_on_realistic_weights() {
        let (k, n) = (2048usize, 512usize);
        let w = heavy_tailed_weights(k, n);

        let (qw_amax, bs_amax, gscale_amax) = quantize_nvfp4(&w, k, n);
        let decoded_amax = dequant_nvfp4(&qw_amax, &bs_amax, gscale_amax, k, n);
        let cos_amax = cosine(&w, &decoded_amax);

        let (qw_mse, bs_mse, gscale_mse) = quantize_nvfp4_mse(&w, k, n);
        let decoded_mse = dequant_nvfp4(&qw_mse, &bs_mse, gscale_mse, k, n);
        let cos_mse = cosine(&w, &decoded_mse);

        println!("realistic weights cosine: amax={cos_amax:.9}, mse={cos_mse:.9}");
        println!(
            "realistic weights mse: amax={:.9e}, mse={:.9e}",
            mse(&w, &decoded_amax),
            mse(&w, &decoded_mse)
        );
        assert!(
            cos_mse >= cos_amax,
            "mse cosine {cos_mse:.9} < amax cosine {cos_amax:.9}"
        );
    }

    #[test]
    fn mse_golden_zero_and_outlier_blocks() {
        let (k, n) = (32usize, 2usize);
        let mut w = vec![0.0f32; k * n];
        let outlier_block = [
            0.010, -0.012, 0.014, -0.016, 0.018, -0.020, 0.022, -0.024, 0.026, -0.028, 0.030,
            -0.032, 0.20, -0.18, 0.16, -0.22,
        ];
        for (offset, &value) in outlier_block.iter().enumerate() {
            w[(16 + offset) * n] = value;
            w[(16 + offset) * n + 1] = -0.5 * value;
        }

        let (qw, bs, gscale) = quantize_nvfp4_mse(&w, k, n);
        assert!(gscale.is_finite() && gscale > 0.0);
        assert!(bs.iter().all(|&b| e4m3_to_f32(b).is_finite()));

        let decoded = dequant_nvfp4(&qw, &bs, gscale, k, n);
        assert!(decoded.iter().all(|v| v.is_finite()));
        for kk in 0..16 {
            for nn in 0..n {
                assert_eq!(decoded[kk * n + nn], 0.0);
            }
        }
    }
}
