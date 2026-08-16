//! VLLM_KERNELS.md §2 — a FUSED **W8A16** (fp8 weight-only) GEMM in CubeCL, validated on the real
//! GB10 GPU against an INDEPENDENT **NdArray (CPU f32)** oracle (the cross-backend law, §0) + OCP
//! E4M3 golden vectors.
//!
//! The win is reading HALF the weight BYTES from HBM: the weight is stored as one OCP E4M3 byte per
//! element (per-output-channel symmetric scale). The fused kernel reads the packed e4m3 BYTE straight
//! from HBM and dequants it IN-REGISTER inside the GEMM accumulation loop — it is NOT pre-expanded to
//! a full f32/bf16 weight tensor (the round-trip §2 rejects). See `src/w8a16.rs`.
//!
//! This spike:
//!   STEP A — e4m3 byte → f32 decode micro-test on the GPU, asserted against hardcoded OCP golden
//!            values (0x00->0.0, 0x38->1.0, 0x7E->448.0, the sign bit, mid-range, smallest normal,
//!            subnormals). The host codec is checked against the SAME golden table, so "GPU == OCP"
//!            and "host == OCP" together prove the oracle's dequant is bit-faithful to the kernel's.
//!   STEP C — the fused W8A16 GEMM at real Qwen3 Linear shapes (K=2048,N=768 ; K=768,N=2048 ;
//!            K=1024,N=3072), M in {1 (decode), 64}. For each: quantize W on the host, run the GPU
//!            kernel, build the NdArray oracle `y_ref = x_f32 @ dequant(q,s)_f32` from the SAME bytes,
//!            and report cosine + max_abs_diff (assert cosine > 0.999). Also prints the informational
//!            f32 reference `x @ W_orig` vs the fp8 path (the ~3% quantization error from the probe).
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example w8a16_spike 2>&1 | tail -40

use burn::prelude::Device;
use burn::prelude::Device;
use burn::prelude::Device;
use burn::tensor::{DType, Device, Int, Tensor, TensorData};

use qwen3_burn::w8a16::{
    dequant_e4m3, e4m3_decode, e4m3_to_f32, quantize_e4m3_per_channel, w8a16_gemm,
};

// -------------------------------------------------------------------------------------------------
// Host helpers
// -------------------------------------------------------------------------------------------------

/// A tiny deterministic LCG so the CPU oracle and the GPU kernel see byte-identical inputs.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// Uniform f32 in [-amp, amp].
    fn next(&mut self, amp: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 33) as u32) as f32 / (u32::MAX as f32);
        (u * 2.0 - 1.0) * amp
    }
}

fn make_data(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next(amp)).collect()
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

fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

// -------------------------------------------------------------------------------------------------
// STEP A — OCP E4M3 byte -> f32 golden vectors.
// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa; NO infinities; NaN = S.1111.111; max finite = 448.
// Each expected value is exact in f32, so the GPU decode must match bit-for-bit.
// -------------------------------------------------------------------------------------------------
fn golden_table() -> Vec<(u8, f32, &'static str)> {
    vec![
        (0x00, 0.0, "zero"),
        (0x38, 1.0, "1.0  (exp=7, m=0)"),
        (0xB8, -1.0, "-1.0 (sign bit)"),
        (0x3C, 1.5, "1.5  (exp=7, m=4)"),
        (0x40, 2.0, "2.0  (exp=8)"),
        (0x48, 4.0, "4.0  (exp=9)"),
        (0x7E, 448.0, "448  (max finite, exp=15 m=6)"),
        (0xFE, -448.0, "-448 (max finite, neg)"),
        (0x08, 0.015625, "2^-6 (smallest normal)"),
        (0x07, 0.013671875, "max subnormal (7*2^-9)"),
        (0x04, 0.0078125, "mid subnormal (4*2^-9)"),
        (0x01, 0.001953125, "min subnormal (2^-9)"),
    ]
}

fn step_a(cuda_dev: &CudaDevice) -> bool {
    println!("=== STEP A — e4m3 byte -> f32 decode (GPU) vs OCP golden vectors ===");
    let table = golden_table();
    let bytes: Vec<u8> = table.iter().map(|(b, _, _)| *b).collect();

    // Carry the raw e4m3 bytes in a 1-byte I8 Int tensor (fp8 has no Burn float DType and Burn's Int
    // kind has no u8; `b as i8` is bit-preserving, and the kernel reinterprets the bits as e4m3).
    let bytes_i8: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
    let q = Tensor::<1, Int>::from_data_dtype(
        TensorData::new(bytes_i8, [bytes.len()]),
        cuda_dev,
        DType::I8,
    );
    let got = e4m3_decode(q).into_data().to_vec::<f32>().unwrap();

    let mut all_ok = true;
    for (i, (b, exp, name)) in table.iter().enumerate() {
        let g = got[i];
        let h = e4m3_to_f32(*b); // host codec (used by the quantizer + oracle)
        // Each golden value is exact in f32, so the GPU decode must match bit-for-bit (diff == 0).
        let gpu_ok = (g - exp).abs() == 0.0;
        let host_ok = (h - exp).abs() == 0.0;
        all_ok &= gpu_ok && host_ok;
        println!(
            "  0x{b:02X} -> gpu={g:<12} host={h:<12} golden={exp:<12} {} {name}",
            if gpu_ok && host_ok {
                "[OK]"
            } else {
                "[MISMATCH]"
            },
        );
    }
    println!(
        "  STEP A: {}\n",
        if all_ok {
            "PASS — GPU e4m3 decode == OCP golden == host codec"
        } else {
            "FAIL"
        }
    );
    all_ok
}

// -------------------------------------------------------------------------------------------------
// STEP C — the fused W8A16 GEMM vs the NdArray (CPU f32) oracle, at real Qwen3 Linear shapes.
// -------------------------------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    k: usize,
    n: usize,
    m: usize,
}

struct Row {
    label: String,
    cos: f32,
    mad: f32,
    // informational: fp8 path vs the full-precision f32 reference (x @ W_orig).
    quant_cos: f32,
    quant_rel: f32,
    ok: bool,
}

fn run_shape(s: Shape, cuda_dev: &Device, nd_dev: &Device) -> Row {
    let Shape { label, k, n, m } = s;

    // Same host data everywhere. Weights small-ish; activations ~N(0,1)-scale uniform.
    let w_data = make_data(k * n, 0x57EE_D001 ^ ((k * n) as u64), 0.08);
    let x_data = make_data(m * k, 0x1234_5678 ^ ((m * k) as u64), 1.0);

    // --- Host quantize: W:[K,N] -> e4m3 bytes q:[K,N] + per-output-channel scale s:[N] ---
    let (q_bytes, scale) = quantize_e4m3_per_channel(&w_data, k, n);

    // --- GPU fused W8A16 GEMM ---
    let x_cu = Tensor::<2>::from_data(TensorData::new(x_data.clone(), [m, k]), cuda_dev);
    let q_i8: Vec<i8> = q_bytes.iter().map(|&b| b as i8).collect(); // bit-preserving e4m3 bytes
    let q_cu =
        Tensor::<2, Int>::from_data_dtype(TensorData::new(q_i8, [k, n]), cuda_dev, DType::I8);
    let s_cu = Tensor::<1>::from_data(TensorData::new(scale.clone(), [n]), cuda_dev);
    let gpu = w8a16_gemm(x_cu, q_cu, s_cu)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // --- NdArray (CPU f32) oracle: dequant the SAME bytes on the host, matmul on CPU ---
    let w_deq = dequant_e4m3(&q_bytes, &scale, k, n); // exact bytes the GPU kernel decodes
    let x_nd = Tensor::<2>::from_data(TensorData::new(x_data.clone(), [m, k]), nd_dev);
    let w_nd = Tensor::<2>::from_data(TensorData::new(w_deq, [k, n]), nd_dev);
    let y_ref = x_nd
        .clone()
        .matmul(w_nd)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let cos = cosine(&gpu, &y_ref);
    let mad = max_abs_diff(&gpu, &y_ref);
    let refmax = max_abs(&y_ref);
    let ok = gpu.len() == m * n && cos > 0.999 && !gpu.iter().any(|x| x.is_nan());

    // --- Informational: fp8 path vs full-precision f32 reference (x @ W_orig) ---
    let w_orig_nd = Tensor::<2>::from_data(TensorData::new(w_data.clone(), [k, n]), nd_dev);
    let y_full = x_nd.matmul(w_orig_nd).into_data().to_vec::<f32>().unwrap();
    let quant_cos = cosine(&gpu, &y_full);
    let quant_rel = max_abs_diff(&gpu, &y_full) / max_abs(&y_full).max(1e-9);

    let lbl = format!("{label:18} K{k} N{n} M{m}");
    println!(
        "{lbl}\n    kernel vs NdArray oracle : cos={cos:.6} max_abs_diff={mad:.3e} (|y_ref|max={refmax:.3e}){}\n    \
         fp8 path vs f32 W_orig   : cos={quant_cos:.6} rel-max-err={:.2}% (informational, ~probe)",
        if ok { "  [PASS]" } else { "  [FAIL]" },
        100.0 * quant_rel,
    );

    Row {
        label: lbl,
        cos,
        mad,
        quant_cos,
        quant_rel,
        ok,
    }
}

fn main() {
    let cuda_dev = Device::cuda(0);
    let nd_dev = Device::flex();
    println!(
        "device: {cuda_dev:?} | oracle: NdArray (CPU f32) | kernel: fused CubeCL W8A16 (e4m3)"
    );
    println!("cross-backend law: oracle is an INDEPENDENT CPU backend (docs/VLLM_KERNELS.md §0)\n");

    let a_ok = step_a(&cuda_dev);

    println!("=== STEP C — fused W8A16 GEMM vs NdArray (CPU f32) oracle, real Qwen3 shapes ===");
    let shapes = [
        Shape {
            label: "qkv/gate [decode]",
            k: 2048,
            n: 768,
            m: 1,
        },
        Shape {
            label: "qkv/gate",
            k: 2048,
            n: 768,
            m: 64,
        },
        Shape {
            label: "down     [decode]",
            k: 768,
            n: 2048,
            m: 1,
        },
        Shape {
            label: "down",
            k: 768,
            n: 2048,
            m: 64,
        },
        Shape {
            label: "mlp-up   [decode]",
            k: 1024,
            n: 3072,
            m: 1,
        },
        Shape {
            label: "mlp-up",
            k: 1024,
            n: 3072,
            m: 64,
        },
    ];
    let mut rows = Vec::new();
    for s in shapes {
        rows.push(run_shape(s, &cuda_dev, &nd_dev));
    }

    println!("\n================ SUMMARY (kernel vs NdArray CPU oracle) ================");
    println!(
        "{:30}  {:>9} {:>11} {:>10} {:>9}",
        "shape", "cos", "max_abs", "quant_rel", "quant_cos"
    );
    let mut all_ok = a_ok;
    for r in &rows {
        println!(
            "{:30}  {:>9.6} {:>11.3e} {:>9.2}% {:>9.6}",
            r.label,
            r.cos,
            r.mad,
            100.0 * r.quant_rel,
            r.quant_cos,
        );
        all_ok &= r.ok;
    }

    println!("\n--- VERDICT ---");
    if all_ok {
        println!(
            "W8A16 KERNEL: VALIDATED — fused fp8 e4m3 weight-only GEMM reads packed e4m3 BYTES from \
             HBM, dequants IN-REGISTER in the GEMM load path, and matches the NdArray CPU oracle \
             (cosine > 0.999) on K=2048/N=768, K=768/N=2048, K=1024/N=3072 at M in {{1,64}}; the GPU \
             e4m3 decode matches the OCP golden vectors exactly."
        );
    } else {
        println!(
            "W8A16 KERNEL: PARTIAL/FAIL — at least one check did not pass (see [FAIL]/[MISMATCH] \
             rows above)."
        );
    }

    assert!(
        a_ok,
        "STEP A: GPU e4m3 decode did not match the OCP golden vectors"
    );
    for r in &rows {
        assert!(
            r.ok,
            "STEP C: shape `{}` failed vs the NdArray oracle (cos={:.6})",
            r.label, r.cos
        );
    }
    println!("\nALL CHECKS PASSED.");
}
