//! NVFP4 decode-GEMV probe.
//!
//! Run on a CUDA build:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example nvfp4_gemv_probe

use burn::tensor::{Tensor, TensorData};
use qwen3_burn::capture::CaptureBackend;
use qwen3_burn::nvfp4::{dequant_nvfp4, nvfp4_gemv_raw, quantize_nvfp4};

type B = CaptureBackend;

fn pseudo(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i.wrapping_mul(1_664_525) ^ seed.wrapping_mul(1_013_904_223)) % 4099;
            (x as f32 / 4099.0 - 0.5) * 0.5
        })
        .collect()
}

fn cpu_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[mm * k + kk] * w[kk * n + nn];
            }
            out[mm * n + nn] = acc;
        }
    }
    out
}

fn metrics(got: &[f32], want: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((g - w).abs());
        dot += (g as f64) * (w as f64);
        ng += (g as f64) * (g as f64);
        nw += (w as f64) * (w as f64);
    }
    (max_abs, dot / (ng.sqrt() * nw.sqrt() + 1e-12))
}

fn check(k: usize, n: usize, m: usize) {
    let dev = Default::default();
    let w = pseudo(k * n, 10 + k + n);
    let x_host = pseudo(m * k, 20 + m + k);
    let (packed_qw, block_scales, gscale) = quantize_nvfp4(&w, k, n);
    let w_deq = dequant_nvfp4(&packed_qw, &block_scales, gscale, k, n);
    let want = cpu_matmul(&x_host, &w_deq, m, k, n);

    let x = Tensor::<2>::from_data(TensorData::new(x_host, [m, k]), &dev);
    let got = nvfp4_gemv_raw(x, &packed_qw, &block_scales, gscale, k, n, m)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let (max_abs, cosine) = metrics(&got, &want);
    println!("K={k} N={n} M={m}: max_abs={max_abs:.4e} cosine={cosine:.8}");
    assert!(
        cosine > 0.999,
        "NVFP4 GEMV cosine too low for K={k} N={n} M={m}: {cosine:.8}"
    );
}

fn main() {
    // Numerics-identity gate: GPU nvfp4_decode_gemv == host dequant_nvfp4(w) then f32 matmul (same
    // E2M1/E4M3 decode). The earlier cubecl codegen blocker (e2m1x2->f32 Line cast) is RESOLVED by the
    // in-kernel manual u8 nibble-unpack (docs/L2C-gemv-cubecl-blocker.md).
    check(256, 128, 1);
    check(512, 256, 4);
    // Larger, real-projection-scale shapes (down K768/N2048, qkv K2048/N768) — the D6 gate exposed a
    // NaN at these on the Fusion path; this checks whether the RAW path handles the shapes.
    check(768, 2048, 1);
    check(2048, 768, 1);
    check(1024, 3072, 1);
    println!("L2C GEMV: ALL PASS (numerics-identity gate vs codec-dequant matmul)");
}
