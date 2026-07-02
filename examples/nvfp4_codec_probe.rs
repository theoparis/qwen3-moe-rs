//! Pure-CPU NVFP4 codec round-trip probe.
//!
//! After adding `pub mod nvfp4;` to `src/lib.rs`:
//!   cargo run --example nvfp4_codec_probe

use qwen3_burn::nvfp4::{dequant_nvfp4, quantize_nvfp4};

#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn uniform(&mut self) -> f32 {
        let bits = 0x3f80_0000 | (self.next_u32() >> 9);
        f32::from_bits(bits) - 1.0
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.uniform()
    }
}

fn main() {
    let cases = [(256usize, 128usize), (64, 64), (512, 32)];
    for (case_idx, (k, n)) in cases.into_iter().enumerate() {
        run_case(case_idx, k, n);
    }
}

fn run_case(case_idx: usize, k: usize, n: usize) {
    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15 ^ ((case_idx as u64) << 32));
    let zero_col = 0usize;
    let outlier_col_a = 3usize.min(n - 1);
    let outlier_col_b = (n / 2).max(1).min(n - 1);

    let mut w = vec![0.0f32; k * n];
    for kk in 0..k {
        for nn in 0..n {
            if nn == zero_col {
                continue;
            }

            let base = rng.range(-0.08, 0.08);
            let ripple = ((kk as f32) * 0.017 + (nn as f32) * 0.013).sin() * 0.02;
            w[kk * n + nn] = base + ripple;
        }
    }

    for &nn in &[outlier_col_a, outlier_col_b] {
        if nn != zero_col {
            for i in 0..4 {
                let kk = (i * 53 + case_idx * 17) % k;
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                w[kk * n + nn] = sign * (8.0 + 3.5 * i as f32 + case_idx as f32);
            }
        }
    }

    let (packed_qw, block_scales, gscale) = quantize_nvfp4(&w, k, n);
    let decoded = dequant_nvfp4(&packed_qw, &block_scales, gscale, k, n);
    let metrics = metrics(&w, &decoded);

    let zero_exact = (0..k).all(|kk| decoded[kk * n + zero_col] == 0.0);
    println!(
        "case {case_idx}: K={k} N={n} bytes={} qw={} bs={} gscale={gscale:e}",
        packed_qw.len() + block_scales.len() + std::mem::size_of::<f32>(),
        packed_qw.len(),
        block_scales.len()
    );
    println!(
        "  max_abs={:.6e} rel_to_max={:.6e} cosine={:.8} zero_col_exact={}",
        metrics.max_abs, metrics.rel_to_max, metrics.cosine, zero_exact
    );

    assert!(
        metrics.cosine > 0.995,
        "NVFP4 round-trip cosine too low: {:.8}",
        metrics.cosine
    );
    assert!(zero_exact, "dead zero column did not dequantize to exact zero");
}

struct Metrics {
    max_abs: f32,
    rel_to_max: f32,
    cosine: f32,
}

fn metrics(reference: &[f32], decoded: &[f32]) -> Metrics {
    assert_eq!(reference.len(), decoded.len());

    let mut max_abs = 0.0f32;
    let mut ref_max = 0.0f32;
    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut dec_norm = 0.0f64;

    for (&a, &b) in reference.iter().zip(decoded) {
        max_abs = max_abs.max((a - b).abs());
        ref_max = ref_max.max(a.abs());
        dot += (a as f64) * (b as f64);
        ref_norm += (a as f64) * (a as f64);
        dec_norm += (b as f64) * (b as f64);
    }

    Metrics {
        max_abs,
        rel_to_max: max_abs / ref_max.max(1.0e-12),
        cosine: (dot / (ref_norm.sqrt() * dec_norm.sqrt()).max(1.0e-30)) as f32,
    }
}
