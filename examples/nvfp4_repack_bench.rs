//! Quantifies the CPU cost of the streamed-expert NVFP4 repack that runs on every routed expert
//! on every decode step (`ExpertSlotPool::fetch_slot` -> `pack_out_in_nvfp4` -> `quantize_nvfp4`).
//!
//! No GPU, no weights, no features required:
//!   cargo run --release --example nvfp4_repack_bench
//!
//! Shapes are the real Qwen3.6-35B-A3B ones from models/config.json:
//!   hidden = 2048, moe_intermediate = 512, 40 layers, top-8 of 256 experts.
//!   gate_up: [2*inner, hidden] = [1024, 2048] -> n=1024, k=2048
//!   down:    [hidden, inner]   = [2048,  512] -> n=2048, k=512

use std::time::Instant;

use qwen3_burn::nvfp4::{quantize_nvfp4, quantize_nvfp4_from_nk_bf16};

/// The exact transpose in `expert_stream::pack_out_in_nvfp4`: [N,K] bf16 bytes -> [K,N] f32.
fn transpose_bf16_to_f32_kn(bf16_bytes: &[u8], n: usize, k: usize) -> Vec<f32> {
    let mut kn = vec![0.0f32; k * n];
    for nn in 0..n {
        for kk in 0..k {
            let off = (nn * k + kk) * 2;
            let bits = u16::from_le_bytes([bf16_bytes[off], bf16_bytes[off + 1]]);
            kn[kk * n + nn] = half::bf16::from_bits(bits).to_f32();
        }
    }
    kn
}

fn synth_bf16(n: usize, k: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n * k * 2];
    let mut state = 0x12345678u32;
    for chunk in bytes.chunks_exact_mut(2) {
        // xorshift -> a small finite bf16 (exponent clamped so the value is never NaN/Inf,
        // which `quantize_nvfp4` asserts against).
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let v = half::bf16::from_f32(((state >> 8) as f32 / u32::MAX as f32) - 0.5);
        chunk.copy_from_slice(&v.to_bits().to_le_bytes());
    }
    bytes
}

fn bench_one(label: &str, n: usize, k: usize, reps: usize) -> (f64, f64) {
    let bytes = synth_bf16(n, k);

    let t0 = Instant::now();
    let mut kn = Vec::new();
    for _ in 0..reps {
        kn = transpose_bf16_to_f32_kn(&bytes, n, k);
        std::hint::black_box(kn.len());
    }
    let transpose_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    let t1 = Instant::now();
    for _ in 0..reps {
        let out = quantize_nvfp4(&kn, k, n);
        std::hint::black_box(out.2);
    }
    let quant_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let old_ms = transpose_ms + quant_ms;

    let t2 = Instant::now();
    for _ in 0..reps {
        let out = quantize_nvfp4_from_nk_bf16(&bytes, k, n);
        std::hint::black_box(out.2);
    }
    let fused_ms = t2.elapsed().as_secs_f64() * 1e3 / reps as f64;

    println!(
        "  {label:8} [n={n:5}, k={k:5}]  OLD transpose {transpose_ms:6.2} + quantize {quant_ms:6.2} = {old_ms:7.2} ms   FUSED {fused_ms:6.2} ms   ({:.1}x)",
        old_ms / fused_ms
    );
    (old_ms, fused_ms)
}

/// Isolate the three things the quantizer inner loop does per element, to find the real hotspot
/// instead of guessing: bf16 decode, the `abs`/max block reduction, the divide, and the E2M1 encode.
fn phase_breakdown(reps: usize) {
    let elems = 2 * 1024 * 1024usize;
    let bytes = synth_bf16(elems / 1024, 1024);
    let vals: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let scale = 0.0123f32;
    let inv = 1.0 / scale;

    let time = |label: &str, f: &dyn Fn() -> u64| {
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps {
            acc = acc.wrapping_add(f());
        }
        std::hint::black_box(acc);
        println!(
            "    {label:34} {:7.2} ms  ({:5.2} ns/elem)",
            t.elapsed().as_secs_f64() * 1e3 / reps as f64,
            t.elapsed().as_secs_f64() * 1e9 / (reps * elems) as f64
        );
    };

    println!("\n  phase breakdown over {elems} f32 elements:");
    time("bf16 decode only", &|| {
        let mut a = 0u64;
        for c in bytes.chunks_exact(2) {
            a += half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32() as u64;
        }
        a
    });
    time("abs/max reduction only", &|| {
        let mut m = 0.0f32;
        for &v in &vals {
            m = m.max(v.abs());
        }
        m as u64
    });
    time("e2m1 encode only (no divide)", &|| {
        let mut a = 0u64;
        for &v in &vals {
            a += qwen3_burn::nvfp4::f32_to_e2m1_bits(v) as u64;
        }
        a
    });
    time("DIVIDE + e2m1 encode", &|| {
        let mut a = 0u64;
        for &v in &vals {
            a += qwen3_burn::nvfp4::f32_to_e2m1_bits(v / scale) as u64;
        }
        a
    });
    time("MULTIPLY by reciprocal + e2m1", &|| {
        let mut a = 0u64;
        for &v in &vals {
            a += qwen3_burn::nvfp4::f32_to_e2m1_bits(v * inv) as u64;
        }
        a
    });
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("NVFP4 streamed-expert repack cost (per expert, single-threaded), reps={reps}");
    let (gu_old, gu_new) = bench_one("gate_up", 1024, 2048, reps);
    let (dn_old, dn_new) = bench_one("down", 2048, 512, reps);

    // 40 layers x top-8 experts, all missing the LRU pool (measured hits=0).
    let experts_per_token = 40.0 * 8.0;
    for (label, per_expert_ms) in [("OLD  ", gu_old + dn_old), ("FUSED", gu_new + dn_new)] {
        let per_token_ms = per_expert_ms * experts_per_token;
        println!();
        println!("  {label} per expert (gate_up + down): {per_expert_ms:.2} ms");
        println!(
            "  {label} per token  (40 layers x 8 experts, 0% cache hits): {:.2} s",
            per_token_ms / 1e3
        );
        println!(
            "  {label} implied ceiling from repack alone: {:.3} tok/s",
            1000.0 / per_token_ms
        );
    }

    phase_breakdown(reps);
}
