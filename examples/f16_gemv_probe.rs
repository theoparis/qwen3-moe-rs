//! Prices a single f16 GEMV at real resident-core shapes (GDN in_proj_qkv [2048->8192],
//! out_proj [8192->2048], full-attn q/k/v/o, shared-expert gate/up/down) through burn's
//! matmul path exactly as `linear3` uses it, so we can attribute the ~950 ms/token resident
//! cost to matmul dispatch vs everything else.
//!
//! Weights are pre-cast to f16 ONCE (matching the new load-time precast), so the loop prices
//! only the matmul, not a per-call weight cast.
//!
//! Run: cargo run --release --features metal --example f16_gemv_probe -- --device metal

use std::time::Instant;

use burn::prelude::Device;
use burn::tensor::{DType, Tensor};

fn main() {
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let x = Tensor::<2>::ones([1, 2048], &device).cast(DType::F16);
    let x8192 = Tensor::<2>::ones([1, 8192], &device).cast(DType::F16);
    let x512 = Tensor::<2>::ones([1, 512], &device).cast(DType::F16);

    // Warmup (autotune/shaders)
    for _ in 0..3 {
        let w = Tensor::<2>::zeros([2048, 8192], &device).cast(DType::F16);
        let _ = x.clone().matmul(w).into_data();
    }

    // (name, input-dim, output-dim, calls-per-token)
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("GDN in_proj_qkv 2048->8192", 2048, 8192, 30),
        ("GDN in_proj_z   2048->2048", 2048, 2048, 30),
        ("GDN in_proj_a   2048->32", 2048, 32, 30),
        ("GDN in_proj_b   2048->32", 2048, 32, 30),
        ("GDN out_proj    8192->2048", 8192, 2048, 30),
        ("full q/k/v/o    2048->2048", 2048, 2048, 40), // 4 proj x 10 layers
        ("shared gate     2048->256", 2048, 256, 30),
        ("shared gate/up  2048->512", 2048, 512, 60), // x2 x 30 layers
        ("shared down      512->2048", 512, 2048, 30),
    ];

    let mut total_pipelined = 0.0f64;
    let mut total_synced = 0.0f64;
    for &(name, k, n, calls_per_token) in shapes {
        // Pre-cast weight once, exactly like the model's load-time precast.
        let w = Tensor::<2>::zeros([k, n], &device).cast(DType::F16);
        let x_in = match k {
            8192 => &x8192,
            512 => &x512,
            _ => &x,
        };
        let reps = 30;

        // Three rounds: round 1 may pay one-time costs (autotune, shader compile);
        // rounds 2-3 show steady state.
        let mut rounds = [0.0f64; 3];
        for r in 0..3 {
            let start = Instant::now();
            let mut last = Tensor::<2>::zeros([1, 1], &device);
            for _ in 0..reps {
                last = x_in.clone().matmul(w.clone()) + last;
            }
            let _ = last.into_data();
            rounds[r] = start.elapsed().as_secs_f64() / reps as f64;
        }
        let pipelined = rounds[2];

        let start = Instant::now();
        let mut last = Tensor::<2>::zeros([1, 1], &device);
        for _ in 0..reps {
            last = x_in.clone().matmul(w.clone()) + last;
            let _ = last.clone().into_data();
        }
        let synced = start.elapsed().as_secs_f64() / reps as f64;

        println!(
            "{name:34} r1 {:8.3} r2 {:8.3} r3 {:8.3} ms   sync-each {:8.3} ms   x{calls_per_token}/tok = {:6.1} ms/tok (r3)",
            p3(rounds[0]),
            p3(rounds[1]),
            p3(rounds[2]),
            p3(synced),
            pipelined * 1e3 * calls_per_token as f64
        );
        total_pipelined += pipelined * calls_per_token as f64;
        total_synced += synced * calls_per_token as f64;
    }
    println!(
        "\nAll resident Linears per token: pipelined {:7.1} ms   sync-each {:7.1} ms",
        total_pipelined * 1e3,
        total_synced * 1e3
    );
}

fn p3(v: f64) -> f64 {
    (v * 1e3 * 1000.0).round() / 1000.0
}
