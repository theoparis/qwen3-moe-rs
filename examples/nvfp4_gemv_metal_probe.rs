//! How fast is ONE streamed expert's NVFP4 decode actually, on the GPU, at decode shapes?
//!
//! The streamed decode profile attributes ~22% of wall clock to `MOE_EXPERTS`, i.e. ~4.9 ms per
//! routed expert. The math is ~3M MACs (~6 MFLOP), which should be microseconds. This probe
//! separates the three candidate explanations:
//!   * real GPU kernel time            -> "pipelined" (many launches, ONE sync at the end)
//!   * per-launch/sync round-trip cost -> "sync each" minus "pipelined"
//!   * the surrounding elementwise ops -> "full expert" minus the two bare GEMVs
//!
//! Run:
//!   cargo run --release --features metal --example nvfp4_gemv_probe -- --device metal

use std::time::Instant;

use burn::prelude::Device;
use burn::tensor::activation::silu;
use burn::tensor::{DType, DeviceKind, Tensor};
use qwen3_burn::nvfp4::quantize_nvfp4;
use qwen3_burn::nvfp4_linear::Nvfp4Linear;

fn make_linear(k: usize, n: usize, device: &Device) -> Nvfp4Linear {
    let mut state = 0x2468_ACE1u32;
    let mut w = vec![0.0f32; k * n];
    for v in w.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *v = ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    let (qw, bs, gscale) = quantize_nvfp4(&w, k, n);
    Nvfp4Linear::from_packed_parts(qw, bs, gscale, k, n, device).with_m_max(8)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device = match args
        .iter()
        .position(|x| x == "--device")
        .map(|i| args[i + 1].as_str())
    {
        Some("metal") => Device::metal(DeviceKind::DefaultDevice),
        Some("wgpu") => Device::wgpu(DeviceKind::DefaultDevice),
        _ => Device::flex(),
    };
    println!("device: {device:?}");

    // Real Qwen3.6-35B-A3B routed-expert shapes.
    let hidden = 2048usize;
    let inner = 512usize;
    let gate_up = make_linear(hidden, 2 * inner, &device); // K=2048 -> N=1024
    let down = make_linear(inner, hidden, &device); // K=512  -> N=2048

    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let x = Tensor::<2>::zeros([1, hidden], &device).cast(DType::F32);

    // Warm up: force shader compile / autotune / allocator priming out of the measurement.
    for _ in 0..10 {
        let gu = gate_up.forward(x.clone());
        let m = gu.dims()[0];
        let g = silu(gu.clone().slice([0..m, 0..inner]));
        let u = gu.slice([0..m, inner..inner * 2]);
        let _ = down.forward(g * u);
    }
    let _ = device.sync();

    let bench = |label: &str, sync_each: bool, f: &dyn Fn()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
            if sync_each {
                let _ = device.sync();
            }
        }
        let _ = device.sync();
        let per = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("  {label:38} {per:7.3} ms/call");
        per
    };

    println!("\nM=1 decode shapes, reps={reps}");
    let gu_pipe = bench("gate_up GEMV, pipelined", false, &|| {
        std::hint::black_box(gate_up.forward(x.clone()));
    });
    let gu_sync = bench("gate_up GEMV, sync each call", true, &|| {
        std::hint::black_box(gate_up.forward(x.clone()));
    });

    let full = |sync_each: bool| {
        let label = if sync_each {
            "full expert (gu+silu+mul+down), sync each"
        } else {
            "full expert (gu+silu+mul+down), pipelined"
        };
        bench(label, sync_each, &|| {
            let gu = gate_up.forward(x.clone());
            let m = gu.dims()[0];
            let g = silu(gu.clone().slice([0..m, 0..inner]));
            let u = gu.slice([0..m, inner..inner * 2]);
            std::hint::black_box(down.forward(g * u));
        })
    };
    let full_pipe = full(false);
    let full_sync = full(true);

    // Does dispatch cost scale with the number of LIVE GPU buffers?
    //
    // The streamed pool keeps `capacity` NVFP4 experts resident as thousands of individually
    // allocated small buffers. In the real run, growing the pool 64 -> 4096 slots made the pool's
    // own `compute` counter go 11.0s -> 30.9s for the *identical* set of expert GEMVs. If wgpu's
    // per-dispatch resource tracking / allocator degrades with live-buffer count, that explains
    // why this probe (which reuses 2 weights) is ~10x faster per expert than production.
    let live: usize = std::env::var("LIVE_BUFFERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if live > 0 {
        println!("\n  allocating {live} extra live expert weights to mimic pool residency ...");
        let mut ballast: Vec<Nvfp4Linear> = Vec::with_capacity(live);
        for i in 0..live {
            // Alternate the two real shapes, as the pool does (gate_up + down per expert).
            ballast.push(if i % 2 == 0 {
                make_linear(hidden, 2 * inner, &device)
            } else {
                make_linear(inner, hidden, &device)
            });
        }
        let _ = device.sync();
        println!("  {} live weight tensors resident", ballast.len());
        let full_pipe_loaded = bench("full expert, pipelined (loaded)", false, &|| {
            let gu = gate_up.forward(x.clone());
            let m = gu.dims()[0];
            let g = silu(gu.clone().slice([0..m, 0..inner]));
            let u = gu.slice([0..m, inner..inner * 2]);
            std::hint::black_box(down.forward(g * u));
        });
        println!(
            "  slowdown from {live} live buffers : {:.2}x  ({:.3} -> {:.3} ms/call)",
            full_pipe_loaded / full_pipe,
            full_pipe,
            full_pipe_loaded
        );
        // The ballast above is resident but never DISPATCHED against. Production differs: every
        // routed expert is a freshly uploaded buffer that gets used exactly once, so any per-buffer
        // first-use cost (bind group creation, lazy init) is paid ~6,500 times per run instead of
        // being amortized. Cycle through distinct weights, one dispatch each, to measure that.
        let pairs = live / 2;
        let cold: Vec<(Nvfp4Linear, Nvfp4Linear)> = (0..pairs)
            .map(|_| {
                (
                    make_linear(hidden, 2 * inner, &device),
                    make_linear(inner, hidden, &device),
                )
            })
            .collect();
        let _ = device.sync();

        let t = Instant::now();
        for (gu_w, dn_w) in &cold {
            let gu = gu_w.forward(x.clone());
            let m = gu.dims()[0];
            let g = silu(gu.clone().slice([0..m, 0..inner]));
            let u = gu.slice([0..m, inner..inner * 2]);
            std::hint::black_box(dn_w.forward(g * u));
        }
        let _ = device.sync();
        let cold_ms = t.elapsed().as_secs_f64() * 1e3 / pairs as f64;
        println!(
            "  full expert, each weight used ONCE     {cold_ms:7.3} ms/call  ({:.2}x vs hot reuse)",
            cold_ms / full_pipe
        );

        // Second pass over the SAME weights: now they are warm, so the delta isolates first-use cost.
        let t = Instant::now();
        for (gu_w, dn_w) in &cold {
            let gu = gu_w.forward(x.clone());
            let m = gu.dims()[0];
            let g = silu(gu.clone().slice([0..m, 0..inner]));
            let u = gu.slice([0..m, inner..inner * 2]);
            std::hint::black_box(dn_w.forward(g * u));
        }
        let _ = device.sync();
        let warm_ms = t.elapsed().as_secs_f64() * 1e3 / pairs as f64;
        println!(
            "  ... same weights, second pass (warm)   {warm_ms:7.3} ms/call  -> first-use cost {:.3} ms",
            cold_ms - warm_ms
        );

        std::hint::black_box((&ballast, &cold));
    }

    println!(
        "\n  per-launch sync round-trip  : {:.3} ms",
        gu_sync - gu_pipe
    );
    println!(
        "  elementwise + down overhead : {:.3} ms",
        full_pipe - gu_pipe
    );
    println!(
        "  extrapolated decode cost    : {:.1} s  ({} experts/token x 40 layers x 17 steps, pipelined)",
        full_pipe * 8.0 * 40.0 * 17.0 / 1e3,
        8
    );
    println!(
        "  ... if every expert syncs   : {:.1} s",
        full_sync * 8.0 * 40.0 * 17.0 / 1e3
    );
}
