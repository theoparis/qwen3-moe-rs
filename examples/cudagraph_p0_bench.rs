//! PHASE 0 of the CUDA-graph plan (docs/cudagraph/DESIGN.md §0b): the synthetic, no-alloc,
//! fixed-buffer kernel-chain microbench that is the STOP/GO launch-overhead gate.
//!
//! This isolates the ONE question CUDA graphs answer: how much per-launch CPU overhead is there to
//! reclaim once the kernels themselves are fixed-shape and host-sync-free? It drives the new capture
//! /replay FFI (component C1) on the RAW `CubeBackend<CudaRuntime>` stream — BELOW Fusion — so the
//! lazy Fusion queue and autotune are never in the path; the captured launch list is determined by
//! code, not by mutable queue state.
//!
//! What it does, per (buffer size, chain length K):
//!   1. Preallocate ONE device buffer `x` (fixed address) — NO allocation inside the captured region.
//!   2. EAGER: fill x = 0.3, then run a chain of K tiny in-place `x = x*0.999 + 0.001` kernels, N
//!      times; wall-clock it.
//!   3. CAPTURE the K-kernel chain ONCE into a graph (records, does not execute), then REPLAY it N
//!      times; wall-clock it.
//!   4. Assert the replayed result is bit-identical to the eager result — proves replay actually
//!      recomputes on the fixed buffer (not a no-op empty graph).
//!
//! It sweeps K at TWO buffer sizes so we see both regimes honestly:
//!   - SMALL (launch-overhead-bound): tiny kernels, host launch cost dominates -> graphs help a lot.
//!   - LARGE (bandwidth-bound): each kernel streams MBs, GPU dominates -> graphs barely help (~1x).
//! The GRPO decode step is bandwidth-bound (the tied-head logits GEMM), so the LARGE column is the
//! honest predictor for it; the SMALL column is the launch-overhead ceiling.
//!
//! THE GATE: report the MAX replay-vs-eager speedup across K and whether it clears ~1.15x. If even a
//! pure many-tiny-kernel chain can't clear ~1.15x after Fusion already collapses launches, the
//! bandwidth-bound GRPO-decode capture never will -> STOP.
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_p0_bench 2>&1 | tail -40

use std::time::Instant;

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};

mod kernels {
    use cubecl::prelude::*;

    /// In-place elementwise step: `x[i] = x[i] * 0.999 + 0.001`. Constants are baked into the PTX
    /// (comptime), so the ONLY kernel argument is the array buffer — no per-launch scalar H2D copy
    /// (which would poison capture by baking a freed host pointer into a graph memcpy node).
    #[cube(launch_unchecked)]
    pub fn fma_step<F: Float>(x: &mut Array<F>) {
        if ABSOLUTE_POS < x.len() {
            x[ABSOLUTE_POS] = x[ABSOLUTE_POS] * F::new(0.999) + F::new(0.001);
        }
    }

    /// Reset the fixed buffer to a literal constant (baked into the PTX). Launched EAGER (never
    /// captured); no scalar arg, so we avoid `F: ScalarArgSettings`.
    #[cube(launch_unchecked)]
    pub fn fill<F: Float>(x: &mut Array<F>) {
        if ABSOLUTE_POS < x.len() {
            x[ABSOLUTE_POS] = F::new(0.3);
        }
    }
}

type Client = cubecl::client::ComputeClient<CudaRuntime>;
type Handle = cubecl::server::Handle;

const THREADS: u32 = 256;

fn count(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(THREADS), 1, 1)
}

fn dim() -> CubeDim {
    CubeDim {
        x: THREADS,
        y: 1,
        z: 1,
    }
}

/// One in-place FMA step on the fixed buffer (eager launch through the normal client path).
fn launch_step(client: &Client, handle: &Handle, n: usize) {
    unsafe {
        let arg = ArrayArg::from_raw_parts::<f32>(handle, n, 1);
        kernels::fma_step::launch_unchecked::<f32, CudaRuntime>(client, count(n), dim(), arg)
            .expect("fma_step launch failed");
    }
}

/// Reset the fixed buffer to 0.3 (eager).
fn reset(client: &Client, handle: &Handle, n: usize) {
    unsafe {
        let arg = ArrayArg::from_raw_parts::<f32>(handle, n, 1);
        kernels::fill::launch_unchecked::<f32, CudaRuntime>(client, count(n), dim(), arg)
            .expect("fill launch failed");
    }
}

/// Block the host until ALL work on the stream has completed. `client.sync()` only RECORDS a fence
/// and returns a future; it must be driven to completion (`block_on`) to actually wait — otherwise we
/// would time host launch-enqueue only, never GPU completion.
fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

fn read_buf(client: &Client, handle: &Handle) -> Vec<f32> {
    let bytes = client.read_one(handle.clone());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Sweep chain length K for a fixed buffer of `n` f32 elements. Returns (max_speedup, any_parity_fail).
fn sweep(client: &Client, label: &str, n: usize, n_iter: usize, ks: &[usize]) -> (f32, bool) {
    // The ONE fixed buffer for this sweep. Its device address is baked into the captured graph and
    // must never move (allocated ONCE, outside any captured region).
    let x = client.empty(n * core::mem::size_of::<f32>());

    // Warm up: compile both kernels + populate the memory pool BEFORE any capture/timing.
    reset(client, &x, n);
    launch_step(client, &x, n);
    block_sync(client);

    println!(
        "{label}: {n} f32 ({} KB/buffer), {} block(s) x {THREADS} threads, N={n_iter}",
        n * 4 / 1024,
        (n as u32).div_ceil(THREADS),
    );
    println!(
        "{:>4}  {:>12}  {:>12}  {:>9}  {:>8}",
        "K", "eager us/it", "replay us/it", "speedup", "parity"
    );
    println!("{}", "-".repeat(56));

    let mut max_speedup = 0.0f32;
    let mut any_fail = false;

    for &k in ks {
        // -------- EAGER: N applications of the K-kernel chain --------
        reset(client, &x, n);
        block_sync(client);
        let t0 = Instant::now();
        for _ in 0..n_iter {
            for _ in 0..k {
                launch_step(client, &x, n);
            }
        }
        block_sync(client);
        let eager = t0.elapsed().as_secs_f64();
        let eager_out = read_buf(client, &x);

        // -------- CAPTURE the K-kernel chain ONCE (records, does NOT execute) --------
        reset(client, &x, n);
        block_sync(client);
        let graph = unsafe {
            client.capture(|| {
                for _ in 0..k {
                    launch_step(client, &x, n);
                }
            })
        };
        block_sync(client);

        // -------- REPLAY: x is still 0.3 (capture didn't run), so N replays == N eager chains -----
        let t0 = Instant::now();
        for _ in 0..n_iter {
            graph.replay();
        }
        block_sync(client);
        let replay = t0.elapsed().as_secs_f64();
        let replay_out = read_buf(client, &x);

        drop(graph); // exercises graph_destroy (cuGraphExecDestroy + cuGraphDestroy)

        let parity_ok = max_abs_diff(&eager_out, &replay_out) < 1e-6;
        any_fail |= !parity_ok;

        let speedup = (eager / replay) as f32;
        if parity_ok && speedup > max_speedup {
            max_speedup = speedup;
        }

        println!(
            "{:>4}  {:>12.2}  {:>12.2}  {:>8.3}x  {:>8}",
            k,
            eager / n_iter as f64 * 1e6,
            replay / n_iter as f64 * 1e6,
            speedup,
            if parity_ok { "OK" } else { "MISMATCH" }
        );
    }
    println!();
    (max_speedup, any_fail)
}

fn main() {
    let device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (no Fusion)\n");

    let ks: [usize; 6] = [1, 4, 8, 16, 32, 50];

    // SMALL: 1024 f32 (4 KB) tiny kernels -> launch-overhead-bound (graphs help most).
    let (small_max, small_fail) = sweep(&client, "SMALL (launch-bound)", 1024, 500, &ks);
    // LARGE: 4M f32 (16 MB) -> each kernel streams 32 MB -> bandwidth-bound (the GRPO-decode regime).
    let (large_max, large_fail) = sweep(&client, "LARGE (bandwidth-bound)", 4 << 20, 50, &ks);

    let any_fail = small_fail || large_fail;
    let max_speedup = small_max.max(large_max);

    println!("=== PHASE 0 GATE ===");
    if any_fail {
        println!(
            "CAPTURE+REPLAY: BROKEN — at least one config replayed to a different result than eager. \
             The captured graph is not faithfully recomputing on the fixed buffer."
        );
    } else {
        println!(
            "capture+replay WORKS: every config replayed bit-identical to eager (max_abs_diff < 1e-6)."
        );
    }
    println!(
        "MAX replay-vs-eager speedup: {max_speedup:.3}x  (small/launch-bound {small_max:.2}x, \
         large/bandwidth-bound {large_max:.2}x)"
    );
    if max_speedup >= 1.15 {
        println!(
            "GATE: GO ({max_speedup:.3}x >= 1.15x) — there IS reclaimable per-launch overhead below \
             Fusion, so capture+replay is worth building (P1 capture arena next)."
        );
    } else {
        println!(
            "GATE: STOP ({max_speedup:.3}x < 1.15x) — even a pure many-tiny-kernel chain can't clear \
             1.15x; the bandwidth-bound GRPO-decode capture never will."
        );
    }
    println!(
        "HONEST NOTE: the small-buffer number is the launch-overhead CEILING (tiny kernels, pure host \
         cost). The large-buffer number (~{large_max:.2}x) is the honest predictor for the \
         bandwidth-bound GRPO decode step — graphs remove host launch latency, not HBM traffic."
    );

    assert!(
        !any_fail,
        "capture/replay correctness gate failed (replay != eager)"
    );
}
