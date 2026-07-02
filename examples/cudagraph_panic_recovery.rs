//! FIX 1 regression test (3-voice review hardening of the PHASE-0 CUDA-graph capture FFI).
//!
//! Proves the unwind-safety guard in `ComputeClient::capture`: if the closure passed to `capture`
//! PANICS between `capture_begin` and `capture_end`, the server must NOT wedge. Before the fix, a
//! panic skipped `capture_end`, leaving `capturing == true` forever AND the CUDA stream stuck in
//! capture mode -> every later op on that stream errored. With the fix, `capture` catches the unwind,
//! aborts the capture (pulls the stream out of capture + discards the partial graph), re-raises the
//! panic, and the server stays fully usable.
//!
//! The test:
//!   1. Warm up (compile kernels, populate the pool) with an eager launch + sync.
//!   2. CAPTURE a closure that launches one step then `panic!`s. Catch the unwind. Assert it DID
//!      panic (the guard re-raises) — i.e. the panic is not swallowed.
//!   3. Prove the server is NOT wedged: a subsequent EAGER launch + sync succeeds and reads back a
//!      finite, expected value (the aborted capture never executed, so the buffer is untouched).
//!   4. Prove capture still works: a FRESH capture (no panic) + replay + sync succeeds and the
//!      replayed result matches an eager reference.
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_panic_recovery 2>&1 | tail -20

use std::panic::{AssertUnwindSafe, catch_unwind};

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};

mod kernels {
    use cubecl::prelude::*;

    /// In-place elementwise step `x[i] = x[i] * 0.999 + 0.001`. Only argument is the buffer (constants
    /// baked into PTX), so it is scalar-free and capturable in P0.
    #[cube(launch_unchecked)]
    pub fn fma_step<F: Float>(x: &mut Array<F>) {
        if ABSOLUTE_POS < x.len() {
            x[ABSOLUTE_POS] = x[ABSOLUTE_POS] * F::new(0.999) + F::new(0.001);
        }
    }

    /// Reset the buffer to 0.3 (literal baked into PTX). Eager only.
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
const N: usize = 1024;

fn count(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(THREADS), 1, 1)
}
fn dim() -> CubeDim {
    CubeDim { x: THREADS, y: 1, z: 1 }
}

fn launch_step(client: &Client, handle: &Handle, n: usize) {
    unsafe {
        let arg = ArrayArg::from_raw_parts::<f32>(handle, n, 1);
        kernels::fma_step::launch_unchecked::<f32, CudaRuntime>(client, count(n), dim(), arg)
            .expect("fma_step launch failed");
    }
}

fn reset(client: &Client, handle: &Handle, n: usize) {
    unsafe {
        let arg = ArrayArg::from_raw_parts::<f32>(handle, n, 1);
        kernels::fill::launch_unchecked::<f32, CudaRuntime>(client, count(n), dim(), arg)
            .expect("fill launch failed");
    }
}

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

fn read_buf(client: &Client, handle: &Handle) -> Vec<f32> {
    client
        .read_one(handle.clone())
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn step(v: f32) -> f32 {
    v * 0.999 + 0.001
}

fn main() {
    let device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | FIX 1 panic-recovery test (RAW CubeBackend<CudaRuntime>)\n");

    let x = client.empty(N * core::mem::size_of::<f32>());

    // 1) Warm up: compile kernels + populate the pool BEFORE any capture (warmup is required).
    reset(&client, &x, N);
    launch_step(&client, &x, N);
    block_sync(&client);

    // Put the buffer in a known state (0.3) for the panic phase.
    reset(&client, &x, N);
    block_sync(&client);

    // 2) Capture a closure that panics partway through. The guard must re-raise (so the panic is
    //    observable) AND leave the server usable.
    println!("[1/3] capturing a closure that panics inside the capture region...");
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _g = unsafe {
            client.capture(|| {
                launch_step(&client, &x, N); // recorded, not executed
                panic!("boom: deliberate panic inside the capture closure");
            })
        };
    }));
    assert!(
        panicked.is_err(),
        "the panic inside capture was swallowed; the guard must re-raise it"
    );
    println!("      panic propagated out of capture (guard re-raised it).");

    // 3) Server must NOT be wedged: an eager launch + sync must succeed. The aborted capture never
    //    executed, so x is still 0.3; one eager step -> step(0.3).
    println!("[2/3] eager launch after the panicked capture (proves the stream is not wedged)...");
    launch_step(&client, &x, N);
    block_sync(&client);
    let after = read_buf(&client, &x);
    let expected = step(0.3);
    let max_diff = after
        .iter()
        .map(|v| (v - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "eager launch after panicked capture produced wrong result (max_diff {max_diff:.3e}); \
         buffer={:?}...",
        &after[..4]
    );
    println!("      eager launch succeeded; result == step(0.3) (max_diff {max_diff:.3e}).");

    // 4) Capture must still work: a FRESH capture + replay must succeed and match an eager reference.
    println!("[3/3] fresh capture + replay after the panic (proves capture still works)...");
    const K: usize = 8;

    // Eager reference from a clean 0.3.
    reset(&client, &x, N);
    block_sync(&client);
    for _ in 0..K {
        launch_step(&client, &x, N);
    }
    block_sync(&client);
    let eager_ref = read_buf(&client, &x);

    // Capture the K-step chain from a clean 0.3, then replay once.
    reset(&client, &x, N);
    block_sync(&client);
    let graph = unsafe {
        client.capture(|| {
            for _ in 0..K {
                launch_step(&client, &x, N);
            }
        })
    };
    block_sync(&client);
    graph.replay();
    block_sync(&client);
    let replay_out = read_buf(&client, &x);
    drop(graph);

    let parity = eager_ref
        .iter()
        .zip(replay_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        parity < 1e-6,
        "fresh capture+replay after panic did not match eager (max_diff {parity:.3e})"
    );
    println!("      fresh capture + replay succeeded; replay == eager (max_diff {parity:.3e}).");

    println!(
        "\nPANIC-RECOVERY: PASS — server is USABLE after a panicked capture \
         (eager launch + fresh capture/replay both succeeded; the stream was not wedged)."
    );
}
