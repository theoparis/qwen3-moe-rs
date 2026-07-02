//! PHASE 3 of the CUDA-graph plan (docs/cudagraph/DESIGN.md §0b C3 + §6): the device-seed counter
//! RNG, option (c).
//!
//! THE PROBLEM (verified): `cubek-random`'s `random()` drew 4 seeds on the HOST and passed them as
//! `ScalarArg::new(seeds[i])` IMMEDIATES into `prng_kernel`. On grid-constant HW (GB10) those scalars
//! lower into the launch params BY VALUE -> frozen into the kernel node at capture -> every CUDA-graph
//! replay reuses identical noise -> degenerate (frozen) sampling.
//!
//! THE FIX (C3, option (c)): the 4 seeds now live in a small DEVICE buffer (`[N_SEEDS] u32`) that the
//! kernel reads via its binding (a STABLE pointer the captured node bakes), not as immediates. Before
//! each replay the HOST writes FRESH seeds into that SAME buffer (`client.write_to_handle`, on-stream,
//! ordered before the replay), so the replayed kernel re-reads new seeds and DECORRELATES — exactly
//! PyTorch's capturable-Philox model, but with cubek's existing TAUS88+LCG core (no Philox, no offset,
//! no in-graph bump kernel; a fresh seed per draw is statistically identical to eager).
//!
//! This bench drives the capturable entry `cubek_random::random_uniform_with_seeds` on the RAW
//! `CubeBackend<CudaRuntime>` stream (below Fusion). It proves, all on real GB10 hardware:
//!
//!   [1] PARITY  — a captured replay reading seed S == an EAGER draw with the same seed S
//!                 (bit-identical): the device-buffer kernel computes exactly the eager RNG.
//!   [2] FROZEN  — replay WITHOUT rewriting the seed buffer => every replay identical (the pre-C3
//!                 "frozen noise" behavior, reproduced here just by not updating the buffer; proves
//!                 the captured pointer is stable and re-read, not a no-op).
//!   [3] DECORR  — replay WITH a fresh-seed host write each replay => consecutive replays differ in
//!                 ~100% of elements (the C3 fix: captured stochastic kernel decorrelates).
//!   [4] EAGER   — the non-captured `random_uniform_with_seeds` still draws fresh per call and is a
//!                 valid uniform[0,1) (eager behavior unchanged by moving the seed to a device buffer).
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_p3_rng_bench 2>&1 | tail -30

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::Runtime;
use cubek_random::{random_uniform_with_seeds, N_SEEDS};
use rand::{rngs::StdRng, Rng, SeedableRng};

type Client = cubecl::client::ComputeClient<CudaRuntime>;
type Handle = cubecl::server::Handle;

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

fn read_f32(client: &Client, h: &Handle) -> Vec<f32> {
    client
        .read_one(h.clone())
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Serialize `N_SEEDS` u32 seeds into little-endian bytes for the device seed buffer.
fn seed_bytes(seeds: &[u32; N_SEEDS]) -> Vec<u8> {
    let mut b = Vec::with_capacity(N_SEEDS * 4);
    for s in seeds {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}

fn draw_seeds(rng: &mut StdRng) -> [u32; N_SEEDS] {
    let mut s = [0u32; N_SEEDS];
    for x in s.iter_mut() {
        *x = rng.random::<u32>();
    }
    s
}

/// Fraction of element positions whose values differ between two draws (0.0 = identical, ~1.0 =
/// fully decorrelated). Uniform[0,1) draws from independent seeds collide with vanishing probability.
fn frac_differ(a: &[f32], b: &[f32]) -> f64 {
    let d = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    d as f64 / a.len() as f64
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Build a 1-D f32 `TensorHandleRef` over a pre-allocated handle of `n` elements.
fn run_uniform(client: &Client, out: &Handle, n: usize, seed: &Handle) {
    let strides = [1usize];
    let shape = [n];
    let dtype = f32::cube_type();
    // SAFETY: `out` is a pre-allocated contiguous n-f32 buffer; `seed` holds N_SEEDS u32.
    let out_ref = unsafe { TensorHandleRef::from_raw_parts(out, &strides, &shape, 4) };
    random_uniform_with_seeds::<CudaRuntime>(client, 0.0, 1.0, out_ref, dtype, seed)
        .expect("random_uniform_with_seeds launch failed");
}

fn main() {
    let device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (no Fusion)");
    println!("=== PHASE 3 GATE: device-seed counter RNG (C3, option (c)) ===");
    println!("seed buffer: [{N_SEEDS}] u32 in DEVICE memory, read via a stable captured pointer\n");

    let n = 1 << 16; // 64K f32 draws
    let n_replays = 8usize;

    // Persistent buffers, allocated OUTSIDE any captured region (their device addresses are baked into
    // the captured graph; the seed buffer's CONTENTS are what the host rewrites per replay).
    let out = client.empty(n * 4);
    let seed = client.empty(N_SEEDS * 4);
    let out_eager = client.empty(n * 4);

    let mut rng = StdRng::seed_from_u64(0xC3_C3_C3);

    // Warmup: write some seeds + run eagerly once so the kernel JIT-compiles before capture.
    let warm = draw_seeds(&mut rng);
    client.write_to_handle(&seed, &seed_bytes(&warm));
    run_uniform(&client, &out, n, &seed);
    block_sync(&client);

    // -------- CAPTURE the random region once (capture_arena handles the lower/upper scalar +
    // linear_view metadata staging; output + seed are pre-allocated so nothing else allocates). ------
    let graph = unsafe {
        client.capture_arena(2, || {
            run_uniform(&client, &out, n, &seed);
        })
    };
    block_sync(&client);

    let mut all_ok = true;

    // =============================================================================================
    // [1] PARITY — captured replay with seed S == eager (non-captured) draw with the same seed S.
    // =============================================================================================
    let s = draw_seeds(&mut rng);
    // Eager reference into a SEPARATE buffer.
    client.write_to_handle(&seed, &seed_bytes(&s));
    run_uniform(&client, &out_eager, n, &seed); // eager launch, NOT captured
    block_sync(&client);
    let eager_ref = read_f32(&client, &out_eager);
    // Captured replay reading the SAME seed S.
    client.write_to_handle(&seed, &seed_bytes(&s));
    graph.replay();
    block_sync(&client);
    let captured = read_f32(&client, &out);
    let parity = max_abs_diff(&eager_ref, &captured);
    let t1 = parity == 0.0;
    all_ok &= t1;
    println!(
        "[1] parity (captured replay@S  ==  eager draw@S): max_abs_diff = {parity:.3e}  -> {}",
        if t1 { "OK (bit-identical)" } else { "MISMATCH" }
    );

    // =============================================================================================
    // [2] FROZEN — replay WITHOUT rewriting the seed buffer => every replay identical (pre-C3 noise).
    // =============================================================================================
    // (seed buffer still holds S from above; do NOT rewrite it.)
    graph.replay();
    block_sync(&client);
    let frozen0 = read_f32(&client, &out);
    let mut frozen_identical = true;
    for _ in 0..n_replays {
        graph.replay();
        block_sync(&client);
        let f = read_f32(&client, &out);
        if frac_differ(&frozen0, &f) != 0.0 {
            frozen_identical = false;
        }
    }
    let t2 = frozen_identical;
    all_ok &= t2;
    println!(
        "[2] frozen (no seed rewrite, {n_replays} replays): all replays identical = {}  -> {}",
        frozen_identical,
        if t2 {
            "OK (stable pointer; replay re-reads the same buffer == old frozen-immediate behavior)"
        } else {
            "UNEXPECTED (replays differed without a seed rewrite)"
        }
    );

    // =============================================================================================
    // [3] DECORRELATION — fresh-seed host write before each replay => consecutive replays differ.
    // =============================================================================================
    let mut prev: Option<Vec<f32>> = None;
    let mut min_frac = 1.0f64;
    let mut sum_frac = 0.0f64;
    let mut pairs = 0usize;
    let mut draws: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_replays {
        let fresh = draw_seeds(&mut rng);
        client.write_to_handle(&seed, &seed_bytes(&fresh)); // on-stream H2D into the SAME buffer
        graph.replay();
        block_sync(&client);
        let cur = read_f32(&client, &out);
        if let Some(p) = &prev {
            let fr = frac_differ(p, &cur);
            min_frac = min_frac.min(fr);
            sum_frac += fr;
            pairs += 1;
        }
        prev = Some(cur.clone());
        draws.push(cur);
    }
    let avg_frac = sum_frac / pairs.max(1) as f64;
    // Independent seeds -> essentially every element differs. Require a strong margin.
    let t3 = min_frac > 0.99;
    all_ok &= t3;
    println!(
        "[3] decorrelation (fresh seed per replay, {n_replays} replays):"
    );
    println!(
        "      consecutive-replay differing-element fraction: min = {:.4}, avg = {:.4}  -> {}",
        min_frac,
        avg_frac,
        if t3 {
            "DECORRELATED (captured stochastic kernel re-reads fresh seeds)"
        } else {
            "FROZEN/CORRELATED (seed got baked as an immediate somewhere!)"
        }
    );
    // Also confirm NO two distinct-seed replays were globally identical (pairwise, not just adjacent).
    let mut any_pair_identical = false;
    for i in 0..draws.len() {
        for j in (i + 1)..draws.len() {
            if frac_differ(&draws[i], &draws[j]) == 0.0 {
                any_pair_identical = true;
            }
        }
    }
    println!(
        "      any two of the {n_replays} fresh-seed replays globally identical: {any_pair_identical} \
         (expected: false)"
    );
    all_ok &= !any_pair_identical;

    drop(graph); // cuGraphExecDestroy + free the capture arena

    // =============================================================================================
    // [4] EAGER unchanged — non-captured draws are fresh-per-call and a valid uniform[0,1).
    // =============================================================================================
    let sa = draw_seeds(&mut rng);
    client.write_to_handle(&seed, &seed_bytes(&sa));
    run_uniform(&client, &out_eager, n, &seed);
    block_sync(&client);
    let ea = read_f32(&client, &out_eager);

    let sb = draw_seeds(&mut rng);
    client.write_to_handle(&seed, &seed_bytes(&sb));
    run_uniform(&client, &out_eager, n, &seed);
    block_sync(&client);
    let eb = read_f32(&client, &out_eager);

    let eager_differ = frac_differ(&ea, &eb);
    let in_range = ea.iter().chain(eb.iter()).all(|&x| (0.0..1.0).contains(&x));
    let mean: f64 = ea.iter().map(|&x| x as f64).sum::<f64>() / ea.len() as f64;
    let mean_ok = (mean - 0.5).abs() < 0.02;
    let t4 = eager_differ > 0.99 && in_range && mean_ok;
    all_ok &= t4;
    println!(
        "[4] eager unchanged: two distinct-seed draws differ frac = {:.4}, in[0,1) = {}, mean = {:.4} \
         -> {}",
        eager_differ,
        in_range,
        mean,
        if t4 { "OK (fresh per call, uniform)" } else { "BROKEN" }
    );

    println!("\n=== SUMMARY ===");
    println!("  [1] captured replay == eager draw (parity)      : {}", yn(t1));
    println!("  [2] no-rewrite replay is frozen-identical        : {}", yn(t2));
    println!("  [3] fresh-seed replay DECORRELATES (the C3 fix)  : {}", yn(t3 && !any_pair_identical));
    println!("  [4] eager RNG unchanged (fresh per call, uniform): {}", yn(t4));

    assert!(t1, "parity (captured replay vs eager) failed");
    assert!(t2, "frozen-replay invariant failed (pointer not stable?)");
    assert!(t3 && !any_pair_identical, "decorrelation failed — seed still frozen under capture");
    assert!(t4, "eager RNG behavior changed");
    assert!(all_ok);
    println!("\nPHASE 3 (C3 device-seed RNG) GATE: GO — captured Tensor::random DECORRELATES across replays.");
}

fn yn(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}
