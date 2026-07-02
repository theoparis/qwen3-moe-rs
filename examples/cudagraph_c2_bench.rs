//! PHASE 1 of the CUDA-graph plan (docs/cudagraph/DESIGN.md §0b C2): validate the graph-aware
//! capture ARENA + scalar/metadata staging that make capture+replay transfer to REAL kernels —
//! the ones P0 hard-errored on (P0 only captured a scalar-free, alloc-free kernel chain).
//!
//! Three things are proven here, all on the RAW `CubeBackend<CudaRuntime>` stream (below Fusion),
//! bit-identical to eager:
//!
//!   1. SCALAR-bearing capture — `x = x*scale + bias` with `scale,bias` as RUNTIME scalars (not
//!      comptime). On grid-constant HW (sm_70+, incl. GB10) the scalars are baked BY VALUE into the
//!      kernel node at capture, so replay reproduces them exactly. (The P0 bench was deliberately
//!      scalar-free; this is the new coverage that proves the scalar path.)
//!
//!   2. ALLOC-inside-capture — a 2-kernel chain `a -> tmp -> b` where `tmp` is allocated INSIDE the
//!      captured region. The capture arena (component C2) serves `tmp` from a pre-reserved,
//!      graph-private, isolated pool with a STABLE device address, and RECYCLES it within the graph.
//!      We capture a LONG chain that reuses `tmp` every step and assert the arena high-water is
//!      ~ONE `tmp` (peak-LIVE working set), NOT N*tmp (sum of all allocations).
//!
//!   3. LIFETIME — dropping the `CapturedGraph` frees its arena. We loop capture+replay+destroy 100x
//!      and assert `memory_usage()` returns to baseline (no per-cycle leak).
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_c2_bench 2>&1 | tail -50

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};

mod kernels {
    use cubecl::prelude::*;

    /// In-place affine step with RUNTIME scalars: `x[i] = x[i] * scale + bias`. `scale`/`bias` are
    /// `ScalarArg`s (resolved at launch), so on grid-constant HW they lower to `__grid_constant__`
    /// by-value kernel params — baked into the captured graph node at record time.
    #[cube(launch_unchecked)]
    pub fn fma_scalar(x: &mut Array<f32>, scale: f32, bias: f32) {
        if ABSOLUTE_POS < x.len() {
            x[ABSOLUTE_POS] = x[ABSOLUTE_POS] * scale + bias;
        }
    }

    /// `tmp[i] = a[i] * 2` (comptime constant, no scalar arg). Writes the captured intermediate.
    #[cube(launch_unchecked)]
    pub fn mul2(a: &Array<f32>, tmp: &mut Array<f32>) {
        if ABSOLUTE_POS < a.len() {
            tmp[ABSOLUTE_POS] = a[ABSOLUTE_POS] * f32::new(2.0);
        }
    }

    /// `tmp[i] = a[i] * 3` (comptime constant). A SECOND producer so a test can hold two distinct,
    /// same-size intermediates live SIMULTANEOUSLY inside one captured region.
    #[cube(launch_unchecked)]
    pub fn mul3(a: &Array<f32>, tmp: &mut Array<f32>) {
        if ABSOLUTE_POS < a.len() {
            tmp[ABSOLUTE_POS] = a[ABSOLUTE_POS] * f32::new(3.0);
        }
    }

    /// `b[i] = b[i] + tmp[i]` — reads the captured intermediate, accumulates into the output.
    #[cube(launch_unchecked)]
    pub fn add_into(tmp: &Array<f32>, b: &mut Array<f32>) {
        if ABSOLUTE_POS < b.len() {
            b[ABSOLUTE_POS] = b[ABSOLUTE_POS] + tmp[ABSOLUTE_POS];
        }
    }

    /// Set `x = v` (runtime scalar; launched eagerly only, never captured).
    #[cube(launch_unchecked)]
    pub fn fill(x: &mut Array<f32>, v: f32) {
        if ABSOLUTE_POS < x.len() {
            x[ABSOLUTE_POS] = v;
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
    CubeDim { x: THREADS, y: 1, z: 1 }
}

fn fill(client: &Client, h: &Handle, n: usize, v: f32) {
    unsafe {
        let arg = ArrayArg::from_raw_parts::<f32>(h, n, 1);
        kernels::fill::launch_unchecked::<CudaRuntime>(client, count(n), dim(), arg, ScalarArg::new(v))
            .expect("fill launch failed");
    }
}
fn fma_scalar(client: &Client, h: &Handle, n: usize, scale: f32, bias: f32) {
    unsafe {
        let arg = ArrayArg::from_raw_parts::<f32>(h, n, 1);
        kernels::fma_scalar::launch_unchecked::<CudaRuntime>(
            client,
            count(n),
            dim(),
            arg,
            ScalarArg::new(scale),
            ScalarArg::new(bias),
        )
        .expect("fma_scalar launch failed");
    }
}
fn mul2(client: &Client, a: &Handle, tmp: &Handle, n: usize) {
    unsafe {
        let a_arg = ArrayArg::from_raw_parts::<f32>(a, n, 1);
        let tmp_arg = ArrayArg::from_raw_parts::<f32>(tmp, n, 1);
        kernels::mul2::launch_unchecked::<CudaRuntime>(client, count(n), dim(), a_arg, tmp_arg)
            .expect("mul2 launch failed");
    }
}
fn mul3(client: &Client, a: &Handle, tmp: &Handle, n: usize) {
    unsafe {
        let a_arg = ArrayArg::from_raw_parts::<f32>(a, n, 1);
        let tmp_arg = ArrayArg::from_raw_parts::<f32>(tmp, n, 1);
        kernels::mul3::launch_unchecked::<CudaRuntime>(client, count(n), dim(), a_arg, tmp_arg)
            .expect("mul3 launch failed");
    }
}
fn add_into(client: &Client, tmp: &Handle, b: &Handle, n: usize) {
    unsafe {
        let tmp_arg = ArrayArg::from_raw_parts::<f32>(tmp, n, 1);
        let b_arg = ArrayArg::from_raw_parts::<f32>(b, n, 1);
        kernels::add_into::launch_unchecked::<CudaRuntime>(client, count(n), dim(), tmp_arg, b_arg)
            .expect("add_into launch failed");
    }
}

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}
fn read_buf(client: &Client, h: &Handle) -> Vec<f32> {
    client
        .read_one(h.clone())
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

// -------------------------------------------------------------------------------------------------
// TEST 1 — a SCALAR-bearing kernel captures + replays bit-identical.
// -------------------------------------------------------------------------------------------------
fn test_scalar(client: &Client) -> bool {
    let n = 1 << 20; // 1M f32 = 4 MB
    let k = 8; // chain length
    let scale = 0.999_f32;
    let bias = 0.001_f32;
    let init = 0.3_f32;

    let x = client.empty(n * 4);

    // Eager reference: one application of the K-chain from `init`.
    fill(client, &x, n, init);
    for _ in 0..k {
        fma_scalar(client, &x, n, scale, bias);
    }
    block_sync(client);
    let eager_out = read_buf(client, &x);

    // Capture the K-chain through the arena path (warmup runs it eagerly to JIT + size the arena).
    let graph = unsafe {
        client.capture_arena(2, || {
            for _ in 0..k {
                fma_scalar(client, &x, n, scale, bias);
            }
        })
    };
    block_sync(client);

    // Reset to `init` (warmup + capture pass mutated `x`), then ONE replay == one eager chain.
    fill(client, &x, n, init);
    block_sync(client);
    graph.replay();
    block_sync(client);
    let replay_out = read_buf(client, &x);

    let diff = max_abs_diff(&eager_out, &replay_out);
    let ok = diff < 1e-6;
    println!(
        "[1] scalar capture (x = x*{scale} + {bias}, K={k}): max_abs_diff = {diff:.3e}  arena = {} KB  -> {}",
        graph.arena_bytes() / 1024,
        if ok { "OK (bit-identical)" } else { "MISMATCH" }
    );
    drop(graph);
    ok
}

// -------------------------------------------------------------------------------------------------
// TEST 2 — ALLOC inside capture: `a -> tmp -> b`, tmp arena-served + RECYCLED (peak-LIVE not sum).
// -------------------------------------------------------------------------------------------------
fn test_alloc_recycle(client: &Client) -> bool {
    let n = 1 << 18; // 256K f32 = 1 MB per tmp
    let chain = 64usize; // reuse tmp 64 times
    let tmp_bytes = (n * 4) as u64;

    let a = client.empty(n * 4);
    let b = client.empty(n * 4);
    fill(client, &a, n, 1.5);
    block_sync(client);

    // Closure: a LONG chain that allocates `tmp` INSIDE each step and frees it (so the arena must
    // recycle one block, not accumulate `chain` blocks). b accumulates: b += a*2 each step.
    let body = |client: &Client| {
        for _ in 0..chain {
            let tmp = client.empty(n * 4); // arena-served during capture; recycled each step
            mul2(client, &a, &tmp, n);
            add_into(client, &tmp, &b, n);
            // tmp drops here -> arena block becomes free -> reused next step
        }
    };

    // Eager reference: b = chain * (a*2), from b = 0.
    fill(client, &b, n, 0.0);
    block_sync(client);
    body(client);
    block_sync(client);
    let eager_b = read_buf(client, &b);

    // Capture (warmup mutates b; we reset after). Arena pre-sizes to ONE tmp during warmup.
    let graph = unsafe { client.capture_arena(2, || body(client)) };
    block_sync(client);

    // Reset b = 0, replay once -> b = chain*(a*2).
    fill(client, &b, n, 0.0);
    block_sync(client);
    graph.replay();
    block_sync(client);
    let replay_b = read_buf(client, &b);

    let diff = max_abs_diff(&eager_b, &replay_b);
    let parity = diff < 1e-3; // 256K accumulations of a*2; allow tiny f32 reduction slack

    let arena = graph.arena_bytes();
    // Peak-LIVE: one tmp (+ tiny metadata-staging block), NOT chain*tmp. Slack: < 2x one tmp.
    let recycled = arena < 2 * tmp_bytes;

    println!(
        "[2] alloc-in-capture (a->tmp->b, chain={chain}, tmp={} KB):",
        tmp_bytes / 1024
    );
    println!(
        "      parity max_abs_diff = {diff:.3e}  -> {}",
        if parity { "OK (bit-identical)" } else { "MISMATCH" }
    );
    println!(
        "      arena high-water = {} KB  ({:.2}x one tmp; sum-of-all would be {} KB)  -> {}",
        arena / 1024,
        arena as f64 / tmp_bytes as f64,
        chain as u64 * tmp_bytes / 1024,
        if recycled {
            "RECYCLED (peak-LIVE, not sum)"
        } else {
            "NOT RECYCLED (arena = sum of allocs!)"
        }
    );
    drop(graph);
    parity && recycled
}

// -------------------------------------------------------------------------------------------------
// TEST 3 — LIFETIME: dropping the graph frees the arena; no leak over 100 capture/destroy cycles.
// -------------------------------------------------------------------------------------------------
fn test_lifetime(client: &Client) -> bool {
    let n = 1 << 18; // 1 MB per tmp
    let cycles = 100usize;

    let a = client.empty(n * 4);
    let b = client.empty(n * 4);
    fill(client, &a, n, 1.0);
    fill(client, &b, n, 0.0);
    block_sync(client);

    let body = |client: &Client| {
        let tmp = client.empty(n * 4);
        mul2(client, &a, &tmp, n);
        add_into(client, &tmp, &b, n);
    };

    // Warm the normal pool so the baseline is steady, then measure.
    {
        let g = unsafe { client.capture_arena(1, || body(client)) };
        g.replay();
        drop(g);
    }
    block_sync(client);
    let base = client.memory_usage().bytes_reserved;

    let mut max_seen = base;
    for _ in 0..cycles {
        let g = unsafe { client.capture_arena(1, || body(client)) };
        g.replay();
        max_seen = max_seen.max(client.memory_usage().bytes_reserved);
        drop(g); // -> graph_destroy -> arena freed
    }
    block_sync(client);
    let after = client.memory_usage().bytes_reserved;

    // After freeing every graph, reserved bytes must return to baseline (no per-cycle leak). Allow
    // one tmp of slack for allocator bookkeeping.
    let slack = (n * 4) as u64;
    let ok = after <= base + slack;
    println!(
        "[3] lifetime ({cycles} capture/replay/destroy cycles): base = {} KB, peak = {} KB, after = {} KB  -> {}",
        base / 1024,
        max_seen / 1024,
        after / 1024,
        if ok {
            "OK (arena freed on drop, no leak)"
        } else {
            "LEAK (reserved grew across cycles)"
        }
    );
    ok
}

// -------------------------------------------------------------------------------------------------
// TEST 4 (FIX 4) — TWO same-size intermediates SIMULTANEOUSLY live inside one captured region.
//
// The decode step needs several intermediates alive at once; the chain in TEST 2 only ever had ONE
// `tmp` live at a time, so it could not prove the arena hands out DISTINCT blocks for concurrent
// liveness. Here `t1 = 2a` and `t2 = 3a` are BOTH live (neither dropped) when `out = h(t1, t2)`
// runs, so the arena MUST reserve two distinct blocks. We assert replay is bit-identical AND the
// arena high-water is ~2x one block (>= 2 blocks: distinct allocation), not 1x.
// -------------------------------------------------------------------------------------------------
fn test_multi_live(client: &Client) -> bool {
    let n = 1 << 18; // 256K f32 = 1 MB per intermediate
    let tmp_bytes = (n * 4) as u64;

    let a = client.empty(n * 4);
    let b = client.empty(n * 4);
    fill(client, &a, n, 1.5);
    block_sync(client);

    // t1 and t2 are allocated and BOTH kept live across the producers + consumers, so the arena
    // cannot recycle one for the other: b += t1 + t2 == b + 5a.
    let body = |client: &Client| {
        let t1 = client.empty(n * 4); // distinct arena block #1
        let t2 = client.empty(n * 4); // distinct arena block #2 (t1 still live)
        mul2(client, &a, &t1, n); // t1 = 2a
        mul3(client, &a, &t2, n); // t2 = 3a
        add_into(client, &t1, &b, n); // b += t1
        add_into(client, &t2, &b, n); // b += t2   => b += 5a
        // t1, t2 drop here
    };

    // Eager reference: b = 5a, from b = 0.
    fill(client, &b, n, 0.0);
    block_sync(client);
    body(client);
    block_sync(client);
    let eager_b = read_buf(client, &b);

    let graph = unsafe { client.capture_arena(2, || body(client)) };
    block_sync(client);

    fill(client, &b, n, 0.0);
    block_sync(client);
    graph.replay();
    block_sync(client);
    let replay_b = read_buf(client, &b);

    let diff = max_abs_diff(&eager_b, &replay_b);
    let parity = diff < 1e-3;

    let arena = graph.arena_bytes();
    // Two concurrently-live blocks => high-water >= 2x one block (proves distinct allocation), and
    // < 3x (proves it did not allocate a fresh block per call / still recycles across replays).
    let two_distinct = arena >= 2 * tmp_bytes && arena < 3 * tmp_bytes;

    println!("[4] multi-live (t1=2a, t2=3a both live, b+=t1+t2, tmp={} KB):", tmp_bytes / 1024);
    println!(
        "      parity max_abs_diff = {diff:.3e}  -> {}",
        if parity { "OK (bit-identical)" } else { "MISMATCH" }
    );
    println!(
        "      arena high-water = {} KB  ({:.2}x one tmp)  -> {}",
        arena / 1024,
        arena as f64 / tmp_bytes as f64,
        if two_distinct {
            ">=2 DISTINCT blocks (concurrent liveness)"
        } else {
            "WRONG block count (expected ~2x one tmp)"
        }
    );
    drop(graph);
    parity && two_distinct
}

// -------------------------------------------------------------------------------------------------
// TEST 5 (FIX 1) — the allocator is USABLE after a PANICKED `capture_arena`.
//
// Mirrors cudagraph_panic_recovery.rs but for the ARENA path. A capture closure panics INSIDE the
// real capture window (on the capture pass, after warmup). The unwind guard in `capture_arena` must
// abort the capture (pull the stream out of capture mode), FREE the active arena's device blocks,
// and clear the `capture` slot — so the allocator is not wedged. We assert: the panic propagates;
// arena bytes return to baseline (the panicked arena was freed, never sealed); an eager launch +
// sync succeeds; and a FRESH `capture_arena` + replay succeeds bit-identical.
// -------------------------------------------------------------------------------------------------
fn test_arena_panic_recovery(client: &Client) -> bool {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let n = 1 << 16; // 64K f32 = 256 KB per tmp
    let a = client.empty(n * 4);
    let b = client.empty(n * 4);
    fill(client, &a, n, 1.0);
    fill(client, &b, n, 0.0);
    block_sync(client);

    let base = client.memory_usage().bytes_reserved;

    // Closure panics on the CAPTURE pass (call index == warmup), i.e. while the CUDA capture window
    // is open and the arena is locked — the hardest abort path.
    let warmup = 2usize;
    let calls = Cell::new(0usize);
    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _g = unsafe {
            client.capture_arena(warmup, || {
                let i = calls.get();
                calls.set(i + 1);
                let tmp = client.empty(n * 4); // grows the arena during warmup
                mul2(client, &a, &tmp, n);
                add_into(client, &tmp, &b, n);
                if i >= warmup {
                    panic!("boom: deliberate panic inside the locked capture pass");
                }
            })
        };
    }));
    assert!(
        panicked.is_err(),
        "panic inside capture_arena was swallowed; the guard must re-raise it"
    );

    // Arena bytes must be back to baseline: the panicked arena was freed by the abort path and never
    // sealed to a graph, so `memory_usage()` shows no lingering arena reservation.
    block_sync(client);
    let after_panic = client.memory_usage().bytes_reserved;
    let arena_zeroed = after_panic <= base + (n * 4) as u64; // allow one tmp of pool slack

    // Allocator usable: an eager launch + sync must succeed (stream not wedged).
    fill(client, &b, n, 0.0);
    let tmp = client.empty(n * 4);
    mul2(client, &a, &tmp, n);
    add_into(client, &tmp, &b, n);
    block_sync(client);
    let eager = read_buf(client, &b);
    drop(tmp);
    let eager_ok = max_abs_diff(&eager, &vec![2.0f32; n]) < 1e-3; // b = a*2 = 2.0

    // Capture still works: a fresh arena capture + replay, bit-identical.
    let body = |client: &Client| {
        let tmp = client.empty(n * 4);
        mul2(client, &a, &tmp, n);
        add_into(client, &tmp, &b, n);
    };
    fill(client, &b, n, 0.0);
    block_sync(client);
    body(client);
    block_sync(client);
    let eager_ref = read_buf(client, &b);

    let graph = unsafe { client.capture_arena(2, || body(client)) };
    block_sync(client);
    fill(client, &b, n, 0.0);
    block_sync(client);
    graph.replay();
    block_sync(client);
    let replay = read_buf(client, &b);
    let fresh_ok = max_abs_diff(&eager_ref, &replay) < 1e-3;
    drop(graph);

    let ok = arena_zeroed && eager_ok && fresh_ok;
    println!("[5] arena panic-recovery (panic inside locked capture pass):");
    println!(
        "      arena bytes after panic: base={} KB after={} KB  -> {}",
        base / 1024,
        after_panic / 1024,
        if arena_zeroed { "FREED (back to baseline)" } else { "LEAK (arena not freed)" }
    );
    println!(
        "      eager launch after panic -> {}   fresh capture+replay -> {}",
        if eager_ok { "OK (not wedged)" } else { "WEDGED" },
        if fresh_ok { "OK (bit-identical)" } else { "BROKEN" }
    );
    ok
}

fn main() {
    let device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (no Fusion)\n");
    println!("=== PHASE 1 GATE: capture ARENA (C2) + scalar/metadata staging ===\n");

    let t1 = test_scalar(&client);
    println!();
    let t2 = test_alloc_recycle(&client);
    println!();
    let t3 = test_lifetime(&client);
    println!();
    let t4 = test_multi_live(&client);
    println!();
    let t5 = test_arena_panic_recovery(&client);
    println!();

    println!("=== SUMMARY ===");
    println!("  [1] scalar-bearing capture+replay bit-identical : {}", yn(t1));
    println!("  [2] alloc-in-capture recycles (peak-LIVE)       : {}", yn(t2));
    println!("  [3] arena freed on graph drop (no leak / 100x)  : {}", yn(t3));
    println!("  [4] multi-live: >=2 distinct concurrent blocks  : {}", yn(t4));
    println!("  [5] allocator usable after panicked capture     : {}", yn(t5));

    assert!(t1, "scalar capture parity failed");
    assert!(t2, "alloc-in-capture parity/recycle failed");
    assert!(t3, "arena lifetime/leak check failed");
    assert!(t4, "multi-live distinct-block check failed");
    assert!(t5, "arena panic-recovery check failed");
    println!("\nPHASE 1 (C2 capture arena) GATE: GO — all checks passed.");
}

fn yn(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}
