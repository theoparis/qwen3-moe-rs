# CubeCL CUDA-Graph Support — Engineering Design

Status: **DRAFT for 3-voice review** (Codex gpt-5.5 high + Gemini 3.1 Pro high + Opus 4.8 high).
Scope locked: full 3-component design, all at once (graph-aware allocator + device-seed RNG + capture FFI).
Grounded in `docs/cudagraph/research-capture-mempool.md`, `research-rng.md`, `research-cubecl-map.md`.

Versions (this repo's lockfile): cubecl `b19859ee`, **cudarc 0.19.8**, cubek `1161040`, burn pinned `5923b1e`.

---

## 0. Scope, payoff, and the honest case

We add CUDA-graph **capture + replay** to the CubeCL/Burn CUDA backend so a fixed-shape, host-sync-free
device region (the GRPO decode step) can be recorded once and replayed per token, removing per-launch CPU
overhead. Three framework components, in cubecl + cubek:

1. **Capture FFI** (`cubecl-cuda`) — bracket the eager stream with `cuStreamBeginCapture/EndCapture` →
   `cuGraphInstantiate` → `cuGraphLaunch`, exposed as a typed `CudaGraph` handle.
2. **Graph-aware allocator** (`cubecl-runtime`) — a capture-mode private pool (PyTorch model) that pins
   every allocation made during capture for the graph's lifetime, so replay VAs stay valid.
3. **Device-seed counter RNG** (`cubek-random`) — move the RNG seed+offset from host `ScalarArg`
   immediates into a device buffer the kernel reads (PyTorch capturable-Philox model), so replays
   decorrelate instead of replaying frozen noise.

**The honest payoff (do not oversell).** The prior 3-voice gate measured the GRPO decode step at **~1.0×
eager** from removing the host sync, because the step is **bandwidth-bound** (the tied-head logits GEMM
streams ~0.6 GB/step), which graphs do not touch — graphs only remove host *launch latency*, which the
Fusion layer already largely collapses. Realistic graph payoff for THIS workload: **~1.1-1.4× at
batch-1/short-context, shrinking toward 1.0× as model size / context / batch grow.** This design is
justified by two things the user is buying, not by the decode number alone:
- **A reusable framework capability** — graph capture benefits *any* launch-bound CubeCL workload (small
  models, micro-batches, many-tiny-kernel pipelines), not just this rollout. It is infrastructure.
- **A correctness prerequisite done right** — the device-seed RNG (C3) and the static-shape decode (§7) are
  independently valuable (a device RNG removes a host sync from *every* stochastic op; static-shape decode
  removes per-step reallocation) even if graph capture is never switched on.

**Gate (measure-first, non-negotiable):** Phase 1 lands greedy capture + a microbenchmark. If the measured
replay speedup at the real GRPO shape is < ~1.15×, STOP and do not build C3/temperature capture — the
device-RNG and static-decode pieces still ship as standalone wins. This keeps the bounded-payoff risk
bounded.

```
  WHY IT'S SLOW NOW                          WHAT EACH PIECE BUYS
  ┌───────────────────────────┐
  │ per token (eager):        │   C1 capture  → removes N× host launch latency (replay = 1 host call)
  │  host: launch K kernels   │   C2 alloc    → makes replay memory-safe (stable VAs)  [enabler, not speed]
  │  gpu : forward + logits   │   C3 rng      → makes temperature capture CORRECT (not frozen noise)
  │        GEMM (0.6 GB read) │   §7 static   → makes the region capturable at all (fixed shape)
  └───────────────────────────┘
   launch latency is the ONLY part graphs remove; it's already small here → bounded payoff.
```

---

## 0b. ⚠️ REVISED APPROACH — after the 3-voice review (supersedes §5-§9 details below)

The 3-voice gate (Codex gpt-5.5 high + Gemini 3.1 Pro high + Opus 4.8 high, all source-verified) found
**C1 sound** but **two P0 defects the v1 draft missed** and **two over-/mis-scoped components**. The
component sections below (§4-§9) are kept as the v1 record; THIS section is the corrected plan. Build to
this, not to them.

**P0-A — the decode loop is NOT capturable by masking attention (Opus, source-verified).** Every per-step
`slice_assign`/`slice` bakes the host loop index as a frozen kernel scalar at capture: the KV write offset
`off = self.filled` (`cache.rs:81-85`), the token/logp/mask writes `slice_assign([.., (lp+t)..], …)`
(`rollout.rs:495-506`), the read-prefix slice. **A graph captured at step `t` replays into column `t`
forever.** So the real static-shape work (§7) is a **device-`pos`-indexed static-cache + loop rewrite**
(scatter KV by a device offset counter, replace every host-`t` `slice_assign` with a device-indexed
scatter), NOT the attention mask. `§8`'s "the driver becomes the capture closure verbatim" is FALSE — this
cache/loop rewrite is the bulk of the work and was the single most under-estimated item.

**P0-B — capture must live BELOW Fusion, not through it (Opus + Codex).** Driving capture through the
`Fusion<CubeBackend>` closure fights the model three ways: (a) C2's handle-pinning clones handles →
`can_mut()` false → **suppresses Burn-Fusion in-place fusion** → the captured plan diverges from warmup
(`handle.rs:10`, `multi.rs:84-91`); (b) autotune resolves **asynchronously** (`tuner.rs:281-314` `block_on`
on miss, applied via channel) so "≥3× warmup" does NOT guarantee the kernel settled — it can swap
mid-replay-set, and there is no freeze hook today; (c) if the Fusion `drain` happens outside the capture
window the graph records empty/partial. **Corrected architecture: capture on the raw
`CubeBackend<CudaRuntime>` stream (bypass the lazy Fusion queue), so the captured launch list is determined
by code, not mutable queue state.** This also dissolves the autotune-refire and empty-graph risks.

**C2 redesign — a dedicated pre-reserved capture ARENA, not `reserve()`-mode + retain-handles (all 3).**
Retaining every `SliceHandle` keeps VAs stable but (i) kills intra-step memory reuse → the arena becomes the
SUM of all step allocations, not the peak-live set → OOM (Gemini/Codex P0), and (ii) cubecl has no
bump-arena, so any `malloc_async` inside capture becomes a graph mem-alloc node whose free never fires →
leak/OOM after N replays (Opus). Build a **separate pre-reserved arena allocator** that sub-allocates
deterministically and recycles freed blocks WITHIN the graph (PyTorch `CUDACachingAllocator::capture_begin`
model), isolated from the general pool until graph destruction, allocated OUTSIDE the Fusion handle graph
(so no `can_mut` perturbation). Pre-size it by a record pass (cubecl exposes only `memory_usage()`, no
graph-mem high-water). Hard-error if capture is requested under a non-`Enabled` persistent-memory config
(`mode()` is config-gated, `memory_manage.rs:338-341` — else a silent no-op → replay corruption).

**C3 simplify — option (c) only: host writes FRESH SEEDS per replay (Opus resolves the Philox debate).**
The Gemini/Codex "Philox is required" objection was to the offset-bump-into-TAUS path, which IS statistically
invalid. But the design's own option (c) sidesteps it: before each replay the host writes 4 fresh seeds
(drawn from the same `StdRng` as eager `get_seeds()`) into the pinned device buffer — each replay is an
independent draw, statistically identical to today's eager behavior, with the existing TAUS+LCG core. **No
offset, no counter, no in-graph bump kernel, no Philox** (Philox only if a single counter stream must be
shared deterministically between graphed AND eager RNG — GRPO needs no such thing). The seed write must be
`cudaMemcpyAsync` on the graph's stream, ordered before `cuGraphLaunch` (not a blocking/pageable copy). This
also covers Codex's multi-RNG-op concern: fresh seeds re-key all ops (each still `ABSOLUTE_POS`-keyed, as
eager) → no cross-op counter overlap.

**C1 fixes (sound otherwise):** add a SECOND Fusion `drain` INSIDE the capture window (after the closure
issues ops, before `end_capture`) or the graph is empty; the real cubecl-native poison sources are
`create_with_data` per-launch H2D for scalars + the autotune `block_on`, NOT cuBLAS; `CubeCount::Dynamic`
is a HARD REJECT (not auto-fallback — it needs a host read).

**Re-phasing (Codex + Opus) — the v1 P1 does too much before the stop/go gate, and §7's own cost rigs it.**
Corrected order:
- **P0 (cheap gate, do FIRST):** capture a SYNTHETIC no-alloc fixed-buffer kernel chain on the raw
  `CubeBackend` (no Fusion, no GRPO), measure replay vs eager. This isolates the *launch-overhead* question
  from everything else. If there's < ~1.15× to reclaim after Fusion already collapses launches, **STOP** —
  the whole GRPO-decode-capture bet fails here, cheaply, before any allocator/cache work.
- **P1 (only if P0 passes):** C1 capture FFI (below Fusion) + the dedicated capture arena (C2) on that
  synthetic chain. The microbench MUST compare captured-static vs **eager-SLICED** (not eager-full-`T_max`),
  or §7's bandwidth cost flatters/rigs the result.
- **P2:** the device-`pos`-indexed static cache + loop rewrite (the real §7) → make the GRPO decode region
  actually capturable. Ship this REGARDLESS — a device-indexed cache removes per-step reshape/realloc and is
  a standalone win.
- **P3:** C3 option (c) → temperature capture. Ship the device-seed RNG REGARDLESS — it removes a host sync
  from every stochastic op.
- **P4:** pool sharing across length buckets (vLLM `graph_pool_handle`); buckets may be needed earlier if
  real prompt lengths vary widely (Codex).

**THE STRATEGIC CALL (the genuine open decision — see §11 R1).** For THIS workload alone (0.6B, N=64, vocab
151,936 logits-GEMM-dominated, bandwidth-bound), all three voices converge that the ~1.1-1.4× upside —
possibly net-NEGATIVE once §7's full-`T_max` bandwidth is added — likely does NOT justify the framework
infra. The recommendation: **build C1 as a standalone framework capability + prove it on the P0 microbench;
ship C3-(c) and the device-indexed static cache as independent wins; do NOT commit to the full GRPO-decode
capture unless (a) the P0 microbench clears the bar AND (b) there's a second launch-bound workload or a
committed upstream path.** This is yours to weigh — you have context (other CubeCL workloads, upstream
intent) the review doesn't.

---

## 0c. P-FINAL STATUS (what is actually built, and what is NOT yet production-gated)

P-final (`examples/cudagraph_pfinal_bench.rs`) assembled the four components (C1 capture/replay, C2
capture arena + **metadata interning**, C3 device-seed RNG, P2 device-`pos` static decode) into an
**actually-CAPTURED** GRPO decode: capture ONE static step, replay `max_new` times, with in-graph `pos`
advance, device-to-device chaining through `Option::take()`-owned persistent buffers, and a per-replay
device-seed write for temperature. It is a **VALIDATED MECHANISM**, not a production-gated path:

- **What is proven (on the GB10):** captured greedy decode is **BIT-IDENTICAL** to the eager static
  decode (`seq_ids` + `completion_mask` + `logp`, including the per-row EOS/pad path); captured
  temperature decode **DECORRELATES** across per-replay seed streams, is reproducible for a fixed seed,
  and (3-voice hardening, FIX 3) is **token-for-token identical to an EAGER temperature decode driven by
  the same seed stream** — a real autoregressive correctness detector, not just seed plumbing. Timing is
  the honest **~1.0×** (bandwidth-bound; graphs remove only host launch latency, which Fusion already
  largely collapses). The deliverable is the **reusable framework capability** (metadata interning +
  in-graph chaining + device-seed RNG), not a speedup for this workload.
- **3-voice robustness hardening applied (P1, not correctness bugs):** an explicit **OOB guard**
  (`assert!(warmup < max_new)` — the capture pass writes column `lp+warmup`, in bounds only if
  `warmup < max_new`); a documented **in-place-VA invariant** + a runtime **VA-stability assert** (each
  persistent buffer's device pointer is snapshotted before capture and asserted unchanged after, catching
  a stray `.clone()` that would relocate a baked VA → frozen/UB); and the temperature-parity detector above.

**REMAINING GATES before any production use** (none of these are wired yet):

1. **bf16 + a real trained model.** Validated only on **f32, random-init** weights. bf16 numerics and a
   calibrated model (where temp=1.0 noise matters without the temp=64 boost the bench uses) are untested.
2. **Graph REUSE across prompts + a 2nd `reset_for_replay` generation on ONE captured graph.** Only
   capture-per-call is tested today; reusing a single captured graph for a fresh generation (the actual
   payoff path) is not.
3. **The GRPO `old_logprob` → PPO step-0 `mean_ratio ≈ 1` gate through the captured rollout at temp>0.**
   Never run (same prerequisite `sampling_device.rs` flags as the eager device path).
4. **Metadata interning is FIXED-TRACE / FIXED-SHAPE-only.** A locked-pass content miss (e.g. autotune
   resolving a *different* kernel between warmup and capture, changing the staged metadata blob) is a
   **SAFE, LOUD ABORT** (hard error, no silent corruption) — the fix is to bump `warmup` so autotune
   settles before capture; it is **not a logic bug**.
5. **`lp` is baked into the graph** (one captured graph per prompt length) → length **buckets** (P4) are
   required for variable prompt lengths. **P4 status:** a real length-`L` prompt is LEFT-PADDED to its
   bucket `B` and the pad columns are masked out by a device `lo = B-L` counter; the bucketized decode
   then **matches the true-length-`L` decode UP TO FP/ARGMAX ROBUSTNESS — bit-identical only when
   `pad_len == 0` (`L == B`).** RoPE is relative, so the uniform `lo`-shift is invariant only in REAL
   arithmetic; in FP, q/k are rotated by the ABSOLUTE angles `(lo+a)` vs `a`, so for `pad_len > 0` a
   near-tie logit can flip the greedy argmax and then autoregressively diverge. The P4 bench therefore
   gates on robustness (`on_frac > 0.98 && on > off + 0.05`), not bit-equality. K buckets also SHARE one
   capture-pool arena (vLLM `graph_pool_handle`) so they cost ~1 (largest) bucket's high-water, not K×.

---

## 1. Background — the 4 blockers this design clears

From the prior gate (`docs/VLLM_KERNELS.md §4`), capturing the decode loop is blocked by, in order:

| # | Blocker | Cleared by |
|---|---------|-----------|
| B1 | No capture/replay API in cubecl (launch is eager per-op) | **C1** Capture FFI |
| B2 | Recycling allocator → replay reads freed/reassigned VAs → corruption | **C2** Graph-aware private pool |
| B3 | `Tensor::random` bakes host seeds as frozen immediates → replay = identical noise | **C3** Device-seed RNG |
| B4 | Decode attention shape grows each step (`filled = lp+t+1`) → not fixed-shape | **§7** static-T_max + device counter + mask |
| B5 | Fusion is a lazy/dynamic queue → captured launch list not stable | **§3** drain + pre-warm (cross-cut) |

---

## 2. Prior art — what we copy (Layer 1, don't reinvent)

- **PyTorch CUDA graphs** (`torch.cuda.CUDAGraph`, `graph_pool_handle()`): the **private memory pool** —
  redirect the caching allocator into an isolated pool during capture; tag blocks with the graph id; never
  return them to the general pool, so VAs are fixed and exclusively owned. Serially-replayed graphs can
  SHARE one pool (vLLM holds ~50 graphs at ~1 graph's memory). Static I/O = `copy_` into fixed buffers,
  never rebind. → **C2.**
- **PyTorch capturable Philox** (`PhiloxCudaState`, `philox_cuda_state(increment)`): seed+offset in DEVICE
  memory; kernel reads via a stable captured pointer; host writes fresh values into the same addresses
  before each replay (O(1)); offset advances by a known increment so ops never overlap. → **C3.**
- **gpt-fast decode capture**: static max-len KV buffer + `input_pos` as a **device** tensor + attention
  mask → ONE graph for the whole decode step. vLLM/TRT-LLM instead hide seqlen via PagedAttention and use
  **batch-size buckets**. For a fixed-N GRPO rollout, gpt-fast's static-shape model is the fit. → **§7.**
- **cudarc 0.19.8**: `result::stream::{begin,end}_capture`, `result::graph::{instantiate,launch,upload,
  destroy}` are the exact `result::` layer cubecl already calls; the deeper FFI
  (`cuGraphExecKernelNodeSetParams`, `cuGraphAddMemAllocNode`, `cuMemPoolCreate`) is in `driver::sys`,
  unwrapped but usable. → **C1.**

---

## 3. The capture/replay lifecycle (the contract)

```
  WARMUP (eager, side stream)         CAPTURE (record, no host sync)        REPLAY (per token)
  ┌──────────────────────────┐        ┌──────────────────────────────┐     ┌────────────────────┐
  │ run the step ≥3× eager:  │        │ drain Fusion queue (flush)   │     │ copy inputs into    │
  │  - JIT-compile kernels    │  --->  │ begin_capture(stream, ThreadLocal)│ │  the FIXED buffers  │
  │  - freeze autotune plans  │        │ run step closure (issues ops)│ --> │ cuGraphLaunch(exec) │
  │  - pre-size the pool       │        │  - alloc → capture pool      │     │ (1 host call)       │
  │  (so capture hits 0 malloc)│        │  - NO sync / NO Dynamic count│     │ read outputs from   │
  └──────────────────────────┘        │ end_capture → instantiate     │     │  the FIXED buffers  │
                                       └──────────────────────────────┘     └────────────────────┘
```

Rules enforced by the API (poison the capture otherwise):
- No host sync inside the closure: no `Command::sync`, no `read_async`/D2H, no `Fence::wait_sync`.
- No `CubeCount::Dynamic` (it does a `block_on` host read at `server.rs:177-191`) — assert Static.
- All allocation inside capture goes to the capture pool (no default-stream `malloc_async`).
- THREAD_LOCAL capture mode (server default) so unrelated host threads aren't policed.

---

## 4. Component 1 — Capture FFI (`cubecl-cuda`)

**The seam.** The single eager submit is `CudaContext::execute_task` → `cudarc::driver::result::
launch_kernel(..., stream.sys, ...)` (`context.rs:297-306`); the raw `CUstream` is the public `Stream.sys`
(`stream.rs:18`). The stream + per-stream allocators live in `Stream`/`MultiStream`/`StreamPool` on
`CudaServer` (the `CudaContext` owns only the `CUcontext`).

**New API (threaded server → channel → client, exactly like `allocation_mode`):**

```rust
// cubecl-runtime ComputeClient (backend-agnostic surface)
impl ComputeClient {
    /// Capture all device work issued by `f` into a replayable graph. `f` must issue only
    /// static-shape, host-sync-free ops (CubeCount::Static, no reads). Allocations during `f`
    /// are pinned to the returned handle's lifetime (see C2).
    fn capture<R>(&self, f: impl FnOnce() -> R) -> (CudaGraphHandle, R);
}
impl CudaGraphHandle {
    fn replay(&self);     // cuGraphLaunch on the server stream (1 host call)
    fn drop(...);         // cuGraphExecDestroy + release the pinned capture pool
}
```

**cubecl-cuda implementation (the wrapper around cudarc):**
1. `server.capture_begin()`: `MultiStream::drain` (flush the Fusion queue, see §3), set the server's
   per-stream allocator into `MemoryAllocationMode::Capture` (C2), call `result::stream::begin_capture(
   stream.sys, ThreadLocal)`.
2. run the closure — eager `launch_kernel` calls now record into the capturing stream as graph nodes.
3. `server.capture_end()`: `result::stream::end_capture(stream.sys)` → `CUgraph`; `result::graph::
   instantiate(graph)` → `CUgraphExec`; move the capture pool's pinned handles into the `CudaGraphHandle`;
   restore `MemoryAllocationMode::Auto`. Return the handle.
4. `replay()`: `result::graph::launch(exec, stream.sys)`.

**Hard problems + mitigations:**
- `malloc_async`/`free_async` (`gpu.rs:162,71`) inside capture become graph-ordered MemAlloc/MemFree nodes.
  *Mitigation:* warmup pre-sizes the pool so capture issues **zero** mallocs (the PyTorch approach); if a
  malloc still occurs, route it to the capture pool's pre-reserved arena (never the default pool).
- The safe cudarc `CudaGraph` is not thread-safe and its `end_capture` also instantiates; we use the
  `result::` layer directly for control over instantiate flags + pool ownership.
- `CubeCount::Dynamic` → hard `assert` (or auto-fallback to eager) when capture is active.

---

## 5. Component 2 — Graph-aware allocator / private pool (`cubecl-runtime`)

**Why.** Graphs bake exact device VAs into nodes. A live `SliceHandle` → fixed `StorageHandle(id+offset)` →
stable address (`gpu.rs:139-154,184`), BUT drop+realloc gives no identity guarantee because `coalesce`
mints fresh handles/offsets (`memory_page.rs:199-206`). So per-step intermediates freed and recycled across
replays would corrupt. PyTorch's fix: a private pool whose blocks are never returned to the general pool
while the graph is alive.

**Design.** Extend `MemoryAllocationMode` (`memory_manage.rs:92`, today `Auto|Persistent`) with `Capture`,
and add a `capture: Option<CapturePool>` to `MemoryManagement`:

```
  reserve(size)  [memory_manage.rs:403]
    ┌─ mode == Capture ─► CapturePool::alloc(size)
    │     - serve from the pre-reserved arena (warmup-sized to the high-water mark)
    │     - RETAIN the minted SliceHandle in capture.live: Vec<SliceHandle>  (never freed/coalesced)
    │     - addresses are therefore stable for the graph's lifetime
    └─ else ─► existing SlicedPool::try_reserve (first-fit)

  capture_end():
    graph_handle.pinned = std::mem::take(&mut capture.live)   // PyTorch private-pool ownership
    // dropping graph_handle later frees the whole arena at once
```

- **Warmup pre-sizing:** run the step eagerly ≥3× under a "record high-water" flag; size the capture arena
  to the peak; then capture allocates from it with zero `malloc_async` (clean capture). Mirrors
  `cuDeviceGetGraphMemAttribute` high-water sizing.
- **Pool sharing across buckets** (optional, vLLM `graph_pool_handle`): a `CapturePoolId` so several graphs
  (e.g. per prompt-length bucket) that are replayed serially share one arena → ~1 graph's memory for many
  graphs. Phase 3.
- **Static I/O:** inputs/outputs are fixed buffers allocated once *outside* capture; per replay we `copy_`
  into them (never rebind a new tensor), so their VAs are stable too.

**Open question (for review):** is retaining every intermediate `SliceHandle` for the graph lifetime an
acceptable memory cost at GRPO decode shapes (small N, short context)? At 0.6B/N=64 the per-step
intermediates are MB-scale; pinning them for one graph is cheap. At large models/long context it grows —
mitigated by pool-sharing + the fact that intermediates are bounded by one step's working set, not T.

---

## 6. Component 3 — Device-seed counter RNG (`cubek-random`)

**Why.** `random()` (`base.rs:23-63`) calls host `get_seeds()` (`base.rs:74-87`, a `static SEED:
Mutex<StdRng>`) and bakes 4 `u32` as `ScalarArg` immediates into `prng_kernel` (`base.rs:48-62`). In
`cubecl-cuda/server.rs` those scalars lower into the launch params (by-value) → **frozen at capture** →
every replay reuses identical noise → degenerate sampling. (Greedy uses no RNG, so greedy capture is safe
without C3 — that's why Phase 1 is greedy-only.)

**Design (PyTorch capturable-Philox model).**
```
  TODAY:   prng_kernel(out, ScalarArg(s0), ScalarArg(s1), ScalarArg(s2), ScalarArg(s3))   // immediates → frozen
  NEW:     prng_kernel(out, &rng_state)   where  rng_state: Array<u32> = [s0,s1,s2,s3, offset]  // DEVICE buffer
           in-kernel: read seed_i + offset from rng_state via the STABLE captured pointer;
                      per-element key = f(ABSOLUTE_POS, offset) keyed counter (subsequence = ABSOLUTE_POS).
  ADVANCE: before each replay the host writes [fresh seeds OR same seed, offset += increment] into the
           SAME rng_state device buffer (O(1) memcpy, PyTorch-style); the captured kernel re-reads it.
```

Three sub-steps, increasing fidelity:
1. **Minimal (unblocks capture):** swap the 4 immediates for a device `Array<u32>` `[seed×4, offset]` bound
   like an output tensor, read in-kernel via the stable pointer. Advance the offset per launch by a
   host-written device buffer (option (c) in `research-rng.md` — O(1) host work, stays in sync with eager).
2. **Correct offset accounting:** advance `offset` by `counter_offset = ceil(nelem / (blocks·threads·
   UNROLL))·UNROLL` (PyTorch `Dropout.cu`) so concurrent ops never reuse a counter.
3. **Optional Philox/threefry swap:** replace the stateful TAUS88+LCG core with a stateless counter-based
   bijection `f(seed, counter)` (Random123). Bonus: fixes the existing vectorization-correlation TODO at
   `base.rs:36` and makes the offset math exact. Defer unless (2) proves insufficient.

burn reaches this via `random_uniform` (`burn-cubecl/.../prng/uniform.rs:16`), seeded at `backend.rs:50`;
the device-buffer state is created once and threaded through (the GRPO Gumbel-max `Tensor::random` is the
consumer).

### P3 status (shipped) and the remaining P-final plumbing gap

**Shipped (P3, validated on GB10 — `cudagraph_p3_rng_bench`).** `cubek-random` now exposes a SEPARATE,
opt-in capturable entry — `random_uniform_with_seeds` → `random_with_seed_handle` → `prng_kernel_seeded`
— that reads its 4 seeds from a caller-owned DEVICE buffer (`[N_SEEDS] u32`) bound as `Array<u32>` (a
stable POINTER the captured node bakes), not from `ScalarArg` immediates. The host rewrites fresh seeds
into that SAME buffer via `ComputeClient::write_to_handle` before each `replay()`, and a captured region
DECORRELATES across replays ([1]-[4] PASS). The DEFAULT eager `random()` is UNCHANGED — it still passes
the 4 seeds as immediates through `prng_kernel` (zero per-call alloc/H2D); the device-buffer path is
opt-in only, so eager `Tensor::random` (incl. the GRPO rollout's Gumbel sampling) pays no regression.

**NOT yet built (P-final).** burn's `Tensor::random` is NOT capturable as-is: it routes to the default
`random()`, which allocates a FRESH internal seed each call and passes immediates — so a captured region
containing `Tensor::random` would freeze its noise. To capture the GRPO Gumbel sampler, P-final must
thread a PERSISTENT, externally-owned "generator handle" (a `[N_SEEDS] u32` buffer allocated OUTSIDE the
capture region) through burn-cubecl's prng path (`random_uniform` → … → `random_with_seed_handle`), so
the host can `write_to_handle` fresh seeds into it per replay. The `cudagraph_p3_rng_bench`
external-handle pattern is the template; the opaque `Tensor::random` path stays on immediates until that
plumbing lands. This is the remaining P-final work.

---

## 7. The dynamic-decode-shape fix (makes the region capturable at all)

The decode forward is NOT fixed-shape today: `cache.update` returns the growing valid prefix
`key/value.slice([0..filled])` (`cache.rs:88-89`) and attention reads `total_seq = lp+t+1`
(`attention.rs:244,269`), so the `q·kᵀ` shape `[b,nh,1,total_seq]` grows by one each step.

**Fix (gpt-fast model):** attention over the FULL static `[B, T_max, ..]` KV buffer (already allocated by
the static cache, `cache.rs with_capacity`) with a **device length counter** `pos` and a **position mask**
that zeroes keys at index ≥ `filled`. Shape becomes constant `[b,nh,1,T_max]` every step → capturable. The
mask + counter live on device (no host offset baked per step).

```
  TODAY (grows):   attn(q[1], K[0..filled], V[0..filled])   shape changes each t  ✗ capturable
  NEW   (fixed):   attn(q[1], K[0..T_max],  V[0..T_max], mask(idx >= pos_dev))     shape constant ✓
                   pos_dev: device counter, ++ per step (also the KV write offset → C2-stable)
```

This is independently valuable (no per-step reshape/realloc) and is the §7 prerequisite for one-graph
decode capture. Cost: attention now scans `T_max` keys every step instead of `filled` — wasted work at
short context (the gpt-fast VRAM/compute tradeoff). For GRPO's bounded `max_new_tokens` this is acceptable;
revisit with length buckets if `T_max` ≫ typical completion.

---

## 8. Public API surface (what the model code calls)

```rust
// One-time, after warmup:
let graph = client.capture(|| decode_one_step(&model, &mut cache, &rng_state, &io_buffers));
// Per token (the hot loop):
for t in 0..max_new {
    write_step_inputs(&io_buffers, t);   // copy_ into fixed buffers (token, pos counter)
    graph.replay();                      // 1 host call, no per-kernel launch
    // outputs are in io_buffers (device); no per-step host read
}
read_results_once(&io_buffers);          // single device→host at the end
```

The GRPO driver `group_sample_cached_device_loop` (already static + host-sync-free) becomes the capture
closure body almost verbatim once §7 lands.

---

## 9. Build order / phasing (incremental, reversible)

| Phase | Deliverable | Gate |
|-------|-------------|------|
| **P1** | C1 capture FFI + C2 capture pool + §7 static-shape attn, **GREEDY only** (no C3). Microbench: eager vs replay at real GRPO shape. | **MEASURE.** If replay < ~1.15× → stop; ship C2/§7 as standalone. |
| **P2** | C3 device-seed RNG (sub-steps 1-2) → temperature capture. Parity: replayed temperature sampling decorrelates + matches eager distribution. | step-0 PPO ratio ≈ 1 through the captured path. |
| **P3** | Pool sharing across length/batch buckets (vLLM `graph_pool_handle`); optional Philox swap. | only if P1 payoff justifies. |

P1 alone proves/refutes the whole bet. Each phase is independently shippable and reversible (capture is
opt-in; eager path untouched).

---

## 10. Test plan

- **C1:** a CubeCL unit test — capture a 2-kernel graph (e.g. `mul2 → add1`), replay, assert bit-identical
  to eager; assert capture with a `CubeCount::Dynamic` op fails loud; assert a host-sync inside capture
  fails loud.
- **C2:** capture a step that allocates intermediates; replay 100×; assert no corruption (outputs stable)
  and that an interleaved eager alloc on the general pool does NOT reuse the graph's pinned VAs (the
  PyTorch-pool invariant). Leak test: dropping the graph frees the arena.
- **C3:** capture a `Tensor::random` step; replay N×; assert the draws DECORRELATE across replays (not
  frozen) AND match an eager reference advancing the same offsets; the existing `device_sample_*` greedy
  parity must still hold.
- **§7:** static-shape masked attention == the growing-prefix attention (bit-parity, the existing
  `cached_matches_uncached_greedy` gate extended to the masked path).
- **End-to-end:** `group_sample_cached_device_loop` under capture == eager (greedy bit-identical; temperature
  distribution + step-0 ratio≈1), on GB10. The microbench reports eager-vs-replay wall-clock at N∈{1,64},
  context∈{128,1024}.

---

## 11. Risks + open decisions (for the 3-voice review)

- **R1 (payoff):** bounded ~1.1-1.4×, possibly ~1.0× — the P1 gate exists to catch this early. Is the
  framework capability + the standalone wins (device RNG, static decode) worth P1's cost on their own?
- **R2 (Fusion stability):** even after `drain` + pre-warm, can the Fusion plan-store / autotune still emit
  a different kernel sequence on a later replay (e.g. a re-autotune)? Need to *freeze* autotune during
  capture, not just pre-warm. Is there a freeze hook, or must we add one?
- **R3 (allocator blast radius):** adding a `Capture` mode touches the core allocator path (`reserve`). Risk
  to all CubeCL users. Mitigation: mode is opt-in + defaults to today's behavior; gate behind a feature.
- **R4 (upstream vs fork):** do we upstream to cubecl/cubek (PR), or carry a patched fork pinned in this
  repo? Upstream is cleaner but slow; a fork unblocks us now. **Decision needed.**
- **R5 (T_max scan cost, §7):** full-`T_max` attention wastes compute at short context. Acceptable for
  bounded `max_new_tokens`; length buckets if not. Measure.
- **R6 (RNG correctness):** RESOLVED by the review → option (c) fresh-seed-per-replay is correct with the
  existing TAUS+LCG (matches eager); the offset/Philox path was the invalid one. Drop it.
- **R7 (input IO sync, Gemini):** the per-replay token/pos writes must be `cudaMemcpyAsync` on the graph's
  stream ordered before launch, or they become the new per-token host stall. Fold into the capture-arena IO.
- **R8 (Fusion empty-capture, Codex/Opus):** the `drain` must be INSIDE the capture window; capturing below
  Fusion (revised §0b) removes this class entirely.
- **R9 (graph lifetime/concurrency, Codex):** drop-while-replay-in-flight, multi-thread replay, stream
  ordering — the cudarc safe `CudaGraph` is not thread-safe; serialize under `lock_device()`.

---

## GSTACK REVIEW REPORT

Plan: `docs/cudagraph/DESIGN.md` — CubeCL CUDA-graph support (capture FFI + graph-aware allocator +
device-seed RNG + static-shape decode). Scope locked with the user: full 3-component design, all at once.

**Runs**

| # | Voice | Model | Status |
|---|-------|-------|--------|
| 1 | Codex | gpt-5.5 high | absorbed |
| 2 | Gemini | 3.1 Pro high (via AGY-USG) | absorbed |
| 3 | Opus | 4.8 high (source-verified) | absorbed |

Inputs: 3 research docs (`research-capture-mempool.md`, `research-rng.md`, `research-cubecl-map.md`), all
prior-art-grounded + cited; all seams verified against pinned source (cubecl `b19859ee`, cudarc 0.19.8,
cubek `1161040`, burn `5923b1e`).

**Findings (consensus → action)**

| Sev | Finding (who) | Action taken (§0b) |
|-----|---------------|--------------------|
| P0 | Decode loop NOT capturable by attention mask — every `slice_assign` freezes the host index (Opus, src) | Re-scoped §7 to a device-`pos`-indexed static-cache + loop rewrite; corrected §8's "verbatim" claim |
| P0 | C2 retain-handles is wrong: kills intra-step reuse → OOM (Gemini/Codex) + flips `can_mut` → suppresses Fusion in-place → autotune refire (Opus) | Redesigned C2 → dedicated pre-reserved capture ARENA, allocated below/outside Fusion |
| P0 | Capture-through-Fusion fights the lazy/dynamic model (can_mut + async autotune + empty-capture) | Architecture moved BELOW Fusion (raw `CubeBackend` stream) |
| P1 | Philox "required" (Gemini/Codex) vs simpler fresh-seed (Opus) | Resolved → C3 option (c) fresh-seed-per-replay; dropped bump/offset/Philox from the critical path |
| P1 | P1 gate does too much before stop/go; §7 cost rigs the benchmark (Codex/Opus) | Re-phased: P0 = synthetic no-alloc microbench first; compare vs eager-SLICED; ship C3/device-cache standalone |
| P1 | Missing poison sources, IO sync, 2nd drain, Dynamic hard-reject, multi-RNG offset, lifetime/concurrency | Folded into §0b (C1 fixes) + R7/R8/R9 |

**VERDICT: REVISE-AND-PROCEED-WITH-GATE.** C1 is sound and well-grounded. C2/§7 had P0 defects, now
redesigned in §0b (capture below Fusion + dedicated arena; device-indexed cache rewrite). C3 is simplified
to the one correct variant. The honest bounded payoff (~1.1-1.4×, possibly net-negative with §7) stands, and
the cheap P0 microbench gate now fronts the plan so the bet is testable before the expensive allocator/cache
work. CROSS-MODEL agreement: all three voices converge on the two P0s and on "capture below Fusion" — strong
signal, but a recommendation, not a decision.

**UNRESOLVED DECISIONS:**
- **Build the full GRPO-decode capture at all?** All three voices flag that for THIS workload alone the
  upside likely doesn't justify the framework infra. Recommended path: build C1 standalone + run the P0
  microbench; ship C3-(c) + the device-indexed static cache as independent wins; commit to full decode
  capture ONLY if the microbench clears ~1.15× AND there's a second launch-bound workload or an upstream
  path. Your call — you have context the review doesn't.
- **Upstream to cubecl/cubek, or carry a pinned fork?** All voices say fork now (upstreaming the allocator +
  below-Fusion capture is a months-long review); upstream once P-phases prove out. Confirm fork.
- **Length buckets in P1 or P4?** If real GRPO prompt lengths vary widely, full-`T_max` waste may force
  buckets earlier (Codex). Depends on the prompt-length distribution — needs your input.
