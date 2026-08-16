# Metal streamed-decode perf findings (M2 Pro 16 GB, Qwen3.6-35B-A3B)

Measured with `examples/qwen35_generate_portable --device metal`, `QWEN35_STREAM_EXPERTS=1`,
prompt `"The capital of France is"`, `QWEN35_MAX_NEW_TOKENS=16`. Every run produced the identical
continuation `" Paris, a city renowned for its rich history, culture, and iconic landmarks."`, so
all deltas are pure performance.

## Headline: it IS the MoE, but it is not the MoE *math*

`QWEN35_PROFILE=1` attribution of a 154s run (profiling syncs at every boundary, which inflates
totals ~17%, but the split is real):

```
MOE_PREFETCH  64.94s (42.1%)   <- streamed expert io + CPU repack
MOE_EXPERTS   34.22s (22.2%)   <- expert GEMVs
GDN_ATTN      22.09s (14.3%)
LM_HEAD       11.49s ( 7.5%)
EMBED          6.27s ( 4.1%)
FULL_ATTN      5.94s ( 3.8%)
MOE_SHARED     3.52s   MOE_SCATTER 3.27s   ROUTER 2.53s   FINAL_NORM 0.01s
```

MoE is ~70% of wall clock. But `examples/nvfp4_gemv_metal_probe.rs` measures one routed expert at
real decode shapes (M=1, gate_up K=2048/N=1024, down K=512/N=2048) in isolation:

```
full expert (gu+silu+mul+down), pipelined   0.435 ms/call
full expert, sync each call                 0.895 ms/call
```

6,520 expert calls per 16-token run x 0.435 ms = **~2.8s of actual expert math**. Production spends
~34s. The kernel is not the problem; roughly 92% of MoE time is everything *around* the math.

## Hypotheses tested and REJECTED

Measured, not assumed — three plausible explanations that turned out to be wrong:

| Hypothesis | Test | Result |
| --- | --- | --- |
| Dispatch cost scales with live buffer count (pool holds 1000s) | `LIVE_BUFFERS=4096` ballast in probe | **1.17x** only |
| Freshly uploaded weights pay a first-use/bind-group cost | cycle N distinct weights, one dispatch each, then re-run warm | first-use = **0.007 ms** |
| E2M1 encoder's linear search over 8 candidates is hot | replaced with comparison chain | **no change** (compiler had already unrolled it) |

## The actual causes

1. **The checkpoint is 68 GB bf16 and can never be page-cache resident in 16 GB.**
   `io` = 14.8s / 2970 misses = **5.0 ms per miss**, i.e. SSD latency on every fetch, zero reuse.
2. **Every routed expert is re-quantized on the CPU, every token.** `upload` = 8.6 ms per miss.
   The pool's hit rate is structurally poor against a 40-layer x 256-expert space.
3. **Memory pressure.** Peak footprint **9.75 GB** with **956,712 page reclaims** on a 16 GB
   machine that already had 1.29 GB of swap in use. This is the most likely explanation for the
   remaining gap between the probe's 0.435 ms/expert and production, and it also explains why
   growing the pool 64 -> 4096 slots made the *identical* set of GEMVs go 11.0s -> 30.9s.

Size budget for why this cannot work as-is:

```
routed expert params : 32.2B  (60.0 GiB bf16)
resident core params : 3.29B  ( 6.13 GiB bf16)   <- larger than turbo-fieldfare's ENTIRE model
per-token active     : 1.01B  ( 1.88 GiB bf16 / 0.47 GiB nvfp4)
full model as NVFP4  : ~18.2 GiB
```

turbo-fieldfare runs Gemma 4 26B-A4B in ~2 GB of RAM on this class of machine because its whole
model is 14.3 GB int4 (fits the page cache), its resident core is ~1.35 GB, and streamed experts are
mmap'd straight into GPU buffers with **zero per-element CPU work**. Here the resident core alone is
6.1 GB and each token drags 1.88 GB of bf16 through a CPU requantizer.

## Fixes landed

| Run | Pool | Sync | io | upload | compute | decode | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 64 | per-expert | 33.0s | 104.0s | 11.0s | 194.5s | 0.08 |
| + bigger pool | 4096 | per-expert | 24.2s | 38.4s | 30.9s | 143.4s | 0.11 |
| + per-layer sync | 4096 | per-layer | 23.6s | 39.4s | 1.3s | 131.9s | 0.12 |

1. **Repack ~2.9x faster per expert (51 ms -> 16 ms)**, in `src/nvfp4.rs`, each pinned bit-identical
   to the code it replaced by a dedicated test:
   - `quantize_nvfp4_from_nk_bf16` — fused the `[N,K]->[K,N]` transpose into the quantizer; the old
     path scattered with stride `n` then immediately gathered back with stride `n`.
   - `exp2i` — `f32_to_e4m3`/`e4m3_to_f32` called libm `powi`/`log2` once per 16-element block
     (131,072 libm calls per gate_up). Powers of two are exact, so build them from exponent bits.
   - `e2m1_bits_finite` — branchless E2M1 encode; the branch ladder blocked vectorization inside the
     packing loop. Biggest single win (21.5 -> 10.7 ms on gate_up).
2. **Pool capacity default 64 -> 4096.** The old tuning comment predated both the O(1) LRU fix and
   NVFP4's 4x smaller slots. Note this trades repack cost against memory pressure — revisit once the
   footprint is fixed.
3. **`QWEN35_STREAM_SYNC=0`** — sync once per layer instead of per routed expert.
4. **`QWEN35_PROFILE=1`** — the attribution above; `sel_w` also now stays on device.

## What did NOT work

**Parallelizing the repack with `std::thread::scope`.** Spawning ~20 threads per expert (~6,400 per
token) was dramatically *slower* on macOS — thread create/join overhead plus contention with Metal's
driver threads. Reverted. If revisited, use a persistent pool created once, never per-call spawns.

## Offline NVFP4 blob store (`src/nvfp4_blob.rs`, `examples/nvfp4_offline_repack.rs`)

Implemented the recommendation above. Pre-quantize every routed expert once into fixed-stride,
page-aligned (16 KiB) per-layer blob files (`layer_{L}.nvfp4` + a `manifest.txt` sidecar); decode
reads a record directly instead of doing bf16-read + CPU transpose/quantize per miss.

`cargo run --release --example nvfp4_offline_repack -- --src models --out models-nvfp4` — resumable
(skips correctly-sized layer files), took **270s** and produced **17 GiB** (vs the predicted 17.2
GiB and 60 GiB of bf16 source), matching `docs/`'s earlier size budget exactly. Point decode at it
with `QWEN35_NVFP4_BLOB_DIR=models-nvfp4`.

First version used `mmap`. **That was wrong** — checked against turbo-fieldfare's own I/O
experiments (`turbo-fieldfare/docs/experiments/summaries/01-model-install-and-expert-io.md`, IO-01):
they measured `mmap` as **3.5x slower per cold read** than explicit `pread` in exactly this regime
(working set > RAM, so every access is a genuine cold fault and the VM layer adds overhead `pread`
doesn't pay) — 0.50 tok/s with `mmap` vs 3.97 tok/s with parallel `pread` in their simulator. Their
IO-06 also independently confirmed the mechanism is memory-system contention, not resident-page
eviction (`mlock` recovered ~0ms). Replaced with `File::read_exact_at` (positional, so many threads
can read the same `File` concurrently with no locking) over a bounded pool (`PREAD_WORKERS = 8`,
chunked so thread count doesn't scale with miss count).

Also ported turbo-fieldfare's **hit-first execution** split (their DEC-18, measured 14.4% win
there): `prefetch_layer_begin` classifies a layer's routed experts into hits (already resident,
LRU-promoted) and misses, and only *starts* background reads for the misses instead of blocking.
The caller runs `expert_forward` for hits immediately -- that GPU dispatch overlaps the still-running
disk reads -- then calls `prefetch_layer_finish` to join the reads and upload, then runs the experts
that were misses. Previously the entire layer blocked on I/O before any expert's GEMV could dispatch.

Measured (16 tokens, `QWEN35_STREAM_SYNC=0`, pool capacity 512, identical output text throughout):

| Run | upload | io | decode | tok/s |
| --- | ---: | ---: | ---: | ---: |
| bf16 checkpoint (prior table's best) | 39.4s | 23.6s | 131.9s | 0.12 |
| + NVFP4 blob store, `mmap`, serial, no hit-first | 0.14s | 8.1s | 60.5s | 0.26 |
| + `pread` (not `mmap`) + hit-first execution | -- | -- | **21.9s** | **0.73** |

**9x over the session's original baseline (0.08 -> 0.73 tok/s)**, with the pool-size inversion noted
earlier now gone (cap=64/512/1024 are all ~22-24s; misses are cheap enough that pool size barely
matters anymore, unlike the mmap/bf16-era where a bigger pool could make things *slower*).

Rebalanced profile at cap=512 (was 42% `MOE_PREFETCH` before any of this):

```
GDN_ATTN      8.11s (25.4%)      MOE_EXPERTS   6.28s (19.7%)
MOE_PREFETCH  4.45s (13.9%)      FULL_ATTN     4.16s (13.0%)
MOE_SCATTER   2.59s ( 8.1%)      ROUTER        2.07s ( 6.5%)
LM_HEAD       1.75s ( 5.5%)      MOE_SHARED    1.58s ( 4.9%)
EMBED         0.95s ( 3.0%)
```

No single bucket dominates anymore -- the remaining time is spread across launch-bound small ops.

## Recommended next step

The blob store's `misses` count is still ~11-12k of ~19.9k total slot accesses even at cap=1024,
i.e. hit rate is still poor -- worth revisiting cache sizing/replacement policy now that a miss is
cheap (turbo-fieldfare uses LFU at only 16-32 slots per layer, not a large flat LRU; their DEC-04
notes I/O fell 166->88ms just from a 16-slot cache with real locality). `GDN_ATTN` and `MOE_EXPERTS`
are now the two largest buckets and neither is compute-bound (the isolated GEMV probe measures 0.435
ms/expert; production is still higher) -- likely per-dispatch launch overhead from many small ops.

## `burn-fusion` investigation (this session): fixed upstream, but not a win here

Re-enabling `fusion` was investigated as the highest-leverage lever for the per-op dispatch overhead
above. Two real, separate blockers were found and both were resolved far enough to get a clean,
correct, full end-to-end run with fusion **on**:

1. **Our own custom kernels didn't compile under fusion at all.** `nvfp4_gemv` /
   `fused_moe_gu2_down_nvfp4` grabbed the raw `CubeTensor` via
   `tensor.try_into_primitive::<burn::backend::Metal>()`, which only type-checks against a raw
   `CubeBackend`; with `fusion` on, `burn::backend::Metal` is `Fusion<CubeBackend<...>>` and the
   primitive type is a `FusionTensor` instead. Fixed by routing both kernels through the existing
   `src/cube_custom_op.rs` bridge (previously CUDA-only) for Metal/wgpu too, gated behind a new
   `metal-fusion-diag` branch in each kernel.
2. **The actual upstream panic** (`stream::execution::ordering.rs`, "Ordering is bigger than
   operations") traces to `tracel-ai/burn#5292`: `FusedReduceLaunch::run` in
   `burn-cubecl-fusion/src/optim/reduce/optimization.rs` resolves the reduce's reference *shape*
   correctly against `outputs` when the reference is a `Concrete(Output(..))`, but unconditionally
   resolves the reference *strides* against `inputs` a few lines later -- for a fused reduce whose
   reference is a Concrete Output (e.g. the decoder's final RMSNorm, an f32-statistics reduction
   over an f16 activation), that walks into the wrong, shorter arg list and panics with an
   index-out-of-bounds inside `resolve_arg`, which then cascades into the ~30 repeated
   `ordering.rs` panics we originally saw (per `#4827`, any panicked fusion-stream task corrupts
   the stream for every subsequent op). Confirmed the exact same code is still present, unfixed, on
   burn's `main` today. The one-line fix (mirror the `shape` dispatch for `strides`) was applied to
   a fork (`theoparis/burn`, branch `fix-5292-on-release`, rebased onto the exact `0.22.0-pre.2`
   release commit so the crates.io-pinned `cubecl`/`cubek` versions still match -- building against
   `main` directly pulls a newer, API-incompatible `cubecl` and is a much bigger yak-shave), wired
   in via `[patch.crates-io]` in `Cargo.toml`, and is ready to open as an upstream PR.

**Result: it works (no panic, output text bit-identical to every prior run), but it's a net loss,
not a win.** Same 16-token run, pool cap=512, blob store enabled:

| | fusion OFF | fusion ON |
|---|---|---|
| decode time | 21.9s | 37.5s |
| tok/s | 0.73 | 0.43 |

Per-bucket, ms/token (fusion off -> on): GDN_ATTN 507->604, FULL_ATTN 260->289, ROUTER 129->148,
LM_HEAD 109->167, MOE_EXPERTS 393->427 -- **every bucket got worse, including the pure-`Tensor`-op
ones (GDN_ATTN, FULL_ATTN, ROUTER) that never touch a custom kernel.** That rules out the
`CubeCustomOp` bridge as the sole cause; burn-fusion's optimization search isn't finding/merging any
fusable op chains in this `M=1`, attention-interleaved-with-sync decode graph, so its stream/planner
bookkeeping is pure overhead here with no compensating win. **Conclusion: leave `fusion` disabled
for normal use** (it already defaults off; `metal-fusion-diag` remains a diagnostic-only feature).
The non-MoE path's ~30x gap to the 15 tok/s target is therefore not fixable by fusion -- the
remaining real levers are hand-written fused Metal kernels (turbo-fieldfare's actual approach: see
its `FusedQKVGEMVTests`/`FusedPostAttentionSetupTests`/`FusedLayerTailTests`) and/or quantizing the
6.1 GiB bf16 resident core to NVFP4.

The upstream fix itself is still worth landing independent of this: it's a correct, verified,
minimal patch for a real (if narrow) bug, and unblocks anyone else hitting the same Metal/wgpu f16
fused-reduce crash.
