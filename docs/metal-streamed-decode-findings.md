# Metal streamed-decode perf findings (M2 Pro, Qwen3.6-35B-A3B)

Measured with `examples/qwen35_generate_portable --device metal`, `QWEN35_STREAM_EXPERTS=1`,
prompt `"The capital of France is"`, `QWEN35_MAX_NEW_TOKENS=16`. Every run below produced the
byte-identical continuation `" Paris, a city renowned for its rich history, culture, and iconic
landmarks."`, so all deltas are pure performance.

## Where the time actually went

| Run | Pool | Sync | io | upload (repack) | compute | decode | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 64 | per-expert | 33.0s | 104.0s | 11.0s | 194.5s | 0.08 |
| + bigger pool | 4096 | per-expert | 24.2s | 38.4s | 30.9s | 143.4s | 0.11 |
| + per-layer sync | 4096 | per-layer | 23.6s | 39.4s | 1.3s | 131.9s | 0.12 |

`upload` is the NVFP4 repack, not PCIe/host->device traffic.

## Fixes landed

1. **Repack cost, ~2.9x per expert (51 ms -> 16 ms).** Three changes in `src/nvfp4.rs`, each
   pinned bit-identical to the code it replaced by a dedicated test:
   - `quantize_nvfp4_from_nk_bf16`: the old path transposed `[N,K] bf16 -> [K,N] f32` with a
     stride-`n` scatter, then `quantize_nvfp4` read it straight back with stride `n`. The source is
     already in the order the quantizer wants, so both passes fuse into one sequential pass with no
     `k*n` f32 scratch.
   - `exp2i`: `f32_to_e4m3`/`e4m3_to_f32` called `2.0f32.powi()` and `log2()` (libm) once per
     16-element block = 131,072 libm calls per gate_up matrix. Powers of two are exact, so these
     are now built from IEEE-754 exponent bits.
   - `e2m1_bits_finite`: the E2M1 encoder's 8-branch boundary ladder was scalar and
     branch-mispredicting inside the packing loop. Summing the boundary predicates as integers is
     branchless and vectorizes. **This was the single biggest win** (21.5 -> 10.7 ms on gate_up).
2. **Pool capacity default 64 -> 4096** (`examples/qwen35_generate_portable.rs`). 64 slots is ~32
   experts across all 40 layers, so it self-evicted and re-paid the repack 2.6x more often than
   needed. The old tuning comment predated both the O(1) LRU fix and NVFP4's 4x smaller slots.
3. **`QWEN35_STREAM_SYNC=0`** (`src/expert_stream.rs`): sync once per layer instead of after each
   of ~320 routed experts per token. Cuts instrumented `compute` 30.9s -> 1.3s. Defaults to the old
   per-expert behaviour; the per-layer sync in `forward_streamed` still bounds dispatch backlog.
4. **`sel_w` stays on device** in `forward_streamed` — it was downloaded per layer only to be
   sliced and re-uploaded. No measurable win, but strictly less work.

`examples/nvfp4_repack_bench.rs` reproduces the repack numbers with no GPU and no weights.

## What did NOT work

- **Parallelizing the repack with `std::thread::scope`.** Spawning ~20 short-lived threads per
  expert (~6,400 per token) made things dramatically worse on macOS — thread create/join overhead
  plus contention with Metal's own driver threads. Reverted. If revisited, use a **persistent**
  pool created once outside the decode loop, never per-call spawns.

## The remaining gap is structural

At 0.12 tok/s against ~15 tok/s for turbo-fieldfare on comparable hardware, micro-optimization is
finished. Two structural problems remain, and they multiply:

1. **The repack should not exist at decode time.** Even fully optimized it is ~39s of the 132s.
   turbo-fieldfare quantizes experts to int4 **offline** into fixed-stride, page-aligned per-layer
   blobs, then mmaps pages straight into GPU buffers with zero per-element CPU work. The fix is
   `MEMORY_STREAMING_PLAN.md`'s offline repack: pre-quantize once, then decode does
   `offset = expert * stride` + mmap slice. This deletes the entire `upload` column.
2. **The checkpoint is 68 GB bf16, so it can never be page-cache resident.** turbo-fieldfare's
   model is 14.3 GB and fits in RAM on a 16 GB Mac, which is most of why its streaming is cheap. A
   pre-quantized NVFP4 repack would be ~17 GB and has a real chance of staying cached on a 32 GB
   M2 Pro, which would also collapse the 24s `io` column.

Separately, ~70s of the 132s is now outside the pool entirely: 680 layer-forwards at ~106 ms each.
That is the non-MoE model (GDN/attention/norms/shared expert) being launch-bound with `fusion`
disabled (see the `burn-fusion` note in `Cargo.toml`). Worth profiling once the streaming path is
fixed, but it is not the top item.
