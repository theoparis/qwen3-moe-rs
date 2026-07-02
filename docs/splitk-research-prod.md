# Split-K / batch-1 MoE-GEMV roofline research (prod)

Web-search research (Gemini 3.1 Pro High, `agy-direct.sh`, 2026-06-29) grounding the
fused-MoE decode GEMV optimization in `src/moe_grouped.rs`
(`gpu::fused_swiglu_gu` / `gpu::fused_swiglu_down`). Sources cited inline.

## 0. The concrete kernel we are optimizing (current state)

`fused_swiglu_gu`: one thread per `(n, ci)` output element of `gu:[N,I]`; the thread runs a
**scalar, loop-carried-dependent K-loop over K = H = 2048**, reading `gate[e,h,ci]` / `up[e,h,ci]`
one bf16 element at a time, accumulating in a single f32 register.

`fused_swiglu_down`: one thread per `(n, hh)` of `out:[N,H]`; scalar dependent K-loop over
**K = I = 768**, reading `down[e,i,hh]` one element at a time.

At decode `M=1, T=1, k=8` ⇒ `N = T*k = 8` assignments ⇒ `gu` launches `N*I = 6144` threads and
`down` launches `N*H = 16384` threads, each doing a long serial dependent dot product. Measured:
**latency-bound at ~43 % of the 273 GB/s GB10 peak** (~117 GB/s achieved). Target ~45 tok/s bf16
roofline. Note the weight memory layout, which dictates which lever applies:
`gate/up` are `[E,H,I]` (reduction axis **H is strided by I**, output axis **I is innermost/contiguous**);
`down` is `[E,I,H]` (reduction axis **I is strided by H**, output axis **H is innermost/contiguous**).

## 1. Why batch-1 expert GEMV is *fundamentally* bandwidth-bound (the roofline)

[vLLM/SGLang `invoke_fused_moe_kernel`] At M=1 the expert FFN is a GEMV: to produce one token's
output against an expert you must stream the **entire** `K×N` weight tile once, doing exactly one
MAC per weight. Arithmetic intensity = `2KN FLOP / 2KN bytes = 1 FLOP/byte`. Modern accelerators
sit at hundreds of FLOP/byte (H100 ≈ 295), so the tensor cores are *starved* and **latency is set
entirely by HBM weight bandwidth**. Corollary: any work that does NOT reduce weight bytes moved
(e.g. M-padding waste on tensor cores) is free; the only thing that matters is *issuing wide,
many-in-flight, coalesced weight loads so the bus stays saturated*. The goal is not "less compute,"
it is "saturate the bus." (Source: HuggingFace MoE kernel writeups; vLLM `fused_moe` source.)

## 2. What the production kernels actually do at small M

- **vLLM / SGLang `fused_moe_kernel` (Triton).** Single grouped-GEMM with `sorted_token_ids` /
  `expert_ids` indirection (exactly the layout `dropless_align` already builds). Tiling
  `BLOCK_SIZE_M/N/K`; at decode `BLOCK_M` is forced to the tensor-core minimum (16/32/64) and the
  lone real token is **zero-padded** into it (≈15/16 rows wasted, but free — see §1). Crucially
  **`SPLIT_K` is hardcoded to 1** in the small-batch path: split-K inside a *dynamic* grouped GEMM
  needs atomic-add / workspace-reduction that is not worth it, so they fill the grid with
  N-dimension tiles × concurrently-active experts instead. `BLOCK_K` (64/128) is chosen for
  divisibility (`K % BLOCK_K == 0`) so the inner loop drops bounds masks → tighter software
  pipelining.
- **TensorRT-LLM low-latency MoE.** Grouped GEMM for prefill, but a dedicated **"small-M" /
  decode GEMV path** for M=1 that *bypasses tensor-core MMA and uses CUDA-core FMA* (no M-padding
  waste). Key techniques: **(a) split-K work decomposition** — partition the reduction (K/hidden)
  dim across CTAs so even one token fills all SMs, then **warp-shuffle (`__shfl_xor_sync`)
  reduction** of partials; **(b) weight-stationary *streaming*** with the **activation vector kept
  stationary in registers/SMEM** while weights stream once; **(c) 128-bit vectorized loads**
  (`ld.global.v4`) to maximize bytes/instruction; **(d) gate+up SwiGLU fusion** — gate/up tiles are
  interleaved/repacked so one CTA loads both and does SiLU·mul **in registers**, never writing the
  intermediate to HBM. (Sources: TRT-LLM docs / PyTorch blog on low-latency MoE; FasterTransformer
  `moe_gemm`.)

## 3. How a CUDA batch-1 GEMV reaches roofline (the transferable techniques)

From the GEMV-roofline search, three orthogonal levers turn a "1 thread / output, serial K-loop"
(exactly our kernel) into a bandwidth-saturating one:

1. **Break the loop-carried FMA dependency → ILP.** `acc += a*x` is a serial chain: the next FMA
   waits on the previous (latency-bound on FMA latency, *not* bandwidth). Unroll the K-loop and keep
   **multiple independent accumulators** (`acc0..acc3`); the compiler then issues independent
   loads+FMAs and hides latency. This is THE fix for a "latency-bound scalar dependent dot product."
2. **128-bit vectorized (`float4` / `bf16x8`) loads → MLP + coalescing.** One wide load fetches
   4–8 elements, cutting instruction count 4–8× and keeping more **bytes in flight**, which is what
   saturates HBM.
3. **K-dimension cooperative reduction (split-K / warp-shuffle).** A *warp* (32 lanes) cooperates on
   one output: each lane sums `K/32` partials, then a 5-step `__shfl_down`/`plane_sum` tree reduces.
   This raises occupancy and concurrent-request density **when there are too few output tiles to
   fill the GPU** — its main job is occupancy, not the dependency.

## 4. The concrete recommendation for `fused_swiglu_gu` / `fused_swiglu_down`

**Primary lever = register-blocking + multi-accumulator ILP + vectorized loads, NOT split-K.**
Reasoning grounded in *our* layout and problem size:

- We already launch 6144 + 16384 threads on a GB10 — **occupancy is not the bottleneck**, so split-K
  (whose main payoff is filling SMs at tiny M) is the *wrong primary lever* here. The bottleneck is
  per-thread: a 2048-/768-long *dependent* chain of *scalar* loads.
- The **output axis is already contiguous** (`ci` innermost in `[E,H,I]`; `hh` innermost in
  `[E,I,H]`), so consecutive threads in a warp already read consecutive weight elements per K-step
  ⇒ **per-warp loads already coalesce**. A naive K-split would instead read *down the strided
  reduction axis* (stride I for `gu`, stride H for `down`) ⇒ 32-way **scattered/uncoalesced**
  gathers — strictly worse. So the matched lever is to widen + parallelize along the contiguous
  output axis, not split the strided K axis.

Apply, per thread, in this order:

1. **Thread-coarsening over the contiguous output axis** (the TRT-LLM register-blocking idea): each
   thread computes a tile of `T_OUT = 4` (or 8) consecutive outputs — `gu[n, ci..ci+4]` (resp.
   `out[n, hh..hh+4]`). Keep `T_OUT` independent f32 accumulators (gives the ILP of §3.1) and load
   the weight stripe `gate[e,h, ci..ci+4]` as ONE **128-bit `Line` load** (§3.2, contiguous ⇒
   legal/coalesced). The shared `x[tok,h]` is loaded **once into a register and reused across the
   `T_OUT` outputs** (weight-stationary's dual: activation-stationary), cutting x re-reads `T_OUT`×.
2. **Vectorize the activation load too:** `x[tok, :]` is contiguous (`stride 1`), so load it as
   `Line<f32, 4>`; the weight reduction-axis load stays scalar/strided (can't vectorize across the
   strided H/I), but the *output-stripe* load is the one that vectorizes.
3. **`#[unroll]` the K-loop by 4–8** with the independent accumulators so the dependent chain is
   broken (§3.1).
4. **Keep gate+up fused** (already done in `fused_swiglu_gu`) — never spill the silu·mul to HBM.

This converts both kernels from "scalar dependent K-loop, 117 GB/s" toward "wide, many-in-flight,
ILP-rich streaming" — the same shape TRT-LLM's small-M CUDA-core path uses to sit near roofline.

**Secondary lever (only if still launch/occupancy-bound after the above, e.g. if N shrinks):**
a true **plane (warp) split-K** with `plane_sum`. To keep its loads coalesced you would first
**pre-transpose the persistent stacks so the reduction axis is innermost** (store `gate/up` as
`[E,I,H]` so the H reduction is contiguous; `down` as `[E,H,I]`). Then a warp-per-output split-K
both coalesces and fills the GPU. This is a bigger change (touches the one-time weight pre-stack in
`stacked_experts_pub`), so it is the *fallback*, not the first move.

## 5. CubeCL feasibility (verified against the patched checkout at `/workspace/cubecl`)

All primitives needed for the primary lever exist **today** in this repo's pinned CubeCL:

- **Vectorized loads — `Line<T>`.** `Lined` / `line_size` infrastructure is present
  (`cubecl-core/src/frontend/container`, `list.rs`, `element/base.rs`). The kernels currently call
  `as_tensor_arg(1)` (line size 1 = scalar). Bumping the **contiguous output axis** tensors to
  `as_tensor_arg(4)` (f32 x) / a bf16 line of 8 (128-bit) gives wide loads, *provided* the innermost
  dim is divisible — `I=768` and `H=2048` are both divisible by 4 and 8, so it is safe. The strided
  reduction-axis reads stay `as_tensor_arg(1)`.
- **Multiple accumulators + comptime unroll.** Plain CubeCL: hold `T_OUT` `f32` registers and use a
  comptime-bounded inner loop / `#[unroll]`; no new primitive. `i_cap`-style comptime sizing is
  already used in `grouped_swiglu`.
- **Warp/plane split-K (the fallback).** Fully supported: `plane_sum`, `plane_shuffle_xor`,
  `plane_shuffle_down/up`, `plane_broadcast`, plus runtime `PLANE_DIM` / `UNIT_POS_PLANE`
  (`cubecl-core/src/frontend/plane.rs`, `topology.rs`), all lowering to CUDA `__shfl_*` /
  `__syncwarp` (`cubecl-cpp/src/cuda/dialect.rs`). Burn itself already uses plane ops on this
  toolchain (`burn-vision` connected-components), so they are proven on the CUDA backend here.
- **Caveat / gotcha.** `plane.rs` warns line size is fixed to 4 at expand time even when
  `PLANE_DIM ≤ 64`; index with the runtime `PLANE_DIM`. And the persistent stacks are deliberately
  **never `into_contiguous`'d** (that copy is what lever (c) exists to avoid), so any `Line`
  vectorization must use the tensor's own strides and only vectorize the innermost contiguous axis —
  exactly the output-coarsening axis recommended in §4, which is compatible. The CUDA-graph capture
  path (`CubeCount::Static`, two launches) is unaffected: register-blocking/vectorization keep the
  grid static; only the per-launch `out_blocks`/`gu_blocks` count changes (divide by `T_OUT`).

## Sources

- vLLM/SGLang fused MoE Triton kernel, SPLIT_K=1 at decode, BLOCK_K divisibility, AI=1 FLOP/byte:
  HuggingFace MoE-kernels writeups + vLLM `fused_moe` source (Gemini grounded search, query set
  `invoke_fused_moe_kernel` / `fused_moe_kernel BLOCK_SIZE_M SPLIT_K`).
- CUDA GEMV roofline: warp-shuffle K-reduction, float4 loads, multi-accumulator ILP vs
  one-thread-per-output (Gemini grounded search).
- TensorRT-LLM low-latency MoE small-M GEMV path, split-K + `__shfl_xor`, weight-stationary
  streaming, 128-bit loads, gate/up SwiGLU repack/fusion, "MoE as Dense GEMM" + TMA on Blackwell:
  PyTorch blog + TRT-LLM docs (Gemini grounded search), URLs in transcript.
- CubeCL capability: verified directly in `/workspace/cubecl` (plane.rs, topology.rs,
  cuda/dialect.rs, container/Line).
