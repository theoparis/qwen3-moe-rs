# Vectorized 128-bit / float4 global loads for the fused-MoE decode GEMV

Research for optimizing `fused_swiglu_gu` / `fused_swiglu_down` in
`src/moe_grouped.rs` toward the ~45 tok/s bf16 roofline. Web-search–grounded
(Gemini 3.1 Pro High via `agy-direct.sh`); sources at the end. Findings
cross-checked against the **pinned** CubeCL (`cubecl 0.10.0-pre.1`, rev
`b19859e`; local `/workspace/cubecl` @ `2cafb76`) and the kernel source.

Device note: the target is **NVIDIA GB10 (Grace Blackwell, aarch64)** with
~273 GB/s **LPDDR5X unified memory** (not HBM). 45 tok/s ⇒ ~6 GB of weights
streamed per token. The web sources talk about H100/HBM at ~3 TB/s; the
*principles are bandwidth-class-independent* — only the absolute GB/s differs.

---

## 1. The core technique: vectorized 128-bit (`LDG.E.128`) loads

At batch-1 decode the GEMV `y = W·x` is **pure weight streaming** — arithmetic
intensity ≈ 0, so the only thing that matters is how fast weight bytes reach
registers, and the kernel is bound by **instruction issue / LSU throughput and
memory latency**, not by raw DRAM bandwidth.

A 128-bit vector load (`float4`/`uint4`/`int4` → SASS `LDG.E.128`) moves
**16 bytes per thread = 512 bytes per warp** in *one* instruction. Versus the
current **16-bit scalar** bf16 loads it buys:

- **8× fewer dynamic load + address-arithmetic instructions** (128b vs 16b),
  un-starving the warp schedulers so the bottleneck moves back onto memory.
- **8× more bytes in flight per instruction** ⇒ by Little's Law
  (`concurrency = BW × latency`) the same warp count keeps far more requests
  outstanding, which is *how* the ~400–600-cycle load latency gets hidden.
- **LSU / MSHR relief**: each tracked miss carries 16 B instead of 2 B, so the
  load pipeline stops throttling (the "LG Throttle" stall) before HBM/LPDDR is
  saturated.
- **Register-file efficiency**: `LDG.E.128` writes 4 contiguous 32-bit regs as
  one wide transaction.

**Important for THIS kernel — coalescing is already fine, vectorization is the
gap.** `fused_swiglu_gu` maps `pos = n·I + ci`, so consecutive threads hold
consecutive `ci` (the *contiguous* inner dim of `gate[E,H,I]`); a warp's 32
weight reads are already coalesced into 64 B. The problem is they are **16-bit
scalar** loads — the warp pulls a sector and uses half of it, and issues H
(=2048) such loads per thread. So the win here is **not fixing coalescing**, it
is **vectorizing** (fewer instructions, more MLP, LSU relief). This is exactly
the "16-bit scalar load is objectively bad … cutting effective bandwidth in
half" case from the sources.

## 2. Alignment requirements (bf16 weights)

- 128-bit load = **8 bf16 elements** (`128 / 16`). Address **must be 16-byte
  aligned**; a misaligned 128-bit load is a hard **`misaligned address`**
  fault on Ampere+/Blackwell (or silent wrong-address / compiler de-vectorizes
  into scalar loads — losing the win).
- `cudaMalloc` base is ≥256-byte aligned. **A row is 16-byte aligned iff the
  inner dim is a multiple of 8.** Here `I=768` and `H=2048` are **both
  multiples of 8**, and every per-expert/per-row offset (`e·H·I`, `h·I`,
  `e·I·H`, `i·H`) is a multiple of 8 elements ⇒ **width-8 line loads are
  automatically aligned**. (Idiom: `reinterpret_cast<uint4>` then unpack to
  `__nv_bfloat162` ×4; in CubeCL this is `Line<bf16>` of size 8.)

## 3. The two ways to apply it to `fused_swiglu_*`

Both current kernels are *one thread per output element, scalar dependent
K-loop*. The reduction dim is the **strided/outer** weight dim in both:
`gu` reduces over `h` (stride `I` in `[E,H,I]`); `down` reduces over `i`
(stride `H` in `[E,I,H]`). Two options:

**(A) Vectorize the contiguous OUTPUT dim — zero layout change.**
Each thread computes a **Line of 8 adjacent outputs**; the weight is read as one
`Line<bf16>` width-8 (128-bit `LDG`), the activation is a broadcast scalar
(`Line::new(xv)`), accumulate `Line<f32>` ×8.
- `gu`: vectorize over `i` (gate/up `[E,H,I]`, `i` contiguous).
- `down`: vectorize over `h` (down `[E,I,H]`, `h` contiguous).
- Pros: no transpose; alignment trivially holds; smallest diff.
- Con: **cuts thread count 8×** (`gu`: 6144→768 = ~24 warps; `down`:
  16384→2048 = ~64 warps) → lower occupancy, which at batch-1 can *hurt*
  latency hiding. Good as a quick A/B; watch Nsight occupancy.

**(B) Vectorize the K REDUCTION dim — textbook batch-1 GEMV (recommended).**
Make K contiguous via a **one-time transpose of the cached stacks**
(`gate/up → [E,I,H]`, `down → [E,H,I]`). Then keep one thread per output and
walk K in steps of 8, loading the **weight** `Line<bf16>(8)` = single 128-bit
`LDG` **and** the activation `Line<f32>` together, `fma`-accumulate a
`Line<f32>`, and reduce the 8 lanes at the very end.
- Vectorizes *both* operands, cuts loop trips 8× (2048→256, 768→96), and
  **keeps the full thread count (6144 / 16384)** → preserves occupancy.
- The one-time transpose folds into the already-planned "pre-stacked contiguous
  weight cache" (no per-call re-stack; the module already forbids per-call
  `into_contiguous` of the stacks).
- bf16 weight at width-8 = one 128-bit load (the bandwidth-dominant operand);
  f32 activation at width-8 = two 128-bit loads, which is irrelevant — `x`
  (8 KB) is tiny and L1/L2-resident.

**(B+) Refill the GPU with split-K + warp-per-row (`plane_sum`).** N = k·T is
only **8** at T=1; even 16384 threads is little work for a multi-SM GPU, which
is *why* it reads latency-bound at 43%. The standard fix (llama.cpp
`mul_mat_vec`, FlashDecoding, cuBLAS tall-skinny GEMV) is **warp-per-output +
split-K**: assign a warp (or several) per output row, each lane vector-loads a
slice of contiguous K, then a `__shfl_down`/`plane_sum` tree reduce; split K
into G chunks to multiply resident warps and saturate every SM. Layer this on
top of (B) once vectorization lands.

**Recommendation:** do **(B)** (transpose-cache + width-8 line loads, both
operands) as the primary change for the bandwidth + occupancy win; keep **(A)**
as a zero-layout-change sanity experiment; add **split-K/`plane_sum`** if Nsight
still shows idle SMs / long-scoreboard stalls at N=8.

## 4. CubeCL feasibility (verified on the pinned rev)

- **Line size is already the `as_tensor_arg` arg** — the repo passes
  `as_tensor_arg(1)` (scalar) *everywhere today*; switch the weights to
  `&Tensor<Line<EW>>` and launch with `as_tensor_arg(8)`; declare `gu`/`out`
  outputs as `Line<f32>`. (`as_tensor_arg(line_size: LineSize)`,
  `launch.rs:184`.)
- **Line math is element-wise**: `Exp for Line<P>` and `Recip for Line<P>`
  exist (`line/ops.rs:247,265`) ⇒ silu `g/(1+exp(-g))` vectorizes over the 8
  lanes; `Line::new(scalar)` broadcasts the activation (`line/base.rs:43`);
  the `fma(a,b,c)` op works on Line primitives (`operation/fma.rs`).
- **`plane_sum` is present and tested at line sizes 1/2/4** (`runtime_tests/
  plane.rs`) for the strategy-(B+) warp reduction. `PLANE_DIM` = warp width.
- **Max line size 8 is supported** (`client.rs:1213` "max is 8 → 1,2,4,8");
  use the `try_tensor_line_size_perpendicular(shape, strides, axis)` helper
  (`lib.rs:160`) to pick the largest safe line size and **fall back to 1**,
  which also enforces the alignment/contiguity requirement automatically.
- **Caveat**: Line vectorization indexes in *line units along a stride-1 axis*.
  The current kernels index the stacks by explicit `i64` stride math — for
  width-8 lines the vectorized axis must be the inner (stride-1) dim. (A) uses
  the existing inner dim; (B) requires the transpose so K becomes inner. The
  64-bit-offset rule still applies (compute the line-unit offset in `i64`).

---

### Sources (Gemini 3.1 Pro High, googleSearch grounding)
- Vectorized 128-bit loads / instruction count / MLP / LSU throttle / "16-bit
  scalar is bad": wingedge777.com; baai.ac.cn (via vertexaisearch grounding
  redirects). Queries incl. `"float4" "LSU" throughput CUDA "memory-bound"`,
  `"float4" "memory-level parallelism" CUDA`.
- bf16 128-bit alignment (16-byte, 8 elems, misaligned fault, multiple-of-8
  rows, `int4`+`__nv_bfloat162` unpack): massedcompute.com; stackoverflow.com.
- Batch-1 GEMV design (1-thread vs warp-per-row, K must be stride-1, split-K,
  multiple-rows-per-warp, llama.cpp `mul_mat_vec` / FlashDecoding / cuBLAS):
  Gemini synthesis (queries `"CUDA" "GEMV" "batch 1" LLM`, `"llama.cpp"
  "mul_mat_vec"`, `"FlashDecoding" "split-K"`).
- CubeCL `Line`/`as_tensor_arg`/`plane_sum`/`#[comptime] line_size`: docs.rs +
  github.com (tracel-ai), cross-checked against the local pinned checkout.
