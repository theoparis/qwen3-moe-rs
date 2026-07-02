# Split-K GEMV for batch-1 (M=1) weight-only decode — research

Research target: turn the fused-MoE decode GEMV in `src/moe_grouped.rs`
(`fused_swiglu_gu` / `fused_swiglu_down`) from a latency-bound, one-thread-per-output
serial-K-loop kernel into a bandwidth-bound, K-parallel kernel. Goal = the ~45 tok/s
bf16 roofline (currently ~43% of the 273 GB/s GB10 peak).

Method: web search only, via `THINK=HIGH /workspace/agy-direct.sh "… Search the web."`
(Gemini 3 Pro, grounded). CubeCL facts cross-checked against the pinned local checkout at
`/workspace/cubecl` (rev `b19859e`, `cubecl v0.10.0-pre.1`).

---

## 1. Why the current kernel is latency-bound (the diagnosis)

Each output element is computed by ONE thread running a serial reduction:

```
float acc = 0;
for (k = 0; k < K; ++k) acc += x[k] * W[k];   // RAW dependency on `acc`
```

- Iteration `k` cannot issue until iteration `k-1`'s FMA retires. An FMA has ~4–6 cycle
  pipeline latency, so the per-thread runtime floor is `K × FMA_latency` regardless of how
  fast memory is — a **dependency chain**, not a throughput problem.
- HBM is saturated only by **Memory-Level Parallelism (MLP)**: Little's Law says
  `in-flight bytes = bandwidth × latency`. To fill 273 GB/s at ~300 ns HBM latency you need
  ~80 KB of loads in flight at all times. A thread stalling 4–6 cycles on every FMA cannot
  keep enough loads outstanding, so the memory bus idles.
- In our kernel at T=1, k=8: `N = T·k = 8`. `fused_swiglu_gu` launches `N·I = 6144` threads
  each with a serial **H=2048** loop; `fused_swiglu_down` launches `N·H = 16384` threads each
  with a serial **I=768** loop. Few threads × long dependent chains = textbook latency-bound.

## 2. The split-K technique + the arithmetic

Split the K (contraction) dimension across the **P = 32 lanes of a warp (CubeCL "plane")**, one
plane per output element. The dot of length K becomes P partial sums, each over K/P:

```
S_t = Σ_{j=0..K/P-1}  x[t + j·P] · W[t + j·P]        (lane t, strided by P)
y   = Σ_{t=0..P-1} S_t
```

Two wins, both decisive at M=1:

1. **Dependency chain shrinks K → K/P.** gu: 2048 → 64 FMAs/lane; down: 768 → 24. The serial
   floor drops ~32×, and with a few unrolled accumulators per lane the chain effectively
   vanishes — the lane is now bound by load throughput, not FMA latency.
2. **Resident threads rise ~32×** (gu: 6144 → ~196 K; down: 16384 → ~524 K). Far more warps =
   far more loads in flight = MLP high enough to saturate HBM. The bottleneck moves off the SM
   pipeline and onto the memory pins — exactly where an M=1 weight-read kernel wants to be.

**Partial-sum reduction (warp-shuffle tree, log2 P = 5 steps for P=32):**

```
acc += shfl_down(acc,16); acc += shfl_down(acc,8); acc += shfl_down(acc,4);
acc += shfl_down(acc,2);  acc += shfl_down(acc,1);   // lane 0 holds the full dot
```

5 register-to-register cycles, no shared memory, no atomics. (Cross-*block* split-K instead
writes P partials to a workspace + a second reduce pass or `atomicAdd` — exllama does this and
it's the source of its decode non-determinism; we do NOT need cross-block split-K here.)

**Roofline check (why this is the right target):** batch-1 GEMV has arithmetic intensity
~1 FLOP/byte (bf16) — ~300× below the GB10 ridge point — so decode time ≈ weight-bytes / BW.
Per layer here ≈ 8 experts × (2·H·I + I·H) × 2 B ≈ 75 MB ⇒ ~0.27 ms/layer at 273 GB/s. The job
of the kernel is purely to keep HBM ≥90% saturated; split-K is the standard way to do it.

## 3. How the production batch-1 kernels do exactly this

- **llama.cpp / ggml `mul_mat_vec` (dmmv/`mul_mat_vec_q`):** `blockDim = (32, nrows)`, **1 warp
  per output row**, lanes interleave over K (thread t → k = t, t+32, …) with 128-bit vector
  loads, dequant in-register, then `__shfl_down_sync` warp reduction; lane 0 writes one output.
- **TensorRT-LLM / FasterTransformer weight-only GEMV:** 1–2 warps/row, fully-unrolled
  `LDG.E.128` loads; when N is small they add **global split-K** (P blocks per row → workspace →
  reduce) to fill all SMs.
- **exllama (GPTQ/EXL2):** aggressive split-K with `atomicAdd` accumulation into the output.
- **Marlin:** opposite extreme — pads M=1 → 16 and runs Tensor-Core MMA with `cp.async`
  prefetch; only worthwhile because compute is free relative to the memory fetch.

The common denominator for a hand-written SIMT batch-1 kernel = **warp-per-output-row +
lane-split-K + shuffle reduction.** That is precisely the shape to adopt here.

## 4. Applying it to `fused_swiglu_gu` / `fused_swiglu_down`

Replace "1 unit per output, serial K-loop" with "1 **plane** per output, lane-split-K +
`plane_sum`":

- **`fused_swiglu_gu`** (output `gu[n,i]`, reduction K = H = 2048): one plane per `(n,i)`.
  Lane `t` accumulates `gacc_t = Σ x[tok,h]·gate[e,h,i]` and `uacc_t` over `h = t, t+32, …`;
  then `g = plane_sum(gacc_t)`, `u = plane_sum(uacc_t)`; lane 0 writes `gu[n,i] = silu(g)·u`.
- **`fused_swiglu_down`** (output `out[n,h]`, reduction K = I = 768): one plane per `(n,h)`.
  Lane `t` accumulates over `i = t, t+32, …`; `acc = plane_sum`; lane 0 writes `acc·sel_w[n]`.

**Critical layout caveat (the one real engineering decision).** The current kernel is already
*coalesced over the OUTPUT dimension*: in `fused_swiglu_gu` adjacent output columns `i` map to
adjacent addresses (`gate` is `[E,H,I]`, stride over `i` = 1), and likewise `down` is `[E,I,H]`
(stride over `h` = 1). The **reduction** dimension is the *strided* one (gate stride over `h` =
I=768; down stride over `i` = H=2048). So a naive "lanes split the reduction dim" mapping makes
the 32 lanes read addresses 768/2048 elements apart — **uncoalesced** (32 separate sectors
instead of one 128-B transaction). Two clean ways out:

  (a) **plane_sum, simplest:** accept the strided per-lane loads. You still get the ~32×
      dependency-chain cut and ~32× MLP, which is the dominant fix at this occupancy; effective
      per-transaction efficiency is lower but the bus is far better fed than today. Lowest code
      risk — start here, measure.
  (b) **2-D block, coalescing-preserving (roofline-faithful):** make the block
      `(PLANE_DIM lanes over an output-column tile, KSPLIT over the reduction)`. Lanes keep the
      stride-1 output-column reads (coalesced as today), and the KSPLIT dimension does the
      K-split; reduce the KSPLIT partials per column through `SharedMemory` + `sync_cube`. This
      preserves 128-B coalescing AND adds K-parallelism — the version that actually reaches the
      bandwidth roofline.

**Cheapest pre-step (no reduction, do this first):** give each thread **4–8 independent
accumulators** (unroll the K-loop by 4–8 into `acc0..acc3`). That breaks the single dependency
chain into 4–8 independent chains (ILP) and recovers much of the latency hiding with a
one-line change and zero layout/occupancy work — a fast way to confirm the kernel is
latency-bound before committing to the full plane rewrite.

## 5. CubeCL feasibility — VERIFIED against the pinned rev

All primitives exist in `cubecl v0.10.0-pre.1` (this repo's pinned `b19859e`), confirmed by
grepping `/workspace/cubecl`:

- Plane ops: `plane_sum`, `plane_prod`, `plane_max/min`, `plane_broadcast`,
  `plane_shuffle{,_down,_up,_xor}`, `plane_inclusive_sum`, `plane_elect`, `plane_ballot`.
- Topology: `PLANE_DIM` (=32 on CUDA), `UNIT_POS_PLANE` (lane id), `CUBE_DIM_X`, `UNIT_POS`,
  `ABSOLUTE_POS`. Barriers: `sync_cube` (block), `sync_plane`. `SharedMemory::<T>::new(SIZE)`
  with a `#[comptime]` size.
- CUDA backend support is real, not theoretical: `crates/cubecl-cuda/src/runtime.rs` registers
  `Plane::Ops` and `Plane::Sync`, with `const_plane_size: 32` and
  `plane_size_min = plane_size_max = warp_size`. `plane_sum` lowers to PTX shuffle/`redux.sync`.
  A runtime test exists at `crates/cubecl-core/src/runtime_tests/plane.rs`.

Launch change in `run_fused_swiglu`: today `CubeDim { x: 256 }` with one unit per output. For
approach (a), launch one plane per output — e.g. `CubeDim { x: 32, y: planes_per_block }`,
`UNIT_POS_PLANE` = lane, the `(n,·)` output index derived from plane id; loop `for h in
(UNIT_POS_PLANE..H).step_by(PLANE_DIM)`, then `let g = plane_sum(gacc)`. Use `PLANE_DIM`, not a
hardcoded 32, for portability (degrades to 1 on the CPU oracle backend so the cross-backend law
still holds). Grid stays `CubeCount::Static` (T,k fixed) ⇒ still CUDA-graph capturable.

Sketch (gu inner loop):

```rust
let lane = UNIT_POS_PLANE;
let mut g = f32::new(0.0); let mut u = f32::new(0.0);
let mut h = lane;
while h < h_dim {
    let xv = x[ /* tok */ x_base + i64::cast_from(h)*xs1 ... ];
    g += xv * f32::cast_from(gate[ g_base + h*gs1 ]);
    u += xv * f32::cast_from(up[   u_base + h*us1 ]);
    h += PLANE_DIM;
}
let g = plane_sum(g); let u = plane_sum(u);
if lane == 0 { let s = 1.0/(1.0+(-g).exp()); gu[pos] = g*s*u; }
```

## 6. Recommendation

1. First, multiple-accumulator unroll (1-line, confirms latency-bound). 
2. Then plane-per-output + `plane_sum` (approach a) — biggest win per unit effort, primitives
   verified present on the CUDA backend. 
3. If still short of roofline, go to the 2-D block + shared-memory K-split (approach b) to
   restore 128-B coalescing of the strided weight reads. Keep `f32` accumulate (parity law) and
   `CubeCount::Static` (graph capture).

---

### Sources (web search, Gemini-grounded)
- llama.cpp / ggml `mul_mat_vec` warp-per-row + shuffle reduce; TensorRT-LLM/FasterTransformer
  weight-only GEMV global split-K; exllama atomic split-K; Marlin M→16 Tensor-Core pad
  (HuggingFace, github.com/ggerganov/llama.cpp, github.com/NVIDIA/TensorRT-LLM, Marlin arXiv,
  exllama PyPI/GitHub) — retrieved via agy web search.
- Split-K arithmetic / Little's Law MLP / `__shfl_down_sync` 5-step tree — agy web search
  (CUDA split-K GEMV grounding set).
- CubeCL plane API + CUDA backend — docs.rs `cubecl`, plus direct verification against the
  pinned local checkout `/workspace/cubecl` (rev b19859e).
