# Split-K warp-reduced GEMV in CubeCL — research for the fused-MoE decode kernels

Target: `src/moe_grouped.rs` kernels `fused_swiglu_gu` / `fused_swiglu_down`.
Goal: lift the M=1, T=1, k=8 decode GEMV from ~43 % of 273 GB/s toward the ~45 tok/s bf16 roofline.

Authoritative source = the **local** CubeCL the project actually compiles against:
`Cargo.toml` redirects `tracel-ai/cubecl` via `[patch]` to `/workspace/cubecl/crates/*`
(git rev `2cafb76`, pkg `cubecl-core 0.10.0-pre.1`). All file:line cites below are from there.

---

## 1. The diagnosis (why it's slow)

Both kernels are **one thread per output element running a scalar dependent K-loop**:

* `fused_swiglu_gu`: thread per `(n,i)` of `gu[N,I]`; serial loop over **K = H = 2048** building
  `gacc`/`uacc`. At N=8, I=768 → **6144 threads ≈ 192 warps**, each a 2048-long dependent FMA chain.
* `fused_swiglu_down`: thread per `(n,h)` of `out[N,H]`; serial loop over **K = I = 768**. At N=8,
  H=2048 → **16384 threads ≈ 512 warps**, each a 768-long chain.

That is the textbook **latency-bound M=1 GEMV** failure mode (confirmed by web search §5): too few
resident warps + a long per-thread dependent chain ⇒ not enough **outstanding memory requests
(memory-level parallelism, MLP)** to fill the HBM pipe, so achieved BW stalls well under peak. Each
weight byte *is* already read exactly once (good), so the fix is not less traffic — it is **more
loads in flight + a shorter dependency chain**.

## 2. The technique — split-K plane (warp) reduction

Assign **one plane (warp = 32 units) per output element** instead of one thread. The 32 lanes split
the contraction dim K: lane `j` accumulates a partial dot product over the strided slice
`k = j, j+32, j+64, …`; then a single `plane_sum` collapses the 32 partials; lane 0 writes the result.

Effect (web §5, NVIDIA CUTLASS split-K / vectorized-memory, Flash-Decoding, vLLM GEMV):
* **32× more in-flight loads** → fills the HBM request queue → higher achieved BW.
* per-thread dependent chain **K → K/32** (2048→64 for gu, 768→24 for down).
* resident warps jump from ~192/512 to ~6144/16384 — kills the low-occupancy stall.
* opens the door to **coalesced / vectorized 128-bit loads** (`Line<T>`, §4).

## 3. CubeCL primitives that exist (verified in source)

Plane ops — `crates/cubecl-core/src/frontend/plane.rs`:
* `plane_sum<E: CubePrimitive>(value: E) -> E`            (:213)  ← the reduction we need
* `plane_prod / plane_max / plane_min`                    (:310/407/433)
* `plane_inclusive_sum / plane_exclusive_sum`             (:245/281)
* `plane_broadcast<E>(value, index) -> E`                 (:35)
* `plane_shuffle / _xor / _up / _down<E>(value, …)`       (:71/110/146/182)
* `plane_elect() -> bool`, `plane_all/any(bool)`, `plane_ballot(bool) -> Line<u32>` (:11/459/483/511)

`plane_sum` is generic over `CubePrimitive`, so it reduces a **scalar OR a `Line<E>` lane-wise** —
confirmed by the runtime test `crates/cubecl-core/src/runtime_tests/plane.rs` (`kernel_sum`, tested
at vectorization 1/2/4).

Topology constants — `crates/cubecl-core/src/frontend/topology.rs`:
* `PLANE_DIM` (warp width, 32 on CUDA), `UNIT_POS_PLANE` (lane id), `PLANE_POS` (plane index in cube),
  `UNIT_POS`, `CUBE_DIM_X`.

Sync — `crates/cubecl-core/src/frontend/synchronization.rs`: `sync_cube()` (:17), `sync_plane()` (:30)
(for the SharedMemory fallback, §4b).

Canonical pattern (verbatim shape of the test kernel + launch):
```rust
#[cube(launch)]
fn kernel_sum<F: Float>(output: &mut Tensor<F>) {
    let val  = output[UNIT_POS as usize];
    let red  = plane_sum(val);
    if UNIT_POS == 0 { output[0] = red; }
}
// launch with one plane per cube:
kernel_sum::launch::<F, R>(&client, cube_count, CubeDim::new_1d(32), handle);
```

Feature gate (CUDA supports it): `client.properties().features.plane.contains(Plane::Ops)`
(`use cubecl_ir::features::Plane;`). The CUDA runtime inserts `Plane::Sync` and `Plane::Ops`
unconditionally — `crates/cubecl-cuda/src/runtime.rs:182,271` — so GB10 has it.

## 4. Vectorized loads — `Line<T>`

`crates/cubecl-core/src/frontend/container/line/base.rs` + `…/line/ops.rs`:
* `Line::<P>::new(val)`, `Line::empty(#[comptime] size)`, `.fill(v)`, `.line_size()`, `.size()` (base.rs:43/103/78/53/135)
* operators `Add`,`Mul`,`AddAssign`,`MulAssign` element-wise, plus **scalar broadcast** `Line<P> * lit` / `+ lit` (ops.rs:25/49/73/103/372/394) — so `acc += x_line * w_line` and `line * scalar` work.
* index a lane `l[k]` to read/write a single element.
* **Caveat:** this rev has **no built-in `Line::sum()` lane-reduce** (only new/empty/fill/size/cmp/and/or).
  To collapse a `Line<L>` to a scalar, sum its lanes by comptime indexing `l[0]+l[1]+…`. (A web answer
  claimed `line.sum()` — **not present in rev 2cafb76**; do not rely on it.)
* Vectorization is set **at launch**: `handle.as_tensor_arg(line_size)` /
  `TensorArg::from_raw_parts::<T>(&h,&strides,&shape, line_size)` (test uses this). Kernel param type
  becomes `&Tensor<Line<EW>>`. Requires the **contiguous (last) dim** divisible by `line_size` and aligned.

Layout fit (which dim is contiguous decides what `Line` vectorizes):
* `gate/up : [E,H,I]` → contiguous in **I** (768 % 8 = 0 ✓) — vectorize the *output* i.
* `down : [E,I,H]` and `x : [T,H]` → contiguous in **H** (2048 % 8 = 0 ✓) — vectorize the *output* h / the x operand.
* bf16 `Line` of 8 = one 128-bit coalesced load (the `float4`/`half8` win of web §5C).

So the two levers are **orthogonal**: `plane_sum` parallelizes the **contraction** dim (H for gu, I for
down); `Line` vectorizes the **contiguous/output** dim. They compose (a plane of 32 lanes, each holding a
`Line<L>` f32 accumulator) but step 1 is the plane reduction alone — it is the direct latency fix.

## 5. How to apply it (concrete)

### gu kernel — split-K over H, plane per `(n,i)`
```rust
#[cube(launch)]
fn fused_swiglu_gu<EW: Float>(x, gate, up, assign_e, assign_tok, gu, h_dim, i_dim) {
    let out_id = CUBE_POS_X * planes_per_block + PLANE_POS;   // one plane per (n,i)
    if out_id < N*I {
        let n = out_id / i_dim;  let ci = out_id % i_dim;
        let lane = UNIT_POS_PLANE;                            // 0..32
        let e = assign_e[n]; let tok = assign_tok[n];
        // ... i64 base offsets as today (gate[e,:,ci], up[e,:,ci], x[tok,:]) ...
        let mut g = f32::new(0.0); let mut u = f32::new(0.0);
        let mut hh = lane;                                    // strided K-loop
        while hh < h_dim {                                    // 64 iters, not 2048
            let xv = x[/*x_base + hh*xs1*/];
            g += xv * f32::cast_from(gate[/*g_base + hh*gs1*/]);
            u += xv * f32::cast_from(up[  /*u_base + hh*us1*/]);
            hh += PLANE_DIM;
        }
        let g = plane_sum(g); let u = plane_sum(u);           // 32→1
        if lane == 0 { let s = 1.0 / (1.0 + (0.0 - g).exp()); gu[out_id] = g*s*u; }
    }
}
```
Launch: `CubeDim{ x:32, y:planes_per_block, z:1 }`, `CubeCount::Static(ceil(N*I/planes_per_block),1,1)`
— still **Static** (grid fixed for fixed T,k), so the CUDA-graph capture path in
`run_fused_swiglu` is preserved. (`PLANE_POS` indexes the plane within the cube; pack e.g. 8 planes =
256 threads.) Keep accumulators **f32** even when `EW=bf16` (cast each load) — matches today's f32-accumulate
parity contract.

### down kernel — split-K over I, plane per `(n,h)`
Identical shape: lane sums `acc += gu[n,ci] * down[e,ci,h]` over `ci = lane, lane+32, …` (24 iters,
not 768), `acc = plane_sum(acc)`, lane 0 writes `out[n,h] = acc * w`.

### Optional follow-on (compose with Line)
Give `gu` a `Line<EW>` over **I** (gate/up contiguous in I): one plane per `(n, i_block)`, each lane
holds a `Line<8>` f32 accumulator, `plane_sum` reduces lane-wise (works on `Line`), lane 0 unpacks &
writes 8 outputs. For `down`, vectorize the `x`/`down` reads over **H**. This adds the 128-bit-load
coalescing on top of the MLP win. Build it only after the plain plane version is measured.

## 6. CubeCL feasibility notes

* ✅ `plane_sum` + topology constants + `CubeDim::new_1d(32)` are all present and CUDA-supported
  (`Plane::Ops`/`Plane::Sync` always inserted by the CUDA runtime). Gate on
  `features.plane.contains(Plane::Ops)` to stay portable.
* ✅ Stays `CubeCount::Static` ⇒ the below-Fusion CUDA-graph capture path is unaffected.
* ✅ Stacks still read **by stride** (no `into_contiguous`, no re-stack) — the plane only changes which
  unit reads which k; offsets stay the i64-before-multiply form already in the file.
* ⚠️ `Line::sum()` does **not** exist in this rev — reduce lanes by comptime indexing.
* ⚠️ **Numerics/parity:** plane reduction changes float summation **order** vs the serial loop. With
  f32 accumulation this stays within the ~1e-8 fp-order tolerance the module already documents against
  the dense oracle, **but** the end-to-end GRPO logprob-parity gate (the file's open P1) must be re-run
  after the change — order change is parity-relevant, magnitude is not.
* Occupancy math: plane-per-output multiplies resident threads 32× (gu 6144→196 608, down 16384→524 288),
  which is exactly the MLP lift a latency-bound GEMV needs to approach the 273 GB/s / ~45 tok/s roofline.

---

## Sources

CubeCL source (local, rev 2cafb76 = what the project builds): `frontend/plane.rs`,
`frontend/topology.rs`, `frontend/synchronization.rs`, `frontend/container/line/{base,ops}.rs`,
`runtime_tests/plane.rs`, `cubecl-cuda/src/runtime.rs`, `cubecl-ir/src/features.rs`.

Web (Gemini 3 Pro, live googleSearch grounding):
* CubeCL plane primitives & topology — github.com/tracel-ai/cubecl, docs.rs/cubecl, thomasantony.com.
* `Line<T>` vectorization / `as_tensor_arg(vectorization)` — cubecl docs.rs + repo gelu/matmul examples.
* Warp-per-row / split-K GEMV rationale (MLP, shorter chain, float4 coalescing) — NVIDIA "Vectorized
  Memory Access" pro-tip, CUTLASS split-K docs, Flash-Decoding (Tri Dao et al.), vLLM/FasterTransformer
  M=1 GEMV kernels.
</content>
</invoke>
