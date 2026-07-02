# L2A.2 — Split-K flash-decode kernel: vetted design blueprint

_Synthesized by the `l2a2-splitk-design-review` Claude Workflow (3 design lenses + 3 adversarial-verify + 1 synthesis = 7 agents). Implement hands-on against this; verify each ranked risk via CPU-oracle parity. NOTE: workflow output was complete._

---

I have the grounding I need. All load-bearing facts are verified against the code: cache is seq-major `[B,T_max,Hkv,D]` (cache.rs:24,79), the current FA-2 kernel is head-major `[B,Hq,Sq,D]` single-thread (flash_attn.rs:58-140), `plane_sum` has **zero** in-repo precedent (grep returns only moe_grouped.rs:322's "NOT plane_sum" disclaimer), moe's split-K reduces cross-warp via `SharedMemory<Line<f32>>+sync_cube`, `VaSnapshot` has a fixed field set `{tok,pos,finished,last,kv}` (capture.rs:143), attention.rs physically repeats+movedims GQA (attention.rs:144-153,265-274), and candle's combine guards live at flash_fwd_kernel.h:618,1237,1245.

Here is the synthesized spec.

---

# L2A.2 — Split-K online-softmax flash-decode kernel: implementation blueprint

## 0. What this is and the one honest claim

Two `#[cube(launch)]` kernels — **`flash_decode_split`** (per-(head,batch,split) FA-2 over a contiguous KV chunk → normalized `O_g` + `lse_g` in global scratch) and **`flash_decode_combine`** (cross-split LSE reweight → `out`) — launched raw on `CubeBackend<CudaRuntime,…>` below Fusion, both `CubeCount::Static`, both `CubeDim{32,1,1}` (one warp, D-partitioned across lanes).

**Numeric claim, stated precisely:** the per-split recurrence is *byte-identical* to `flash_attn.rs:105-127` (same `m/alpha/p/l/acc`), and the split-K LSE merge is the *algebraically exact* global softmax. It is **NOT bit-identical** to `flash_attn.rs`, because the per-key `q·k` dot is reduced with a `plane_sum` butterfly instead of the sequential `for d in 0..D`, and the merge adds cross-split mul/adds. Expect ~1e-6 rel (the same concession `moe_grouped`'s split-K carries). Gate on token-identity + max-abs/max-rel + top-k overlap, never cosine alone. **S=1 is a tolerance anchor, not an equality anchor** (the plane butterfly alone breaks bit-equality even at one split).

## 1. The core architectural decision (resolves the three designs' disagreement)

The register budget at D=256 *forces* D-partition across the warp (a full-head-per-thread is 512 f32 → spill, F8). D-partition makes the acc update embarrassingly lane-parallel but forces a **cross-lane reduction of the QK dot per key**. A per-key `SharedMemory+sync_cube` reduction (moe's pattern) would put a block barrier inside the loop-carried KV scan — catastrophic. So the dot reduction *must* be a warp shuffle (`plane_sum`), which has **no in-repo precedent** — that is why it is risk #1.

**Latency/occupancy is hidden by the grid (`grid.z = S`, many blocks), NOT by intra-block KWARPS warps.** This is the decisive choice: it deletes the perf design's partial-empty-stripe NaN (F1-perf) and its undersized/mis-indexed smem buffer (F3-perf) *by construction* — there is no intra-block merge in the primary path. The KWARPS+smem variant is documented in §7 as a profiling-gated follow-on, with the guards those bugs require.

## 2. Kernel signatures

```rust
mod gpu {
    use cubecl::prelude::*;

    // PASS 1 — one warp per (q_head, batch, kv_split). D-partitioned across the 32 lanes.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_decode_split<EW: Float>(
        q:         &Tensor<Line<f32>>,   // [B, 1,     Hq,  D]  decode activation, lined over D
        k:         &Tensor<Line<EW>>,    // [B, T_max, Hkv, D]  KV-cache key, CACHE-NATIVE (GQA NOT expanded)
        v:         &Tensor<Line<EW>>,    // [B, T_max, Hkv, D]  KV-cache value, cache-native
        pos:       &Tensor<i32>,         // [B] DEVICE upper bound: visible keys are [lo_b ..= pos_b]  (INDEX, not count)
        lo:        &Tensor<i32>,         // [B] DEVICE lower bound: first visible key (0 = causal/uniform)
        o_accum:   &mut Tensor<f32>,     // [S, B, Hq, D]  per-split NORMALIZED output O_g
        lse_accum: &mut Tensor<f32>,     // [S, B, Hq]     per-split log-sum-exp lse_g
        scale:     f32,                  // 1/sqrt(D)                         (ScalarArg)
        n_rep:     u32,                  // Hq/Hkv                            (ScalarArg)
        #[comptime] head_dim: u32,       // 128 or 256  → derives dpl, v, line_rounds
        #[comptime] split_len: u32,      // T_max / S   (host constant, T_max % S == 0 asserted)
        #[comptime] n_splits: u32,       // S
    );

    // PASS 2 — one warp per (q_head, batch). D-partitioned; loops S in registers, NO cross-lane reduce.
    #[cube(launch)]
    pub fn flash_decode_combine(
        o_accum:   &Tensor<f32>,         // [S, B, Hq, D]
        lse_accum: &Tensor<f32>,         // [S, B, Hq]
        out:       &mut Tensor<f32>,     // [B, 1, Hq, D]
        #[comptime] head_dim: u32,
        #[comptime] n_splits: u32,       // comptime → the `for g in 0..S` merge unrolls
    );
}
```

Host wrapper `flash_decode_raw(q,k,v,pos,lo, scale, bucket) -> out` on `CaptureBackend`: dtype-dispatch `match k.dtype { BF16 => launch::<half::bf16,..>, F32 => launch::<f32,..> }` (moe pattern); **assert `hq % hkv == 0`, `head_dim ∈ {128,256}`, `D % (32*V) == 0`**; picks the bucket (→ S, split_len); holds the two scratch handles persistently (§6). Single-stream callers pass `pos:[1]`, `lo` = zeros[1] with B=1; **assert single-stream/uniform when a real `lo` is absent** (F4).

**Launch:** `CubeCount::Static(hq, bsz, S)` / `Static(hq, bsz, 1)`, `CubeDim{x:32,y:1,z:1}`; `q.as_tensor_arg(V_f32)`, `k/v.as_tensor_arg(V_ew)`, `pos/lo/o_accum/lse_accum/out.as_tensor_arg(1)`.

## 3. Grid, CubeDim, and the D-partition / Line layout (fixes F2-perf)

- **Grid split kernel:** `x = q_head` (0..Hq), `y = batch` (0..B, also the pos/lo index), `z = kv_split` (0..S). All extents are config/comptime constants per bucket → one constant capturable shape. `grid.z = S` is the SM-fill lever.
- **CubeDim:** `{32,1,1}` — exactly one warp = one plane, so `plane_sum` spans precisely the D-reduction and cannot straddle two heads/splits.
- **Line width is derived, NEVER probed:** `V = min(D/32, 128bits/sizeof(dtype))`; `dpl = D/32` (dims per lane); `line_rounds = dpl / V`.

| case | D | dpl=D/32 | V | line_rounds | f32/lane owned |
|---|---|---|---|---|---|
| 30B bf16 / f32 | 128 | 4 | 4 | 1 | 4 |
| 35B f32 | 256 | 8 | 4 (128-bit cap) | **2** | 8 |
| 35B bf16 | 256 | 8 | 8 | 1 | 8 |

The perf design's "one width-8 f32 line covers D/32=8 dims" is **illegal** (f32 line caps at 4 = 128-bit) and would silently cover 128 of 256 dims. Each lane owns `line_rounds` lines of width `V`; lane `ℓ`, round `r` owns line index `r*32+ℓ` → consecutive lanes read consecutive 16-byte lines ⇒ one coalesced `LDG.E.128` per lane per round. Use the **Line tensor's own reported strides in line units** (moe pattern), do NOT reuse the element-stride formula ×V (F7-corr).

## 4. Split kernel body (device-pos early-exit + FA-2 recurrence)

```
h  = CUBE_POS_X;  b = CUBE_POS_Y;  g = CUBE_POS_Z;  lane = UNIT_POS_X;
kv_h = h / n_rep;                                        // GQA in-register, no repeat (A4)

// ---- device-pos bound (F2): loop bound is a DEVICE read, never a host scalar ----
let n_keys = u32::cast_from(pos[b]) + 1;                 // visible keys [lo_b ..= pos_b]  (pos is an INDEX)
let lo_b   = u32::cast_from(lo[b]);
let start  = max(g*split_len, lo_b);
let end    = min((g+1)*split_len, n_keys);               // last split clamps to n_keys (F5 coverage)

// q into registers: line_rounds lines of width V (lane ℓ, round r → line r*32+ℓ over D)
// m,l,acc init:  m=-1e30 (finite sentinel), l=0, acc[round]=Line::empty(V).fill(0)

// ---- FA-2 recurrence, warp-distributed; start>=end ⇒ loop body never runs (block-granular early-exit) ----
for kj in start..end {                                   // warp-uniform bounds ⇒ zero divergence
    // per-lane partial dot over its dpl slice, then all-reduce over the 32 D-lanes
    let mut lane_partial = f32::new(0.0);
    for r in 0..line_rounds {                            // comptime unroll
        let kl = Line::<f32>::cast_from(k[k_line_base + r*32 + lane]);   // bf16→f32 in-register
        let prod = q_lines[r] * kl;                      // V-wide elementwise
        for c in 0..V { lane_partial += prod[c]; }       // sum the V components
    }
    let s = plane_sum(lane_partial) * scale;             // full scalar dot, UNIFORM on all 32 lanes

    // online softmax — IDENTICAL to flash_attn.rs:116-126, warp-uniform scalars
    let m_new = max(m, s);
    let alpha = (m - m_new).exp();
    let p     = (s - m_new).exp();
    l = alpha * l + p;
    for r in 0..line_rounds {                            // acc update is lane-parallel over D (no reduce)
        let vl = Line::<f32>::cast_from(v[v_line_base + r*32 + lane]);
        acc[r] = acc[r] * Line::empty(V).fill(alpha) + vl * Line::empty(V).fill(p);
    }
    m = m_new;
}

// ---- UNCONDITIONAL sentinel/normalized write EVERY step (F2: scratch is reused; never early-`return`) ----
if l > 0.0f32 {
    let inv = 1.0f32 / l;
    for r in 0..line_rounds { o_accum[o_line_base + r*32 + lane] = acc[r] * Line::empty(V).fill(inv); }
    if lane == 0 { lse_accum[g,b,h] = m + log(l); }      // exp(lse_g) = Σ_{k∈g} exp(s_k)
} else {                                                 // empty split (past pos OR front left-pad)
    for r in 0..line_rounds { o_accum[o_line_base + r*32 + lane] = Line::empty(V).fill(0f32); }
    if lane == 0 { lse_accum[g,b,h] = -1.0e30f32; }      // finite sentinel — NEVER m+log(0)=-inf
}
```

## 5. Combine kernel (candle's guards ported — fixes F3-corr / F1-perf)

Grid `Static(Hq,B,1)`, `CubeDim{32,1,1}`, D-partitioned. No cross-lane reduction — the O-sum is per-D-element.

```
lse_max = -1e30;  for g in 0..S { lse_max = max(lse_max, lse_accum[g,b,h]); }   // redundant per-lane, S≤64
if lse_max <= -1.0e30f32 { write O=0; return; }                                 // candle:1237 (all-empty)
let mut lse_sum = f32::new(0.0);
for g in 0..S { lse_sum += (lse_accum[g,b,h] - lse_max).exp(); }
if !(lse_sum > 0.0f32) { write O=0; return; }                                   // candle:1245 (sum==0 || NaN)
let lse_logsum = log(lse_sum) + lse_max;
for r in 0..line_rounds { acc[r] = Line::empty(V).fill(0f32); }
for g in 0..S {                                                                  // comptime unroll
    let sc = (lse_accum[g,b,h] - lse_logsum).exp();                             // =0 for empty splits
    for r in 0..line_rounds { acc[r] += o_accum[g-slice] * Line::empty(V).fill(sc); }
}
for r in 0..line_rounds { out[out_line_base + r*32 + lane] = acc[r]; }
```

Exactness: `sc_g·O_g = (Σ_{k∈g} exp(s_k)V_k)/Z` with `Z = Σ_g exp(lse_g)`, so `Σ_g sc_g·O_g = (Σ_k exp(s_k)V_k)/(Σ_k exp(s_k))` = the single-pass `acc/l`.

## 6. Scratch buffers + capture safety (reconciles F1-corr vs F3-capture)

The two critiques disagreed; the code decides it. `moe_grouped::run_fused_swiglu` **allocates its `gu`/`out` intermediates inside the captured closure via the client pool** and does **not** register them in `VaSnapshot`, and `cudagraph_moe_decode_bench` captures/replays that correctly — so pooled intra-arena intermediates get baked pool VAs that are stable on replay. `VaSnapshot` (capture.rs:105-140) has a **hardcoded** field set; "add o_accum to VaSnapshot" is unimplemented machinery.

**Decision (union-safe):** allocate `o_accum:[S,B,Hq,D]` + `lse_accum:[S,B,Hq]` **once at build**, sized to **max S over all context buckets**, hold the handles persistently in the flash-decode state, pass them to every step, and **fully overwrite them every step** (the unconditional sentinel write in §4 guarantees no stale partials). Add an explicit VA-equality assertion for the two handles to the flash-decode wrapper's own verify (belt-and-suspenders, since `VaSnapshot` won't track them). Nothing host-derived enters the captured region: grid is constant per bucket; the only per-step-varying inputs are `pos`/`lo`/K/V contents; launch is raw `CubeBackend`, not the `CubeCustomOp` Fusion bridge (A3). K/V are read cache-native — no `into_contiguous`, no GQA repeat inside the step.

## 7. Register budget for head_dim 256 + the KWARPS/smem variant

**Primary, D=256:** `dpl=8` → `q_reg` 8 f32/lane (line_rounds=2 lines of 4) + `acc` 8 f32/lane + `m,l,alpha,p,s` ~5 uniform + transient k/v lines (not held across keys). **~20-24 f32/lane, no spill** vs the naive 512 f32/thread. bf16 D=256 (V=8, line_rounds=1) is the same 8 f32/lane owned.

**smem-LSE-merge (variant only, if profiling shows the combine tail / block count starves latency):** add `y = KWARPS` warps, warp `w` scans its split's keys interleaved `kj = start+w, +KWARPS, …`. At block end each warp normalizes `O_w=acc_w/l_w`, `lse_w=m_w+log l_w` into `SharedMemory<Line<f32>>` sized `KWARPS * 32 * line_rounds` lines (**not** `[KWARPS]` — F3-perf: index `w*(32*line_rounds) + r*32 + lane`, or all 32 lanes race last-writer-wins and corrupt 31/32 of O), `sync_cube`, warp 0 merges. **Mandatory guard (F1-perf):** a warp with `end-start <= w` gets zero keys → `l_w=0`; you must write `O_w=0` (do NOT divide) and `lse_w=-1e30` (finite, not `-inf`), or `0/0=NaN` and `0*NaN=NaN` poisons the whole output. This is why the primary uses `grid.z=S` instead.

## 8. Hard preconditions (coupled changes, not in the kernel)

- **A4 attention.rs rewrite:** delete the `unsqueeze_dim(3).repeat(...).flatten(2,3)` GQA expansion (attention.rs:144-153, 265-274), the `movedim(1,2)`, and the K/V `into_contiguous`; pass cache-native `[B,T_max,Hkv,D]` + `n_rep`. If the physical repeat stays while the kernel also does `kv_h=h/n_rep`, K/V are double-indexed (silent wrong).
- **Context bucketing:** geometric T_max buckets {1K,4K,16K,64K}, each a separately captured graph; S sized so ~64-256 keys/split at the ceiling and Hq·B·S ≥ a couple SM waves. **KV buffer sized to the active bucket ceiling; scheduler promotes to the next bucket's graph before pos reaches the ceiling** — the kernel cannot assert context overflow on device (F8-corr).
- **Ragged GRPO** is a *precondition for GRPO use*, not optional: per-row `pos[b]` (index) + `lo[b]` (first valid). `pos` is an index (`n_keys=pos+1`); if the harness stores counts (`cache.rs seq_len()` returns `filled`), convert. Gate on the plan's CRITICAL logprob-parity regression against the mask+full-scan reference.

## 9. RANKED silent-wrong risks — verify these first against a CPU f32 oracle

1. **`plane_sum` lane-set / lowering on sm_121 at this pin.** Zero in-repo precedent — moe *deliberately* avoided plane ops (moe_grouped.rs:322). If it silently reduces a sub-warp or wrong mask, **every score `s_k` is wrong**. *Verify first, in isolation:* a micro-kernel where `plane_sum(lane_id) == 496` on all 32 lanes on the real GB10; then a single-(head,split) dot vs sequential CPU. Fallback: manual `plane_shuffle_xor` butterfly (offsets 1,2,4,8,16).
2. **Stale-partial merge + `-inf` sentinel.** Empty/past-pos splits must write `lse_g = l>0 ? m+log(l) : -1e30` and `O_g=0` **unconditionally every step** — a literal `if start>=end { return; }` leaves the *previous, longer-context* step's finite partials in the reused scratch → combine merges stale → silent wrong; and `m+log(0)=-inf` → `NaN`. *Verify:* decode at a short pos immediately after a long pos (stale finite scratch); parity + no NaN.
3. **Combine all-empty / NaN guards.** Port candle:1237 (`lse_max==sentinel → O=0`) and candle:1245 (`lse_sum==0 || NaN → O=0`). *Verify:* fully-masked ragged row (`lo>pos`) → O=0, no NaN into downstream layers.
4. **Ragged GRPO: cross-pad attention + index/count off-by-one.** Scalar `pos` with no per-row `lo` makes real tokens attend left-pad columns → wrong logprobs/ratio/KL, no crash. *Verify:* GRPO logprob-parity on a left-padded ragged batch vs the mask+full-scan reference (the plan's CRITICAL gate); assert single-stream when `lo` absent.
5. **Line-width vs D/32 at head_dim 256.** `V=min(D/32, 128b/dtype)`, `line_rounds=(D/32)/V`; f32 D=256 needs `V=4, line_rounds=2`. A single width-8 f32 line silently covers 128 of 256 dims. *Verify:* D=256 **f32 and bf16** parity — and make sure the f32 oracle path itself isn't the 128-of-256 truncation.
6. **GQA double-index / stride units.** `kv_h=h/n_rep` on cache-native `[B,T_max,Hkv,D]` using the **Line tensor's own line-unit strides**; requires the A4 repeat-drop. *Verify:* GQA ratio-8 parity (n_rep=8 on both 30B 32/4 and 35B 16/2).
7. **Split coverage gap.** `S*split_len` must cover `[0,T_max)`; assert `T_max % S == 0` comptime AND clamp the last split to `n_keys`. *Verify:* `n_keys` not a multiple of `split_len` → tail keys still scanned.
8. **VA-stability of scratch.** `o_accum`/`lse_accum` allocated once at max-S, held persistently, fully overwritten; add an explicit VA-equality assert (VaSnapshot won't track them). *Verify:* capture buffer-move mutation test extended to the two scratch handles.

**Numeric-gate caveat (not a bug, a test-design correction):** not bit-identical to `flash_attn.rs` (plane butterfly + merge reorder f32); S=1 is a ~1e-6 tolerance anchor, not equality. For tight debugging, optionally build a "reduction-order-matched" CPU oracle that mimics the butterfly + split order.

**Perf-goal risks (real, but NOT silent-wrong — do not let them block the correct kernel):**
- **n_rep× KV re-read at long context (capture F-1):** one q-head per block re-reads the shared kv-head 8×; at the 64K bucket a kv-head's K (~32 MB) exceeds L2 → 8× HBM exactly where flash-decode should win. **Follow-on:** GQA-group packing (grid `x=kv_head`, loop the n_rep q-heads reusing each K/V line once). Register-feasible at **D=128 (30B, the primary flash-decode beneficiary)**; **spills at D=256 (35B)** where it's acceptable because GDN carries long context and only 10/40 layers are full-attn. Keep the unpacked path for D=256.
- **1-warp occupancy / serial per-key chain (F6-corr/F6-capture):** hidden by `grid.z=S` (many blocks); tune S per bucket. KWARPS+smem variant (§7) only if Nsight shows starvation.
- **Idle CTAs at short ctx in a big bucket (all F7):** bounded by geometric bucketing; must clear the P0.4 empty-split probe before relying on the static grid.

**Key files for the implementer:** `src/flash_attn.rs` (FA-2 reference recurrence + `flash_attention_raw` launch pattern), `src/moe_grouped.rs:336-503` (Line/split-K/smem idioms; note :322 = no plane_sum precedent, and `run_fused_swiglu` :783-923 = pooled intra-arena intermediates), `src/capture.rs:105-174` (VaSnapshot fixed field set), `src/cache.rs:24,79` (cache-native `[B,T_max,Hkv,D]`), `src/attention.rs:144-153,265-274,335-336` (GQA repeat + movedim to delete for A4), `/workspace/candle/candle-flash-attn/kernels/flash_fwd_kernel.h:618,1237,1245` (empty-LSE sentinel + all-empty combine guards).
