# L2C — NVFP4 SIMT decode-GEMV: vetted design blueprint

_Synthesized by the `l2c-nvfp4-decode-gemv-design` Claude Workflow (3 lenses + 3 adversarial-verify + synthesis = 7 agents). Implement hands-on/subagent against this; verify each ranked risk vs an f32 reference + the bf16 token-parity gate. Output was complete._

---

I have verified every load-bearing claim against source. Here is the synthesized blueprint.

---

# L2C — NVFP4 SIMT decode-GEMV: synthesized implementable blueprint (GB10 / sm_121)

Greenfield: `src/nvfp4.rs` (codec + kernel) + `src/nvfp4_linear.rs` (Linear wrapper + `QuantLinear` enum). No NVFP4 source exists yet. Everything below is grounded in verified source; the three designs agreed on the skeleton and disagreed on ~9 concrete points — each resolution is called out with the finding it kills.

**Verified facts this rests on** (active cubecl is the local `/workspace/cubecl` via `Cargo.toml:270` `[patch]`, not the `b19859e` git rev):
- `e2m1x2` is `#[repr(transparent)] struct e2m1x2(u8)` = **1 byte** (`/workspace/cubecl/crates/cubecl-common/src/float/fp4.rs:28`). ⇒ a `Line<e2m1x2>` of width `WV=8` is **8 bytes = 64-bit**, and 8 bytes = 16 fp4 = **exactly one 16-block**. (Kills codec-lens "WV=8 → 128-bit"; confirms perf-lens F6.) A 128-bit load needs `WV=16` = 2 blocks/scale-decodes.
- Packing (`fp4.rs:199-211`): `a = e2m1::from_f32(first) & 0x0F` (**low nibble = even K = 2i**), `b = (e2m1::from_f32(second) << 4) & 0xF0` (**high nibble = odd K = 2i+1**). Consecutive along K.
- `Line::<f32>::cast_from(Line<e2m1x2>@WV)` widens to `WV * packing_factor(2) / 1 = 2·WV` f32 (`cubecl-core/src/frontend/element/cast.rs:20-22`), element `2j`=low nibble of byte `j`, `2j+1`=high. So `vals[p]` ↔ `K = blk*16 + p` directly.
- E2M1 saturates out-of-range/Inf/NaN → **±6, silently** (`fp4.rs:32` MAX=6.0; float4 crate `lib.rs:65-73`). ⇒ a zero scale gives `W/0=inf → e2m1=±6 → dequant 6·0=0`: **whole block silently zeroed, no NaN crash**. This is the dangerous failure mode below.
- `w8a16_gemm` reaches the GPU **through `CubeCustomOp`** (`src/w8a16.rs:291`) = the Fusion bridge; `capture.rs:3-4,26` mandates raw `CubeBackend<CudaRuntime,f32,i32,u8>` below Fusion. ⇒ the existing FP8 fallback is **NOT capture-runnable** (confirms codec-lens F2).
- Dual-backend raw-launch pattern to mirror: `moe_grouped.rs:1010-1099` (`FusedSwigluBackend` for both `Cuda` and `CubeBackend<…>`). Typed exotic-dtype `TensorArg::from_raw_parts::<e2m1x2>/<e4m3>`: `nvfp4_gemm_probe.rs:453-476,519-543`. plane_sum + comptime-array-in-register + runtime-scalar discipline: `flash_decode.rs:45,62-83`. Persistent VA-stable buffers: `capture.rs:105-174`.

---

## 1. Codec — host f32 → NVFP4, run ONCE at load (`src/nvfp4.rs`)

Two-level, per **output column** n over 16-wide K-blocks. Mirrors `quantize_e4m3_per_channel` (`w8a16.rs:83-119`) including its `.max(f32::MIN_POSITIVE)` floor — **which the codec/perf/gate designs all dropped (F1/A1/A1, unanimous CRITICAL).**

Constants: `E2M1_MAX=6.0`, `E4M3_MAX=448.0`, `E4M3_MIN_NORMAL = 2f32.powi(-6) = 0.015625`.

```
fn quantize_nvfp4(w: &[f32], k: usize, n: usize) -> (Vec<u8> qw[N,K/2], Vec<u8> bs[N,K/16], f32 gscale)
  assert all finite (w8a16.rs:88 discipline)
  amax = max|w|                                        // (calibrated: see hook)
  gscale = (amax / (E2M1_MAX*E4M3_MAX)).max(f32::MIN_POSITIVE)   // = amax/2688, FLOORED (F1 fix)
  for n, for block b (16 K-vals of column n):
    bamax = max_{k in b} |W[k,n]|
    if bamax == 0.0:                                    // dead/padded block (F1 fix)
        bs[n,b] = f32_to_e4m3(E4M3_MIN_NORMAL); write q=0 for all 16 vals; continue  // dequant≡0, exact
    sb_ideal = bamax / (E2M1_MAX * gscale)             // lands ≤448 (bamax≤amax)
    bs[n,b]  = f32_to_e4m3( sb_ideal.max(E4M3_MIN_NORMAL) )         // FLOORED before encode
    // RECONSTRUCT-then-QUANTIZE (codec-lens crux): quantize against the ROUNDED byte, not the ideal,
    // so E4M3 rounding is absorbed into the E2M1 choice → per-elem error bounded by E2M1 grid alone.
    S_b = e4m3_to_f32(bs[n,b]) * gscale                 // S_b > 0 guaranteed
    for k in b: q[k,n] = e2m1::from_f32(W[k,n] / S_b)   // nearest on {0,±.5,±1,±1.5,±2,±3,±4,±6}
  pack q along K with e2m1x2::from_f32_slice semantics (low=even K, high=odd K) → qw[N, K/2]
```

- **Layout (column-major, transposed from Burn `[K,N]` ONCE here):** `qw:[N,K/2]` e2m1x2, `bs:[N,K/16]` E4M3, `gscale:[1]` f32. Bytes/weight `= 1/2 + 1/16 = 0.5625 B` = 3.56× < bf16, 1.78× < FP8. Block `b` of column `n`: e2m1x2 bytes `[b*8, b*8+8)`, K-vals `[b*16, b*16+16)`, scale at `bs[n*(K/16)+b]`.
- **Calibration hook (D6, mandatory for sensitive tensors):** the only change is `amax → calibrated stat` (AWQ/SmoothQuant/percentile-clipped from a calibration corpus). Structure unchanged. **Re-derive `gscale` from the POST-calibration amax** (kills A3: clipping below the true amax else saturates outliers to 448).
- **One canonical codec** (`quantize_nvfp4` + `dequant_nvfp4` inverse) shared by host quantizer, CPU oracle, golden vectors, and — bit-identically — the kernel decode (the w8a16 discipline). Reject non-finite up front.

---

## 2. Kernel — `#[cube(launch)]` SIMT GEMV (`src/nvfp4.rs mod gpu`)

One **warp per output column**, 32 lanes split-K over the K/16 blocks, M batch rows amortized in-register (each weight byte read once regardless of M≤8), plane_sum reduction. Mirrors `flash_decode.rs` + `moe_grouped.rs`.

```rust
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_decode_gemv(
    x:      &Tensor<f32>,          // [M, K] activations f32
    qw:     &Tensor<Line<e2m1x2>>, // [N, K/2] packed weight, Line width WV=8 over K (contiguous)
    bs:     &Tensor<e4m3>,         // [N, K/16] E4M3 block scales (K/16 innermost => coalesced)
    gscale: &Tensor<f32>,          // [1] persistent FP32 global scale (device-read; capture-safe)
    out:    &mut Tensor<f32>,      // [M, N]
    m_dim:  u32,                   // RUNTIME batch (<= M_MAX): guards x-reads/out-writes only
    #[comptime] k:      u32,       // reduction dim (one compiled kernel per K)
    #[comptime] blocks: u32,       // K/16
    #[comptime] wv:     u32,       // 8 (one 16-block per lane-step, 64-bit LDG)
    #[comptime] m_max:  u32,       // register-array bound: SET = the fixed decode batch (1 for greedy)
) {
    let col  = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;   // output column n
    let lane = UNIT_POS_X;                              // 0..32, split-K
    if col < out.shape(1) {                            // TAIL GUARD (C2 fix)
        let g = gscale[0];
        let mut acc = Array::<f32>::new(m_max);        // comptime-sized => stays in registers
        #[unroll] for m in 0..m_max { acc[m] = f32::new(0.0); }
        let lines_per_col = blocks;                     // WV=8 => 1 line == 1 block == 1 scale
        let mut blk = lane;
        while blk < blocks {                            // lane owns blocks {lane, lane+32, ...}
            let line = qw[(col*lines_per_col + blk) as usize];       // ONE 64-bit coalesced LDG (F4 fix)
            let vals = Line::<f32>::cast_from(line);                 // width 2*WV=16; vals[p] <-> K=blk*16+p
            let s = f32::cast_from(bs[(col*blocks + blk) as usize]); // E4M3 scale, coalesced across lanes
            let k0 = blk * 16;
            #[unroll] for m in 0..m_max {
                let mut part = f32::new(0.0);
                #[unroll] for p in 0..16u32 {
                    // m < m_dim guards OOB x-read for unused rows when m_max>1 (C1 fix)
                    if m < m_dim { part += vals[p] * x[(m*k + k0 + p) as usize]; }
                }
                acc[m] += s * part;                     // block scale factored out: 1 mul/block (B2 fix)
            }
            blk += 32;
        }
        #[unroll] for m in 0..m_max {
            let full = plane_sum(acc[m]);               // sum 32 lanes' partials => full column dot
            if lane == 0 && m < m_dim { out[(m*out.shape(1) + col) as usize] = g * full; } // gscale once
        }
    }
}
```

- **grid/CubeDim:** `CubeCount::Static(N.div_ceil(COLS_PER_CTA), 1, 1)`, `CubeDim { x:32, y:COLS_PER_CTA, z:1 }`. `COLS_PER_CTA` (4–8) tunes warps/CTA for occupancy; N is large (attn N~1–6K, dense mlp N~3–12K, lm_head N=248320) ⇒ thousands of warps saturate SMs even at M=1.
- **m_max resolution (kills C1):** `m_max` is **comptime and set equal to the fixed decode batch** (1 for greedy single-stream — the true target). `acc[m_max]` then stays in registers (the `flash_decode.rs:62` `Array::new(dpl)` discipline). `m_dim` runtime only guards OOB `x`/`out`. For a batched-serving build, set `m_max=8` and pay the unrolled cost; one compiled kernel per `(K, m_max)`. **Never a runtime loop bound on the register array** (that spills to local memory).
- **Weight coalescing (kills F4):** each lane issues **one `Line<e2m1x2>` width-8 load** = its whole 16-block; adjacent lanes read contiguous 8-byte spans ⇒ real 256-B coalesced warp transaction. The scalar `as_tensor_arg(1)` byte-loop the designs wrote is a stride-8 gather (~50% efficiency) — must vectorize.
- **Typing (kills A4/E1):** carry `qw`/`bs` as persistent 1-byte `I8` Burn `CubeTensor`s (VA-stable, no Burn e2m1x2/e4m3 DType), and at launch build the typed args over their handles with `TensorArg::from_raw_parts::<e2m1x2>(&qw.handle, &[k/2,1], &[N,k/2], 8)` / `::<e4m3>(&bs.handle, &[k/16,1], &[N,k/16], 1)` (the probe idiom, `nvfp4_gemm_probe.rs:519-543`). Decode with `cast_from` — **not** `reinterpret`. `gscale` is a persistent `[1]` f32 buffer read as `gscale[0]` (capture-safe; ScalarArg is an acceptable simplification only because the weight is static).
- **Host launch:** raw `CubeBackend<CudaRuntime,f32,i32,u8>` below Fusion (mirror `moe_grouped.rs:1069-1097` / `flash_decode_raw`). Provide a `Cuda` Fusion impl too for eager.

### Numerics identity (for the oracle, kills F6)
Kernel computes `out[m,n] = g · Σ_lanes Σ_{b∈lane} S_b·Σ_{p<16}(vals_p·x[m,k0+p])`. The Tier-1 CPU oracle **must replicate this exact block-factored accumulation** (S_b out of the 16-term block dot, gscale once at the end) to claim bit-exactness "modulo lane-reduction order." A per-element-dequant oracle (`Σ W_hat_k·x_k`) is a *different* rounding sequence — assert a tight ULP bound there, not bit-equality.

---

## 3. Accuracy gate (D6, decisive) — codec-correctness first

Reconciles the plan's §6C token-identity mandate with critique D1 (free-run greedy 100%-identical is chaotic and even FP8 may not meet it).

- **Tier 1 — codec round-trip (offline, host):** OCP/E2M1 golden vectors; assert kernel dequant == block-factored host oracle bit-for-bit (mod lane order); per-tensor round-trip rel-max within E2M1 grid tolerance. **Must include an all-zero-block tensor and an outlier-heavy tensor** — these are the F1/A1 traps the same-codec cosine test cannot catch. Add a w8a16 STEP-A-style micro-probe: `i8 → from_raw_parts::<e2m1x2> → Line::<f32>::cast_from` vs golden nibbles (packed-e2m1x2 SIMT cast is unexercised in-repo; only scalar e4m3 is proven — F9/E2).
- **Tier 2 — per-layer (like `w8a16_linear.rs:220-279`):** `Nvfp4Linear` vs bf16 Linear on real Qwen3 shapes at M=1 (and M=8 if batched): cosine, rel-max-err, **and the argmax-margin distribution** (top1−top2 gap). Flag any layer regressing vs the FP8 baseline.
- **Tier 3 — THE GATE (per tensor):**
  1. **Statistical (primary, predicts generalization):** teacher-forced next-token **top-1 agreement rate + argmax-margin** vs the bf16 model over a fixed *held-out* corpus (not one string). This is the instrument D1 argues for; it survives near-tie chaos.
  2. **Acceptance smoke (the plan's hard mandate):** captured greedy string **token-identical** to the bf16 greedy string on the known-good 30B prompt set (`vllm_infer.rs:21`). A tensor passes only if both hold. PPL/KL are secondary telemetry only.
- **Perf co-gate (P0.5 Nsight):** `dram__throughput` high, scheduler-stall low, on real 30B shapes **with the vectorized WV=8 load**. If ALU/unpack-bound at M=1 (2 cvts fp4→f16→f32 + scale, nothing to amortize at batch-1 — B1/F5), NVFP4 loses to FP8 even when numerically fine ⇒ demote on perf too.

---

## 4. FP8 fallback — per-tensor `QuantLinear`, capture-fixed (`src/nvfp4_linear.rs`)

```rust
enum QuantLinear<B> { Nvfp4(Nvfp4Linear<B>), Fp8(W8A16Linear<B>), Bf16(Linear<B>) }
// common forward(x:[M,K]) -> [M,N]; layout-preserving; bias-free for Qwen3
```
Selected at **load** from a checked-in per-tensor manifest `{tensor -> nvfp4|fp8|bf16}` produced by the gate. Demotion ladder per tensor: NVFP4 → (fails token-id or P0.5) FP8 → (fails) bf16.

**CRITICAL prerequisite (kills F2):** `W8A16Linear::forward` runs through `w8a16_gemm`'s `CubeCustomOp` bridge (`w8a16.rs:291`) — **cannot execute inside the captured region**, and its `[K,N]` flat kernel re-reads columns at M>1. Before the FP8 fallback is real for capture, port `w8a16_gemm` to a **raw `CubeBackend` GEMV** (dual-backend trait, exactly `moe_grouped.rs:1010-1099`). Treat "FP8 fallback exists" as blocked on this port, not done.

**Scope (S1, non-negotiable):** decode/inference-only, M = the fixed decode batch (1 for greedy). NEVER in the GRPO grad recompute (logprob-parity break, identical rule to `w8a16.rs:27-39`). The batched GRPO rollout decodes `[n,1]` with `n=prompts×group ≫ 8`, so this is a **batch-1 serving / greedy lever, not a rollout lever**. MoE experts (routed) stay on `moe_grouped.rs`'s fused gather-GEMV; NVFP4-ing them is a *separate* extension of `fused_swiglu_*_splitk` (add block-scaled e2m1x2 dequant-in-load), sequenced to avoid the flagged L1/L2C conflict on `moe_grouped.rs` (plan line 426).

---

## 5. Ranked highest-risk decisions — verify FIRST, against an f32 reference + the bf16 model

**Accuracy (which weights 4-bit vs FP8 vs bf16) — verify before any perf work:**

1. **lm_head + MoE router-gate placement (the #1 accuracy risk).** E2M1 = 1 mantissa bit; a K≈2048 dot leaves ~1–3% correlated logit error → argmax flips on the 248K head. A **router top-8-of-256 flip is categorically worse than an lm_head flip** — it reroutes to a *different expert*, changing which weights are even read (F3). **Default: router-gate → bf16 (tiny, most brittle); lm_head → FP8** (2× win on the single largest dense per-step byte source, proven cosine>0.999, far safer on argmax than NVFP4 — resolves the F3 "gating out the biggest tensor hollows the win" tension without betting argmax on 4-bit). Verify per-tensor token-identity + margin vs bf16; only promote to NVFP4 if it passes.
2. **Codec scale floor (F1/A1, silent whole-block zeroing).** Floor `gscale` at `f32::MIN_POSITIVE` and `bs` at `E4M3_MIN_NORMAL`, special-case `bamax==0`. Without it, outlier-heavy tensors (exactly NVFP4's target) divide-by-zero → blocks silently read 0, and the same-codec cosine oracle passes while the model diverges. Verify with the all-zero-block + outlier golden vectors (Tier 1).
3. **Calibration + reconstruct-then-quantize.** Naive amax flips tokens; use calibrated scale selection for every tensor kept at NVFP4, and quantize against the *rounded* E4M3 S_b. Verify Tier-2 cosine/margin improves vs naive amax.
4. **Nibble order + column-major transpose + block indexing (silent-if-wrong, F9/E2).** low=even/high=odd, `[N,K/2]` not `[K,N]`, block b ↔ line b ↔ scale b. Lock with byte-exact kernel-vs-host dequant + the packed-e2m1x2 SIMT cast micro-probe **before** the GEMV.

**Perf — verify after accuracy holds:**

5. **P0.5 BW-vs-ALU at M=1, with the vectorized WV=8 load** (B1/F4/F5). The whole 3.56× thesis is unproven until Nsight shows DRAM-bound; if ALU-bound, prefer FP8. The coalescing property only exists with the vectorized load.
6. **x activation L2 traffic at large N** (B3): GEMV re-reads `x[M,K]` once per column; at lm_head N=248K this can make the kernel L2/activation-bound after the weight shrinks. If P0.5 shows it, add a per-CTA `SharedMemory` x-stage shared across the `COLS_PER_CTA` warps (budget `m_max·K·4` B ⇒ ~8 KB at M=1/K=2048; bounds m_max·K under the ~48 KB smem limit).
7. **Capture invariants (F2 + VA):** the raw-backend FP8 port (item in §4) must exist for the fallback to be capture-usable; `qw`/`bs`/`gscale` allocated once, never reallocated / `into_contiguous`'d (VaSnapshot, `capture.rs:142-174`).

**Minor:** split-K idles lanes when `K/16 < 32` (K<512 attn projections) — correct (idle lanes add 0 to plane_sum), just wasteful on small tensors; fine. `k`/`blocks`/`wv`/`m_max` comptime ⇒ one compiled kernel per unique K — the `Static` grid tolerates the per-K kernel set.

Files to create: `/workspace/qwen3-burn-manin-grpo/src/nvfp4.rs` (codec + `mod gpu` kernel + raw/Fusion launchers), `/workspace/qwen3-burn-manin-grpo/src/nvfp4_linear.rs` (`Nvfp4Linear` + `QuantLinear` enum + gate manifest load). Templates: `/workspace/qwen3-burn-manin-grpo/src/w8a16.rs`, `src/w8a16_linear.rs`, `src/moe_grouped.rs` (dual-backend trait), `src/flash_decode.rs` (raw launch + plane_sum), `examples/nvfp4_gemm_probe.rs` (from_raw_parts typing).
