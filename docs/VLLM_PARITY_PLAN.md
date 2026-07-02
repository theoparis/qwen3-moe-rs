# vLLM-Parity Inference Plan (GRPO rollout speed on GB10 / sm_121)

Status: **REVIEWED + LOCKED** — eng-review (4 sections) + Codex gpt-5.5-high + Gemini 3.1-Pro-high
outside voices. Scope: user chose the full vLLM-class set of pillars; sequencing chosen **bandwidth-first**
(the two biggest parity-safe levers first; device-side decode loop + CUDA graphs gated behind a
measurement). See `## GSTACK REVIEW REPORT` at the bottom.

## 0. RESUME STATE — paused 2026-06-28 (read this first)

This plan was partly executed and **measured**, which CORRECTED the bandwidth-first thesis below. State:

**DONE + committed (`grpo-phase-a`):** Phase-0 parity gate + fp8 probe; static KV cache (~14× cache-update);
Fusion custom-op bridge (GO) + typed safe-wrapper (`src/cube_custom_op.rs`); FlashAttention-2 kernel
(`src/flash_attn.rs`, correctness-validated, not production); fused W8A16 fp8 GEMM (`src/w8a16.rs`,
correct but **not GRPO-deployable**); dynamic batch-shrink (`group_sample_cached_shrink`, **2.03×**).

**TWO measurements overturned the thesis:**
1. **fp8 is NOT the GRPO-rollout lever** (VLLM_KERNELS §2): forward-only fp8 in the rollout breaks the PPO
   logprob-parity invariant, and the batched rollout (M = P·G > 1) never hits the M=1 bandwidth win — fp8
   is a batch-1 *serving* lever. So the §1 "fp8 = biggest lever" thesis is WRONG for GRPO.
2. **Decode breakdown** (`examples/rollout_decode_bench.rs`, production vocab, N=64, T=144): host-side
   sampling/argmax **43%**, transformer fwd 42% (launch/bandwidth-bound), host-sync `[N,vocab]` read 8%,
   **attention-SDPA only ~5%**. So attention/FlashAttention is NOT the bottleneck at this length.

**⇒ EVIDENCE-GROUNDED ROADMAP (resume here, in order):**
- **(A) Device-side sampling/logprob** — ✅ **DONE (`src/sampling_device.rs`, `group_sample_cached_device`):
  3.6-4.8× end-to-end decode speedup**, host boundary 38.9 MB/step → `[N]`, greedy bit-parity + raw
  pre-warp logp (3-voice-gated, mathematically sound). Device argmax/logsumexp + Gumbel-max (unfiltered —
  which is GRPO-*mandatory*, not a limitation: top-p/k truncation makes sampling off-policy). REMAINING
  before it goes into the trainer: (i) an on-CUDA step-0 `mean_ratio ≈ 1` gate at temperature>0 through
  `group_sample_cached_device` (the existing test only checks logp from the *same* logits, not the
  trainer's separate full-forward recompute); (ii) wire `grpo_step` over to it. Follow-ons: device top-k/p
  filtering; f64 Gumbel tail (negligible).
- **(A2) Fully device-side decode loop** — the actual prerequisite for (B), bigger than (A). (A) removed the
  `[N,vocab]` *bandwidth* but the residual `[N]` token device→host copy for the EOS/finished check is still
  **3 stream-syncs/step** (the latency wall). Capturing needs: device-side EOS/finished tracking + on-device
  `token ∈ eos` reduce, a preallocated `[N, lp+max]` device token buffer (slice-assign, NOT `Tensor::cat`),
  a fixed decode length (drop the host `all-finished` break → mask instead), device-counter positions, and
  exactly one device→host at the end. ✅ **DONE (`group_sample_cached_device_loop`):** static-counter +
  host-sync-free (zero device→host inside the loop), greedy bit-parity. Eager speedup ~1.0× (structural).
- **(B) CUDA-graph capture** — ⛔ **BLOCKED on framework-level CubeCL work AND low-payoff** (3-voice review
  with cubecl source). CubeCL exposes no cudaGraph capture API (cudarc has the FFI, unused); the `Fusion`
  layer is a lazy/dynamic op queue (no stable launch list); `Tensor::random` bakes host seeds as frozen
  immediates → replay = identical noise → degenerate stochastic sampling (only greedy is capture-safe); no
  graph-aware allocator (replay corruption); and decode attention grows per step (`filled=lp+t+1`, not
  fixed-shape). Even if built, decode is BANDWIDTH-bound (the tied-head logits GEMM streams ~0.6 GB/step,
  which graphs don't touch) → ~1.1-1.4× at best, → 1.0× as model/context grow. **NOT the next lever.**
- **(B′) The real next decode lever = the bandwidth-bound logits GEMM** — the tied-head `[H,V]` projection
  streams the full vocab matrix each step (~0.6 GB at 0.6B, more at scale) and dominates the remaining ~2 s.
  Chunked/fused logits, a device-side sampler that never materializes full `[N,V]` (Gumbel-max over a
  streamed logit tile), or vocab-parallel reduction are the bandwidth levers. Measure first.
- **(C) Flash-Decode** (make `src/flash_attn.rs` fast: split-K + warp-parallel + bf16 + the padding mask)
  — LAST, because attention is ~5% at T=144 BUT is O(T)/step so its share grows; re-profile at production
  `max_new_tokens` (T≈1k), where it becomes the next lever. Also needs the padding mask (Opus P0) before
  it can replace SDPA on the ragged GRPO path.
- **Gate before wiring batch-shrink into `grpo_step`:** an on-GB10 temperature parity check (kept-row
  `old_logprob` match + step-0 PPO ratio ≈ 1) — sm_121 has batch-dependent kernel history.

(The "fp8-storage / FlashAttention bandwidth-first" framing in §1-§5 below is the PRE-measurement plan;
keep for context, but the roadmap above supersedes it for GRPO.)

## 1. Goal & thesis

Make the autoregressive **rollout** (the dominant share of GRPO step wall-clock) fast on GB10's 273 GB/s
unified LPDDR5X. A GRPO rollout is a **fixed batch of P×G sequences that start together and finish at
staggered lengths** — not vLLM's async request stream. The outside voices (both models, independently)
collapsed the design to one truth: **decode is HBM-bandwidth-bound and CPU↔GPU-sync-bound, not
launch-bound** — so the two biggest, parity-safe levers are (1) reading **fewer weight bytes** (fp8
storage) and (2) not materializing the **O(S²) attention scores** (FlashAttention). The flashy pillars
(CUDA graphs, dynamic batch-shrink) only pay off inside a fully device-side, host-sync-free decode loop,
which is the hardest/riskiest work on pre-1.0 CubeCL + uncharacterized sm_121 — so it is **gated behind a
measurement** that proves host-sync/launch overhead is the ceiling after the bandwidth wins land.

## 2. The non-negotiable invariant: logprob parity (+ the canonical harness)

GRPO correctness requires the rollout's **behavior-policy** logprobs to equal what training scores. Two
traps the outside voices flagged (P0, both models):

- **Behavior-policy match (Codex/Gemini P0).** PPO `old_logprob` must be the logprob of the policy that
  *sampled* the token. If the rollout samples under fp8, recomputing `old_lp` in bf16 makes the step-0
  ratio `P_bf16/P_fp8 ≠ 1` — you penalize the RL for quantization drift. **Resolution:** the rollout
  **samples in bf16** (fp8 is *storage only*, dequantized to bf16 before the MMA), so the behavior policy
  is bf16 and `old_lp` is exact. No decoupled recompute, no off-policy bias.
- **Silent finite corruption (cross-cutting).** sm_121 has a documented silent (finite, no-NaN) matmul
  corruption; fp8 dequant, a hand-written attention/MoE kernel, or a mis-written cache can each reproduce
  silent-wrong-logits that surface as degraded GRPO reward, never a crash.

**The canonical equivalence harness (Phase 0, the load-bearing gate; Codex P0).** ONE audited semantic
path for masks/positions/routing/dtype/cache that rollout, `old_lp`, and the training recompute all share,
checked by an always-on CI gate asserting ALL of:
1. per-token logprob max & mean error ≈ 0 (not just "mean ratio ≈ 1" — that hides token-local corruption),
2. sampled-token agreement under fixed RNG,
3. route-id agreement (MoE),
4. KV-cache vs no-cache equivalence,
5. shrink vs no-shrink equivalence (when batch-shrink lands),
6. long-context Manim-prompt cases (not just short prompts).
Every later pillar must pass this before any speed benchmark.

## 3. Perf model (why bandwidth-first)

GB10 roofline over ~3.3B active params (30B-A3B) — decode is HBM-bandwidth-bound:

```
  precision   tok/s (roofline)   lever                         needs fp8 tensor cores?
  f32         ~21                baseline                       —
  bf16        ~41   (~2x)        half the weight bytes          no  (works today for MoE)
  fp8 storage ~80   (~2x again)  half again, dequant->bf16 MMA  NO  <- the win is BYTES, not MMA
```

Reading fp8 *bytes* and dequantizing to bf16 in-register captures the full bandwidth win using standard
**bf16** tensor cores — so it does NOT depend on sm_121's unvalidated fp8 MMA (`tensor_cores_per_sm[121]
= None`), and it keeps sampling in bf16 (parity-exact). Forcing e4m3 `q_matmul` would need activation
quant (added latency) for *zero* decode gain (Codex P1/9, Gemini P1/8). **Attention:** reference SDPA
writes the O(S²) score matrix to HBM; on long Manim prefill that out-costs the KV `cat` — so a tiled
FlashAttention kernel is a co-equal Phase-1 lever (Codex P1/8, Gemini P0/9).

## 4. Architecture — revised dependency graph (bandwidth-first)

```
 PHASE 0  canonical parity harness  +  spikes (fp8-storage probe, flash-attn probe)
            │   (gates everything numeric; sm_121 silent-corruption defense)
            ▼
 PHASE 1  fp8 weight-STORAGE (bf16 dequant+MMA+sampling)    ║   FlashAttention / tiled-attn kernel
   (~2x weight bandwidth, parity-exact, no fp8 MMA)         ║   (kills O(S^2) prefill HBM)
            │  [LANE A: quant/loader]                        │  [LANE B: attention] — INDEPENDENT
            └───────────────────────┬────────────────────────┘
                                     ▼
 PHASE 2  static in-place KV cache  (host-scalar slice_assign at lp+t; kills O(T^2) Tensor::cat)
                                     │
                                     ▼
   ┌─────────────── MEASURE: wall-clock decomposition ───────────────┐
   │  prefill | decode | reward | old_lp recompute | backward/update  │  + per-token host-sync cost
   │  + per-layer attention vs router vs expert (Codex P1/8)          │  ⇒ is host-sync/launch the ceiling?
   └─────────────────────────────────┬───────────────────────────────┘
                                     ▼  (gated by the measurement)
 PHASE 3+  DEVICE-SIDE DECODE LOOP (no host-read-per-token: device EOS/sampling/append)
   ├─ dynamic batch-shrink (scatter-write KV to original rows; needs a scatter kernel)
   ├─ CUDA-graph capture (device-tensor cache offsets, bucketed shapes; needs a CubeCL fork)
   ├─ MoE grouped-GEMM kernel (MoE models only; behind moe_probe)
   └─ full PagedAttention + scheduler + speculative-decode (serving / async-RL only)
```

Key coupling the outside voices exposed: batch-shrink (dynamic shapes) and the `lp+t` host-scalar KV
offset both **break CUDA-graph capture** — graphs need a device-side, host-sync-free loop with
device-tensor offsets + a scatter kernel + bucketed shapes. So all three are one coupled Phase-3 effort,
entered only if the measurement says host-sync/launch overhead is the ceiling.

## 5. Phases

### Phase 0 — Canonical parity harness + de-risk spikes (~3-5 days, FIRST)
- Promote the greedy bit-parity harness (`tests/grpo_rollout.rs`) into the always-on **canonical
  equivalence gate** of §2 (all 6 assertions). This is THE defense against silent sm_121 corruption.
- **fp8-storage probe:** quantize one Qwen3 `Linear` to E4M3 weights, dequant→bf16 in the matmul, compare
  logits vs an f32/bf16 oracle (per-token max-abs + cosine) on a fixed batch incl. a long prompt.
- **flash-attn probe:** a minimal tiled attention `#[cube]` kernel vs reference SDPA on one layer; pin the
  sm_121 causal-mask handling (the fused-mask bug is why we're on reference SDPA today).
- Why first: every numeric pillar must clear the gate; the two probes de-risk the two biggest levers
  before integration. Risk: a green probe that doesn't generalize — the gate stays always-on.

### Phase 1 — fp8 weight-storage + FlashAttention (the two biggest levers; two parallel lanes)
- **Lane A — fp8 weight-storage (~4-6 days):** store Qwen3 weights as E4M3 (per-channel/block symmetric
  scales), dequantize to bf16 in-register, run the existing batch-safe bf16 MMA; **sampling stays bf16**
  (behavior policy = bf16 → `old_lp` exact, no decoupled recompute). ~2× weight bandwidth at decode.
  Fallback if a probe regresses: keep weights bf16 (no win) — never fp8 MMA on sm_121.
- **Lane B — FlashAttention/tiled-attn kernel (~1.5-3 weeks):** a `#[cube]` tiled attention that never
  materializes the `[S,S]` scores to HBM (online-softmax), correct causal mask on sm_121. Replaces
  reference SDPA for prefill (and decode). The harder of the two; gated by the Phase-0 attn probe.
- **Test:** canonical gate (per-token logprob err, fixed-RNG sampled-token agreement, long-context);
  cosine>0.99999 vs the SDPA oracle for attention; reward-curve sanity for fp8.

### Phase 2 — Static in-place KV cache (~3-5 days, low risk)
- Replace `KVCache::update`'s `Tensor::cat` (`cache.rs:53`) with a pre-allocated
  `[N, T_max, kv_heads, head_dim]` buffer, written via `slice_assign` at host-scalar offset `lp+t`. Kills
  the O(T²) per-step realloc/copy. (Host-scalar offset is fine WITHOUT graphs; the device-tensor-offset +
  scatter version is deferred to Phase 3, where graphs need it — Gemini P1/9.)
- **Test:** bit-identical greedy decode vs the `cat` path AND vs no-cache `group_sample`.

### MEASURE (gate to Phase 3) — wall-clock decomposition
Instrument the GRPO step: prefill / decode / reward / `old_lp` recompute / backward+update, the per-token
host-sync cost, and per-layer attention vs router vs expert (Codex P1/8). Decide: is host-sync/launch
overhead actually the ceiling now that bandwidth+attention are fixed? Only then build Phase 3.

### Phase 3+ — Device-side decode loop + gated heavy pillars (gated by the measurement)
- **Device-side decode loop** (the prerequisite for everything below): device-side EOS detection,
  sampling, token-append, and reward plumbing so there is **no host read per token**. Without this, the
  host-sync tax (~20-50µs/token) eats the kernel/graph savings (Codex P1/8, Gemini P0/10).
- **Dynamic batch-shrink:** stop forwarding finished rows; scatter-write generated KV back to original
  rows (needs a scatter kernel, not `slice_assign` — Gemini P2/10). Lazy/threshold compaction only.
- **CUDA-graph capture:** device-tensor cache offsets + bucketed/padded N + a capture-safe CubeCL fork
  (no `cuMemAlloc`/host-memcpy in the captured region) + a pinned graph memory pool.
- **MoE grouped-GEMM kernel** (MoE models only, behind `moe_probe`): on-device align+sort → `#[cube]`
  block-per-expert-segment fused-SwiGLU grouped GEMM (`CubeCount::Dynamic` device-driven grid). Forward
  only — SKIP the autodiff Backward (rollout is inference; LoRA-gated training recompute uses the oracle
  path). Caveat: at GRPO decode batch nearly all 128 experts are touched, so its "load only touched
  experts" win largely evaporates and fp8-storage captures the HBM win first — build only if a benchmark
  shows full-E weight-read (not precision) is the bottleneck (Codex P1/8).
- **Full PagedAttention + scheduler + speculative decode:** serving / async-RL only; wrong-regime for a
  fixed synchronized P×G batch. Build only if the goal shifts; enter spec-decode via host-side n-gram /
  SPEC-RL prior-epoch-prefix (no second model, survives policy drift), logprobs always from the verify
  forward.

## 6. Test strategy

The canonical gate of §2 runs on EVERY rollout/cache/quant/attn/graph change (silent corruption is not a
Phase-0-only concern — Codex P0). Per pillar: (fp8) per-token logprob err + cosine>0.99999 + reward-curve;
(flash-attn) cosine>0.99999 vs SDPA + long-context; (static KV) bit-parity vs cat and no-cache; (device
loop / shrink) shrink-vs-no-shrink bit-parity + correct scatter write-back; (graphs) replay-vs-eager
bit-parity across buckets + a deliberate buffer-move negative test; (MoE kernel) cosine>0.99999 vs oracle
+ exact-no-drop + GB10 bench; (spec/paged) rejection-sampler marginal-distribution test.

## 7. Top risks

1. **Silent sm_121 numerical corruption** (cross-cutting). Defense: the always-on canonical gate + per-
   pillar cosine/bit-parity + the Phase-0 probes. Highest-priority, every phase.
2. **Behavior-policy parity break** if fp8 ever touches rollout *sampling*. Mitigation: fp8 is
   storage-only, sampling stays bf16 — locked into the Phase-1 design.
3. **Host-sync per token is the real ceiling**, not kernel launches — graphs are worthless until the
   decode loop is device-side. Mitigation: the MEASURE gate before Phase 3.
4. **FlashAttention causal-mask correctness on sm_121** (the fused-mask bug exists). Mitigation: the
   Phase-0 attn probe + hand-written mask.
5. **CUDA-graph capture needs a pre-1.0 CubeCL fork** that re-breaks on every burn/cubecl bump; the
   `lp+t` host-scalar offset and dynamic shrink shapes are non-capturable as-is.
6. **MoE kernel value evaporates at GRPO decode batch** — gate behind a benchmark proving weight-read,
   not precision, is the bottleneck.

## 8. NOT in scope (explicitly deferred)
- **Full PagedAttention block manager / request-admission scheduler / speculative decoding** — serving &
  async-RL only; a fixed synchronized P×G batch gains nothing. Revisit only if the goal shifts to serving.
- **fp8 tensor-core MMA** — storage-only + bf16 dequant captures the bandwidth win without the sm_121 risk.
- **MoE kernel autodiff Backward** — rollout is inference-only; LoRA-gated training recompute uses the
  autodiff oracle path; frozen expert GEMMs need no input-grad.
- **int4 / AWQ / GPTQ / fp8 KV cache / activation-fp8** — low value or parity-risky; QuantMode is
  symmetric-only (no zero-point) in the pinned CubeCL anyway.
- **CUDA graphs + dynamic batch-shrink + device-side loop** — deferred behind the MEASURE gate (chosen
  bandwidth-first sequencing), not dropped.

## 9. What already exists (reuse, don't rebuild)
- **MoE forward paths** (`src/moe.rs`): `forward_oracle` (dense oracle), `forward_routed` (host
  token-routing, 2.7× decode), `forward_routed_ondevice` (on-device capacity, ~11× batched) — the MoE
  kernel (Phase 3) reuses the on-device align+sort idea (`one_hot`/`cumsum`/`scatter`).
- **bf16 inference** — works today (the RmsNorm/combine dtype fixes shipped this session).
- **Greedy bit-parity harness** (`tests/grpo_rollout.rs`) — already compares no-cache `group_sample` vs
  `group_sample_cached`; the canonical gate (Phase 0) extends it.
- **Batch-safe `linear3`** (2-D GEMM, dodges the sm_121 broadcast bug) — the fp8 dequant matmul builds on it.
- **Reference SDPA** (`decoder.rs`) — the FlashAttention kernel replaces it (and the parity oracle uses it).
- **`KVCache`** (`cache.rs`), **`group_sample_cached`** rollout (`src/grpo/rollout.rs`) — Phases 2 & 3 modify these.
- **CubeCL kernel surface** (`/workspace/cubecl` @ b19859ee): `#[cube(launch)]`, `CubeCount::Dynamic`,
  `SharedMemory`, `Atomic`, `cmma` tensor cores, E4M3/E5M2 — all primitives the kernels need exist.

## 10. Failure modes (new codepaths)
| Codepath | Realistic failure | Test? | Error handling? | Silent? |
|---|---|---|---|---|
| fp8 dequant matmul | scale overflow / wrong per-channel axis → wrong logits | canonical gate + cosine | n/a (numeric) | **yes → gate catches it** |
| FlashAttention kernel | causal-mask off-by-one on sm_121 → attends future tokens | cosine vs SDPA + long-ctx | n/a | **yes → gate catches it** |
| static KV `slice_assign` | wrong `lp+t` offset → stale/overwritten KV | bit-parity vs cat | bounds check | yes → bit-parity catches it |
| device-loop EOS (P3) | device EOS misfire → never stops / stops early | shrink-vs-no-shrink parity | max-len cap | partial |
| KV scatter write-back (P3) | scatter to wrong original row → cross-sequence contamination | scatter unit test + parity | n/a | **yes → critical gap until tested** |
| CUDA-graph replay (P3) | replay against moved buffer / stale scalar → garbage logits | replay-vs-eager + buffer-move | n/a | **yes → critical gap until tested** |
Critical gaps (silent + no handling until their tests exist): fp8 dequant, flash-attn mask, KV scatter,
graph replay — all gated by the canonical harness, which is why Phase 0 ships first.

## 11. Parallelization
| Step | Modules | Depends on |
|---|---|---|
| Phase 0 harness + probes | `tests/`, `src/grpo/` | — |
| P1 Lane A fp8-storage | `src/load.rs`, `src/decoder.rs` (linear) | Phase 0 |
| P1 Lane B FlashAttention | `src/decoder.rs` (attention), CubeCL kernel | Phase 0 |
| P2 static KV | `src/decoder.rs` (`cache.rs`) | Phase 0 |
| Phase 3+ | `src/grpo/rollout.rs`, kernels, CubeCL fork | MEASURE gate |
- **Lane A (fp8) and Lane B (flash-attn) are independent** (quant/loader vs attention math) → parallel
  worktrees. P2 (KV) touches `decoder.rs` like Lane B — sequence P2 after Lane B or coordinate the file.
- Execution: Phase 0 first (blocks all). Then Lane A ∥ Lane B. Then P2. Then MEASURE → Phase 3.

## 12. Implementation Tasks
- [ ] **T1 (P0, human ~1d / CC ~2h)** — tests — canonical equivalence gate (6 assertions, always-on CI).
  - Surfaced by: §2 / Codex P0 "parity gate too narrow". Files: `tests/grpo_rollout.rs`. Verify: `cargo test`.
- [ ] **T2 (P0, human ~1d / CC ~2h)** — quant — fp8-storage probe (one Linear E4M3→bf16 vs oracle).
  - Surfaced by: §5 Phase 0 / Gemini P1/8. Files: `examples/fp8_probe.rs`. Verify: cosine + max-abs print.
- [ ] **T3 (P1, human ~1d / CC ~3h)** — attention — tiled-attn probe vs SDPA on one layer (sm_121 mask).
  - Surfaced by: §5 Phase 0 / Codex P1/8, Gemini P0/9. Files: `examples/attn_probe.rs`. Verify: cosine.
- [ ] **T4 (P1, human ~4-6d / CC ~1-2d)** — quant — fp8 weight-storage loader + bf16-dequant matmul.
  - Surfaced by: §5 Phase 1 Lane A. Files: `src/load.rs`, `src/decoder.rs`. Verify: canonical gate + reward.
- [ ] **T5 (P1, human ~2-3wk / CC ~1wk)** — attention — FlashAttention `#[cube]` kernel (online softmax).
  - Surfaced by: §5 Phase 1 Lane B. Files: CubeCL kernel + `src/decoder.rs`. Verify: cosine vs SDPA + long-ctx.
- [ ] **T6 (P1, human ~3-5d / CC ~1d)** — cache — static in-place KV (`slice_assign`, kill `cat`).
  - Surfaced by: §5 Phase 2. Files: `cache.rs`. Verify: bit-parity vs cat + no-cache.
- [ ] **T7 (P1, human ~2d / CC ~4h)** — bench — GRPO-step wall-clock decomposition + per-token host-sync.
  - Surfaced by: MEASURE gate / Codex P1/8. Files: `examples/grpo_bench.rs`. Verify: prints the breakdown.
- [ ] **T8 (P2, deferred)** — device-side decode loop, batch-shrink, CUDA graphs, MoE kernel, paging — gated by T7.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found | 11 findings (1 P0 correctness), all folded |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | reviewed | 13 issues, 4 silent-corruption gaps gated by the parity harness |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **CODEX (gpt-5.5 high):** caught the decoupled-PPO **off-policy bug** (P0/9 — fp8-sampled tokens scored
  under bf16 ⇒ ratio ≠ 1), fp8-is-storage-not-MMA, FlashAttention as a missing early prereq, and
  host-sync-per-token as the real ceiling. All folded into the bandwidth-first spine.
- **CROSS-MODEL:** Codex and Gemini agreed on all six major findings (decoupled-PPO broken, fp8 storage
  not MMA, FlashAttention early, host-sync ceiling, batch-shrink↔graphs conflict, parity gate too narrow)
  — **no tension**, unusually strong consensus. Gemini added the static-KV-offset↔graph and KV-scatter
  catches. The plan reflects both.
- **VERDICT:** ENG CLEARED — bandwidth-first plan locked, all outside-voice findings folded; the 3 scope
  decisions (full pillar set → full-scope-sequenced → bandwidth-first ordering) are resolved by the user.
  Ready to implement **Phase 0** (canonical parity harness + fp8/flash-attn probes).

NO UNRESOLVED DECISIONS
