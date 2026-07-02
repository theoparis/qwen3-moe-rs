# Path to 80 tok/s — Qwen3-30B-A3B single-stream decode on GB10

3-model council review COMPLETE (Codex gpt-5.5 high + Opus 4.8 xhigh source-verified + Gemini 3.1 Pro high)
+ prod-impl research. Question: how do we go from a MEASURED **0.73 tok/s** to **80 tok/s** for 30B-A3B
greedy decode on a single GB10? **Verdict up front: 80 single-stream is a STRETCH (89% of the roofline);
the honest landing is ~55-70 tok/s, and 80 needs speculative decoding or W4/NVFP4. The plan below is
corrected after the council found the binding constraint mislabeled and the "built" inventory overclaimed.**

## 0. The bandwidth roofline — CONFIRMED by all 3 voices (recomputed from config)

Config (`src/moe.rs:76`): H=2048, L=48, E=128, top-K=8, I=768, GQA 32q/4kv/128hd, untied lm_head [2048,151936].
GB10/DGX Spark LPDDR5X = **273 GB/s** (verified). One expert = 3·H·I = 4.718M params.

| Per-token weight read | bytes | 273 GB/s **roofline** |
|---|---|---|
| **Full-dense (all 128 experts)** — runs NOW | ~60.4 GB | **~4.5 tok/s** |
| **Top-8 routed, bf16** | 3.62(exp)+1.81(attn)+0.62(head) ≈ **6.08 GB** | **~45 tok/s** |
| **Top-8, fp8 EXPERTS ONLY** (attn+head stay bf16) | ≈ **4.27 GB** | **~64 tok/s** ⚠ below 80 |
| **Top-8, fp8 experts+attn+head** | ≈ **3.04 GB** | **~90 tok/s** |

⚠ **Two corrections the original draft missed:** (1) experts-only fp8 caps at **64** — you MUST fp8 attn +
lm_head too to clear 80. (2) **KV-cache read is omitted**: ~negligible at 1k ctx but **~4 GB/token at the
40,960 max context** (GQA), which drags every ceiling down at long context. The lm_head can't be chunked for
exact greedy argmax (needs all 151,936 logits), so its 0.62 GB is a hard floor unless fp8'd.

**Roofline ≠ throughput.** All 3 voices: batch-1 MoE sustains only ~60-75% of the byte-roofline (small GEMV
shapes, routing/top-k, gather/scatter, 48-layer sequencing, dequant-scale reads, residual launches). So the
**~90 roofline → ~55-70 sustained.** 80 = 89% of roofline = not honest as a committed forecast.

## 1. CORRECTED diagnosis — at 0.73 tok/s the model is LAUNCH-bound, not bandwidth-bound

The original draft said "the dense expert read (~58 GB) is the dominant cost." **All 3 voices corrected this:**
0.73 tok/s = **1.37 s/token**, but the dense byte-floor is only **0.22 s/token** (4.5 tok/s). The model runs
at **~44 GB/s effective = ~16% of peak** — i.e. **6.2× under even the dense ceiling.** That 6× gap is
**kernel-launch + host-sync overhead, not bytes**: `forward_oracle` (`moe.rs:286`) issues ≈ **31,000 eager
Fusion launches/token** (48·(4 attn + 128·3 expert GEMMs + combines)) + a per-layer route host-sync ≈ 44
µs/launch. **Textbook launch-bound.**

Consequence for the plan: **fp8 (a bandwidth lever) does NOTHING while the path is launch-bound.** It only
pays off AFTER the launch tax is removed by graph capture. This reorders the levers (see §3).

## 2. HONEST built-vs-unbuilt inventory — the council's critical correction

The original draft marked levers (A)/(C)/(D)/(E)/(F) "BUILT." Opus source-verified that **the specific paths
the plan leans on each have a disqualifying issue for 30B captured INFERENCE decode at T=1:**

| Lever | Reality (verified) |
|---|---|
| **(A) top-8 MoE decode** | ⚠ **The capturable top-8 block does NOT exist.** `forward_routed` (`moe.rs:365`) reads only 8 experts ✓ but does a device→host sync **every layer** (`moe.rs:375`, 48 syncs/tok) → **uncapturable**. `forward_routed_ondevice` (`moe.rs:476`) + `forward_grouped` (`moe_grouped.rs:424`) **re-stack ALL 128 experts** (+ grouped casts to f32) every call → at T=1 they read ≥128 experts, WORSE than dense. The "grouped wins at T=1" claim is true for the *kernel*, false for the *as-built wrapper*. |
| **(B) fp8 W8A16** | Kernel BUILT + M=1 decode is its winning regime (RL-parity objections don't apply to greedy gen) ✓ — but **integration UNBUILT**: no `W8A16Linear`, no fp8 load/quantize wired into the attn/expert/head GEMMs. |
| **(C) device sampling** | BUILT + greedy drop-in ✓, but the 3.6-4.8× was the GRPO **batched** shape; at single-stream `[1,vocab]` it saves ~0.6 MB/step — **modest**; real value is removing a sync for capture. |
| **(D) CUDA-graph + (E) static KV + (F) device-pos decode** | ⚠ **Built for the DENSE `Qwen3ForCausalLM` only.** `Qwen3MoeForCausalLM` has NO `forward_with_cache_static*`; `generate_greedy` (`moe.rs:630`) uses the legacy O(T²) cat cache. **None of D/E/F is wired to the 30B MoE.** |

⇒ **Lever A is a real BUILD, not a wiring job:** a post-load **pre-stacked (fp8-packed) contiguous expert
cache** + a **fixed-shape single-token 8-expert gather kernel** (no per-layer host sync). And D/E/F must be
**ported from the dense model to the MoE** + the MoE block made capturable. This is the bulk of the work.

## 3. CORRECTED lever order (graph BEFORE fp8 — the launch tax binds first)

1. **(A) Build the capturable top-8 MoE decode block** — single-token 8-expert gather + pre-stacked expert
   cache. Attacks BOTH the 16× expert bytes AND the 16× expert launches (384→24 GEMMs/layer). **Biggest lever.**
2. **(D) Port CUDA-graph + static-decode to the MoE** — CONCURRENT with A, not last. Kills the ~6× launch
   tax (the binding constraint at 0.73). Until captured the path is launch-bound, so D is worth FAR more than
   the "~1.0×" the draft claimed. **Expected after A+D: ~30-40 tok/s (bf16, now bandwidth-bound).**
3. **(B) fp8 weights — experts + attn + lm_head** (not experts-only; that caps at 64). **Expected: ~50-67.**
4. **(C) device sampling + EOS** on-device (remove the residual per-step token sync for clean capture).
5. **MEASURE sustained DRAM bytes/token with Nsight after each lever** — don't trust the roofline.
6. **For an honest 80: speculative decoding** (EAGLE/EAGLE-3/Medusa/n-gram) — the ONLY lever past the
   single-token bandwidth wall (K accepted tokens per 1 weight read; need ~1.45×, EAGLE clears 2-3× on
   greedy). **MoE caveat (Opus):** a K-token verify batch routes to the UNION of the tokens' top-8 (up to 8K
   distinct experts) → reads >8 experts; net win needs high acceptance + expert overlap. OR **W4/NVFP4**
   (GB10 is Blackwell — sub-fp8 weights halve bytes again) if accuracy holds.

## 4. Honest achievable (council consensus)

| Stage | tok/s | note |
|---|---|---|
| now (full-dense, eager) | **0.73** | launch-bound, 16% of peak |
| (A) top-8 bf16, EAGER `forward_routed` | **~10-20** | still launch + 48-sync bound (NOT the 45 roofline) |
| (A+D) top-8 captured (gather + graph) | **~30-40** | bf16, now bandwidth-bound |
| (+B) fp8 experts+attn+head | **~50-67** | ×~1.5-1.8 |
| (+C/D polish) | **~55-70 sustained** | the honest single-stream landing |
| **(+ spec-decode or W4/NVFP4)** | **~70-80+** | the only honest path to 80 |

**Prod grounding (`docs/perf-research.md`, sources inline):**
- **Published 30B-A3B single-stream on DGX Spark / GB10:** bf16 **~30**, fp8 **~44-55**, Q4_K_M (Ollama)
  **~57**, NVFP4 **~65 (today's published ceiling)**. **Nobody has publicly shown 80 tok/s single-stream on
  this hardware class** — 80 is above every measured number, not just above our roofline estimate.
- vLLM/SGLang/TRT-LLM fused-MoE: `moe_align_block_size` + grouped-GEMM reads **only the top-k active
  experts** at batch-1 (0-token experts never leave HBM); **W8A16 fp8 weight-only is the universal batch-1
  choice** (block-wise 1×128 for MoE accuracy); all three **CUDA-graph the MoE step**. TRT-LLM on Blackwell
  runs **"MoE as dense GEMM"** at small token counts (grouped-GEMM launch overhead dominates) ⇒ a *fused MoE
  GEMV*, not a grouped-GEMM scheduler, is the right batch-1 kernel — the draft's "grouped wins at T=1" is naive.
- **Efficiency calibration:** llama.cpp on Strix Halo saturates **~80-93% of effective BW** (the target);
  MLX ~40-50%. fp8 W8A16 (Marlin/Machete) measures **1.6-1.9×** decode (not clean 2× — attn/KV stay 16-bit),
  E4M3 keeps **>99%** accuracy.
- **Spec-decode + the MoE dilution (quantified):** EAGLE-3/MTP give 2-4× on *dense*, but the union-of-experts
  penalty (a K-token verify routes to the union of top-8 sets — ~15 experts at K=2, ~29 at K=4 of 128)
  dilutes it to **~1.2× net, sweet-spot K=2** for this MoE.

## 5. WAVE-1 IMPLEMENTATION RESULTS (built + 3-voice-reviewed + measured)

Three independent building blocks built in parallel, each 3-voice-gated (Codex 5.5 + Gemini 3.1 Pro + Opus
4.8 source-verified). All correct + committed (`af4a6e7`); the review reshaped the path to the roofline.

**(C) Measurement — the launch-bound diagnosis is now EMPIRICALLY CONFIRMED** (`examples/decode_perf_bench.rs`,
real 30B on GB10, steady-state):

| path | tok/s | GB/tok | eff GB/s | % peak | verdict |
|---|---|---|---|---|---|
| oracle (dense, 128 experts) | 0.673 | 60.42 | 40.7 | **15%** | LAUNCH-BOUND |
| routed (host top-8, 48 syncs/layer) | **5.72** | 6.06 | 34.7 | **13%** | LAUNCH-BOUND |
| ondevice (re-stacks 128 @ T=1) | 1.03 | 60.42 | 62.2 | 23% | LAUNCH-BOUND |

The killer proof: **routed reads 10× fewer bytes yet has LOWER effective GB/s** (34.7 vs 40.7) — if
bandwidth-bound, cutting bytes 10× would approach peak; instead it's pinned at 13% by the 48 host-syncs/layer.
**All paths 13-23% of peak ⇒ overhead-bound, not bandwidth-bound.** Top-8 routing is already an 8.5× win
(0.67→5.72) but capped by launches/syncs — so capturing a sync-free top-8 path (Block A + graph) is the lever.

**(A) Capturable top-8 MoE decode** (`src/moe_decode.rs`) — **CORRECT + the bandwidth lever is REAL, but a
capturable SCAFFOLD, not the roofline kernel:**
- Opus kernel-traced `Tensor::select(0, ids)`: reads ONLY the k indexed `[H,I]` slabs (one thread/output
  element, strided input, no `into_contiguous`) — the other E−k experts are never addressed. The O(k)
  bandwidth win is real (NaN-poison proves the read-set; the kernel trace proves the byte-traffic). NOT fatal.
- `select_assign(Add)` with repeated token indices is **race-free** (kernel-proven: parallel over the feature
  dim, serial in-thread accumulate over the scatter dim) — Gemini's race worry refuted.
- **CAVEAT:** `select` is a *materializing* gather (alloc + write [N,H,I], then cast + matmul re-read) ⇒ ~3×
  the minimal traffic, at cache dtype (no fp8) ⇒ realized ~**5.3×** weight-byte cut, NOT ~16×. Still
  launch-bound eager (~50-70 launches/layer vs oracle's ~650). **The roofline needs a FUSED gather-GEMV
  (read each byte once into the MAC, fp8-packed).** GATES before Wave-2: CUDA **bf16 parity** (target dtype;
  tests are f32/CPU) + **Nsight `dram__bytes_read`**.

**(B) `W8A16Linear`** (`src/w8a16_linear.rs`) — **codec bit-faithful, but under-gated:**
- per-channel symmetric e4m3, i8-carries-bits reinterpret (verified to the C++ `reinterpret_cast` + size
  guard), bf16 round-trip — all kernel-verified correct; cosine 0.9996 vs bf16.
- **GATE (unanimous):** per-GEMM cosine on benign synthetic weights does NOT gate generation — real
  outlier-heavy Qwen3 weights + 48-layer compounding can drift the lm_head argmax. **Needs end-to-end
  PPL/logit-agreement on a real checkpoint** before integration. Decode/M=1-only (M>1 re-reads the column =
  O(M·K·N), fatal for prefill); naive no-split-K kernel under-saturates at small N (occupancy risk).

**Reshaped Wave-2 (gated):** (1) capture Block A (the launch-bound→captured transition the bench shows is
where the 8.5× routing win currently caps) + port the dense static-decode/CUDA-graph stack to the MoE; (2)
build the **fused gather-GEMV** (fp8-packed) for the real roofline; (3) gate fp8 on end-to-end PPL; (4) Nsight
each lever. The naive blocks are correct capturable scaffolds, NOT wired into the 30B until the gates pass.

## 6. LEVER (c) RESULTS — fused gather-GEMV (built + 3-voice-reviewed + 30B-validated)

The measured journey on the real 30B (single-stream greedy decode, GB10):

| stage | tok/s | % of 273 GB/s | what changed |
|---|---|---|---|
| dense oracle (start) | 0.73 | 15% | reads all 128 experts/token |
| static decode (Wave-2 Step-1, fix-b) | 6.45 | 14% | top-8 routing, single-owner contiguous storage, no OOM |
| + CUDA-graph capture (Step-2) | 6.66 | 15% | launch tax removed (decode_topk still materializes) |
| **+ fused gather-GEMV (lever c) + capture** | **19.38** | **43%** | reads k experts by `id·stride` from the stacks, NO slab |

**= 26× from the dense start.** Lever (c): a dedicated M=1 fused gather-GEMV (`moe_grouped.rs`
`fused_swiglu_gu`/`fused_swiglu_down`, generic over bf16/f32) that reads gate/up/down ONCE from the
persistent `[E,H,I]`/`[E,I,H]` stacks by `expert_id·stride` — no `[N,H,I]` materialized slab, no host
re-stack. `decode_topk_fused` == oracle (block-output parity <1e-4, token-identical on the 30B).

**3-voice verdict (Codex 5.5 + Gemini 3.1 Pro + Opus 4.8 source-verified): correct + committable.**
- **Honest perf attribution (Opus corrected the agent's framing):** the **3.0×** is the materializing-oracle
  baseline → fused+captured. Lever (c)'s ISOLATED win is **2.47× eager** (15.89 vs 6.43) + the genuine
  **capture-arena 121 MB → 1 MB** collapse (the slab round-trip is gone); capture adds ~1.22× on top.
  **43% of peak is honest, even conservative** (the `gu` round-trip is 24 KB, trivial).
- **Correct (Opus, kernel-traced):** strides right (each tensor's own `.stride()`), accumulate race-free
  (kernel writes `[N,H]`; the k-combine is the external scatter-add == oracle), bf16→f32 matches the oracle.
  The Gemini/Codex **OOB worry is overstated** — the position is guarded and `route_topk` guarantees the
  expert id ∈ 0..E-1 (documented as the bounds invariant).
- **Fixed before commit:** the `_prec` footgun (Opus's #1 risk) — `decode_topk_fused` ignored `prec` and
  always f32-accumulates; now **asserts `Precision::F32`** (a `Bf16` caller would silently diverge from the
  oracle = the GRPO rollout-vs-recompute trap). Raw-CubeBackend launch documented (below-Fusion / same-stream
  / pinned-lifetime — the capture satisfies it).
- **The corrected roofline (decisive):** on the bf16 top-8 byte model, **`273/6.06 ≈ 45 tok/s is the absolute
  ceiling** — the council's 55-70 target is NOT reachable at bf16; it needs fp8 (fewer bytes). 19.38 is ~43%
  of the **bf16** roofline. The kernel is **latency-bound** (Opus: too few threads each on a long *dependent*
  K-loop; coalescing is already good, the `gu` round-trip is trivial) — so the **#1 next lever is a split-K /
  warp-per-output, register-blocked, vectorized GEMV** (toward the ~45 bf16 roofline), and **fp8 comes AFTER**
  the kernel is bandwidth-bound (and is inference-only — fp8 breaks GRPO parity per the repo docs).

## 7. SPLIT-K KERNEL — built + measured (workflow + 3-voice), and the decisive finding

A split-K GEMV was built via a 5-phase workflow (AGY-USG research → design → 3-voice design review →
serialized 30B build/measure → 3-voice impl review). The kernel does everything the (c) review prescribed:
`fused_swiglu_{gu,down}_splitk` (`src/moe_grouped.rs`) — 2-D `CubeDim::new_2d(32, KSPLIT)`, 32 lanes coalesce
the contiguous output axis as one 128-bit `Line<8>` load, KSPLIT warps split the strided K-reduction summed
cross-warp via `SharedMemory<Line<f32>>` + `sync_cube`, V independent f32 accumulators break the loop-carried
FMA chain. f32-exact, scalar kernels kept as the `V==1` fallback. **Numerics == oracle (token-identical),
all 6 council verdicts GO_WITH_FIXES (fixes applied).**

**Measured (30B, independently re-run): 19.38 → 21.03 tok/s, 43% → 47% peak** (1.49× capture / ~1.09× over
the prior fused). Real, but modest.

**THE DECISIVE FINDING (the workflow's real payoff): the MoE GEMV is NOT the decode bottleneck at N=8.**
Bandwidth-ideal is ~22 ms/tok (6.06 GB/tok ÷ 273 GB/s); we sit at ~47.5 ms/tok, so **~26 ms/tok of residual
is OUTSIDE the expert GEMV** — the 48-layer launch sequencing, attention, the lm_head, and the gu→down global
round-trip — plus **under-occupancy**: at N=8 the grid is only 24 (gu) / 64 (down) blocks, leaving >70% of
the SMs idle, so single-stream decode is occupancy-bound no matter how good the GEMV is. A better expert GEMV
hit its block-level ceiling here. ⇒ **The path to ~45 is NOT a better GEMV** — it's the non-GEMV decode terms
(attention fusion / fewer launches / the lm_head) and/or **fp8** (halve the bytes → ~90 roofline), and/or
**larger batch** (more tokens → more blocks → the GEMV + split-K finally matter). This empirically reframes
the roadmap: kernel micro-opt of the expert GEMV is done; the remaining ~2× is elsewhere.

## GSTACK REVIEW REPORT

| Review | Trigger | Runs | Status | Findings |
|--------|---------|------|--------|----------|
| Codex gpt-5.5 high | council voice 1 | 1 | issues_found | Math right as roofline; 88% efficiency is the weak claim; 0.73 = ~44 GB/s effective (overhead > dense read); reorder graph w/ routing; fp8 must include attn+head (64 cap); grouped-GEMM-at-T=1 naive (TRT-LLM runs MoE-as-dense); KV unaccounted; **honest 50-70, 80 needs spec-decode/W4** |
| Gemini 3.1 Pro high | council voice 2 (AGY-USG) | 1 | issues_found | Host tax = 1.15 s/tok = 5× the dense read → lever order BACKWARDS (graph+sampling first); KV omission → 71 ceiling@4k; scatter-gather → ~75% efficiency; fp8-experts-only=65; grouped-GEMM-at-T=1 anti-pattern; **realistic 50-65, 80 needs spec-decode** |
| Opus 4.8 xhigh | council voice 3 (source-verified) | 1 | issues_found | Math CONFIRMED from config; **diagnosis mislabeled — LAUNCH-bound (16% peak, ~31k launches/tok), not bandwidth**; **"BUILT" overclaimed — capturable top-8 MoE block does NOT exist (forward_routed uncapturable/48 syncs; ondevice+grouped re-stack 128); graph/static stack is DENSE-ONLY; fp8 un-integrated**; fp8 must cover attn+head; **honest 55-70, 80 needs spec-decode** |
| Prod-impl research | agy-direct (Google) | 1 (5 queries) | done | GB10 273 GB/s confirmed; **published DGX-Spark 30B-A3B: bf16 ~30, fp8 ~44-55, NVFP4 ~65 (ceiling) — none at 80**; vLLM/SGLang/TRT-LLM fused-MoE top-k batch-1 + W8A16 + CUDA-graph; llama.cpp ~80-93% effective BW; spec-decode diluted to ~1.2× (K=2) on MoE (`docs/perf-research.md`, 290 lines) |
| Wave-1 impl review | Codex 5.5 + Gemini 3.1 Pro + Opus 4.8 (per-block) | 3 voices × 2 blocks + measurement | done | Both blocks CORRECT but under-gated (§5). Opus kernel-proved `select` reads only k slabs (lever real) + scatter-add race-free; but materializing gather → realized ~5× not ~16×, needs fused gather-GEMV; W8A16 codec bit-faithful but needs end-to-end PPL gate. Launch-bound EMPIRICALLY CONFIRMED (oracle 15% peak, routed 8.5× but still 13%) |
| Wave-2 + lever (c) | Codex 5.5 + Gemini 3.1 Pro + Opus 4.8 source-verified | 3 voices each on the OOM fix + the fused kernel | done | OOM fix = (b) contiguous slot-storage (prod-confirmed, `docs/WAVE2_STATIC_DECODE.md`); static decode + capture token-identical; **fused gather-GEMV CORRECT + committable (§6): 19.38 tok/s @ 43% peak, 3× over materializing**. Opus corrected: 3.0× is capture-attribution (c's isolated win 2.47× + arena 121→1MB); bf16 roofline ~45 tok/s (55-70 needs fp8); kernel is LATENCY-bound → next lever = split-K/vectorize; fixed the `_prec` GRPO-parity footgun |

- **CROSS-MODEL: unanimous, no contradiction.** All 3 independently: (a) bandwidth math is correct as a
  ROOFLINE but ~60-75% sustained is the real number; (b) the 0.73 tok/s is **launch/host-overhead-bound, not
  bandwidth-bound** (the draft's central mislabel) ⇒ **graph capture must come WITH top-8 routing, not after
  fp8**; (c) fp8 must cover **experts + attn + lm_head** (experts-only caps at ~64); (d) the
  grouped-GEMM-"wins-at-T=1" framing is naive (prod uses a fused MoE GEMV / MoE-as-dense); (e) **80 tok/s
  single-stream is a stretch (89% of roofline) — honest landing 55-70; 80 requires speculative decoding** (or
  W4/NVFP4 + all-weight fp8 + short context). Opus adds the load-bearing correction the other two couldn't
  (no code access): **the "built" levers don't apply to the 30B MoE** — the capturable top-8 block is unbuilt
  and the graph/static-decode stack is dense-model-only, so lever A is a real kernel+cache build and D/E/F are
  a real port, not wiring. **Prod research closes it with measurements: the published single-stream ceiling
  for 30B-A3B on this hardware class is ~65 tok/s (NVFP4); nobody has shown 80** — so 80 is above the state of
  the art, not just above our estimate. fp8+top-8+graph+device-sampling lands ~62-72; NVFP4 + light EAGLE-3/MTP
  (K=2, ~1.2× on MoE) → ~80-87 is the only demonstrated route past it.
- **VERDICT:** YES, token-generation efficiency can be massively improved (0.73 → ~55-70 tok/s is a real,
  bandwidth-grounded target) — but **80 single-stream is NOT honest without speculative decoding or W4/NVFP4.**
  The path: build a capturable top-8 MoE decode block (new gather kernel + pre-stacked fp8 expert cache) →
  port CUDA-graph/static-decode to the MoE (concurrent) → fp8 experts+attn+head → measure with Nsight → then
  spec-decode for the last ~15 tok/s. The bandwidth math is the load-bearing frame; the binding constraint
  today is launch overhead.
- **WAVE-1 (built + measured, §5):** the launch-bound diagnosis is now EMPIRICALLY CONFIRMED on the real 30B
  (oracle 0.673 tok/s = 15% peak; routed reads 10× fewer bytes yet LOWER eff GB/s ⇒ overhead-bound). Three
  building blocks landed + 3-voice-reviewed: the capturable top-8 decode block (`select` kernel-proven to
  read only k slabs, scatter-add race-free — but a materializing scaffold realizing ~5× not ~16×, so the
  roofline needs a FUSED gather-GEMV) and `W8A16Linear` (codec bit-faithful, gated on end-to-end PPL). Both
  are correct capturable scaffolds; Wave-2 integration is GATED on bf16 parity + Nsight (A) and end-to-end
  PPL (B). The keystone risk (does the gather actually cut DRAM bytes) is RESOLVED yes.

**UNRESOLVED DECISIONS:**
- Target reset: commit to the **honest ~55-70 tok/s** (top-8 capture + fp8-everything, no new research risk),
  OR keep **80 as a hard OKR** and take on **speculative decoding** (EAGLE/MTP — diluted on MoE) and/or
  **W4/NVFP4** (Blackwell, accuracy risk) as additional scope? The council says 55-70 is plan-backed; 80 is a
  stretch needing one of those two bets. User's call on whether 80 is a hard number or a direction.
