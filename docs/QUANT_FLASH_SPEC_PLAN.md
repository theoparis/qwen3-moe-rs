# Qwen3.6-35B-A3B port + NVFP4 + Flash-Decode + MTP — engineering plan (GB10)

Two-lane plan: **(Lane 1)** port the Qwen3.6-35B-A3B *hybrid* architecture (Gated DeltaNet linear attention +
shared-expert MoE + MTP block + multimodal text extraction) so the engine can run the target at all, and
**(Lane 2)** build the three decode levers (NVFP4, flash-decode, capture) — **proven on Qwen3-30B-A3B in
parallel**, then re-applied to the 35B once the port lands. Kernels are CubeCL **SIMT**, porting battle-tested
math (FlashInfer / vLLM / Dao-flash-attn / cubek-quant / candle), staying on Burn/CubeCL.

**HF candle is now cloned at `/workspace/candle` (v0.11.0)** and the port-mapping below is grounded against its
actual source (§8a). The reuse rule: candle's **CUDA kernels are sm80/sm90/CUTLASS and won't run on sm_121** —
we reuse the **math + Rust model-wiring**, not the binaries. Three things candle does NOT have (greenfield):
**NVFP4/E2M1**, **mRoPE sectioning**, **the MTP block**.

**This is a v2 rewrite after the 3-voice external review (Codex GPT-5.x high + Gemini 3.1 Pro high + Opus 4.8
high) + HF-config verification overturned the v1 framing.** v1 assumed a standard-attention 48-layer model; the
target is a hybrid Gated-DeltaNet multimodal MoE the engine cannot currently load. See `## GSTACK REVIEW
REPORT`.

Companion: `PERF_80TOKS_PLAN.md`, `perf-gap-vs-prod.md`, `longctx-decode-findings.md`, `VLLM_KERNELS.md`,
`rust-tooling-survey.md`.

---

## 0. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | Scope | All 3 levers, unblocked-first |
| D2 | Spec-decode | **MTP first-class** — and it is a **full trained block** (`mtp.layers.0.self_attn/mlp.experts/fc`), not a logits head |
| D3 | Target | **Qwen3.6-35B-A3B** (`Qwen/Qwen3.6-35B-A3B`, Apache-2.0) |
| D4 | Kernel strategy | Port the algorithm into CubeCL SIMT; sm_121 tensor-core pin = separate upstream track |
| D5 | Architecture | **Port the Qwen3.6 hybrid architecture FIRST** (Lane 1) — the engine can't run it today |
| D6 | Quant | ~~NVFP4-first~~ → **FP8-first (RESOLVED 2026-07-01, 3-voice unanimous)**: real-35B D6 gate showed NVFP4-dense collapses (repetition) while FP8 ≈ bf16, and NVFP4-over-FP8 is <~1.1× at occupancy-bound decode. The per-token-identity gate was INVALID for PTQ (kept only for spec-decode); use teacher-forced KL/top-1/PPL. See UNRESOLVED DECISIONS + docs/specs/M-B-nvfp4-gate-plan.md |
| D7 | Sequencing | **Two parallel lanes** — 35B architecture port + kernel levers proven on 30B-A3B; levers re-apply to 35B after the port |

---

## 1. The target model — verified reality (HF config.json + safetensors index)

`Qwen/Qwen3.6-35B-A3B` is **not** a standard-attention MoE. Verified facts (authoritative HF config, cross-
checked by Codex + Gemini):

```
architectures: ["Qwen3_5MoeForConditionalGeneration"]    model_type: qwen3_5_moe   (MULTIMODAL: vision_config present)
text_config:
  num_hidden_layers: 40
  layer_types: [linear_attention ×3, full_attention] × 10   ← 30 GDN linear-attn + 10 full-attn (full_attention_interval: 4)
  linear_conv_kernel_dim: 4, linear_key_head_dim: 128, linear_num_key_heads: 16,
  linear_num_value_heads: 32, linear_value_head_dim: 128, mamba_ssm_dtype: float32   ← Gated DeltaNet / SSM
  num_attention_heads: 16, num_key_value_heads: 2, head_dim: 256, partial_rotary_factor: 0.25,
  rope: mrope_interleaved (mrope_section [11,11,10]), rope_theta 1e7                  ← full-attn geometry (NOT 30B's)
  num_experts: 256, num_experts_per_tok: 8, moe_intermediate_size: 512,
  shared_expert_intermediate_size: 512                                                ← 2× experts + a SHARED expert
  mtp_num_hidden_layers: 1, mtp_use_dedicated_embeddings: false                        ← native MTP, a full block
  vocab_size: 248320, hidden_size: 2048, dtype: bfloat16
```

**Engine gap (current engine = Qwen3-30B-A3B: GQA full-attn, 48 layers, 128 experts, head_dim 128, no shared
expert, no GDN, no MTP, no vision).** To run the target, Lane 1 must build, from scratch:
- **Gated DeltaNet linear-attention decode** (30/40 layers) — recurrent matrix state + short conv; the
  dominant compute; no existing code. **This is bigger than all three levers combined.**
- **Shared-expert MoE + 256-expert routing** (every layer adds a shared expert to the top-8).
- **Full-attention layers** with head_dim 256, `partial_rotary_factor 0.25`, interleaved **mRoPE**.
- **MTP block** (its own self-attn + MoE + fc), not a head.
- **Multimodal text-model extraction** (load the text tower of a `…ForConditionalGeneration` checkpoint).
- **Config + sharded loader** for `qwen3_5_moe`.

### 1a. What this does to the three levers on the 35B
- **GDN linear attention is constant-time per token** (rolling state, no KV growth) — so the model is *designed*
  for long context, and the O(pos) cliff only afflicts the **10/40 full-attention layers**. ⇒ **Flash-decode's
  value on the 35B is smaller than on the 30B** (where all layers are full-attn). Flash-decode's big win is on
  the 30B (Lane 2) and on the 35B's 10 full-attn layers' long-context scaling.
- **The dominant 35B bandwidth term is the MoE** (256 experts top-8 + shared, every layer) → **the quant lever
  (NVFP4/FP8) is the most central lever for the 35B**, more than flash-decode.
- **MTP rollback must restore GDN recurrent/conv state, not just KV columns** (Codex F7, Gemini #3).

---

## 2. The binding constraints (the frame for the kernels)

### 2a. sm_121 tensor-core wall — UPDATED by P0.3 (the NVFP4 path is NOT blocked)
At our pins (cubecl `b19859ee`+5 cuda-graph commits, cubek `1161040`), **CMMA/WMMA bf16/f32 matmul fails to
compile** (`CmmaInstructionUnavailable`, proven by `cubek_attn_spike`). candle's FA (sm80 CUTLASS), burn FA3,
cubek-matmul all hit it on the bf16/f32 path.

**P0.3 + P0.3b (DONE, 3-voice each — `docs/P0.3-scaled-mma-findings.md`, `docs/P0.3b-tiled-nvfp4-findings.md`)
re-mapped the NVFP4 tensor-core picture precisely:**
- **The instruction works:** the E2M1 block-scaled MMA (`mma.sync...mxf4nvf4.block_scale...m16n8k64...e2m1.e2m1.f32.ue4m3`)
  JIT-compiles + launches + is bit-correct for the TOP half of a tile on sm_121 (P0.3 `max_abs_diff=0.0`). **The
  "NVFP4 blocked until a pin bump" assumption is dead.**
- **But the canonical full tile is a cubecl BUG (P0.3b, NO-GO):** the natural single-`execute_scaled` path is
  correct for rows 0-7 but **wrong for rows 8-15 on the e4m3/scales_factor=4 route** (positive scales: rows0-7=0.0,
  rows8-15=55.0 FAIL; tiled M128N128K2048 FAIL). 3-voice root-cause (0.7-0.9): the **e4m3 scale-lane mapping for
  `scale_vec::4X` in `cubecl-cpp/.../cuda/ptx/mma.rs`** (the runtime test only covers ue8m0/factor-2, so e4m3/4X
  bottom-half was never exercised). The smem trap is NOT the blocker (2880 B fits 99 KiB).
- **⇒ NVFP4 tensor-core PREFILL + MTP-VERIFY are BLOCKED by a fixable cubecl codegen bug** → **P0.3c** (fix
  `ptx/mma.rs` scale-register packing + add an all-16-rows e4m3/4X runtime test; **this is genuine "improve
  cubecl" work, the unanimous #1 path**), with **bf16-prefill defer as the safe fallback (path d)**.
- **NVFP4 DECODE is SIMT (no tensor cores) → UNAFFECTED and remains the critical-path lever** (bandwidth-bound;
  m16n8 wastes 15/16 of M at M=1). The "pin bump" track is dead; the real gate is the P0.3c cubecl fix.

### 2b. Decode is occupancy/latency-bound, NOT purely bandwidth-bound (the corrected core bet)
`PERF_80TOKS_PLAN.md` §7 (measured, 30B): at N=8 single-stream decode **>70% of SMs are idle**, the GEMV is
latency-bound on a dependent K-loop, and **~26 ms of the 47.5 ms/tok is OUTSIDE the expert GEMV** (attention,
lm_head, 48-layer sequencing). **All three reviewers: NVFP4 is a bandwidth lever applied to an occupancy-bound
regime → ~1.3×, not 1.5–2×.** Corrected honest landings:

| lever | 30B short-ctx × | 30B long-ctx × | the real win |
|---|---|---|---|
| **NVFP4 decode-GEMV** | **~1.3×** (≈21→~27–30 tok/s) | ~1× | weight bytes, but capped by occupancy + the 26 ms non-GEMV residual |
| **Flash-decode O(pos)** | ~1.0–1.3× | **2–3× at 1K → fixes OOM at 32K+** | the +100 ms/tok long-ctx attention cliff |
| **Capture** | already in the 21 baseline | — | do NOT re-multiply; the 21 tok/s captured number already includes the ~1.3–1.5× capture win |

**The v1 "45–55 sustained / 65–70 NVFP4" target is withdrawn — it double-counted capture and ignored §7.**
Honest 30B targets: short-ctx **~27–35 tok/s** (NVFP4 + the non-GEMV cuts), long-ctx **flash-decode removes the
cliff**. "Candle-level 65–70" is NOT a committed target until a CubeCL microbench shows >70% GB10 bandwidth on
the *actual routed expert shapes* (Codex F15). Atlas's ~70 NVFP4 is a custom-CUDA existence proof, and its
Qwen3.6 recipe is **FP8, not NVFP4** (Codex) — a flag on NVFP4-for-this-model accuracy.

---

## 3. Two-lane sequencing (D5 + D7)

```
 PHASE 0 — de-risk probes (cheap, first, parallel)
 ├─ P0.1  Download + inspect Qwen3.6-35B-A3B (config, safetensors index, MTP weight shapes, GDN params)
 ├─ P0.2  Factor the per-bench capture harness into a reusable src/ helper
 ├─ P0.3  E2M1 scaled-MMA probe on sm_121 (does the instruction compile+run? — answers prefill/verify only)
 ├─ P0.4  No-op/empty-split CUDA-graph overhead probe (cost of launching idle CTAs at small pos in a T_max grid)
 └─ P0.5  One-layer NVFP4 GEMV + Nsight on real 30B expert shapes (is it bandwidth-bound or ALU-bound? — F1/F2 gate)
            │
   ┌────────┴───────────────────────────────────────────────────────────────────────┐
   ▼                                                                                  ▼
 LANE 1 — Qwen3.6-35B-A3B ARCHITECTURE PORT (the target; the bulk of the work)      LANE 2 — KERNEL LEVERS on 30B-A3B
 │ L1.1 config + sharded loader (qwen3_5_moe), multimodal text-tower extraction      │ L2.A flash-decode O(pos) (full-attn)
 │ L1.2 full-attention layer: head_dim 256, partial-rotary 0.25, mRoPE               │   A1 bf16-in/f32-acc variant
 │ L1.3 Gated DeltaNet linear-attention decode (recurrent state + conv) ★ biggest    │   A2 split-K + warp + DEVICE-pos loop
 │ L1.4 shared-expert MoE + 256-expert routing                                       │   A3 raw-CubeBackend launch (off Fusion bridge)
 │ L1.5 MTP block (self_attn + MoE + fc)                                             │   A4 wire into attention.rs; drop GQA repeat
 │ L1.6 end-to-end greedy parity vs HF reference (text-only)                         │   A5 padding mask + per-row seq_len (ragged GRPO)
 └────────────────────────────────────────────────────────────────────────────────  │ L2.B capture-in-vllm_infer (after A3)
                                                                                      │ L2.C NVFP4 (calibration-gated): codec → GEMV → Linear
                                                                                      │       fallback: FP8 w8a16 if NVFP4 fails the token-id gate
   └──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               ▼
 PHASE 2 — CONVERGE on 35B (after Lane 1 loads + Lane 2 kernels validated on 30B)
 ├─ flash-decode → the 10 full-attention layers       ├─ NVFP4/FP8 → the 256 experts + shared + attn + lm_head (the dominant 35B lever)
 ├─ MTP block + n-gram-probe-first verify machinery    └─ capture: reconcile with GDN state + MTP dynamics (see §7)
```

Lane 1 and Lane 2 run in parallel (disjoint: new model modules vs the 30B kernels). Phase 2 converges. MTP is
**last and conditional** on the n-gram probe (Codex F9, mandatory).

---

## 4. Lane 1 — the Qwen3.6 hybrid architecture port (NEW, the bulk)

### 4.1 Gated DeltaNet linear-attention decode (L1.3) ★ the load-bearing new kernel
30 of 40 layers. GDN keeps a **recurrent matrix state** `S[d_k, d_v]` per head + a short causal conv
(`linear_conv_kernel_dim 4`) over the input projections; decode is a constant-time-per-token state update +
readout (no KV growth). This is a Mamba2/GatedDeltaNet-class kernel — **not expressible as the existing SDPA**.

```
 decode token t, per linear-attn layer, per head h (16 k-heads / 32 v-heads):
   q,k,v,β,α = projections(x_t)           ← + short conv (kernel 4) over the last 4 tokens (conv state)
   S_t = α_t · S_{t-1} + β_t · (k_t ⊗ (v_t − S_{t-1}ᵀ k_t))   ← gated delta rule (recurrent matrix state)
   o_t = S_tᵀ q_t                          ← readout;  state S persists across tokens (the rollback hazard)
```

- **Port from (grounded against candle):** the **gated-delta-rule spec is HF `modeling_qwen3_next` / FLA `fla`**
  (authoritative — candle has no GatedDeltaNet). But candle gives strong, verified references for the *pieces*:
  - **delta rule:** `candle/candle-transformers/src/models/rwkv_v7.rs:405-430` implements it almost 1:1 —
    matrix state `(n_heads, head_size, head_size)` in **f32** (`:83`), `new_state = state*w(decay) + state@ab +
    v⊗k` where `state@ab` is the `−(k⊗Sᵀk)` delta correction, readout `out = state@r = Sᵀq`. Maps directly to
    `S_t = a·S_{t-1} + b·(k⊗(v−Sᵀk)), o=Sᵀq`. (Gap: RWKV-7 uses token-shift not a depthwise conv; GDN's per-head
    `a=exp(−softplus)` decay + `beta` gate differ — take those from FLA.)
  - **short-conv state carry:** `mamba2.rs:262-324` (`apply_conv1d` window roll + elementwise-decay update).
  - **hybrid cache (the 30 GDN + 10 attn mix):** `lfm2.rs:142-201` — one `Cache` carrying both attn-KV and
    conv/recurrent state per layer + `reset()` semantics. The cleanest wiring template for this model.
  Implement the recurrent update as a SIMT CubeCL kernel; state lives in a persistent device buffer
  (capture-stable, like the KV cache), accumulated in **f32** (`mamba_ssm_dtype: float32`).
- **State management:** add a GDN-state cache alongside the KV cache (conv ring buffer + `S` matrix per head
  per layer). Decode updates it in place (capture-safe); **prefill** must run the chunked form or a sequential
  warmup to build `S` before the first decode step.
- **Tests:** recurrent decode must equal the chunked prefill at the boundary; cosine vs an HF/FLA CPU reference
  on a real prompt; numerically `mamba_ssm_dtype: float32` (accumulate state in f32).

### 4.2 Full-attention layers (L1.2)
10/40 layers. head_dim **256** (not 128), GQA 16q/2kv, `partial_rotary_factor 0.25` (RoPE applies to the first
64 of 256 dims), interleaved **mRoPE** (`mrope_section [11,11,10]`). Reuse `src/attention.rs` structure but add
partial-rotary + mRoPE to `src/rope.rs`. Flash-decode (Lane 2) targets these layers.

### 4.3 Shared-expert MoE + 256 routing (L1.4)
Every layer: top-8 of **256** routed experts **plus a shared expert** always active (`shared_expert_intermediate_size
512`). Extend `moe.rs`/`moe_grouped.rs` routing to 256 + add the shared-expert path (a dense FFN added to the
routed combine). The fused gather-GEMV + persistent expert stacks generalize (more experts = bigger stacks).

### 4.4 MTP block (L1.5) — see §7 (it's the spec-decode lever, built last).

### 4.5 Config + loader + multimodal extraction (L1.1)
New `Qwen3_5MoeConfig`, sharded `qwen3_5_moe` loader, extract the text tower (skip vision weights for text-only
inference). Verify greedy parity vs HF transformers on a fixed prompt (L1.6) before any perf work on the 35B.

---

## 5. Lane 2A — O(pos) flash-decode (proven on 30B)

Today `src/attention.rs:494` reads the full `T_max` KV buffer every step (O(T_max)), masks the future, runs a
non-fused SDPA, and physically `repeat()`s GQA 4→32. `src/flash_attn.rs` is a correct FA-2 foundation (cosine
1.0 vs CPU oracle) but single-thread-per-cube (~10–100× too slow), f32-only, Fusion-bridge, no mask.

### Data flow + the review's hard corrections folded in
```
 grid = (num_q_heads, num_kv_splits)   num_kv_splits sized to T_max (CONSTANT ⇒ capturable)
 inner loop bound = pos read from a DEVICE tensor   ← NOT a host ScalarArg (F2: a host scalar bakes a stale range at capture)
 splits entirely > pos early-exit (block-granular, uniform per block ⇒ minimal warp divergence; cost = idle CTAs, F7)
 kv_head = q_head / n_rep   GQA broadcast in-register (no physical repeat)
 online-softmax (m,l,acc; acc*=α) → cross-split LSE merge
```

- **A2 device-pos (F2, conf 7):** the loop bound is a device value (`let n = pos[0]`), so one captured graph
  serves every length. A host `ScalarArg` would replay a stale range — silent over/under-read.
- **A3 raw-CubeBackend launch (F2, conf 7):** `flash_attn.rs` currently goes through `CubeCustomOp::<CudaRuntime>`
  (a Fusion-bridge op) — **cannot run inside the below-Fusion captured region.** Port to a raw-`CubeBackend`
  launch (the pattern `moe_grouped` fused kernels already use in `cudagraph_moe_decode_bench`). This is
  prerequisite work for B, stated explicitly now.
- **A2 register budget (F8, conf 6):** head_dim 256 (35B) / 128 (30B) → `q_reg+acc` is 512/256 f32/thread →
  local-mem spill caps occupancy. Plan: vectorized 128-bit `Line` loads, split D across the warp lanes, stage
  `acc` in shared memory for the merge. Without a register-budget plan, split-K may not clear even its modest
  short-ctx multiplier.
- **A5 ragged GRPO (F11, conf 6):** the scalar device-`pos` bounds a *single-stream* loop. A ragged GRPO batch
  needs a **per-row `seq_len` buffer** (vLLM `seq_lens`), not a scalar. So GRPO either keeps the mask + full
  scan (correct, no O(pos) win) or builds the per-row path. State this; don't conflate single-stream and ragged.
- **A4** drop `into_contiguous` + physical GQA repeat in `attention.rs`.
- **Context bucketing (F7):** geometric `T_max` buckets (1K/4K/16K/…) so short-ctx capture doesn't launch a
  32K-sized grid of idle CTAs (`longctx-decode-findings.md`). In scope for the captured path.

### Tests
NdArray f32 stable-softmax oracle (cosine + **max-abs/rel + top-k overlap**, not cosine alone); pos ∈ {1, 800,
4K, 32K}; head_dim 128 (30B) + 256 (35B); bf16 + f32; **CRITICAL** GRPO ragged padding-mask logprob-parity
regression; capture buffer-move mutation test; Nsight `dram__bytes_read` O(pos); **empty-split overhead probe
(P0.4)** must show idle-CTA cost is tolerable at short ctx before relying on the static grid.

---

## 6. Lane 2B/2C — capture + NVFP4 (proven on 30B)

### 6B. Capture in vllm_infer
`capture_arena(warmup, f)` → `CapturedGraph::replay()` exists; `cudagraph_moe_decode_bench.rs` is the working
template. `vllm_infer.rs` is eager: runs on `Cuda=Fusion` (must be raw `CubeBackend`) and does a per-step D2H
read to sample (`:130–146`, capture poison). **B1** raw backend, **B2** on-device argmax/Gumbel (`sampling_device.rs`)
into persistent buffers, **B3** `capture_arena`/`reset_for_replay`/`replay`. **P0.2 first:** factor the
hand-rolled harness into `src/` (DRY — `vllm_infer`, benches, and `rollout.rs:551` all need it). **Do NOT
re-multiply the capture speedup** — the 21 tok/s baseline already includes it (F1). Capture's job here is to make
`vllm_infer` match the bench, not to add a new 1.3×.

### 6C. NVFP4 (D6: NVFP4-first, calibration-gated)
NVFP4 = E2M1 (4-bit) + per-16 **E4M3 block scale** + FP32 global scale = 72 bits/16 weights = **0.5625 B/weight,
3.56× smaller than bf16** (not 4×, F3/Codex). Extend the `w8a16.rs` dequant-in-load pattern to 4-bit + block
scale, adopting **cubek-quant** `dequantize_symmetric_packed_value` + native `e2m1x2` packing.

**Candle grounding (§8a):** candle has **NO E2M1/NVFP4** (only F8E4M3 + the E8M0 MX-scale type) — the *format*
is greenfield (cubek-quant supplies e2m1x2). But candle's **`fast_mmvq.rs:161-214` is the structural reference**
for the decode GEMV: a batch≤8 quantized matmul-vector that dynamic-quantizes the activation (to Q8_1) and does
**dequant-in-load dot** against block-scaled 4-bit weights — exactly our shape. The `BlockQ4K` layout
(`k_quants.rs:153-158`: super-block scale + 6-bit packed sub-block scales + 4-bit values) is the scaffold to
adapt to NVFP4's (block-16, E4M3 scale, no min). Port the structure to CubeCL SIMT (candle's is cudarc/PTX,
BLOCKED on sm_121).

```
 q4:[K/2,N] (e2m1x2)   bs:[K/16,N] (e4m3)   gscale:f32
 one warp per output col n (split-K over rows):
   unpack e2m1x2 → v0,v1 ; s = e4m3_to_f32(bs[kk/16,n])·gscale ; acc += x·(v·s)   ← f32 MAC, never materialize bf16
 cross-split sum (SharedMemory<f32>)
```

**The review's hard NVFP4 gates (Gemini #2, Opus F5, Codex F4 — unanimous):**
- **Calibration, not naive `amax/qmax`.** PTQ NVFP4 on a 248K-vocab lm_head + MoE gating logits flips greedy
  tokens. Use a calibrated codec (AWQ/SmoothQuant/Hadamard-style scale selection, per the RedHatAI/vLLM NVFP4
  recipe), not the v1 naive per-block max. Atlas ships FP8 (not NVFP4) for this model — a warning.
- **Gate on per-token-identity, not PPL.** PPL is an average and can pass while greedy argmax flips on near-tied
  tokens (F5). Gate = the captured greedy string is **token-identical** to the bf16 greedy string on a fixed
  prompt set (the 30B has a known-good string, `vllm_infer.rs:21`). PPL is secondary.
- **Prove bandwidth-bound first (P0.5).** One-layer NVFP4 GEMV + Nsight: if `dram__throughput` is low and
  scheduler stalls high (ALU/unpack-bound at N=8, F1/F2/Codex), NVFP4 won't beat FP8 at batch-1 — **fall back to
  FP8 `w8a16`** (already validated). Coalesce the block-scale stream (F3) — `[K/16,N]` per-warp-column reads can
  be uncoalesced.
- **M>1 footgun (F9):** the SIMT dequant-GEMV re-reads weights M× at M>1 (the w8a16 W8A32 trap). NVFP4 for the
  **MTP verify** batch (M=K) would invert the byte win — verify needs the scaled-MMA path (P0.3) or accepts the
  penalty. NVFP4 is decode-only (M=1); inference-only (breaks GRPO parity like fp8).
- **C4 prefill** (gated on P0.3 + the 99-KiB check): a GO on the scaled-MMA *instruction* probe does **not**
  prove a tiled NVFP4 prefill GEMM fits in 99 KiB smem (the TRT-LLM CUTLASS trap, F10/Gemini #6). Assert smem
  before building C4.

---

## 7. MTP speculative decoding (Phase 2, last, conditional)

**MTP is a full trained block** (`mtp.layers.0.self_attn.*`, `mtp.layers.0.mlp.experts.*`, `mtp.fc.weight`,
Codex F6), not a logits head. It proposes K draft tokens; the target verifies in one M=K forward; accepted
tokens advance >1 per weight read.

```
  last hidden ─► MTP block (self_attn + MoE + fc) ─► K draft tokens
        │                                   build verify batch [tok_pos, d1..d_{K-1}]  (M=K)
        ▼                                              ▼
   accept longest matching prefix ◄── ONE target forward over K-batch (routes to UNION of top-8 → >8 experts)
        └─ on mismatch: take target token + ROLL BACK both KV columns AND GDN state to pos+j
```

**The load-bearing correctness changes (unanimous):**
- **`select_assign(Add)` rollback (F3, Opus, conf 8 — the most dangerous silent bug).** `cache.rs:119`'s `Add`
  == assign **only** because each column is written once over a zero-init buffer. Speculative re-writes
  **accumulate**. `rollback_to(pos)` must **zero columns `[pos+j..]`**, not just decrement `filled`. Mandatory.
- **GDN state rollback (F7/Gemini #3).** A rejected draft has already mutated the GatedDeltaNet recurrent matrix
  `S` + conv ring. Rollback must **restore `S` and the conv state** to pos+j, not just KV. ⇒ snapshot/restore
  GDN state per speculative step. This is the single biggest MTP risk on this model and v1 missed it entirely.
- **MTP vs capture are partly dynamic (F4).** Variable accept-length j → host control flow (a per-step D2H of
  the accept count = capture poison); variable expert-union size → non-static verify shape. Reconcile by
  capturing the fixed sub-steps (draft, verify-at-fixed-K, rollback-to-bucket) and keeping the accept decision
  on host, OR pad to K (which erodes the spec win). Measure; don't assume capture+MTP compose for free.
- **Verify dilution + M>1 (F8/F9 + perf-gap §7):** union-of-experts ~15 at K=2 (256-expert: similar union math)
  + shared expert + the M>1 re-read → net likely **~1.2× or less**. K=2 only.
- **n-gram probe FIRST is mandatory (Codex F9).** Build the n-gram/prompt-lookup draft (no model) to prove the
  **verify-batch + cache-rollback + GDN-state-rollback machinery** and quantify multi-column-write tolerance
  *before* the full MTP block. If the machinery is wrong, the n-gram probe localizes it (vs the full MTP where a
  divergence could be draft math, verify routing, rollback, or state aliasing).

**Cardinal invariant:** spec-decode output is **token-identical to greedy** (acceptance is speed-only). Plus
bit-exact KV+GDN-state rollback. Measured acceptance + net tok/s K=2.

---

## 8. What already exists (reuse, don't rebuild)

| Need | Exists | Plan |
|---|---|---|
| FA-2 recurrence + GQA broadcast | `src/flash_attn.rs` (cosine 1.0, single-thread, f32, Fusion-bridge) | Extend: split-K + bf16 + mask + **raw-backend launch** + **device-pos** |
| Capture mechanism | `capture_arena`/`CapturedGraph`, `cudagraph_moe_decode_bench.rs` | Wire into vllm_infer; factor harness to `src/` |
| Dequant-in-load GEMM | `src/w8a16.rs` (FP8, validated) | Narrow to 4-bit + block scale (NVFP4); **FP8 is the fallback** |
| NVFP4 dequant primitive | cubek-quant `dequantize_symmetric_packed_value`, `e2m1x2` | Adopt directly |
| Split-K vectorized GEMV | `moe_grouped.rs::fused_swiglu_*_splitk` | Template for flash-decode merge + NVFP4 GEMV |
| Device sampler | `src/sampling_device.rs` | On-device argmax/Gumbel for capture |
| Static KV + device-pos write | `src/cache.rs` | Read side (flash) + **rollback-zeroing** (MTP) + **GDN-state cache** (new) |
| MoE routing + fused gather-GEMV | `src/moe*.rs` | Extend to 256 experts + shared expert |
| E2M1 scaled-MMA codegen | cubecl `compile_scaled_mma` (sm_121-gated, untested) | Probe (P0.3); NVFP4 prefill if it clears the 99-KiB check |
| GDN linear attention | **no local code**; candle refs (§8a): `rwkv_v7.rs` (delta rule), `mamba2.rs` (conv), `lfm2.rs` (hybrid cache) | New (L1.3) — biggest build, but **port-with-references**, not pure greenfield |
| mRoPE | **greenfield** (candle absent); partial-rotary ref `phi3.rs`/`glm4_new.rs` | New + extend `src/rope.rs` |
| Shared+routed MoE wiring | candle `qwen2_moe.rs:229-254` (shared_expert + sigmoid gate) | Port wiring; extend `src/moe*.rs` to 256 + shared |
| MTP block, multimodal loader, qwen3_5_moe config | **greenfield** (candle absent for MTP) | New (Lane 1) |

---

## 8a. Candle reuse map (verified against `/workspace/candle` v0.11.0)

The reuse rule: candle's CUDA is **sm80/sm90/CUTLASS** (flash-attn sm80, flash-attn-v3 sm90, quantized GEMV
cudarc/PTX) — **none runs on sm_121**, so we reuse the **algorithm** (port to CubeCL SIMT) or the **Rust
model-wiring**, never the binaries.

| Workitem | candle reference (file:line) | Verdict | What to port |
|---|---|---|---|
| Flash-decode split-K, hdim256 | `candle-flash-attn/build.rs:8-60`, `kernels/flash_fwd_launch_template.h:39-49`, `flash_api.cu:161` | algorithm (CUDA blocked) | split-KV partition + LSE-rescale combine, hdim256 bf16 tiling; **supply your own num_splits heuristic** (candle pins =1) |
| Attention wiring | `candle-transformers/src/models/qwen3.rs:225-352` | model-wiring | 3-path dispatch, KV cache, rope, q/k-norm, GQA |
| NVFP4 format | — (`dtype.rs:29,37` only F8E4M3 + E8M0) | **ABSENT — greenfield** | nothing; build E2M1 fresh (cubek-quant `e2m1x2`) |
| NVFP4 decode GEMV | `quantized/fast_mmvq.rs:161-214`, `k_quants.rs:153-158` | algorithm (CUDA blocked) | MMVQ batch≤8 dequant-in-load dot; Q4_K block layout as scaffold |
| Gated DeltaNet (delta rule) | `rwkv_v7.rs:405-430` (state `:83`, f32) | algorithm + wiring | `state=state*w + state@ab + v⊗k`, readout `state@r=Sᵀq`; per-head matrix state, f32 accum, 1-token step |
| GDN short-conv + state | `mamba2.rs:262-324` | algorithm + wiring | depthwise conv window roll + decay update |
| Hybrid attn+recurrent cache | `lfm2.rs:142-201` | model-wiring | one Cache carrying attn-KV + conv/recurrent state per layer; reset semantics — **the template for 30 GDN + 10 attn** |
| GDN authoritative spec | — (candle ABSENT) | external | **HF `modeling_qwen3_next` / FLA** for the gated-delta-rule decay/beta gates |
| Routed MoE | `qwen3_moe.rs:98-178` | model-wiring | gate→softmax→top-k→per-expert gather |
| Shared + routed expert | `qwen2_moe.rs:229-254` (closest); `deepseek2.rs:736-811` | model-wiring | shared_expert + sigmoid shared_expert_gate alongside routed |
| Partial-rotary 0.25 | `phi3.rs:58-88`, `glm4_new.rs:18,34` | model-wiring | rotate first `partial_dim`, pass-through the rest |
| mRoPE | — (`qwen3_vl/text.rs` plain rope) | **ABSENT — greenfield** | mRoPE `mrope_section [11,11,10]` sectioning fresh |
| MTP / speculative | — | **ABSENT — greenfield** | MTP block + verify/rollback fresh |

**Net:** candle de-risks **the levers** (flash-decode algorithm, NVFP4-GEMV structure, attention/MoE/partial-rotary
wiring) **and meaningfully de-risks the GDN port** (rwkv_v7 delta-rule + mamba2 conv + lfm2 hybrid cache are real
references, upgrading L1.3 from "greenfield" to "port-with-references"). **NVFP4 format, mRoPE, and MTP are the
only fully-greenfield pieces** — candle has nothing for them.

## 9. NOT in scope (deferred, with rationale)

- **NVFP4 prefill MMA** — behind P0.3 + the 99-KiB check; decode is the BW-bound win.
- **Vision tower** — text-only inference; extract the text tower, skip vision weights.
- **Paged KV / continuous batching** — multi-request only; single-stream gets nothing.
- **Porting policy to candle** — framework migration; staying on Burn/CubeCL (your directive).
- **fp8 KV cache** — only at true long context; defer until flash-decode proves O(pos).
- **EAGLE/Medusa** — MTP (native trained block) chosen.
- **NVFP4/FP8 in the GRPO trainer** — inference-only (parity break).
- **Adopting cubek-attention / burn FA3** — NO-GO on sm_121 at our pins.
- **"candle-level 65–70 tok/s" as a committed target** — withdrawn until a microbench proves >70% GB10 BW on
  routed expert shapes (Codex F15); kept as a direction, not an OKR.

---

## 10. Failure modes (silent-bug-first, per the review)

| Codepath | Silent failure | Guard (mandated) |
|---|---|---|
| MTP rollback `select_assign(Add)` (F3) | re-written column **accumulates** → corrupt KV | `rollback_to` **zeroes** `[pos+j..]`; token-identical-to-greedy gate |
| MTP GDN-state rollback (F7) | stale recurrent `S`/conv → divergence after first reject | snapshot/restore GDN state; bit-exact rollback test |
| flash-decode host-pos at capture (F2) | bakes a stale KV range → over/under-read | read `pos` from a **device** tensor; buffer-move mutation test |
| flash-decode Fusion-bridge in capture (F2) | op can't run below Fusion → capture abort/corrupt | raw-`CubeBackend` launch (A3) |
| NVFP4 PTQ accuracy (F5/F4) | PPL passes, greedy argmax flips on the 248K head | **calibrated codec + per-token-identity gate**, not PPL |
| NVFP4 ALU-bound at N=8 (F1/F2) | "win" measures ~1.3× and reads as broken | P0.5 Nsight gate; FP8 fallback |
| MTP capture dynamics (F4) | per-step accept-count D2H poisons the graph | capture fixed sub-steps; measure, don't assume |
| GDN prefill/decode boundary | recurrent state mismatched vs chunked prefill | boundary-equality test (recurrent == chunked) |

**Critical (silent + load-bearing):** MTP `Add`-rollback, GDN-state rollback, flash-decode host-pos — each has a
mandated test above; none ships without it.

---

## 11. Parallelization (worktree lanes)

| Lane | Modules | Depends |
|---|---|---|
| P0 | models/, new probe examples, src/cache.rs | — |
| L1 (35B port) | new src/qwen3_5/* (GDN, mRoPE, shared-MoE, MTP, loader) | P0.1 |
| L2A (flash-decode) | src/flash_attn.rs, src/attention.rs, src/cache.rs (read side) | P0.2/P0.4 |
| L2B (capture) | examples/vllm_infer.rs, new src/capture helper | P0.2, L2A.A3 |
| L2C (NVFP4) | src/nvfp4*.rs, src/w8a16*.rs, src/moe_grouped.rs | P0.3/P0.5 |

**Launch L1 + L2A + L2C in parallel worktrees** (disjoint). **L2B after A3.** **Phase 2 (MTP + converge) last.**
**Conflict flag:** L2A and MTP both edit `src/cache.rs` (read-side vs rollback) — coordinate. L2C and L1 both
edit `src/moe_grouped.rs` (NVFP4 GEMV vs shared-expert/256 routing) — sequence or coordinate.

---

## 12. Implementation tasks

- [ ] **P0.1 (P1, CC: ~45min)** — models — download + inspect Qwen3.6-35B-A3B (config, safetensors index, MTP/GDN shapes). _Verify: shapes documented; loader requirements enumerated._
- [ ] **P0.2 (P1, CC: ~2h)** — capture — factor per-bench capture harness into reusable `src/` helper. _Verify: moe_decode bench uses it, still token-identical._
- [ ] **P0.3 (P1, CC: ~2h)** — cubecl — E2M1 scaled-MMA probe on sm_121 (compile+run+numerics) + assert smem ≤99 KiB for a tiled shape. _Verify: GO/NO-GO + smem headroom._
- [ ] **P0.4 (P1, CC: ~2h)** — flash — empty-split CUDA-graph overhead probe (idle-CTA cost at small pos, T_max grid). _Verify: idle-CTA ms/tok at pos=32 in 32K grid._
- [ ] **P0.5 (P1, CC: ~3h)** — nvfp4 — one-layer NVFP4 GEMV + Nsight on real 30B expert shapes (BW-bound vs ALU-bound). _Verify: `dram__throughput` + stall %; FP8-vs-NVFP4 decision._
- [ ] **L1.1–L1.6 (P1, CC: ~1–2wk)** — qwen3_5 — config + loader + text extraction; full-attn (head_dim 256, partial-rotary, mRoPE); **GDN linear-attention decode + state ★**; shared-expert MoE + 256 routing; MTP block; greedy parity vs HF. _Verify: text-only greedy matches HF on a fixed prompt._
- [ ] **L2A.A1–A5 (P1, CC: ~2d)** — flash_attn/attention — bf16 variant; split-K+warp+**device-pos**; **raw-backend launch**; wire-in + drop GQA repeat; mask + per-row seq_len. _Verify: oracle + Nsight O(pos) + CRITICAL GRPO parity._
- [x] **L2B.B1–B3 (P1, CC: ~4h)** — vllm_infer — raw backend + on-device sampler + capture/replay (no re-multiply). _Verify: captured == eager greedy._
- [ ] **L2C.C1–C3 (P1, CC: ~1.5d)** — nvfp4 — calibrated codec + golden vectors; fused 4-bit dequant-in-load GEMV (coalesced block scales); NVFP4Linear + load. _Verify: CRITICAL per-token-identity gate on 30B; Nsight bytes; **FP8 fallback if it fails**._
- [ ] **C4 (P2, CC: ~1d, gated P0.3)** — nvfp4 — NVFP4 prefill scaled-MMA (or defer to pin). _Verify: prefill numerics + smem fit._
- [ ] **MTP.1 (P1, CC: ~1d)** — spec_decode — n-gram/prompt-lookup draft + verify-batch + **rollback-zeroing + GDN-state rollback** machinery. _Verify: CRITICAL token-identical-to-greedy + bit-exact KV+state rollback._
- [x] **MTP.2 (P2, CC: ~2–3d, conditional)** — spec_decode — full MTP block (self_attn+MoE+fc) draft; K=2 verify; measured acceptance. _Verify: token-identical + measured net tok/s; capture-dynamics reconciled._

## M-B.5 — UN-PARKED (2026-07-02): true NVFP4 experts to run `nvidia/Qwen3.6-35B-A3B-NVFP4`
### (v2 after 3-voice eng review — Codex + Gemini 3.1 Pro + Opus 4.8 xhigh, all AGREE-WITH-CHANGES)

**Why un-parked:** the Hadamard NO-GO applied to *producing* NVFP4 via PTQ. NVIDIA ships a
modelopt-**calibrated** MIXED_PRECISION checkpoint (near-lossless: MMLU-Pro 85.0 vs 85.6 bf16) —
we *consume* their weights, voiding the accuracy blocker. Footprint need now exists: ~22GB on
disk (3.06×), 262K-context KV headroom, NVIDIA's own DGX-Spark recipe (which also uses MTP K=3).
Honest expectation: **footprint/official-artifact lever, ~parity tok/s** (occupancy-bound).

**Checkpoint format (verified from config.json + hf_quant_config.json + index + card):** FP8
dense (130 GDN/attn projections: e4m3 `weight` [N,K] + `weight_scale` + `input_scale`);
W4A16-NVFP4 group-16 (256×40 experts with **3-way SPLIT per-expert gate/up/down tensors**,
shared expert, **lm_head**: packed `weight` u8 + e4m3 `weight_scale` + f32 `weight_scale_2` +
`input_scale`); `mtp.*` bf16 (ignored by quant); router/norms bf16; names under
`model.language_model.*`; 124,468 tensors; local at models/qwen3.6-35b-a3b-nvfp4/.

### Load-bearing design pins (from the review — implement exactly)
1. **REPACK TO OUTPUT-MAJOR NIBBLES AT LOAD (Opus F1, the kernel-saving pin).** Modelopt packs
   2×e2m1 per byte along the REDUCTION axis; the C2 fused kernels Line-vectorize over the
   OUTPUT axis (one `Line<u8>` = V consecutive output channels at fixed k; moe_grouped.rs:721,
   762) — native packing would destroy coalescing and force a from-scratch kernel. Instead the
   loader repacks: byte = (output-channel 2j, 2j+1) nibble pair at reduction index k; kernel
   decodes 2 output nibbles per byte, each scaled by ITS OWN (channel, k/16) block scale.
2. **KSPLIT == 16 == NVFP4 block size, asserted as a kernel invariant.** Split-K stride k =
   ky + 16·j means block index = j exactly: no split boundary ever falls mid-scale-block, no
   scale re-fetch. (SPLITK_KSPLIT_GU/DOWN are already 16 — moe_grouped.rs:1330.)
3. **Scales stay SEPARATE: per-16 e4m3 block scale × f32 global (per-expert `[E]` gscale for
   ExpertNvfp4; scalar for lm_head).** NO folding of the global into e4m3 (Emax=8 would
   under/overflow — Gemini F2). Block-scale stream fetched once per 16-K step; stage via
   warp-collective/smem if per-lane sub-word loads hurt (Gemini F5).
4. **input_scale:** FP8-dense arm DROPS it — we run f32 activations, strictly HIGHER precision
   than their calibrated fp8-activation target (accuracy-positive, documented). NVFP4/W4A16
   arm: whether vLLM's weight-only path uses input_scale (or folds anything at export) is a
   **B5.R hard question**; resolved empirically by the B5.0 named-reference golden regardless.
5. **Shared expert routes through the dense Nvfp4Linear path** (all-token dense GEMV), never
   the gather-GEMV (Gemini F3). lm_head via the plane-sum nvfp4_decode_gemv (M=1; index math
   verified safe at N=248320 — Opus F3) + vocab%16 assert.
6. **No-bf16 mechanism (named — Opus F2):** expert params are lazy-uninit at init; B5.0 SKIPS
   set_param3 for experts, builds ExpertNvfp4 straight from checkpoint bytes, sets the params
   to the tolerated [1,1,1] placeholders. NEW constructors required: `Nvfp4Linear::
   from_packed_parts` + `W8A16Linear::from_packed_parts` (today both only quantize FROM bf16);
   fp8 dense bytes need [N,K]→[K,N] transpose (view ok, q not contiguous-forced) and
   scalar→[N] scale expansion (16KB/layer — negligible; Gemini's in-register-scalar
   alternative REJECTED as complexity for no measurable bandwidth win).
7. **Prefill/TTFT (Gemini F6):** no bf16 experts exist to fall back to; per-expert NVFP4 GEMV
   is m≤8-capped. v1: chunk prefill M by 8 through the per-expert GEMV (correct, slow) +
   measure TTFT; if unacceptable, transient per-layer dequant-to-bf16 for prefill (memory
   spike bounded to one layer) as the documented fallback. TTFT gate added below.

### Tasks
- [x] **B5.R (research gate)** — docs/specs/M-B.5-prior-art.md: modelopt pack axis +
  nibble order, scale shapes/strides, W4A16 input_scale verdict w/ vLLM code quotes, fp8 dense
  scale granularity, SIMT e2m1 unpack recipes. _Blocks B5.0 golden + B5.3 only._
- [x] **B5.1 ExpertNvfp4 sidecar** (7160f00) — stacked
  output-major-repacked `[E,…]` u8 + e4m3 block scales + `[E]` f32 gscales, single-owner,
  ConstantRecord/autodiff-None, built last. Tensor-name allowlist counts (experts×layers×3 +
  shared×2… exact expected totals + no-leftover-quant-tensors check — Codex F11).
- [x] **B5.2 fused NVFP4 gather-GEMV ★** (6a8f42f, 26/26 GPU ladder) — extend C2 per
  pins 1-3: output-major nibble decode (2 channels/byte), per-(channel, k-block) e4m3 scale ×
  per-expert gscale, split-K KSPLIT=16 invariant assert, deterministic reshape+sum combine
  preserved, T≤16 gate + scalar fallback guards. Correctness ladder: (1) unpack golden ==
  host codec bit-exact incl. the repack roundtrip; (2) **gate/up fusion-offset test** (fused 2I
  stack vs two unfused GEMVs — the silent gate/up-swap catcher, Opus F5); (3) fused-vs-host
  parity cos>0.9999 @35B expert shapes; (4) end-to-end token identity vs dequant-to-bf16
  reference. Nsight bandwidth/occupancy/scale-overhead counters recorded (Codex F12).
- [x] **B5.0 loader** (43c3dca, bit-exact external golden) — `load_nvidia_nvfp4()`: name remap + 3-way-split→fused-2I-stack repack +
  output-major nibble repack + no-bf16 mechanism (pin 6) + fp8 dense adapter (pin 6) + mtp/
  router/norms bf16 paths. GOLDEN (blocks on B5.R): our host dequant over REAL repacked bytes
  == a **named external reference** (vLLM/modelopt python dequant run once over the real ckpt
  — never our-interpretation-squared; Codex F3); + **one-full-layer parity** (attn fp8 + NVFP4
  MoE + shared + norms) vs the reference (Codex F5). Peak host RAM + peak VRAM instrumented
  during load, not just steady-state (Codex F6).
- [x] **B5.0c staged fallback artifact** (NVFP4_DEQUANT_TO_FP8=1) — dequant-experts-to-
  fp8 mode reusing the PROVEN fp8 pipeline end-to-end: isolates loader correctness from kernel
  work + yields a runnable official-artifact build at ~40GB while B5.2 hardens (Codex F9).
- [x] **B5.3 wire + gates** (0938bd1 + 3381fbf) — NVFP4 sidecar arm in forward_static/forward_impl beside fp8;
  shared/lm_head per pin 5; capture re-gate (G2/G3-style static parity + captured identity +
  Qwen35VaSnapshot + zero-alloc + REPEATED capture/replay across positions — Codex F7).
  QUALITY GATES: teacher-forced 188-pos vs the bf16 original (bar ≈ FP8's top1 97.9%/KL
  0.0055) + router top-8 set-agreement + **lm_head-specific**: top-k/rank agreement,
  stop-token (151643/151645) rank checks, tail-token sampling, margin-bucketed (Codex F4) +
  free-run coherence + long-context smoke. FOOTPRINT GATE: device in-use ≤ ~26GB (vs 40.3
  fp8). PERF: captured tok/s vs 8.96 + **TTFT gate** (pin 7).
- [ ] **B5.4 MTP smoke** (FOLLOW-UP: mtp is bf16 and untouched, but mtp+nvfp4 composition unexercised — the mtp examples need a QUANT=nvfp4 arm) — mtp bf16 (2c parity + one MODE=perf run); note: 2g mtp-fp8
  quantization remains available unchanged (mtp is dense bf16 in this ckpt).

### RESULTS (2026-07-02, all gates GPU-run on the real checkpoint)
| Gate | Result |
|---|---|
| Footprint | **22.5GB device in-use** (vs 40.3 fp8 / 71 bf16); host HWM 22.7GB; load ~4 min |
| Golden | bit-exact vs the independent python reference on real bytes (incl. stop-token rows) |
| Greedy | **byte-identical to the bf16 original** (16 tok); first-token top5 rankings identical |
| Captured (G3 ladder) | PASS — **11.78 tok/s captured** / 7.35 eager-static; identity + VA + zero-alloc + 2nd-prompt green; **1.32× over the 8.96 fp8 captured baseline** |
| Teacher-forced (188 pos) | top1 89.9%, KL 0.0374, high-margin 97.8% — calibrated NVFP4 sits at the SAME logit-fidelity floor as naive (M-B.2b VALIDATED: E2M1-per-16 floor is real; near-lossless holds at TASK level per NVIDIA's evals + greedy behavior) |
| TTFT | not yet isolated (5-tok prompt fine; long-prompt host-dequant prefill cost = follow-up with B5.4) |
35B decode journey: 0.91 → 4.85 (fused fp8) → 8.96 (captured fp8) → **11.78 tok/s (captured NVFP4)**.

### Failure modes (silent-first)
| Risk | Guard |
|---|---|
| Nibble order / pack-axis wrong → plausible-garbage | B5.0 golden vs NAMED external reference on REAL bytes, before kernels wire |
| Scale layout transposed / weight_scale_2 direction | golden asserts the FOLDED PRODUCT vs reference (not factors separately) |
| gate/up silently swapped in the fused 2I stack | B5.2 fusion-offset test vs unfused two-GEMV reference |
| lm_head 248K-vocab tail/stop-token flips | dedicated lm_head gate battery (top-k, stop-token rank, tail sample) |
| Split-K boundary mid-scale-block | KSPLIT==16 invariant assert |
| Packed-byte vectorization misalignment | I%V-style guards + scalar fallback + parity ladder |
| W4A4 assumption creep (their group_1 acts=4-bit) | we run W4A16; input_scale per pin 4 |
| Prefill TTFT collapse (m≤8 chunking) | TTFT gate + documented transient-dequant fallback |
| Load-time OOM (repack double-buffering) | peak RAM/VRAM instrumentation in B5.0 |

### Parallel lanes
Lane A (now): B5.1 + B5.2 on synthetic weights (our codec, layout per pins). Lane B (now):
B5.R research. Lane C (after B5.R): B5.0 + golden + B5.0c. Converge: B5.3 → B5.4. Conflict:
B5.1/B5.2 and B5.0 all touch src/nvfp4*.rs + moe_grouped.rs — Lane C sequences after A merges.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | issues_found | Step 0 scope reduced; v1→v2 rewrite after target-model reality overturned the framing; 1 P0 + 6 P1 + 8 P2 folded |
| Eng Review (M-B.5) | `/plan-eng-review` | M-B.5 un-park section (NVFP4 ckpt) | 1 | issues_found→folded | 3-voice (Codex+Gemini+Opus xhigh) AWC×3; 2 HIGH (kernel axis redesign→output-major repack pin; no-bf16 mechanism+missing ctors) + 12 MED/LOW folded as pins 1-7 + gate battery; Gemini scalar-broadcast concern REJECTED w/ rationale |
| Outside Voice — Codex | GPT-5.x high (`codex exec`, web) | Independent 2nd opinion | 1 | issues_found | 15 findings; verified GDN/hybrid + MTP-is-a-trained-block + Atlas-uses-FP8-for-3.6 + candle-FA sm80-only; NVFP4 ~1.3×; n-gram-probe mandatory; prove on 30B first |
| Outside Voice — Gemini | 3.1 Pro high (agy-direct, web) | Independent 2nd opinion | 1 | issues_found | 6 findings; **surfaced the Gated-DeltaNet hybrid reality (verified)**; NVFP4 ALU-bound at low occupancy; PTQ-NVFP4 accuracy needs QAD; scaled-MMA 99-KiB trap |
| Outside Voice — Opus 4.8 | high (fresh-context agent) | Independent 2nd opinion | 1 | issues_found | F1–F11; NVFP4 multiplier contradicts §7 (~1.3× not 1.5–2×, target double-counts capture); `select_assign(Add)` rollback silent bug; capture/Fusion + host-pos gaps; calibration + token-identity gate |
| Candle-grounding pass | `/workspace/candle` v0.11.0 cloned + mapped (agent) | Ground the "reuse HF math" premise | 1 | done | §8a: candle CUDA is sm80/sm90 (blocked sm_121) → reuse algorithm/wiring only; flash-decode (`flash_fwd_splitkv` hdim256) + NVFP4-GEMV (`fast_mmvq`) + shared-MoE (`qwen2_moe`) + partial-rotary (`phi3`) reusable; **GDN de-risked via `rwkv_v7` delta-rule + `mamba2` conv + `lfm2` hybrid cache**; **NVFP4/mRoPE/MTP greenfield** (candle absent) |

- **CANDLE GROUNDING (this pass):** candle is now local and the port-mapping is verified against it (§8a). It
  confirms the reuse premise for the levers (algorithm + wiring), upgrades the GDN port from "greenfield" to
  "port-with-references" (`rwkv_v7`/`mamba2`/`lfm2`), and pins the only fully-greenfield pieces (NVFP4 format,
  mRoPE, MTP). It also re-confirms Codex's finding that **all candle CUDA is sm80/sm90 — unusable on sm_121** —
  so "candle-level efficiency" comes from porting candle's math to CubeCL SIMT, not from candle's kernels.
- **CODEX:** verified the target is `qwen3_5_moe` hybrid (40 layers, 30 GDN linear-attn + 10 full-attn, 256+shared
  experts, head_dim 256, mRoPE, trained MTP block); Atlas's Qwen3.6 recipe is **FP8 not NVFP4**; candle-flash-attn
  ships only sm80 kernels. Recommends: prove levers on 30B first, n-gram-probe-first mandatory, cut the 65–70 target.
- **CROSS-MODEL: unanimous, mutually corroborating, no contradiction.** All three independently overturned the v1
  framing: (a) the target is a **hybrid Gated-DeltaNet multimodal MoE the engine cannot run** — flash-decode helps
  only 10/40 layers, MTP rollback must restore SSM state (Gemini+Codex; HF-config-verified); (b) decode is
  **occupancy/latency-bound, not bandwidth-bound** → NVFP4 ~1.3× and the v1 45–55/65–70 target double-counts capture
  and is withdrawn (all three); (c) **NVFP4 PTQ accuracy is unsafe without calibration**, gate on token-identity not
  PPL, FP8 is the proven fallback (all three; Atlas-uses-FP8 corroborates); (d) **`select_assign(Add)` rollback is a
  silent corruption** + GDN-state rollback is unbuilt (Opus+Codex); (e) **flash-decode capture needs device-pos +
  raw-backend launch**, not host ScalarArg + Fusion bridge (Opus); (f) **MTP is a full trained block** and partly
  fights capture (Codex+Opus); n-gram-probe-first is mandatory. Divergence only on emphasis (Codex: cut the target;
  Opus: target-optimistic-but-build-sound) — folded as: withdraw the committed target, gate on Nsight/token-identity.
- **VERDICT:** ENG CLEARED with a v2 rewrite — **build-sound, sequencing sound (two-lane, unblocked-first), targets
  corrected to honest.** Plan is ready to execute as sequenced (P0 probes → Lane 1 GDN port + Lane 2 levers on 30B →
  Phase 2 converge + MTP last). The single biggest underweighted risk now surfaced and folded: **the target is a new
  hybrid architecture, not a faster 30B — Lane 1 (GDN port) is the bulk, and decode is occupancy-bound so quant is a
  ~1.3× lever, not 1.5–2×.**

**UNRESOLVED DECISIONS:**
- ~~**NVFP4 vs FP8 final call is conditional on P0.5**~~ **RESOLVED 2026-07-01 → DEFAULT TO FP8** (3-voice
  unanimous: Codex + Gemini + Opus). D6 gate ran on the real 35B via fake-quant (docs/specs/M-B-nvfp4-gate-plan.md):
  NVFP4-dense (cos ~0.996) collapses into a repetition loop; FP8-dense (cos >0.9996) stays ~as-good-as-bf16.
  Two findings: (1) the D6 free-run "token-identical to greedy" gate is INVALID for PTQ (chaotic — even FP8
  fails it; it conflated the SPEC-DECODE/MTP invariant §7 with PTQ quality §6C). Correct gate = teacher-forced
  KL + top-1 + router top-8 set-agreement + ΔPPL + a task slice, on a corpus, on the FULL config (experts
  quantized). (2) At occupancy-bound decode NVFP4-over-FP8 is realistically <~1.1× end-to-end; NVFP4's only
  edge is footprint (~20 vs ~35 GB, not binding on GB10 128GB single-stream). ⇒ **Lane 2C pivots NVFP4-first →
  FP8-first** (`w8a16` GEMM + `QuantLinear::Fp8` already exist; remaining = raw-backend port for capture +
  wire at load). NVFP4 stays a research upside, pursued ONLY if Hadamard-calibrated full-config NVFP4 matches
  FP8 on the corpus gate AND the footprint is actually required.
- **M-B.2b HADAMARD EXPERIMENT RAN (2026-07-02) → NO-GO, unanimous (Codex+Opus CONCUR; Gemini's
  like-for-like MSE demand-probe satisfied).** Randomized-Hadamard rotation (D_s·H_g, g=128
  aligned to the per-16 blocks, (layer, input-site)-shared seeds, statistical clip; 3 seeds +
  amax/clip + MSE ablations; codec + golden tests in src/nvfp4.rs, PREC=hadamard in the D6
  harness) was verified APPLIED AND EFFECTIVE AT GAUSSIANIZING (the 3.5·rms clip stopped binding
  post-rotation) yet did NOT improve accuracy: dense-only Hadamard+clip top1 90.4–92.0%/KL
  0.038–0.045, Hadamard+MSE top1 92.0%/KL 0.051 & cos 0.9961 == plain-MSE cos, vs plain-MSE
  92.6%/0.0327 and FP8 95.7%/~0.006. ⇒ the E2M1 per-16-block RESOLUTION FLOOR (~0.9955–0.9961
  cos on this checkpoint's already-Gaussian-ish blocks) dominates, NOT within-block outliers —
  the "Hadamard is the real lever" hypothesis is FALSIFIED for this model. NVFP4 stays PARKED
  (accuracy path exhausted at 4-bit weight-only; also the deployment ceiling <~1.1× over FP8 and
  non-binding footprint stand, per this section). Codec kept as research infrastructure.
- **MTP-under-capture composability** — whether to capture fixed sub-steps + host accept decision, or pad-to-K
  (erodes the spec win). Resolve by measurement during MTP.1 (n-gram probe).
- **M-B.5 auto-decided items (2026-07-02, autonomous /goal mode — user may veto during implementation):** (1) fp8 per-tensor scale expanded to [N] on load (in-register scalar path rejected: 16KB/layer is not a real bandwidth cost); (2) prefill strategy = m≤8-chunked per-expert GEMV first, transient per-layer dequant only if the TTFT gate fails.
