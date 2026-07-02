# vLLM-Parity Custom Kernels — design specs (prior-art-verified + dual-reviewed)

Companion to [VLLM_PARITY_PLAN.md](VLLM_PARITY_PLAN.md). The custom CubeCL kernels that gate the remaining
speed wins, each: designed against the real `/workspace/cubecl` API, adversarially hardened for sm_121
silent corruption, **verified against canonical references** (FlashAttention-2/Flash-Decode, vLLM
`apply_w8a16` + `moe_align_block_size`, Marlin, gpt-fast), and **reviewed by Codex gpt-5.5 high + Gemini
3.1 Pro high** (report at the bottom). All produced in a GPU-less design pass — every kernel is honestly
**NEEDS-SPIKE → MULTI-SESSION**.

## 0. Two cross-cutting laws (both reviewers, independently)

**Law 1 — the real critical path is the Burn *Fusion custom-op bridge*, not any single kernel.** The
project's default backend is `Fusion<CubeBackend<CudaRuntime>>`. A `#[cube]` kernel cannot be reached from
the model without either a Fusion custom-op registration or rebuilding the stack on the raw `CubeBackend`.
This boundary **blocks fp8 AND MoE**, and can break graph capture (hidden alloc/autotune/dynamic dispatch),
and may force bf16 materialization that *erases the bandwidth win*. **So the first spike is not a kernel —
it is: get one minimal fused custom op running inside the real rollout path with stable shapes.** Until
that exists, no kernel is on the critical path to a faster correct rollout. (Codex "highest risk"; Gemini
"build-order deadlock".)

**Law 2 — the oracle must be a DIFFERENT backend AND kernel-specific (cosine alone is a trap).** Same-
sm_121 oracles can be corrupted identically to the kernel (the silent batched-matmul bug), so oracles run
on NdArray/CPU — but CPU-f32 ≠ hardware-bf16-MMA (non-associative truncation), so a blanket
`cosine>1−1e-6` *falsely fails* valid bf16 kernels over long context. Use **kernel-specific** oracles +
metrics beyond cosine (max abs/rel error, top-k overlap, KL on the sampled distribution, exact routing-
metadata checks, adversarial imbalance):
- **FA:** a high-precision (f64/f32) stable-softmax reference; tolerance sized for bf16 input truncation.
- **fp8:** exact **OCP E4M3 encode/decode golden test vectors** + an f32 matmul reference (don't trust the
  hand-rolled `log2/floor` estimator).
- **MoE:** a deterministic CPU reference that checks **routing ids / `sorted_token_ids` / per-expert counts
  *before* logits**, then the weighted top-k combine.
- **Graph:** eager-CUDA + CPU spot checks + **mutation tests** (bit-exact-eager proves capture *stability*,
  not correctness, if both share a corrupt kernel).

## 0b. The Fusion custom-op bridge — RESOLVED (GO), with production rules

Law 1's blocker is **cleared**. `examples/fusion_bridge_spike.rs` proves a hand-written `#[cube(launch)]`
kernel runs on the default `Cuda = Fusion<CubeBackend<CudaRuntime>>` backend, on-device, bit-exact
(max_abs_diff=0 vs the Burn-ops reference, GB10). The pattern: register an `OperationIr::Custom` + an
`Operation::execute()` that does `HandleContainer::get_float_tensor::<CubeBackend>()` (FusionTensor →
CubeTensor) → launch the kernel → `register_float_tensor()` back into the stream. 5 burn-internal crates
(`cubecl`, `burn-cubecl`, `burn-cubecl-fusion`, `burn-fusion`, `burn-ir`) are pinned to burn's exact rev so
types unify. Reviewed by **Codex gpt-5.5 high + Opus 4.8 xhigh + Gemini 3.1 Pro high** — all GO, sound for
out-of-place ops, no P0 in the spike. Opus verified against the *actually-pinned* burn `5923b1e`.

**Production rules every real kernel MUST follow (the "manual tri-contract" — the unanimous P0/P1 risk;
the fusion engine cross-validates NONE of these):**
1. **Declare every input in BOTH** `OperationStreams::with_inputs([...])` AND `CustomOpIr::new(inputs=[...])`.
   Omit one → the JIT GCs/reorders it → silent stale-handle read / illegal memory access / `"Should have
   handle"` panic on the server thread.
2. **The output `TensorIr{shape,dtype}` must EXACTLY match the buffer `execute()` allocates.** They are two
   disconnected computations (the lazy `FusionTensor` shape vs `execute`'s real alloc); a drift = silent
   numeric/shape corruption downstream. (Critical for `[M,K]·[K,N]→[M,N]`.)
3. **Thread dtype → element type + byte size everywhere.** The spike hardcodes `size_of::<f32>()` +
   `launch::<f32>`; a bf16 tensor flows in tagged BF16 (2 B) but an `n*4` buffer / f32 kernel → OOB read.
4. **Packed int weights via `get_int_tensor`, not `get_float_tensor`** (panics on Int handles). fp8/e4m3 has
   no float-dtype carrier → smuggle as `i8`/`u8` bytes (a quantized/int tensor) + a separate scale tensor.
5. **Never `into_contiguous` a pre-packed/swizzled weight** — it triggers a layout-fixing copy that destroys
   the packing. Pass as-is; handle strides in the kernel.
6. **Scalars live in the `Operation` struct** (`CustomOpIr` is tensors-only), captured by value.
7. **`execute()` touches ONLY the raw CubeCL client + the passed `HandleContainer`** — never the fusion
   `client` (re-entrancy under the server lock → `BorrowMutError`/deadlock).
8. **Rev lockstep:** the 5 deps must move with burn's rev; skew fails LOUD (compile error / `TypeId`
   downcast panic), except a rare id-allocation-model change that the type system won't catch.

**NEXT (infra, before any kernel):** a typed `custom_op<const N_IN, const N_OUT>(...)` safe-wrapper that
*enforces* rules 1-2 (cross-validates the declared inputs/outputs against the closure's allocation +
launch), so a kernel cannot silently violate the tri-contract. Built + 3-voice-reviewed, then the kernels.

## 1. Attention — prefill (FA-2) AND decode (Flash-Decode); the rollout needs BOTH

**STATUS: correctness VALIDATED, not production-ready** (`src/flash_attn.rs`, commit on `grpo-phase-a`).
A custom FA-2 online-softmax kernel matches an NdArray CPU oracle to ~1e-6 on decode+prefill/short+long/
GQA/head_dim 64/128, via the safe-wrapper. 3-voice gate: correctness sound; gaps before it can replace
SDPA — (1) **causal-only, no padding mask** → wrong on the left-padded ragged GRPO path (Opus P0);
(2) f32-only (bf16 latent/guarded — model is f32-typed today); (3) single-thread → ~10-100× slower than
cuBLAS/CMMA SDPA (a regression); (4) `into_contiguous` copies the KV cache. Perf (Flash-Decode split-K +
CMMA) + mask + bf16 are the follow-ons. The design below stands.



**Readiness:** NEEDS-SPIKE → MULTI-SESSION. **Correction (both reviewers):** the single "one thread per
query row, FA-2" design is wrong for the **decode-dominated** rollout. At decode `q_len=1`, so query-tiling
launches only `batch*heads` blocks and serializes the KV scan — the GB10 sits idle, and one-thread-per-row
gives up tensor cores entirely.
- **Prefill** (prompt, `Q=S`): FlashAttention-2 (split over Q). Port the **FA-2 CUDA / CUTLASS-CuTe** MMA
  tiling (not the Triton tutorial literally — it's educational, not a perf target).
- **Decode** (`Q=1`, growing KV): **Flash-Decode / split-K over the KV cache** — partition the KV across
  blocks, each computes a partial `(m, l, acc)`, then a reduction combines. Port `Dao-AILab/flash-attention
  flash_decode` or **FlashInfer** decode kernels. This is the rollout priority.

**Canonical math (FA-2 recurrence, f32 state):** `m_new=max(m_old, max_k s_k)`; `α=exp(m_old−m_new)`;
`p_k=exp(s_k−m_new)`; `l=α·l+Σp_k`; **`acc=α·acc+Σp_k·V_k`**; final once `O=acc/l`. `scale=1/sqrt(head_dim)`
applied before the max. Flash-Decode = the same recurrence per KV-split + a cross-split log-sum-exp merge.

**Must-honor corrections:** the **`acc*=α` rescale** (most-dropped term); mask before the max; **global**
`q_idx` (`q_offset=total_seq−seq_len`); the **all-masked-*tile* NaN guard** — not just masked rows: a
causal query can hit a later K-tile where every `kj>q_idx`, giving `m_old==m_new==−inf` → `exp(NaN)`; skip
tiles with zero valid keys or special-case `m==−inf`; **64-bit** global offsets.

## 2. fp8 weight-storage — a FUSED W8A16 kernel (not dequant→linear3)

**STATUS: a CORRECT fused GEMM, but NOT deployable in the GRPO rollout** (`src/w8a16.rs`, committed; 3-voice
gate). It reads e4m3 *bytes* + dequants in-register (validated vs an NdArray oracle, cosine 1.0; e4m3 codec
proven OCP-faithful in cubecl source). Opus 4.8 (tracing the trainer) found two deployment blockers that
**change the plan**: (1) **breaks GRPO logprob parity (P0)** — forward-only (no backward) → can only live in
the no-grad rollout → rollout fp8 vs f32 recompute → per-layer quant error compounds into a logprob shift
that silently biases the gradient (gate on end-to-end **logprob parity**, not GEMM cosine); (2) **wrong
regime (P1)** — the batched rollout decodes `[n,1]` with `n = prompts×group_size > 1`, where the flat kernel
re-reads the weight M× and loses to bf16 `linear3`; the half-the-bytes win is a **batch-1 serving** win the
rollout never reaches. It's W8A**32** (f32 activations); split-K/CMMA + the e4m3 loader/`W8A16Linear` are
unbuilt. **⇒ Plan correction:** fp8-storage is a batch-1 *serving* lever, not the GRPO-rollout decode lever
the plan assumed; the real rollout levers are the static KV cache (done), Flash-Decode, and batch-shrink.
The design below stands for serving / a future both-sides-fp8-with-STE training path.



**Readiness:** NEEDS-SPIKE → MULTI-SESSION; gated by Law-1 (Fusion bridge). **Correction (both reviewers,
Critical):** "dequant to bf16 then feed the existing `linear3`" **does not deliver the win** — if CubeCL
materializes the bf16 weight in HBM, traffic becomes fp8-read + bf16-write + bf16-read, *worse* than bf16.
The dequant **must be fused into the GEMM's shared-memory/register load path** (read fp8 from HBM → dequant
in-register → tensor-core MMA, f32 accum). This is a **custom fused W8A16 GEMM** (Marlin/Machete), not a
reuse of `linear3`.

**Port:** Marlin (`IST-DASLab/marlin`) / Machete fused W8A16; the dequant math from vLLM
`per_tensor_dequantize`. **Pin ONE scale convention + golden tests** (Codex): store `scale = amax/448`
(per output channel), dequant `w = q * scale`; do NOT mix with vLLM's `inv_scale=448/amax` (dequant
`q/inv_scale`) — one canonical field name, OCP-E4M3 golden encode/decode vectors. E4M3 max = 448; use
CubeCL's **native E4M3** cast, not the estimator.

## 3. MoE grouped-GEMM — dropless layout + the full top-k combine

**STATUS: CORRECT + DROPLESS + PARITY-SAFE, but currently SLOWER than the batched rollout it targets**
(`src/moe_grouped.rs`, committed; 3-voice gate). The dropless `moe_align_block_size` layout is proven sound
(Opus, arithmetic walk) and the numerics equal the dense oracle (cosine 1.0) — so it's parity-safe for the
rollout (forward-only → no-grad rollout, recompute uses the oracle's backward), UNLIKE fp8 §2. **But it hits
the same wrong-regime trap as fp8:** this scalar, zero-weight-reuse kernel reads fewer weight bytes than
dense only at `T < E/k = 16`; the batched rollout decodes `T = P·G ≫ 16`, where it re-reads each expert's
weights `k·T`× (~8× more traffic than `forward_routed_ondevice`, no tensor cores) → ~10-50× SLOWER. Plus
~7 GB/layer/step of per-call `stacked_experts` re-stacking, an `I`-sized per-thread local spill, and an
atomic-sink padding combine. To win it needs a **weight-stationary CMMA rewrite + cached pre-stacked
weights** (unbuilt), and there is **no MoE GRPO trainer/rollout path** yet (dense-only). **⇒ The right
batched-MoE-decode path is weight-reuse (dense `forward_oracle` is dropless + parity-safe + reuses weights),
not this no-reuse compact GEMM — until the weight-stationary tiling lands.**

**Readiness:** NEEDS-SPIKE → MULTI-SESSION; gated by Law-1. **Approach + corrections:**
- **Dropless** compact layout (not `forward_routed_ondevice`'s fixed-stride `expert*C`, which drops tokens
  → corrupts GRPO parity): `count_e=oh.sum(0)` → `round_up(count_e, BLOCK_M)` → `base_e=ExclusiveCumsum(...)`
  → `sorted_token_ids` + per-block `expert_ids` (`−1` sentinel for pad) + `num_tokens_post_padded`; buffer
  `T*k + E*(BLOCK_M−1)`. Port vLLM `moe_align_block_size_kernel` + `fused_moe_kernel`.
- **The top-k combine is underspecified and load-bearing (Codex High):** sorting alone is not enough —
  carry `(source_token, expert, topk_slot)` identity, apply the **router weight** per (token,expert), and
  scatter K expert outputs back to the same token with a **deterministic accumulation order**.
- **64-bit on ALL intermediate index math (Codex), not just the final offset:** padded counts, prefix
  sums, `num_tokens_post_padded`, block offsets, expert weight strides — `E·C·H=128·16384·2048=4.3e9>2^31`;
  CubeCL defaults to u32. vLLM/DeepGEMM/CUTLASS cast `.to(int64)` "to prevent overflow in stride*offset".
- **Load imbalance (Gemini High):** `CubeCount::Dynamic` block-per-segment alone → tail latency when one
  expert is hot; a stream-K / persistent scheduler is needed for balanced occupancy.

## 4. CUDA-graph capture — a runtime-architecture spike, not the 4th kernel

**STATUS: IMPLEMENTED — the capability is built + GB10-validated as a MECHANISM** (branch
`cuda-graph-support` on cubek/cubecl + the repo; full design + phase log in `docs/cudagraph/DESIGN.md`).
The four blockers this section listed are each CLEARED, every phase agent-built → 3-voice-gated (Codex
gpt-5.5 high + Gemini 3.1 Pro high + Opus 4.8 high, source-verified) → committed:
- **No cudaGraph API** → C1/P0: capture/replay FFI **below Fusion** on raw `CubeBackend` (cubecl `23a0d5c`).
- **Fusion lazy/dynamic queue** → captured below Fusion + **metadata interning** stabilizes the per-op
  trace (cubecl `eac3454`); residual: a mid-capture autotune divergence is a *safe loud abort* (bump
  warmup), not a hard freeze-autotune hook.
- **`Tensor::random` frozen seeds** → C3/P3: opt-in device-seed RNG, captured replay DECORRELATES (cubek
  `b132af5`).
- **No graph-aware allocator** → C2/P1: graph-private capture arena (peak-live recycling, no leak; cubecl
  `e38e3fa`) + the shared lp-bucket pool (P4, `2cafb76`, 2.49× saving).
- **Decode attention not fixed-shape** → P2: device-`pos` static cache + masked full-`T_max` attention +
  device length counter (`fd017bc`); the host-scalar `slice_assign` is replaced by a device-`pos` scatter.

Pass criterion MET: P-final captures one decode step + replays per token — **greedy bit-identical to eager**,
temperature 192/192 bit-identical under a fixed seed + decorrelating, with the **VA-stability guard = the
buffer-move mutation test** (`c98f630`). lp-bucket + left-pad covers variable prompt length (P4).

**The section's PAYOFF call HELD (measured-confirmed):** decode is bandwidth-bound (the tied-head logits
GEMM streams ~0.6 GB/step), graphs only cut launch latency → captured replay is **~1.05× small → ~1.0× at
scale**, exactly the predicted band. **⇒ The real next decode lever is still the bandwidth-bound logits GEMM
(chunked/fused logits, stream-tile Gumbel-max), not CUDA-graphs** — the durable GRPO-rollout wins are the
standalone pieces (device sampler 3.6-4.8×, static KV, batch-shrink). So this is built-and-correct but a
~1.0× lever for THIS workload; its durable value is the reusable, upstreamable framework capability.

**REMAINING (production gates, `DESIGN.md §0c`):** validated on **f32/random-init** only → bf16 + a real
trained model; the **GRPO `old_logprob` → step-0 PPO `mean_ratio ≈ 1`** gate through a captured temperature
rollout (never run); shared pool is **serial-replay only** (asserted at capture, not enforced at replay);
the freeze-autotune hook. Metadata interning is **fixed-trace/fixed-shape** (not a general dynamic-shape
capability).

_(Historical: this section was originally written NO-GO/BLOCKED before the framework was built. The build
cleared the feasibility blockers; the payoff assessment was correct and is unchanged.)_

**Readiness:** NEEDS-SPIKE (GO/NO-GO) → MULTI-SESSION. **Reframe (both reviewers):** this is a runtime-
architecture problem, not a kernel. Beyond autotune-freeze, **Burn's async memory allocator + JIT +
data-dependent dispatch** can each invalidate or silently corrupt a captured graph (pool resize,
out-of-order free, a fallback branch, a Dynamic grid). Treat it as a separate spike.
- SPIKE-0 pass criterion = `EndCapture` non-null → **replay == eager** + a **buffer-move mutation test**
  (not `status==ACTIVE`, which is necessary-not-sufficient; and not bit-exact-eager alone, which proves
  *stability* not correctness).
- Port gpt-fast's **device `input_pos`** KV write (the Phase-2 host-scalar `slice_assign` is provably non-
  capturable) + vLLM **bucketed batch sizes + pad-to-bucket**; warmup ≥3 on a side stream; non-default
  stream; graph-private pool; freeze autotune; steady-state forward allocates nothing.

## 5. Revised build order (gated by Law 1, then a measurement)

0. **Fusion custom-op bridge spike** — get ONE minimal fused `#[cube]` op callable from the real rollout
   path with stable shapes (or stand up the raw-`CubeBackend` path). **Nothing below is on the critical
   path until this works.** (Both reviewers; supersedes the old "fp8 first".)
1. **Measure** the real rollout bottleneck (prefill vs decode-attention vs MoE vs logits vs host-sync) —
   the `VLLM_PARITY_PLAN.md` MEASURE gate. Decode/MoE often dominate, not prefill.
2. Then, by the measurement: **Flash-Decode** (decode attention, standalone — bypasses the Fusion graph),
   **fused W8A16** (if decode is weight-bandwidth-bound), **MoE grouped-GEMM** (on MoE models). FA-2 prefill
   only if prefill dominates.
3. **CUDA graphs** last (the runtime spike), after device-`input_pos` + a fixed kernel set exist.

## GSTACK REVIEW REPORT

| Review | Trigger | Runs | Status | Findings |
|--------|---------|------|--------|----------|
| Prior-art verification | agy-direct (GitHub/arXiv/Google) | 1 | done | 4 kernels grounded; fp8/MoE/cuda-graph corrected vs references |
| Codex Review | gpt-5.5 high (inlined) | 1 | issues_found | 12 findings, 3 Critical — all folded |
| Gemini Review | 3.1 Pro high (agy-direct) | 1 | issues_found | 6 findings, 4 Critical — all folded |
| §4 CUDA-graph feasibility-vs-built (this session) | plan-eng-review; absorbs the 7-phase build gates | 7 phases × 3 voices | resolved | §4 IMPLEMENTED — all 4 blockers cleared (P0-P4) + pass criteria met + committed; §4's ~1.0× payoff call confirmed-by-measurement; stale NO-GO STATUS corrected |

- **CROSS-MODEL:** strong consensus, **no contradiction** — both independently flagged (a) fp8 dequant→
  GEMM round-trip kills the bandwidth win (must fuse), (b) the NdArray-f32 oracle is insufficient (need
  kernel-specific), (c) the Fusion custom-op bridge is the true blocker / build-order is wrong, (d) the
  one-thread-per-row attention is a perf dead-end. Divergence only on *which* comes first after the bridge:
  Gemini → Flash-Decode; Codex → measure then choose. Folded as: bridge spike → measure → choose.
- **VERDICT (kernel specs):** prior-art-verified + dual-reviewed and **ready to implement as gated
  spikes**, in the revised order. Honest readiness unchanged: all NEEDS-SPIKE → MULTI-SESSION, each behind
  its kernel-specific cross-backend oracle.
- **VERDICT (§4 CUDA-graph, this session):** **Q: "can we implement CUDA graph as defined in §4?" → A: YES,
  and it is implemented + GB10-validated as a mechanism.** All 4 §4 blockers cleared by the framework build
  (P0-P4 on `cuda-graph-support`, each 3-voice-gated); the pass criterion (replay==eager + buffer-move
  mutation test) is met; §4's own ~1.0× bandwidth-bound payoff prediction is measured-confirmed. The §4
  STATUS line was stale (NO-GO/blocked, pre-build) and is now corrected to IMPLEMENTED. The remaining work
  is **value-vs-effort, not feasibility**: the GRPO-decode replay is ~1.0×, so the durable rollout levers
  stay the standalone pieces (device sampler 3.6-4.8×, static KV, batch-shrink); production gates
  (bf16/trained model, the captured-temperature step-0 PPO `mean_ratio≈1`, serial-replay) remain per
  `docs/cudagraph/DESIGN.md §0c`.

**UNRESOLVED DECISIONS:**
- Production-gate the captured decode (bf16 + trained model + the PPO step-0 `mean_ratio≈1` gate at temp>0)
  to make it trainer-usable, OR leave it as a validated mechanism + reusable framework capability and keep
  the GRPO-rollout speed on the standalone levers? The ~1.0× payoff for this bandwidth-bound workload argues
  for the latter; the framework capability (upstreamable to tracel-ai/cubecl) argues either way. User's call.
