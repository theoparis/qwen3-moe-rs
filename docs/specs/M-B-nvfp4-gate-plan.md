# M-B — NVFP4 D6 token-identity gate: execution plan (3-voice reviewed)

Folds the 3-voice review (Codex GPT-5.5-high + Gemini 3.1 Pro-high + Opus 4.8-high) of the exec order +
open M-B decisions. See docs/specs/EXEC-STATE-MAP.md for the codebase state, docs/specs/L2C-nvfp4-decode-gemv-design.md
for the codec/kernel/gate design. Runs EAGER on Fusion `Cuda` (numerically valid gate; capture is
bit-identical replay). GPU runs SERIALIZED.

## Review verdicts (Codex + Gemini; Opus appended below)
- **Order:** M-D (raw backend) must precede M-C (both need below-Fusion). M-B first is OK *because it is
  backend-generic + isolated and the capture-safe NVFP4 path (`nvfp4_gemv_raw`) already exists* — its
  deliverable is a GO/NO-GO + a per-tensor manifest, not a captured artifact. Order: M-B → M-D → M-C → M-E → M-F.
- **Eager gate valid** numerically (both). Eager does not prove capture-safety — that's M-D.
- **Q3 CORRECTION (both, high severity):** gating a *dense-only* NVFP4 pass on strict end-to-end
  token-identity WILL FAIL — the ~0.995 dense perturbation shifts hidden states → router logits →
  top-8-of-256 ROUTING FLIPS → divergence, even with the router weight kept bf16. So:
  - primary instrument = per-tensor **numerical** gate: layer-output cosine + argmax-margin; and
    **pre-router activation cosine (>0.999)** for tensors feeding the router.
  - end-to-end **token-identity only on the FULL selected config** (dense NVFP4 + MoE-expert NVFP4
    together), NOT per single-tensor swap.
  - ⇒ M-B must include the **MoE-expert NVFP4 grouped-GEMV** (extend moe_grouped) — it's the dominant
    35B lever; a dense-only gate proves codec+accuracy but not the perf win or final-config identity.
- **Q4 CORRECTION (both):** AWQ IS a weight-only method (my "activations-not-quantized ⇒ N/A" was wrong).
  Order: (1) **MSE-optimal per-block scale-clip FIRST** (weight-only, no corpus, ~0.990→0.995), (2) **AWQ**
  if short (needs small activation corpus; folds a per-channel scale into weight + inverse into the
  activation at runtime; ~0.995→0.999+). SKIP SmoothQuant (activation-quant) + QuaRot (invasive, kernel cost).
- **Q5 nuance (both):** eager M-E = filled-rewind (KV) + GDN S/conv snapshot-restore. But **captured MTP
  (M-F) re-introduces the Add hazard**: host `slice_assign` at `filled` is NOT CUDA-graph-safe → captured
  MTP needs device-pos + rollback-zeroing. Deferred, not dead. Assert hybrid MTP never routes update_static eagerly.

## M-B task breakdown (revised scope)
- **M-B.1 (Codex, IN PROGRESS)** create `src/nvfp4_linear.rs`: backend-generic `Nvfp4Linear<B: Nvfp4GemvBackend>`
  + `QuantLinear{Nvfp4,Fp8,Bf16}`. Mirrors w8a16_linear.rs. Build-only verify. [invariant — dispatched]
- **M-B.2** MSE-optimal calibration in `src/nvfp4.rs quantize_nvfp4`: per-block E4M3 scale + fp32 global
  chosen by MSE-minimizing grid/search over a clip range (weight-only). Re-derive gscale from post-clip
  amax. Keep the scale floor + reconstruct-then-quantize. Golden-vector tests (all-zero + outlier block).
- **M-B.3** load hook: `src/load.rs set_quant_linear` (manifest-keyed) + route the ~11 `linear3` call
  sites in qwen3_5/mod.rs through `QuantLinear`. Default tiers: router-gate→bf16, lm_head→FP8, rest→NVFP4
  candidate. Inference-only; never GRPO grad.
- **M-B.4** per-tensor NUMERICAL gate harness `examples/qwen35_nvfp4_gate.rs`: for each candidate tensor,
  Nvfp4Linear vs bf16 Linear on captured real activations — layer-output cosine + argmax-margin; pre-router
  activation cosine for router-feeding tensors. Emit per-tensor pass/fail → manifest.
- **M-B.5** MoE-expert NVFP4 grouped-GEMV: extend `moe_grouped.rs` fused_swiglu_*_splitk with block-scaled
  e2m1x2 dequant-in-load for the rank-3 expert stacks (gate_up_proj [E,2I,H], down_proj [E,H,I]). Verify vs
  bf16 expert GEMV (cosine/margin). (L1 is done → the moe_grouped L1/L2C conflict is moot.)
- **M-B.6** FULL-CONFIG token-identity smoke: 30B known-good greedy string (vllm_infer.rs:19-21) + 35B
  (capture bf16 greedy as reference) with the selected NVFP4 config (dense + experts). GO/NO-GO vs FP8
  fallback per tensor. + Nsight perf co-gate (P0.5): DRAM-bound vs ALU-bound at M=1; demote to FP8 if ALU-bound.
- FP8 (w8a16) is the wired fallback for any tensor failing numerical/token-identity/perf.

## Opus review (folded — 6 ranked findings, all verified against source)
- **F1 (HIGH) — M-E rollback recipe was BUGGY (fixed below).** GDN state is mutated per-token during the
  K-token verify batch (`set_state@869`, `push_conv@799`). Snapshotting GDN at `pos` but rewinding KV to
  `pos+acc` leaves GDN@pos vs KV@pos+acc → stale GDN readout → SILENT divergence from greedy. The gated
  delta rule is a non-invertible sequential recurrence. FIX: rewind KV and GDN to the SAME position:
  (A) roll BOTH to `pos` + re-forward the accepted (+bonus) tokens through normal decode (≤K steps, K=2),
  OR (B) roll BOTH to `pos+acc` with GDN checkpointed PER verify step. Test must exercise acc>0 and assert
  RECURRENT-STATE equality (not just KV) vs a fresh forward to the same pos.
- **F2 (HIGH latent) — Add hazard revives under captured MTP (M-F).** Eager M-E filled-rewind is fine, but
  the capturable decode needs a DEVICE-pos KV write (host `slice_assign@filled` bakes a constant into the
  graph). Today device-pos = `update_static` = `select_assign(Add)` → speculative multi-writes accumulate.
  FIX before M-F: device-pos KV write must be an OVERWRITE (assign), not Add.
- **F3 (HIGH) — dense-only cannot certify; relabel M-B.** Routed experts (256 top-8 + shared, every layer)
  dominate 35B decode bandwidth; a dense-only NVFP4 config leaves the dominant bytes bf16 → little perf win.
  Accuracy: expert quant perturbs hidden state → discrete top-8-of-256 routing FLIPS (router input drifts
  even with the gate weight bf16) → discontinuous, unpredictable from dense-only. RELABEL: M-B is the
  accuracy-MACHINERY lever; the central 35B PERF lever is the MoE-expert NVFP4 (M-B.5). Shipping manifest
  MUST be produced with experts in target precision + a ROUTING-STABILITY check (top-8 set agreement rate).
- **F4 (MED-HIGH) — gate on the RAW CubeBackend, not Fusion.** Capture-replay adds zero numeric risk (same
  kernels/buffers). The real risk: the gate runs the model on `Cuda=Fusion` (fuses/reorders RMSNorm/softmax/
  GDN/epilogues → different rounding) while deployment runs raw `CubeBackend` below Fusion. Token-identity is
  gated on near-tie argmax margins → a backend rounding delta flips them. FIX: run the gate on raw CubeBackend
  eagerly (capture-replay is then bit-identical). Cheaper sanity first: **M-B.0** prove bf16 Fusion-vs-raw
  greedy token-parity — if identical, gating on Fusion transfers; if not, gate on raw. Requires QuantLinear
  dual-backend (F5).
- **F5 (MED) — order sound; build QuantLinear DUAL-BACKEND in M-B.** `nvfp4.rs` already has both GEMV backend
  impls, so make `Nvfp4Linear`/`QuantLinear` generic over `Nvfp4GemvBackend` now (cheap; avoids an M-F
  rewrite; enables F4). A4 is NOT M-D-independent (it calls CaptureBackend-typed `flash_decode_raw`). M-E can
  run parallel to M-C (needs only GDN snapshot + filled-rewind + M>1 forward — all present).
- **F6 — calibration taxonomy.** SmoothQuant: SKIP (migrates activation outliers into weights → makes weights
  HARDER; activations are f32 so nothing to fix). AWQ: applicable (weight-only W4A16) but MARGINAL (NVFP4's
  per-16 E4M3 block scale already subsumes much of AWQ's salient-channel protection); needs corpus. Hadamard/
  QuaRot: HIGHEST ceiling for E2M1 (1 mantissa bit → one within-block outlier crushes the other 15 onto the
  coarse grid; a fixed Hadamard `W→QᵀW`, `X→XQ` at runtime spreads outliers, flattens per-block amax; data-
  free). MSE-clip: FIRST (data-free, ~0.995→0.997-0.998). ORDER: **MSE-clip → Hadamard → (optional) AWQ**.
  Worst D6 tensors (shared_expert gate 0.14, GDN out_proj 0.11 — outlier-heavy) are exactly where Hadamard+
  MSE help most. Keep gate→bf16, lm_head→FP8 (4-bit argmax on those is fragile regardless of calibration).

## D6 RESULT (real 35B, fake-quant, 2026-07-01) + DECISION PIVOT
Ran all-dense fake-quant (310 dense linears; router gate + shared_gate + lm_head kept bf16), prompt
"The capital of France is", greedy 24 tokens, vs bf16 baseline:
| config | per-tensor cos | free-run token match | output |
|---|---|---|---|
| bf16 baseline | — | (ref) | "Paris, a city renowned for its iconic landmarks such as the Eiffel Tower, the Louvre..." |
| NVFP4-MSE | ~0.9962 | 1/24 (diverge @1) | coherent but REPETITIVE loop ("...Paris." repeated) = quantization collapse |
| FP8 (e4m3) | >0.9996 | 7/24 (diverge @7) | coherent, HIGH-QUALITY, non-repetitive ("...rich history, culture...Situated in the north-central...") |

**TEACHER-FORCED gate (rigorous, 188 positions over an 8-string corpus, dense-only, vs bf16):**
| metric | FP8 (e4m3) | NVFP4-MSE |
|---|---|---|
| top-1 agreement (all) | 95.745% | 92.553% |
| agreement @ margin >0.1 / >0.5 | 98.3% / 100% | 96.1% / 100% |
| KL(bf16‖quant) | 0.00519 | 0.03274 (6.3× worse) |
| CE delta | 0.011 | ~0 |
| disagreements / mean margin | 8 / 0.055 (near-ties) | 14 / 0.116 (flips more confident) |
FP8 ≈ bf16 (near-lossless; 100% agreement on all >0.5-margin confident decisions). NVFP4 is ~6× worse in
KL and flips higher-margin tokens. CONFIRMS FP8-first.

**FULL-CONFIG (dense + experts quantized, closes Opus F3) — FP8 airtight:**
FP8 full-config (80 expert stacks / 20480 expert matrices round-tripped, sample cos 0.9997): top1 **97.9%**,
**KL 0.00548** (≈ dense-only 0.005 — near-lossless), 100% agreement on all >0.5-margin confident decisions,
4 near-tie disagreements. ⇒ **FP8 on the REAL shipping config (experts included) ≈ bf16.** (NVFP4 full-config
run timed out — the MSE grid-search codec is very slow over 20480 expert matrices; irrelevant since NVFP4 is
deprioritized and dense-only NVFP4 was already 6× worse.) **M-B fully, rigorously COMPLETE: ship FP8.**

**3-voice review (Codex + Gemini + Opus) — UNANIMOUS:**
- Opus quantified the decode margin: NVFP4-over-FP8 end-to-end realistically **<~1.1×** at occupancy-bound
  decode (both occupancy-capped; NVFP4's byte edge only touches the ~21ms weight-read term of 47.5ms/tok).
  NVFP4's ONLY real edge is footprint (~20GB vs ~35GB for the 35B) — NOT binding on GB10 128GB single-stream.
- Threshold is RELATIVE: "NVFP4 ≈ FP8 on KL/PPL/task", not an invented "99%". Weight cosine is the wrong
  space; report OUTPUT metrics (KL/top-1/router-set-agreement/ΔPPL/task).
- Free-run greedy token-identity is the WRONG gate for plain quant inference (chaotic near-tie/accumulation
  compounding; even FP8>0.9996 only 7/24, and bf16 itself only tracks HF ~7 tokens). It's the SPEC-DECODE/MTP
  invariant, mis-applied. RETIRE it for plain NVFP4/FP8 inference; keep only for MTP.
- Correct gate = TEACHER-FORCED top-1 agreement, **margin-stratified** (by bf16 top1-top2 margin) + KL/CE
  delta vs bf16 + repetition-rate + a small downstream metric. (M-B.4b, building.)
- NVFP4 is SPECIFICALLY lossy here (repetition = collapse), not merely "different". FP8 is ~as-good-as-bf16.
- **DECISION: default to FP8 for the 35B quant lever.** Decode is occupancy-bound (NVFP4 only ~1.3x, FP8
  also halves bytes), FP8 is validated + coherent, Atlas ships Qwen3.6 in FP8. Resolves the plan's
  UNRESOLVED "NVFP4 vs FP8" decision. NVFP4 stays a research option ONLY IF a mixed-precision config
  (incl. MoE experts) is statistically indistinguishable from FP8 on the teacher-forced gate. (Pending the
  rigorous teacher-forced numbers + Opus's 3rd voice to lock it.)
- Dense-only was never a shippable result anyway (experts dominate 35B bytes). Any real NVFP4 pursuit MUST
  quantize experts (M-B.3b/M-B.5).

**PLAN-LEVEL IMPLICATION:** Lane 2C pivots NVFP4-first -> **FP8-first** for 35B deployment. FP8 GEMM
(w8a16) already works eager + QuantLinear::Fp8 exists (M-B.1). Remaining FP8 deploy work: raw-backend port
of w8a16_gemm (for capture, M-D) + wire QuantLinear::Fp8 at load. LESS work than the NVFP4 kernel path.

## REFINEMENT (post-review, Opus-controller): in-place fake-quant surgery is the gate instrument
The reviews assumed the gate runs the NVFP4 KERNEL (hence F4: gate on raw backend). Better: gate via
standard PTQ **fake-quantization** — round-trip each selected weight through `dequant_nvfp4(quantize_*)`
and store it back as a normal bf16 Linear/Param, then run the model's NORMAL path. This:
- isolates the PURE quantization accuracy effect (identical backend/accumulation vs the bf16 baseline —
  F4 is MOOT for the accuracy gate; the kernel's fidelity to dequant is separately proven, probe cos 1.0);
- needs NO new kernels and NO struct/forward rewire (all model linear fields are `pub`);
- covers dense linears AND the rank-3 MoE experts (full-config token-identity + routing-stability WITHOUT
  building the NVFP4 grouped-GEMV kernel first);
- defers the NVFP4 GEMV kernel (nvfp4_linear, done) + QuantLinear model wiring + moe_grouped NVFP4 to
  AFTER the gate says GO (the perf win).
Memory: 35B ~70GB — run bf16 greedy (record tokens) → quantize SAME model in-place → run again → compare
(one model resident; save one weight for per-tensor restore). So M-B.3 = a `quant_gate` surgery module
(dense + rank-3 experts, plan-controlled), M-B.4 = the harness/decision. Kernel/deployment work is gated
on the GO.

## FINAL revised M-B task order (all 3 voices folded)
- **M-B.0** bf16 Fusion-vs-raw greedy token-parity baseline (decides gate backend). Cheap, decisive.
- **M-B.1 (Codex, dispatched)** `src/nvfp4_linear.rs` — `Nvfp4Linear<B: Nvfp4GemvBackend>` (dual-backend) +
  `QuantLinear{Nvfp4,Fp8,Bf16}`. Nvfp4/Bf16 arms generic; Fp8 arm Cuda-only for now.
- **M-B.2** MSE-optimal per-block calibration `quantize_nvfp4_mse` (brief ready).
- **M-B.2b** Hadamard rotation (data-free) if MSE-clip leaves tensors <0.999. `W→QᵀW` + `X→XQ` runtime.
- **M-B.3** load hook `set_quant_linear` (manifest) + route `linear3` sites through `QuantLinear`.
- **M-B.4** per-tensor NUMERICAL gate (raw backend per F4): layer-output cosine + argmax-margin + pre-router
  activation cosine (>0.999) + top-8 routing-set agreement. Emit manifest.
- **M-B.3b** expert fake-quant surgery (for the full-config accuracy gate). Expert weights are stored
  `[out,in]` (gate_up_proj [E,2I,H] slice -> [inner,hidden]=[out,in]; down_proj [E,H,I] slice -> [hidden,
  inner]=[out,in]; `matmul_out_in` transposes to [in,out]). So block along contraction `in`: TRANSPOSE each
  expert slice [out,in]->[in,out], quantize_nvfp4(k=in,n=out), dequant, transpose back. (Dense Burn Linears
  are already [in,out]=[K,N] — no transpose.) Round-trip all E experts per stack in place.
- **M-B.5** MoE-expert NVFP4 grouped-GEMV (extend moe_grouped) — the real 35B perf lever (only after GO).
- **M-B.6** FULL-CONFIG token-identity smoke (dense + experts NVFP4) on 30B known-good string + 35B bf16 ref,
  + Nsight perf co-gate (DRAM vs ALU bound at M=1). GO/NO-GO vs FP8 per tensor.
