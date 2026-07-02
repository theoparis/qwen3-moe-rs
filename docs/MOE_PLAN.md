# Qwen3-MoE + fast GRPO-rollout — engineering plan (reviewed)

Status: reviewed via `/plan-eng-review` + outside voice (Codex gpt-5.5 high, Gemini 3.1 Pro high).
Branch `grpo-phase-a`. Companion to [GRPO_PLAN.md](GRPO_PLAN.md). Every architectural number is cited
to a primary source. Scope decisions made in review: **rollout/inference first (MoE training deferred to
a LoRA-gated B3)**; perf goal = measurable our-workload target; the custom grouped-GEMM kernel is the
fast path, precision-first. The §4-§6 performance/training design was revised after the outside voice
(see the review report at the end).

## 0. Goal & locked acceptance criteria

Add **Qwen3-MoE** support to the qwen3-burn engine and make the **GRPO rollout** fast on a single
NVIDIA GB10 (Grace-Blackwell, sm_121, 128 GB unified, ~273 GB/s, ~100 BF16 TFLOPS). Primary target
**Qwen3-30B-A3B** (30B total / ~3.3B active); `Qwen3-235B-A22B` is out of single-GB10 scope.

Acceptance (four gates):
1. **Correctness (mandatory, blocks every perf claim):** MoE forward matches HF transformers
   logits/log-probs, cosine > 0.9999 (the bar the dense model uses in `tests/ref`).
2. **Beat the in-repo naive baseline (primary deliverable):** rollout decode tok/s at the real GRPO
   shape (P×G, capped `max_new_tokens`) on GB10, stated as "Nx over the f32 / dense-oracle baseline",
   device-synced.
3. **Beat HF-transformers-eager on the same GB10** (same precision/batch/prompts).
4. **Stretch (reported, not pass/fail):** decode tok/s as a fraction of SGLang on Qwen3-30B-A3B on the
   same GB10 and of the fp8 roofline (~80 tok/s); target band ≥ 50% of SGLang.

Non-goals: beating every engine; expert-parallel / multi-GPU; PagedAttention; full-param 30B GRPO
training (deferred B3, LoRA-only).

## 1. Where we are (current engine)

Dense-only Qwen3. Each `Qwen3DecoderLayer` (src/decoder.rs:368) holds `mlp: Qwen3MLP<B>`
(src/decoder.rs:419) — SwiGLU through the **batch-safe 2-D `linear3`** (src/linear2d.rs) that dodges a
CubeCL CUDA batched-matmul corruption bug on sm_121 (docs/ARCHITECTURE.md, examples/matmul_probe.rs).
Attention (GQA + per-head QK-RMSNorm + RoPE), KV cache, sampling, and the GRPO rollout/trainer are
reusable unchanged. No router, no experts, no MoE config.

## 2. Verified Qwen3-MoE spec (the parity target)

Source: HF `config.json` for `Qwen/Qwen3-30B-A3B` & `Qwen/Qwen3-235B-A22B`, `modeling_qwen3_moe.py`
@ transformers v4.51.0, Qwen3 Technical Report (arXiv:2505.09388). Both outside voices upheld this spec.

| field | 30B-A3B | 235B-A22B | note |
|-------|---------|-----------|------|
| hidden_size | 2048 | 4096 | |
| num_hidden_layers | 48 | 94 | |
| num_attention_heads / kv | 32 / 4 | 64 / 4 | GQA, head_dim **128 explicit** (2048/32=64≠128) |
| num_experts | 128 | 128 | |
| num_experts_per_tok (top-k) | 8 | 8 | |
| moe_intermediate_size | 768 | 1536 | per-expert SwiGLU inner dim |
| norm_topk_prob | true | true | renormalize the kept 8 |
| decoder_sparse_step / mlp_only_layers | 1 / [] | 1 / [] | **every layer is MoE** |
| shared expert | **none** | **none** | Qwen3 drops the Qwen2.5-MoE shared expert |
| router_aux_loss_coef | 0.001 | 0.001 | training-only |
| tie_word_embeddings | false | false | separate `lm_head.weight` |

**Routing reference (HF literal):** `router_logits = gate(x)` (`Linear(hidden→128, bias=false)`);
`probs = softmax(router_logits, dim=-1, fp32)` over all 128; `(w, idx) = top-8(probs)`; `norm_topk_prob`:
`w /= w.sum(-1)` (fp32, divisor < 1); cast after; `out = Σ_{e∈top8} w_e · SwiGLU_e(x)`, no shared term.

**Math lineage (papers read):** DeepSeekMoE fine-grained routing
([2401.06066](https://arxiv.org/html/2401.06066)) **without** shared expert, Mixtral-style top-k softmax
combine ([2401.04088](https://ar5iv.labs.arxiv.org/html/2401.04088)), Switch-style `Σ f_i·P_i`
load-balance loss at **training only** ([2101.03961](https://ar5iv.labs.arxiv.org/html/2101.03961)),
confirmed in the Qwen3 report ([2505.09388](https://ar5iv.labs.arxiv.org/html/2505.09388) §2). At
inference the aux loss does not run.

## 3. Architecture (Burn)

Homogeneous all-MoE model type (not a per-layer enum): `decoder_sparse_step=1, mlp_only_layers=[]` make
every layer sparse for both checkpoints, and `#[derive(Module)]` on an enum injects the variant name
into the weight key path (`mlp.Sparse.gate.weight`) — which breaks the loader. Mixed dense/sparse is
deferred behind that enum until a checkpoint needs it.

```
Qwen3MoeConfig            // Qwen3Config fields + num_experts, num_experts_per_tok,
                          //   moe_intermediate_size, norm_topk_prob, decoder_sparse_step,
                          //   mlp_only_layers; preset qwen3_30b_a3b(); tie_word_embeddings=false
Qwen3MoeRouter<B>         // { gate: Linear(hidden→128, bias=false) } — router GEMM forced F32
Qwen3MoeSparseBlock<B>    // { router, experts: Vec<Qwen3MLP<B>>(len 128, inner=768), Ignored meta }
Qwen3MoeDecoderLayer<B>   // self_attn (REUSED), mlp: Qwen3MoeSparseBlock, 2× RmsNorm
Qwen3MoeModel<B> / Qwen3MoeForCausalLM<B>   // mirror the dense pair; lm_head untied, always present
```

Experts as `Vec<Qwen3MLP>` (positional keys map automatically). For the Tier-2 custom kernel (§4b) the
128 per-expert weights also need a **stacked `[E,…]` view** — provide it as a derived stacked tensor at
load time (a 128-key→1-tensor pack), keeping the `Vec` for the oracle. Reuse `Qwen3MLP`,
`Qwen3Attention`, `RmsNorm`, `linear3`, KV cache, and `load_weights_sharded` verbatim. (Code-quality
note: the MoE model/layer wrappers duplicate the dense pair; a generic-over-FFN layer is awkward under
Burn's derive, so a thin parallel type is the accepted lesser evil — see review report.)

## 4. Routing + the fast-MoE path (revised per outside voice)

### 4a. Skip the 128-way softmax at inference

HF does `softmax_128(fp32) → top-8 → renorm`. That renormalized top-8 is **algebraically identical** to
`softmax over just the top-8 logits` — the full denominator cancels (Gemini's correction). So at
inference we do **not** exponentiate all 128: take the **top-8 by iterated argmax over the raw logits**
(argmax is on-device; `topk`/`sort` are host-sync on CubeCL), then softmax over those 8 in fp32. The
parity test asserts this equals HF's renorm path within cosine > 0.9999, with **deterministic non-tie
fixtures** (PyTorch `topk` tie-ordering ≠ argmax masking; rare on continuous softmax but pinned in
tests). The full 128-way softmax is needed only for the training aux loss (deferred, §6).

```
x[T,H] → router.gate (linear3, F32) → logits[T,128]
  → top-8 via 8×{ argmax-over-LOGITS → record → mask_fill(-inf) }   (no 128-way exp, no topk/sort)
  → softmax over the 8 selected logits (fp32)  ==  HF renorm(softmax_128)[top8]   → gate_w[T,8]
  → out = Σ_{e∈top8} gate_w_e · SwiGLU_e(x)
```

### 4b. Performance tiers (honest — capacity-scatter dropped)

The prior draft's on-device capacity gather/scatter is **a non-goal**: both outside voices showed a
no-drop fixed capacity `C=T` makes each expert's buffer `[T,H]` and computes the full `E×dense` FLOPs
(zero savings — Gemini), while `C<T` silently drops tokens (corrupts RL/parity — Codex); and a
per-expert loop of 128 2-D GEMMs is launch-bound (~18k launches/decode-step × 48 layers).

- **Tier 0 — precision (the reliable rollout win; ships first).** bf16 (unblock the RmsNorm
  DTypeMismatch, B0) then fp8. Decode is HBM-bandwidth-bound, so this is hardware-grounded and
  independent of any kernel. GB10 roofline: f32 ≈21 / bf16 ≈41 / fp8 ≈80 tok/s over ~3.3B active.
- **Tier 1 — correctness oracle ONLY (not a perf or training path).** Rung-1 dense-masked loop:
  every expert over all tokens via `linear3`, weighted-sum the top-8. Trivially correct, on-device,
  the numerical reference. But `E×dense` FLOPs, launch-heavy, and it materializes `E×` activations
  → **OOMs any training batch** (Gemini). Use only as the small-scale oracle.
- **Tier 2 — custom CubeCL grouped-GEMM kernel (THE fast MoE path).** A `#[cube(launch)]` kernel that
  builds per-expert **segment offsets** on-device (`one_hot` + `cumsum`, no host-sync), runs **one
  variable-length segmented GEMM** (no fixed capacity → no token drop), then unpermute + weight-combine,
  with a hand-written `Backward` for the deferred training path. This is the only structure giving real
  compute savings (~k×dense) **and** a single fused launch. **Gated behind a feasibility spike**
  (`moe_probe`, §8): the broadcast batched-matmul kernel is buggy on sm_121, so the spike must prove a
  *packed/custom* grouped GEMM is correct here first. References: MegaBlocks block-sparse
  ([2211.15841](https://ar5iv.labs.arxiv.org/abs/2211.15841)), ScatterMoE `scatter2scatter`
  ([2403.08245](https://arxiv.org/html/2403.08245)), SGLang align&sort.

## 5. Performance — roofline, ROI, honest caveats

ROI order: **(1) bf16 (fit 30B in 128 GB + ~2×), (2) the Tier-2 custom kernel via the spike,
(3) fp8, (4) CUDA-graph capture.** Two caveats both reviewers raised: (a) at small decode batch the
Tier-2 win is **launch-elimination** (one kernel vs 128), at larger rollout batch most experts get ≥1
token so weight traffic approaches the full ~30B and **precision dominates** — the kernel wins on
compute, not bandwidth; (b) `~k×dense` is a FLOP argument, not a guaranteed wall-clock number. **Every
speedup claim is gated on the GB10 benchmark (Gate 2), never asserted.** bf16 inference is BLOCKED today
(RmsNorm DTypeMismatch) and is a hard prerequisite to even load 30B (f32 ≈120 GB > 128 GB) — Phase B0,
which must also produce an **end-to-end memory budget** (weights + casts + duplicated Burn params +
lm_head + KV + activations + kernel scratch), not just "60 GB of weights fits."

## 6. GRPO + MoE training (DEFERRED to B3 — design notes, corrected)

Training is out of the near-term scope (decision: rollout/inference first; full-param 30B GRPO doesn't
fit — AdamW state ≈ 240 GB → LoRA-only or offload). Recording the corrected design so B3 starts right:

- **Train-inference routing mismatch:** the primary fix is **identical deterministic routing
  (same code + precision) between rollout and the policy-gradient recompute**, so they route to the
  same experts and the router trains normally — *not* forcing stored indices. **Rollout Routing Replay
  (R3)** (record rollout expert indices, force them in the backward) is a **fallback** for residual
  divergence, with the caveat (both reviewers) that forcing indices can **starve/bias the router
  gradient**, and the backward must still recompute the 128-way router probs under current weights to
  differentiate the gate.
- **Aux load-balance loss:** only meaningful if the router is **trainable** (with LoRA, decide whether
  the router is a LoRA target). The coefficient is an **RL hyperparameter to tune**, not the 1e-3
  pretraining default; the training forward needs the full 128-way softmax (no skipping it there).
- The fast-rollout / differentiable-backward split is industry standard (the repo already mirrors it).

## 7. Loader changes

- **Router-key collision (bug to fix):** `create_safetensors_store_causal_lm` (src/load.rs) applies
  `\.weight$ → .gamma`, turning `mlp.gate.weight` into `mlp.gate.gamma`, and the `_proj\.gamma$ →
  _proj.weight` rule does **not** restore it. → add `mlp\.gate\.gamma$ → mlp.gate.weight`, **anchored to
  the exact router path and shape-asserted `[num_experts, hidden]`** (Codex: don't add a loose regex to
  a fragile chain). Experts (`mlp.experts.{j}.{gate,up,down}_proj.weight`) already round-trip.
- **Sharded load reused:** `load_weights_sharded` union-coverage check fails loud on any missing param;
  add a test asserting every `mlp.gate` + `mlp.experts.{j}.*` key is applied.

## 8. Verification plan (gates)

- **Gate 1a — router unit test:** fixed logits; assert (top-8 argmax-over-logits → softmax-8) matches a
  torch reference (`softmax_128 → topk(8) → renorm`) within tol, on **non-tie fixtures**; assert it
  **differs** from the wrong topk-then-softmax-of-raw order.
- **Gate 1b — oracle vs kernel:** Tier-2 grouped-GEMM output == Tier-1 oracle within cosine > 0.99999.
- **Gate 1c — HF logit parity (CI, CPU):** tiny random Qwen3-MoE built in transformers v4.51.0 (config
  must satisfy `head_dim` / head divisibility — name the constraints: e.g. hidden 64, 8 q / 4 kv heads,
  head_dim 16, 8 experts, top-2, moe_inter 32, vocab 256); dump logits + intermediates; load same
  safetensors into Burn; compare layer-by-layer (router probs ≈, top-k idx exact on non-tie, final
  logits cosine > 0.9999) on NdArray.
- **Gate 1d — real-weights smoke (GB10):** Qwen3-30B-A3B (bf16) first-token logits cosine > 0.9999 and
  identical greedy argmax for N tokens vs HF.
- **`moe_probe` (de-risks Tier 2):** a `matmul_probe`-style smoke that proves `cumsum`/`one_hot`/
  `scatter(Add)`/`argmax` run **on-device** (no host readback) on sm_121 AND that a packed/custom
  grouped GEMM is numerically correct there. Tier-2 go/no-go.
- **Gate 2/3/4 — perf (GB10, device-synced):** rollout decode tok/s; Tier-2 vs Tier-1 oracle, bf16 vs
  f32; ≥ HF-eager; report fraction of SGLang + fraction of the ~80 tok/s fp8 roofline. Always GB10.

## 9. Phased rollout

- **B0 — bf16 inference unblock + memory budget (prerequisite).** Fix RmsNorm DTypeMismatch (f32-cast
  norm inputs, or fix Fusion dtype tracking); produce the end-to-end bf16 memory budget proving 30B
  fits. Own tests. *Blocks everything.*
- **B1 — correct Qwen3-MoE forward (Tier-1 oracle) + parity.** Config/router/sparse-block/model +
  loader fix; Gates 1a-1d. Inference path only (training deferred).
- **B2 — the fast path.** `moe_probe` spike → if green, the Tier-2 custom grouped-GEMM kernel +
  stacked-weights pack; benchmark (Gates 2-4). fp8 (B2.5) alongside. If the spike is red, fall back to
  precision-only wins and re-scope the kernel.
- **B3 — MoE GRPO training (deferred, LoRA-gated).** Deterministic-routing consistency (R3 fallback) +
  trainable-router-aware aux loss + Tier-2 `Backward`. Only when MoE *training* is greenlit.
- **Later — CUDA-graph capture; mixed dense/sparse enum (only if a real mixed checkpoint appears).**

## 10. Risk register

| risk | sev | mitigation |
|------|-----|------------|
| 30B f32 doesn't fit; bf16 inference blocked (RmsNorm) | **high** | B0: bf16 storage + f32 norms + end-to-end memory budget; own test |
| Tier-2 custom kernel feasibility on sm_121 (grouped GEMM + on-device offsets) | **high** | `moe_probe` go/no-go BEFORE building; precision-only fallback; Tier-1 oracle always available |
| Per-expert loop launch-bound; capacity-scatter gives no no-drop savings | **high** | dropped as a path; Tier-2 fused kernel or precision-only — never the 128-GEMM loop for decode |
| Tier-1 oracle materializes E× activations → OOM if used for training | medium | oracle is inference/small-scale only; training uses the Tier-2 kernel (B3) |
| Loader router-key collision → random-init router | medium | anchored `mlp.gate.gamma$ → mlp.gate.weight` + shape assert + key-coverage test |
| Full-param 30B GRPO doesn't fit (Adam ~240 GB) | medium | scope to rollout/inference; B3 training is LoRA-only/offload |
| Tiny HF parity model won't construct in v4.51.0 (divisibility/RoPE) | low | name the head_dim/divisibility constraints in Gate 1c |
| bf16/scatter add-order ≠ HF `index_add_` | low | accumulate combine in fp32; assert cosine > 0.9999 |

## 11. Open decisions (to confirm)

Resolved in review: **training scope** = rollout/inference first (B3 LoRA-gated); **perf strategy** =
custom kernel via spike, precision-first. Still to confirm (both have a recommendation):
1. **Authorize B0 bf16-inference unblock?** *Recommend yes* — it's the #1 rollout lever and the
   prerequisite to load 30B at all.
2. **Confirm Qwen3-30B-A3B as the sole target** (235B is multi-device, out of single-GB10 scope)?
   *Recommend yes.*

## 12. Citations

- Qwen3-MoE: HF `Qwen/Qwen3-30B-A3B` & `Qwen3-235B-A22B` config.json; transformers v4.51.0
  `modeling_qwen3_moe.py`.
- Math: DeepSeekMoE [2401.06066](https://arxiv.org/html/2401.06066) · Mixtral
  [2401.04088](https://ar5iv.labs.arxiv.org/html/2401.04088) · Switch
  [2101.03961](https://ar5iv.labs.arxiv.org/html/2101.03961) · Qwen3 report
  [2505.09388](https://ar5iv.labs.arxiv.org/html/2505.09388).
- Fast kernels: MegaBlocks [2211.15841](https://ar5iv.labs.arxiv.org/abs/2211.15841) · ScatterMoE
  [2403.08245](https://arxiv.org/html/2403.08245) · SGLang fused MoE align&sort · vLLM
  `moe_align_block_size`.
- RL+MoE: R3 (rollout routing replay) for MoE-RL routing mismatch; aux-loss handling in
  TRL/OpenRLHF/verl; DeepSeek-V3 aux-loss-free routing (background).
- Rust prior art: candle `qwen3_moe.rs` (naive per-expert loop, semantics reference); mistral.rs
  Qwen3-MoE (candle-based, inference). No Burn-native MoE exists.
- Burn/CubeCL (rev 5923b1e): `burn-tensor` base.rs/numeric.rs/orderable.rs (gather/scatter/select/
  cumsum/one_hot/argmax on-device; topk/sort host-sync); `burn-backend` IndexingUpdateOp (Add-only);
  `examples/custom-cubecl-kernel` (`#[cube(launch)]` + autodiff Backward).

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | held_scope | HOLD SCOPE; full plan held (rejected reduce/defer); 3 rigor notes kept as review notes (below) |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found | gpt-5.5 high SOUND_WITH_GAPS + Gemini 3.1 Pro high FLAWED — capacity-scatter flaw, R3 over-claim |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | clean | 2 arch (loader bug, homogeneous type) + 1 DRY + 2 test-gaps; perf strategy revised; both open decisions now resolved (yes/yes) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (backend) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | n/a |

- **CODEX:** gpt-5.5 (high) SOUND_WITH_GAPS — Rung-2 index construction hand-waved; R3 not the obvious
  fix; loader regex too broad; bf16 end-to-end memory unproven. All folded into the revision.
- **CROSS-MODEL:** Both outside voices independently flagged the same Rung-2 flaw (no-drop capacity = no
  FLOP savings; per-expert loop launch-bound) → high-confidence; the verified spec (softmax-before-topk,
  no shared expert, all-layers-MoE) was upheld by both. The plan was revised: capacity-scatter dropped,
  custom grouped-GEMM kernel elevated to the fast path behind a `moe_probe` spike, precision-first,
  128-way-softmax skipped at inference, R3 demoted to a fallback with the router-gradient caveat.
- **CEO (HOLD SCOPE):** premise challenge run (MoE-30B vs the Rust-GRPO wedge); user held the full
  plan. Three rigor notes to apply at implementation time (not folded into the design above, by user
  choice): (1) **Observability** — per-step expert-load distribution (router-collapse early-warning),
  a dropped-token counter asserted 0, and a device-synced rollout/reward/gradient three-budget timing
  breakdown; (2) **promote silent-failure guards to gated assertions** — capacity overflow counter,
  a "no host readback in the hot loop" check inside `moe_probe`, non-tie router parity fixtures;
  (3) **write the Tier-2 stop-condition explicitly** — "moe_probe red → precision-only, re-scope the
  kernel" (the custom CubeCL kernel is a path-dependency on a pre-1.0 churning API; kernel
  reversibility ~3/5).
- **DECISIONS (resolved):** both open decisions confirmed YES — B0 bf16-inference unblock authorized;
  Qwen3-30B-A3B is the sole target.
- **IMPLEMENTATION (B1 — shipped + verified):** `src/moe.rs` (Qwen3MoeConfig/SparseBlock/DecoderLayer/
  Model/ForCausalLM) implements the Tier-1 oracle (fp32 softmax over experts → iterated-argmax top-k →
  renorm → dense-masked weighted SwiGLU sum), reusing the dense attention/norms/KV-cache; `src/load.rs`
  adds the MoE loader with the anchored router-key remap + a single-file union-coverage guard (no silent
  random-init). **18 lib tests pass** (routing parity vs a host reference, invariants, determinism,
  top-1 combine == selected expert, no-renorm branch, cache-vs-no-cache parity, end-to-end generate) and
  a CUDA `moe_smoke` ran the forward + greedy generate on the **real GB10/sm_121**. Reviewed by **Codex
  gpt-5.5 high (CORRECT_WITH_NITS)** + **Gemini 3.1 Pro high** + a **4-agent Claude adversarial workflow
  (CORRECT_WITH_NITS, no bugs)** — all three independently confirmed the routing/combine/loader/shapes
  are correct vs HF; folded the fp32-router-cast hardening, the silent-failure loader guard, and the
  combine/no-renorm/cache-parity test gaps. B0 (bf16 weight storage; needs the 30B weights to validate)
  and B2 (custom grouped-GEMM kernel; its on-device routing ops already proven on GB10 by the smoke)
  remain per the plan's phasing.
- **VERDICT:** CEO (HOLD SCOPE) + ENG CLEARED; B1 implemented + verified (CPU + GB10) + cross-model
  reviewed. No Design/DX review required (backend systems plan).

NO UNRESOLVED DECISIONS
