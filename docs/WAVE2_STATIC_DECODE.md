# Wave-2 Step-1: static-decode port to the MoE + the OOM fix

Plan for 3-voice review (Codex 5.5 high + Gemini 3.1 Pro high + Opus 4.8 high) + prod-repo similarity search
(`docs/moe-weight-layout-research.md`). Reviews the partial Wave-2 Step-1 port AND the blocking OOM fix.

## What was built (compiles + loads the 30B; partial agent, recovered)

The dense `Qwen3ForCausalLM` has a full static-decode stack (`forward_with_cache_static_pre`, P2/P4);
`Qwen3MoeForCausalLM` had none. Step-1 ported it (src/moe.rs):
- `MoeStaticDecode<B>` — companion built ONCE post-load: `expert_caches: Vec<MoeExpertCache>` (one Block-A
  cache per layer) + precomputed rope `freqs` + `arange_tmax`.
- `Qwen3MoeForCausalLM::forward_with_cache_static_pre(input_ids, pos, cache, sd, prec)` — one token `[B,1]`
  → `[B,1,V]`, each layer = dense static-attention path REUSED VERBATIM (`Qwen3Attention::forward_with_cache_static_pre`: full-`T_max` masked attention + device-`pos` KV) → `MoeExpertCache::decode_topk` for the MoE block.
- `build_expert_caches()` = `self.layers.iter().map(|l| MoeExpertCache::from_block(&l.mlp)).collect()`.
- `generate_greedy_static(...)` — device-`pos` greedy driver over the static path.
- Tiny tests: `static_decode_matches_eager_greedy_tiny`, `static_forward_matches_eager_cache_logits_tiny` (green).

## THE BLOCKER — the pre-stacked cache DUPLICATES the experts → 30B OOM

Validation on the real 30B PANICKED: `can't allocate buffer of size 502054912` at
`MoeExpertCache::from_block` → `stacked_experts` → `cat`. Root cause: `from_block` builds a fresh contiguous
`[E,H,I]` copy of every expert (gate/up/down) per layer. For the 30B that is **128 experts × 48 layers × 3 ≈
58 GB (bf16) of DUPLICATE** held alongside the already-loaded ~60 GB model ⇒ ~118 GB on a 119 GB unified box
(GB10) ⇒ OOM mid-build. The tiny-model tests + the f32/CPU parity never surface it; it appears only at 30B
scale on real hardware. (Opus's Wave-1 review flagged the cache as "materializing" — a 3× constant at small
scale; at 30B the upfront duplication is an outright OOM.)

A second consequence: the **bf16 parity gate** wants to run eager `generate_greedy` (needs per-expert weights)
AND static `generate_greedy_static` (needs the cache) on the SAME loaded model — they cannot both be resident
at 30B. So even running ONLY the static path OOMs (model 60 + cache 58 > 119) unless the cache REPLACES the
per-expert storage.

## The fix options (the decision to verify, prod-grounded)

- **(a) MOVE, not copy** — build the contiguous `[E,H,I]` cache, then FREE the per-expert Linear weights
  (the static decode path needs only the cache). Total stays ~60 GB. Smallest change; but the eager
  oracle/routed paths then can't run on the same model (they need per-expert weights) ⇒ parity must compare
  static tokens vs a SEPARATE-process eager run (or the known `moe_generate` output), or run on the 15B.
- **(b) LOAD-CONTIGUOUS from the start** — the loader writes each expert's safetensors shard DIRECTLY into a
  pre-allocated `[E,H,I]` slot, so per-expert tensors never exist separately (one copy, ever). Cleaner +
  matches prod (vLLM `FusedMoE` `w13_weight`/`w2_weight` + a slot weight_loader), but a loader refactor that
  touches the eager paths (they'd index the contiguous buffer too).
- **(c) NO stack — gather per-expert with pointer arithmetic** — keep per-expert weights, no contiguous copy
  at all; the decode kernel reads the k routed experts by expert-stride pointer (vLLM/SGLang fused-MoE GEMV).
  Zero duplication, but needs a custom kernel (the fused gather-GEMV Wave-1 already said is the roofline path).

## Questions for the council
- Which fix do well-tested MoE engines actually use (vLLM/SGLang/llama.cpp/ktransformers)? Is the canonical
  pattern (b) load-contiguous-into-slots, or (c) per-expert + pointer-gather? Is (a) move-not-copy a sound
  interim, or a smell (two layouts to maintain)?
- Is holding the experts ONCE (no duplicate) sufficient to fit + decode the 30B on the 119 GB GB10, or is fp8
  / expert-offload also needed? (60 GB model with the experts stored once should fit; verify.)
- The bf16 parity gate at 30B: separate-process eager-vs-static comparison, or validate parity on the 15B-A2B
  (fits with the duplicate) + only smoke-test the 30B static path? Which is the honest gate?
- Does the `MoeStaticDecode`-holds-`Vec<MoeExpertCache>` design survive the fix, or should the cache BE the
  model's expert storage (so there's one owner)? Is reusing the dense static-attention path verbatim correct
  for the MoE (same GQA), or are there MoE-specific attention differences?
- Does the contiguous layout matter for Step-2 CUDA-graph capture (fixed base pointers), favoring (b)/(a)
  over (c)?

## GSTACK REVIEW REPORT

| Review | Trigger | Runs | Status | Findings |
|--------|---------|------|--------|----------|
| Codex gpt-5.5 high | council voice 1 | 1 | issues_found | **(b) load-contiguous** — make contiguous `[E,..]` the model's real storage (vLLM `FusedMoE` `w13_weight`/`w2_weight` + slot loader), eager+static read it, fused kernel reads same by `id·stride`. (a)=temp smell; (c)=kernel not storage. One bf16 copy fits 119GB. Parity=separate-process LOGITS not tokens. **Catch: zip over layers/expert_caches needs length asserts (silent layer-skip = catastrophic); verify QK-norm/RoPE config, not just GQA** |
| Gemini 3.1 Pro high | council voice 2 (AGY-USG, after 429-retry) | 1 | issues_found | **Skip to (c)** — building a 58GB contiguous stack to feed the *materializing* `decode_topk` is a dead-end crutch; jump to the fused pointer-gather kernel on the original weights. (a)=hack, (b)=prod-correct but "massive loader rewrite". Sidecar = "abomination". One copy fits; KV <5GB. Parity=15B + 30B smoke |
| Opus 4.8 high | council voice 3 (source-verified) | 1 | issues_found | **TIE-BREAK: (c) does NOT skip the stack** — `select(0,ids)` AND `base+id·stride` BOTH require a contiguous E-major base, so the layout is mandatory; (c) is a downstream compute swap. **(b) is the right end state but a REAL refactor** (loader is declarative `apply_to`, no per-expert loop — `load.rs:244`). **(a) move-then-free is structurally OK (attention is a separate field) but a BET on CubeCL pool reclaim** (size-class mismatch may not free 58GB) — gate on a measured drop. Memory ~65GB≪119, no fp8. Parity=15B same-proc + 30B cross-proc/golden. Attn reuse correct (caveat: MoE passes `None` for P4 `lo` → no left-pad batches) |
| Prod-repo search | agy-direct (Google) | 5 queries (429-backoff) | done | **(b) CONFIRMED across every well-tested engine** (`docs/moe-weight-layout-research.md`, primary-source line refs): vLLM `unquantized_fused_moe_method.py::create_weights` pre-allocs `w13_weight=Parameter(empty(E,2I,H))` + `weight_loader` does `param.data[expert_id].copy_(shard)` (`routed_experts.py:646/659`); SGLang identical (`fused_moe_triton/layer.py:839`); llama.cpp merges experts into one stacked tensor at convert + `ggml_mul_mat_id` indexes slices in place. Per-expert tensors NEVER exist separately → no 2× copy ever. (c) `base+id·stride` is the batch-1 *compute* on top of the (b) buffer. Stable `data_ptr()` = capturable; per-step `cat` = uncapturable. 1 copy fits 119GB w/ ~50GB headroom |

- **CROSS-MODEL: a real split, RESOLVED by Opus's source verification.** Codex+Opus say **(b) contiguous storage**; Gemini says **skip to (c)**. Opus's code-grounded reframe settles it: the contiguous `[E,H,I]` base is **mandatory for both `decode_topk`'s `select` AND the future fused gather-GEMV** (`base+id·stride`), so **(c) is a downstream compute swap, not a storage fix — you cannot skip the layout** (refutes Gemini's core claim). All three AGREE: the duplicate is the whole bug (one copy ≈ 65 GB ≪ 119 GB, **no fp8/offload needed**); the `MoeStaticDecode` `Vec<MoeExpertCache>` sidecar is wrong (expert storage belongs in the block, single owner); **(a) move-not-copy is not a product fix**; dense static-attention reuse is correct (caveat: no P4 left-pad on the MoE path). Opus corrects the plan's premise — the loader is *declarative*, so **(b) is a real refactor** (Param<3>, custom slot-loader, rewrite the 4 eager forwards), not the small change implied. Parity reconciled: **15B same-process logit parity (the `decode_topk`/bf16 path) + 30B cross-process or golden-vs-`moe_generate` (the 30B numerics)** — single-process eager-vs-static at 30B is physically impossible.
- **VERDICT:** **Fix = (b)** — a single-owner contiguous `[E,H,I]` expert store in `Qwen3MoeSparseBlock`, populated slot-wise at load (vLLM `FusedMoE` pattern), read by eager + static + the future fused kernel. **(a) per-layer move-then-free is an acceptable STOPGAP to land a 30B result fast, but ONLY gated on a measured ~58 GB resident-memory drop** (else the CubeCL pool didn't reclaim and you're forced to (b) anyway). **Do NOT jump to (c)** — it presupposes the same contiguous cache; build it on top after the static port validates. Must-fix regardless: the layer/cache **length assertion** (Codex), the **QK-norm/RoPE config check** (Codex), and move the cache into the block as the single owner (all three). Memory analysis says the 30B fits once the duplicate is gone — no fp8/offload this lever.

**DECISION (user):** **(b) straight — the contiguous slot-loaded storage refactor, no (a) stopgap.** Confirmed
unanimously by the council AND every prod engine (vLLM/SGLang/llama.cpp all load-contiguous-into-slots). The
build: replace `Qwen3MoeSparseBlock`'s `Vec<Qwen3MLP>` experts with persistent `gate_stack/up_stack/down_stack
[E,..]` `Param` fields; a custom expert loader that slice-writes each `mlp.experts.{j}.*` shard into slot `j`
(Burn analogue of `param.data[expert_id].copy_`, with the PyTorch→Burn Linear transpose); rewrite the 4 eager
forwards (`forward_oracle`/`forward_routed`/`forward_fast`/`forward_routed_ondevice`) + `proj_weights` +
`decode_topk` to slice the stacked param instead of indexing `experts[ei]`; delete the per-call `cat` in
`stacked_experts`. Must-fix alongside: the layer/cache **length assert** (silent layer-skip) and the
**QK-norm/RoPE config check**. Parity gate: 15B same-process logits + 30B cross-process/golden. Step-2 capture
(the fused gather-GEMV `base+id·stride`) builds on this contiguous store afterward.

NO UNRESOLVED DECISIONS
