# Long-context decode — the O(T_max) flaw + the fix (3-voice reviewed)

Context: prepping `vllm_infer` to test `Qwen/Qwen3-30B-A3B-Instruct-2507` (256K native context) at long
`--max-tokens`. The critical code is the static-decode attention. SGLang research: `docs/sglang-engine-research.md`.

## The flaw — our static decode is O(T_max), not O(pos) (confirmed in code, 3-voice unanimous)

`src/attention.rs:494` `t_max = key.dims()[1]` is the FULL static KV buffer width; `cache.update_static`
returns the entire `[B, T_max, kv_heads, head_dim]` (not a `0..filled` slice); the SDPA runs DENSE over all
`T_max` columns, masking the future with `idx.greater(pos) → -inf`. **Masking hides columns; it does not skip
the work.** So decode is O(T_max) PER STEP, position-independent. In `vllm_infer.rs:167` `T_max = lp +
max_tokens`, so a large `--max-tokens` makes *every* step pay the full width from step 1.

**Empirical:** `--max-tokens 1024` → 8.00 tok/s vs ~14-15 at short context (~2× slower at 1K).

**The real cause of the 1K drop is NOT KV bytes** (Opus, source-verified): at 1K the KV read is ~1/8 of the
6 GB/tok weights. It is the **non-fused reference SDPA** (`attention_fallback`, deliberately chosen because the
fused kernel mishandles the broadcast mask on sm_121 — a sequence of separate matmul/mask/exp/sum/div/matmul
kernels each sweeping `[.,32,1,T_max]`) **plus the GQA `repeat()` materialization** (4→32 heads, n_rep=8, ~0.8
GB/step round-trip of pure waste — a real GQA broadcasts the KV head). KV *bytes* dominate only past **~8K**
(with the GQA 8×) / ~64K (without). At true 256K: ~24.5 GB/tok KV + OOM from the `[B,256K,..]×2×48` buffer +
the 8× GQA expansion.

## The fix — a custom CubeCL flash-decode kernel (the priority long-context lever)

Not expressible as Burn tensor ops: a `matmul` over `[.,128,T_max]` always does `T_max` work; `slice(..pos)`
→ dynamic shape → breaks CUDA-graph capture + needs a host `pos` (sync); masking saves nothing. **No Burn op's
trip count is a device scalar while its shape stays static.**

The portable mechanism (SGLang/FlashInfer's, all 3 voices): a **from-scratch CubeCL flash-decode kernel** —
- grid/block = `(num_q_heads, num_kv_splits)` with `num_kv_splits` sized to `T_max` (CONSTANT ⇒ captured launch
  config stays static — capturable; a naive `grid = ceil(pos/block)` would change per step and BREAK capture),
- each block loops `j in 0..pos_device` (the device `pos`/`seq_len` VALUE bounds the KV loop = O(pos)),
- **GQA-broadcast** the KV head (kills the `repeat`), **online-softmax** (kills the non-fused fallback),
- splits whose range is entirely `> pos` **early-exit** cheaply.

This fixes O(T_max)→O(pos) AND both constant-factor overheads (non-fused SDPA + GQA repeat) in ONE kernel. The
cache's device-`pos` scatter (`cache.rs:111-145`) + `reset_for_replay` already give the capture-safe WRITE
side; only the READ/attention kernel is missing.

**Priority order (Codex):** (1) kill physical GQA expansion in decode (immediate constant-factor win), (2) the
O(pos) flash-decode kernel, (3) fp8 KV (mandatory at true 256K — even O(pos) KV traffic is large), (4) paged KV
for batching/serving. **Context-position bucketing** (geometric T_max buckets, switch graphs at 1K/4K/16K/...)
is a ≤2× cheap interim FOR THE CAPTURE PATH only — it does nothing for the current eager example (T_max is
already sized exactly to the run).

## Instruct-test gotchas (fixed)

- **`--chat`** wraps the prompt in Qwen ChatML (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`).
  Tokenization VERIFIED: the specials encode as single ids (151644/151645), no BOS injected, stop on 151645
  (not emitted). Template is the canonical Qwen3 single-turn, non-thinking. ✓
- **`--chat` now auto-applies the instruct sampling defaults** (temp 0.7 / top_p 0.8 / top_k 20 unless
  overridden) — greedy at long `--max-tokens` loops and never emits EOS (the "won't stop" failure). Fixed.
- **The instruct repo ships no `tokenizer.json`** (only `merges.txt`) → `Llm::from_dir` would error at load;
  fixed by copying the base `qwen3-30b-a3b/tokenizer.json` (identical Qwen3 tokenizer) into the instruct dir.

## Highest risk

Testing the 256K-native instruct on the O(T_max) path and wrongly concluding the MODEL/runtime is slow, when
the attention is doing fake long-context work from token 1 (amplified ~8× by the GQA repeat). The path that
works at 1K will not survive a genuine long instruct generation (unusable + OOM). The flash-decode kernel is
the structural fix.
</content>
