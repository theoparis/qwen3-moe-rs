# MTP.1 — n-gram speculative-decode probe: PASSED (2026-07-01)

Plan §7 mandates an n-gram/prompt-lookup probe FIRST to prove the speculative-decode **machinery**
(verify-batch + KV rollback + GDN-state rollback) before the full trained MTP block. **DONE + verified on
the real Qwen3.6-35B hybrid.**

## Result (`examples/qwen35_mtp_ngram.rs`, MAX_NEW_TOKENS=48 K=4, prompt "The capital of France is")
```
TOKEN_IDENTITY PASS matched=48/48
SPEC_STATS drafted=35 accepted=3 acceptance_rate=0.086 steps=45 verify_batches=12 verify_tokens=35
```
- **Cardinal invariant holds:** spec-decode output is TOKEN-IDENTICAL to plain bf16 greedy (48/48).
- **Machinery exercised:** 12 verify-batches (M>1 forward) with both accept (3) and reject (32) paths →
  the dual rollback ran repeatedly and never corrupted the output.
- Low acceptance (8.6%) is expected for a trivial n-gram draft on a short factual prompt; irrelevant to the
  correctness goal.

## What this validates
- **verify-batch:** `forward_prec` over M>1 tokens (the union verify shape) is correct.
- **KV rollback:** `KVCache::rewind(filled)` — the hybrid full-attn path uses slice_assign + the 0..filled
  prefix, so rewind (no zeroing) is correct (the plan's `select_assign(Add)` accumulate hazard does NOT
  apply to the hybrid eager path; that path is the OLD dense model / captured path only — Opus F2).
- **GDN-state rollback (the single biggest MTP risk, Opus F1):** `GdnStateCache::snapshot()/restore()` +
  `Qwen3_5HybridCache::snapshot_gdn/restore_gdn` bit-exactly restore the GatedDeltaNet recurrent matrix S
  + short-conv ring. The corrected recipe — **rewind KV AND GDN to the SAME pos, then re-forward the
  committed tokens** (Opus-F1 option A) — produces token-identical output. A single pre-batch snapshot +
  KV-to-pos+acc (the original buggy recipe) would have diverged silently.

## Scope / next
- This is a CORRECTNESS probe: it re-forwards committed tokens after rollback (no perf win). That's the
  plan's intent (validate the machinery cheaply first).
- **MTP.2 (Phase-2, conditional):** the full trained MTP block (`mtp.layers.0.self_attn/mlp.experts/fc`) as
  the draft, K=2, measured acceptance + net tok/s. For a PERF win, GDN rollback should move to per-step
  checkpointing (Opus-F1 option B) to avoid the re-forward cost. Under CUDA-graph capture, the device-pos KV
  write must be an OVERWRITE, not select_assign(Add) (Opus F2).
- Primitives: `src/cache.rs` (rewind/snapshot/restore + 3 unit tests). Driver: `examples/qwen35_mtp_ngram.rs`.
