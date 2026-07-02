# M-D: Qwen3.6-35B captured decode — LOCKED design (3-voice reviewed, 2026-07-01)

Review: Codex GPT-5.5-high + Gemini 3.1 Pro-high + Opus 4.8-high (repo-grounded), all
AGREE-WITH-CHANGES on packet R2; changes folded in below. Architecture: **Option C** (unanimous) —
parallel static entrypoints beside the untouched eager paths (repo precedent: `Qwen3Attention`
eager vs `_static_pre` siblings). Option B (unify in-place + deep-copy snapshots) REJECTED 3/3:
would mutate the MTP.1-verified functional semantics and tax the eager path.

## The invariant conflict (accepted premise)
CUDA-graph replay requires every persistent buffer the graph touches to keep a STABLE VA across
replays (in-place writes only). MTP snapshot/rollback relies on cheap handle-clone snapshots +
functional replace (`set_state`), where in-place writes under a live shared handle either COW-move
the VA (breaking the graph) or corrupt the snapshot. These cannot share one API; they get two.

## Components

### 1. GdnStateCache static mode (src/cache.rs, additive)
- `init_static(&mut self, device)`: allocate the S buffer `[B, Hv, Dk, Dv]` f32 zeros ONCE; set
  `static_mode = true`. Conv ring is already in-place (`push_conv`, constant-range shift).
- `step_static(...)`: recurrent update reading prev S via CLONE (never `take()` — the buffer must
  survive), computing `new_state` into scratch, then `drop(prev_clone)` BEFORE a full-range
  `take()+slice_assign` copy-back into the persistent buffer (read-before-write discipline;
  Codex R2-F3, Opus R2-F4). NO ping-pong double-buffering — a captured graph bakes BOTH read and
  write VAs (Opus R2-F4). Cost accepted: ~2 MB × 30 layers ≈ 60 MB/step memcpy.
- **Static GDN prefill (Opus R2-F1, CRITICAL)**: `forward_prefill_recurrent` ends in
  `set_state`-replace → re-prefill after `reset_for_replay` MOVES the S VA and breaks the graph.
  Add a static prefill variant that writes its FINAL state in place into the `init_static` buffer
  (interior prefill steps may stay functional; only the final write must land in-place).
- Snapshot guard (Gemini R2-F3, Codex R2-F1, Opus R2-F5): `snapshot()` hard-`assert!` (release-
  visible) `!static_mode`. Mechanical enforcement: cache holds `Rc<()>`; snapshots would hold a
  clone; `step_static` asserts `Rc::strong_count == 1`. (Burn exposes no handle refcount; VA drift
  is only detectable after the fact via VaSnapshot — the Rc token makes the contract a real
  precondition.)
- `reset_for_replay`: zero S + conv IN PLACE (VA kept) — exists (cache.rs:402-409), keep.

### 2. Static full-attn step (src/qwen3_5/mod.rs, new fn on Qwen3_5FullAttention)
NOT a drop-in of the 30B template (Opus R2-F9): Qwen3.5 needs **partial rotary** (rotary_dim =
head_dim/4 = 64 → `rope_freqs(64, 1e7)` + `apply_rope_partial` with `compute_rope_embeddings_pre`),
the packed q+gate projection with `sigmoid(output_gate)` multiply AFTER movedim, and `ql3` fp8
projections. Structure from `Qwen3Attention::forward_with_cache_static_pre_lp` (attention.rs:465):
`KVCache::update_static` (device-pos select_assign(Add) into fixed `[B,Hkv,T_max,256]`) +
`arange_tmax > pos` −inf mask + dense masked SDPA over T_max. NO flash-decode in the captured
step v1 (host pos-branch + no device length bound; T9 later; eager keeps flash). The step never
reads `seq_len()`/`filled` (verified safe, Opus R2-F10).

### 3. Static MoE step (src/qwen3_5/mod.rs, new static-only method)
Fused device-routed branch only. **Hoist `assign_tok`** (`Tensor::arange` at mod.rs:878/925 is a
per-step H2D staging = capture poison; Opus R2-F2) — precompute once outside the step (depends
only on tokens/top_k), mirroring the 30B `decode_topk_pre` hoisting (moe_decode.rs:142-147).
Preflight (build-time, BEFORE eager prefill/warmup) release-visible `assert!`: fp8 sidecar present,
stacks non-placeholder, T ≤ fused max — so capture cannot silently hit the host-loop fallback
(mod.rs:953). Panic message dumps routing shape/dtype/T (Gemini R2-F5). The shared eager
`forward_impl` keeps its fallback untouched (Opus R2-F6).

### 4. Step contract (forward_decode_static_pre)
`(tok [B,1] Int, pos [1] Int DEVICE, &mut Qwen3_5HybridCache, prec, freqs64, arange_tmax,
assign_tok)` → writes f32 logits into `DecodeState.last` in place (write_last_in_place already
casts; no new bf16→f32 slice_assign site — Opus R2-F10). PROHIBITED inside the step:
`into_data/to_vec/into_scalar` (D2H), `from_data/from_floats/arange` (H2D staging), env reads,
shape changes, `set_state`, pos-dependent host-range slice_assign (constant-range is fine).

### 5. VaSnapshot extension (src/capture.rs) — HARD G3 PRECONDITION (Opus R2-F3)
Current VaSnapshot is hardcoded to ModelCache KV (capture.rs:115-140) — GDN-BLIND. Extend (generic
or new struct) to enumerate: per-full-layer KV VAs, per-GDN-layer S + conv VAs, tok/pos/finished/
last/emit VAs, over `Qwen3_5HybridCache` (hybrid reset_for_replay exists, cache.rs:541-548).
Also assert allocator telemetry: memory_usage() unchanged across replays (zero new allocations —
Gemini R2-F4, Codex R2-F2); log arena_bytes.

### 6. Capture driver (examples/cudagraph_qwen35_decode_bench.rs)
Mirror cudagraph_moe_decode_bench + vllm_infer --capture: eager prefill closure (variable-shape
KV update cols 0..lp + static GDN prefill final-state in-place write) → captured step closure
(device argmax → EOS mask_where/bool_or → scatter_emit_to_tok → forward_decode_static_pre →
write_last_in_place) → CapturedDecoder::build/decode_n. v1: ONE bucket T_max=1024,
`assert!(prompt_len + max_new <= T_max)` at build (Opus R2-F8). K=1; shaped so K can be added
later (NOT currently parameterized — Opus R2-F7). Warmup ≥ 3 (autotune keys cached pre-capture).

## Gate ladder (revised per R2)
- G1  Fusion-vs-raw free-run parity — RAN: 0-6 identical, diverges @7 (coherent alt continuation).
- G1b Teacher-forced cross-backend gate (D6 methodology): forced 21-tok sequence, per-position
      top1/margin/top5 both backends. BENIGN if raw top1 agrees at all confident positions and
      mismatches are small-margin ties; else bisect (QWEN35_DEBUG_LAYERS) before proceeding.
- G2a VA/COW microgate (Codex R2-F5): unit test — N step_static calls, S/conv VAs stable; negative
      test — snapshot() in static mode panics (Rc token).
- MTP.1 RE-GATE EARLY (Codex R2-F4): run examples/qwen35_mtp_ngram.rs 48/48 immediately after the
      cache.rs additive changes land (again at the end if cache.rs is touched later).
- G2  Eager-driven static-step parity: drive forward_decode_static_pre in a host loop ≥64 tok,
      token-identical vs raw eager (T1 freerun raw tokens), bf16 AND fp8; report max/mean logit
      delta alongside identity (Codex R2-F6).
- G3  Captured: replay ≥48 tok token-identical vs G2; VaSnapshot(+GDN) clean; zero-alloc assert;
      SECOND-PROMPT leg: reset_for_replay → re-prefill → re-capture-or-replay, compare vs a FRESH
      static cache run (content test, not just VA — Codex R2-F8).
- G5  fp8-under-capture: sidecars quantized on the raw backend (build LAST, no to_device after —
      burn-module-skip-workaround), captured fp8 token-identical vs eager fp8, tok/s vs the 4.85
      eager baseline = THE M-D HEADLINE.

## Task order
T1b (teacher-forced gate, in flight) → T2 cache.rs static mode + static GDN prefill + G2a tests +
MTP.1 re-gate → T3 static full-attn (partial-rope pre) → T4 static MoE (hoisted assign_tok +
preflight) + forward_decode_static_pre → G2 → T5 VaSnapshot extension → T6 capture driver → G3 →
T7 fp8 quant generalization → G5. Coding: Codex subagents (Gemini agentic mode flaky today);
review per task; GPU gates hands-on, serialized.
