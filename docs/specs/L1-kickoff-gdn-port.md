# Lane 1 kickoff — Qwen3.6-35B-A3B hybrid architecture port (the multi-week core)

Phase 0 is essentially complete (P0.1/P0.2/P0.3/P0.3b done+reviewed; P0.3c cubecl fix closing). Lane 1 is the
bulk: make the engine LOAD + RUN Qwen3.6-35B-A3B (text-only, greedy parity vs HF). Operating decisions baked in:
prefill **bf16**, decode **NVFP4-SIMT** (later), MTP/NVFP4-TC deferred. Grounded in the verified weight map
(`docs/qwen36-weights-P0.1.md`) + candle refs (`docs/QUANT_FLASH_SPEC_PLAN.md` §8a).

## Increment order (each = one coding-subagent task + 3-voice review on the critical ones)

### L1.1 — config + loader + text-tower extraction (FIRST; non-GPU-heavy, unblocks all)
- New `Qwen3_5MoeConfig` (parse the verified `config.json`: 40 layers, `layer_types`, 256 experts top-8,
  shared expert, head_dim 256, partial_rotary 0.25, mRoPE `[11,11,10]`, `mtp_num_hidden_layers:1`).
- Sharded safetensors loader: strip `model.language_model.` prefix; **skip `model.visual.*`** (text-only);
  load `experts.gate_up_proj`/`down_proj` (fused) into the persistent stacks; per-layer dispatch on
  `layer_types[N]` (linear_attention vs full_attention).
- **Gate:** loads all 26 shards without error; tensor shapes match config; param count ≈ 35B.

### L1.2 — full-attention layer (10/40) — reuse, low risk
- head_dim 256, GQA 16q/2kv, **partial_rotary 0.25** (rotate first 64 of 256; ref candle `phi3.rs`/`glm4_new.rs`),
  interleaved **mRoPE** (greenfield — `mrope_section [11,11,10]`; extend `src/rope.rs`), q/k-norm. Reuse
  `src/attention.rs` structure + candle `qwen3.rs` wiring.
- **Gate:** one full-attn layer matches an HF reference forward (cosine) on a fixed input.

### L1.3 — Gated DeltaNet linear-attention decode (30/40) ★ THE hard kernel
- Weights: `in_proj_qkv/a/b/z`, `A_log`, `dt_bias`, `conv1d.weight`, `norm.weight`, `out_proj.weight`.
- Decode recurrence (port from candle `rwkv_v7.rs:405-430` delta-rule + `mamba2.rs:262-324` conv/state;
  authoritative math = HF `modeling_qwen3_next` / FLA): per head, f32 state `S[d_k,d_v]`:
  short causal conv (kernel 4) on q/k/v → `S_t = a·S_{t-1} + b·(k⊗(v − Sᵀk))` (gated delta rule; `a` from
  `A_log`+`dt_bias` softplus decay, `b` the write gate) → `o = norm(Sᵀq) ⊙ gate(z)` → `out_proj`.
- **State cache** alongside KV (ref candle `lfm2.rs:142-201` hybrid cache): persistent per-head `S` matrix +
  conv ring buffer; capture-stable (in-place update like the KV cache). Prefill builds `S` via the chunked
  form or a sequential warmup.
- **Gate:** recurrent decode == chunked prefill at the boundary; cosine vs an HF/FLA CPU reference; f32 state
  accumulation (`mamba_ssm_dtype: float32`). **This is the critical-item 3-voice review target of Lane 1.**

### L1.4 — shared-expert MoE + 256 routing
- Extend `src/moe*.rs` 128→256 + add the always-on shared expert (`shared_expert.{gate,up,down}_proj` +
  sigmoid `shared_expert_gate`); ref candle `qwen2_moe.rs:229-254`. Fused gather-GEMV generalizes.

### L1.5 — MTP block (later; the spec-decode lever)
- Full attn+MoE layer + `fc` + `pre_fc_norm_{embedding,hidden}` (EAGLE-style embed⊕hidden fusion) + `mtp.norm`,
  shares `lm_head`. (Draft path; verify/rollback per the MTP plan — n-gram probe first.)

### L1.6 — end-to-end greedy parity
- text-only greedy output matches HF transformers on a fixed prompt (the Lane-1 acceptance gate) before any
  35B perf work.

## Coding-subagent dispatch model (per the user's directive)
Each increment → a Codex (or Gemini agy-sub) coding subagent with the weight map + candle file refs + the gate;
I build/run on the GB10; critical items (L1.3 GDN, L1.6 parity) get the Codex+Gemini+Opus 3-voice review.
**Serialize 35B runs** (memory note: 35B is memory-heavy). Build on 30B-comparable shapes where possible first.

## Honest scope
L1.1/L1.2/L1.4 are days each; **L1.3 (GDN) is the multi-week heart** (new recurrent kernel + state cache +
capture-safety + numerics). This is why D5 chose architecture-first. Lane 2 (flash-decode, capture-in-vllm_infer,
NVFP4-decode-SIMT) runs in parallel on 30B and re-applies to 35B after L1 lands.

## L1.1 CLOSED (Opus GO + gate-verified) — CONTRACTS the forwards MUST honor
L1.1 loader+skeleton verified: `L1.1 LOAD-VERIFY: PASS` (712 mapped / 333 visual-skipped / 0 missing / 0 orphan
/ 0 mismatch / 35.505B params). Opus 4.8 authoritative review (read load.rs + weight doc): **GO — semantically
correct to build forwards on** (no a↔b swap; layer dispatch gate-proven exact; vision-skip clean; no name→field
collisions). (Codex review degraded by a sandbox read-failure; Gemini review failed on a foreground-timeout —
not re-run, since the gate + Opus GO settle a skeleton; full 3-voice resumes on L1.3/L1.6.)

**Carry-forward contracts (silent-wrong-model traps — a shape-check is blind to these):**
1. **Fused-expert orientation (L1.4, #1 risk):** `experts.gate_up_proj` is `[E, 2I, H]`, `experts.down_proj`
   is `[E, H, I]` — loaded **un-transposed = PyTorch `[E, out, in]`**, the OPPOSITE handedness of the existing
   `moe_grouped.rs` stacks (`[E, in, out]`). The L1.4 forward MUST use the `[out,in]` grouped-GEMM convention
   and split `[gate; up]` along dim-1 (first `I`=gate, next `I`=up). **Reusing the old gather-GEMV as-is →
   silent wrong output, zero shape errors.**
2. **`attn_output_gate: true` (L1.2):** `q_proj` out-dim is `num_heads·head_dim·2 = 8192` — it emits
   `[query; output-gate]`. The full-attn forward must split q into query + gate and apply the (sigmoid) output
   gate to the attention output. `o_proj` in-dim is the un-gated `4096`.
3. **mRoPE interleaved (L1.2):** `mrope_interleaved: true` is parsed-then-dropped from the config struct — the
   L1.2 forward must hardcode interleaved mRoPE with `mrope_section [11,11,10]`, `partial_rotary 0.25` (rotate
   the first 64 of 256 head dims).

## L1.2 REVIEW — one confirmed bug (Opus, 95%), fix queued as L1.2-fix
Opus contract-check verdict: contracts 2-5 CORRECT (QK-norm-before-RoPE ✓, partial-RoPE first-64 θ=1e7 ✓,
GQA 2→16 kv-major ✓, causal + scale=head_dim^-0.5 ✓). **Contract #1 (attn_output_gate) BROKEN:** HF views
`q_proj` out as `[B,S,16,512]` then `chunk(2, dim=-1)` → **per-head interleaved** `[query(256); gate(256)]`
within each head. Current code does a **block split** (`flat[0:4096]`=query / `[4096:8192]`=gate) → scrambles
both tensors (even heads=queries 0-7, odd=gates). Shape-correct → smoke PASSed → would FAIL L1.6 parity.
**Fix:** reshape to `[B,S,num_heads,2*head_dim]` FIRST, then slice the last dim `0..head_dim`=query,
`head_dim..2*head_dim`=gate. Apply after L1.3 (mod.rs locked by the L1.3 coder). Also: forward hardcodes
PARTIAL_ROTARY_FACTOR/ROPE_THETA as consts (fine for default config, ignores overrides — minor).
Residual to re-verify at L1.6: mRoPE collapses to 1D RoPE for text (correct) but diverges for vision positions.

## L1.4 MoE — 3-voice GO (contract CORRECT vs HF). Two L1.6 watch-items:
Opus verbatim-verified all 5 axes vs HF: fused [E,out,in] + x·Wᵀ ✓, gate-first BLOCK split (HF chunk(2)
contiguous, not interleaved) ✓, softmax→top-k→renorm (softmax before top-k, on probs) ✓, sigmoid-gated shared
expert ADDED to routed ✓, no orientation mixup ✓. Self-consistent gate (cosine 0.99999970) confirms a RIGHT
contract (not a shared bug, unlike GDN). WATCH at L1.6:
1. PRODUCTION-PATH ORIENTATION: the L1.4 reference forward consumes HF-native [E,out,in]; but the production
   decode kernel `src/moe_grouped.rs` expects TRANSPOSED [E,in,out] (gate/up as two separate [E,H,I] stacks,
   down [E,I,H]). Wiring real fused `gate_up_proj [E,2I,H]` into the gather-GEMV needs per-expert block-split
   (gate=rows[0:I], up=[I:2I]) + transpose. Verbatim drop-in = silent-wrong. Single-expert numeric probe at L1.6.
2. Router gate forced F32 in Rust vs HF bf16-then-upcast-in-softmax → may flip a borderline 8th-expert pick on
   real bf16 weights. Minor top-k divergence to watch.
