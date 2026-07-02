# P0.1 — Qwen3.6-35B-A3B weight inventory + loader/module map

From `model.safetensors.index.json` (1045 tensors, 71.90 GB, 26 shards). This is the concrete contract for the
Lane-1 port (loader L1.1, full-attn L1.2, GDN L1.3, shared-MoE L1.4, MTP L1.5). Config in `docs/QUANT_FLASH_SPEC_PLAN.md` §1.

## Top-level namespaces
- `model.language_model.*` — the 40-layer text tower (the part we run).
- `model.visual.*` — 27-block ViT + merger (**skip for text-only inference**; 333 tensors).
- `mtp.*` — the multi-token-prediction block (19 tensors).
- `lm_head.weight`, `model.language_model.embed_tokens.weight` — untied (`tie_word_embeddings: false`).

## Per-layer structure (40 layers; `layer_types` = 3 linear : 1 full, repeating)

### GDN / linear-attention layers (30) — `model.language_model.layers.N.linear_attn.*`
| weight | role (Gated DeltaNet) |
|---|---|
| `in_proj_qkv.weight` | projects hidden → q,k,v for the delta rule |
| `in_proj_a.weight`, `in_proj_b.weight` | the `a` (decay gate) and `b` (write/beta gate) projections |
| `in_proj_z.weight` | output gate `z` (gated-norm gating) |
| `A_log` | log-parameterized SSM decay (Mamba-style; `a = exp(−softplus(...)·exp(A_log))`) |
| `dt_bias` | timestep (Δt) bias |
| `conv1d.weight` | depthwise short causal conv, kernel `linear_conv_kernel_dim=4` (over q/k/v) |
| `norm.weight` | gated RMSNorm on the readout |
| `out_proj.weight` | output projection |
+ `input_layernorm.weight`, `post_attention_layernorm.weight` (per layer).
**Decode state to carry (capture-stable buffers):** the recurrent matrix `S` per head + the conv ring (last 3
inputs). Maps to candle `mamba2.rs` (A_log/dt/conv carry) + `rwkv_v7.rs` (delta-rule update) + `lfm2.rs` (hybrid
cache). Authoritative math: HF `modeling_qwen3_next` / FLA.

### Full-attention layers (10) — `model.language_model.layers.N.self_attn.*`
`q_proj`, `k_proj`, `v_proj`, `o_proj` + `q_norm`, `k_norm` (QK-RMSNorm). GQA 16q/2kv, head_dim 256,
partial_rotary 0.25, mRoPE. Matches candle `qwen3.rs`. Flash-decode (Lane 2A) targets these.

### MoE (every layer) — `model.language_model.layers.N.mlp.*`
| weight | role |
|---|---|
| `gate.weight` | router (hidden → 256) |
| `experts.gate_up_proj`, `experts.down_proj` | **fused** routed-expert weights (gate_up stacked) — note the grouped/fused layout (no per-expert split in names) |
| `shared_expert.gate_proj/up_proj/down_proj.weight` | the always-on shared expert (SwiGLU) |
| `shared_expert_gate.weight` | sigmoid gate scaling the shared-expert output |
Top-8 of 256 routed + shared. Matches candle `qwen2_moe.rs:229-254` wiring. Extend `src/moe*.rs` (128→256 +
shared); note `experts.gate_up_proj`/`down_proj` are **fused stacks** — fits the existing persistent-stack
gather-GEMV (`moe_grouped.rs`).

### MTP block (1) — `mtp.*`
`mtp.pre_fc_norm_embedding.weight` + `mtp.pre_fc_norm_hidden.weight` → `mtp.fc.weight` (fuses the token
embedding ⊕ the last hidden state — EAGLE-style), then a full layer: `mtp.layers.0.{input_layernorm,
self_attn.{q,k,v,o}_proj + q/k_norm, post_attention_layernorm, mlp.{gate, experts.gate_up_proj/down_proj,
shared_expert.*, shared_expert_gate}}`, `mtp.norm`, sharing `lm_head`. **Confirms MTP is a full trained block,
not a logits head** (Codex F6). Draft path: `fc(norm(embed) ⊕ norm(hidden)) → attn+MoE layer → norm → lm_head`.

## Loader implications (L1.1)
- Strip `model.language_model.` prefix → layer modules; **skip `model.visual.*`** for text-only.
- Dispatch per layer on `config.layer_types[N]` (linear_attention vs full_attention).
- `experts.gate_up_proj`/`down_proj` are fused grouped tensors → load straight into the persistent expert stacks.
- 71.9 GB bf16 → ~36 GB at fp8 / ~18 GB at NVFP4 (the quant lever's payoff on this model; experts dominate).
