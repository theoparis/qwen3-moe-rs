# Architecture

Qwen3 is a decoder-only transformer. This crate implements it in Burn with a focus on (1) a
batch-safe `Linear` for the CubeCL CUDA backend and (2) bf16 mixed-precision training. The model
code in `src/` is backend-generic; the bf16 + batch-safety work targets the CUDA backend.

## Module layout

| File | Responsibility |
|------|----------------|
| `src/decoder.rs` | `Qwen3Config`, `Qwen3Model`, `Qwen3DecoderLayer`, `Qwen3MLP` (SwiGLU), `Qwen3ForCausalLM` (tied LM head, generation) |
| `src/attention.rs` | Grouped-query attention with QK-norm + RoPE; reference scaled-dot-product attention |
| `src/linear2d.rs` | `Precision` enum + `linear3` — the batch-safe / bf16 Linear seam |
| `src/rope.rs` | Rotary position embeddings (θ = 1e6) |
| `src/cache.rs` | KV cache for autoregressive generation |
| `src/load.rs` | Hugging Face `safetensors` weight loading |
| `src/tokenizer.rs` | Tokenizer wrapper (`tokenizers` crate) |

## Forward data flow (one decoder layer)

```
x ─► RMSNorm ─► self-attn ─────────────────────────► + residual ─► ...
                │  q/k/v_proj = linear3(prec)            ▲
                │  reshape + q_norm/k_norm (f32)          │
                │  RoPE (f32)                             │
                │  scaled-dot-product attn (softmax f32)  │
                │  o_proj = linear3(prec) ────────────────┘
x ─► RMSNorm ─► MLP (SwiGLU) ───────────────────────► + residual
                │  gate = silu(linear3(prec))
                │  up   = linear3(prec)
                │  down = linear3(prec)  on (gate * up)

final RMSNorm ─► tied logits: hidden @ embed_tokens.weightᵀ (f32) ─► CrossEntropyLoss
```

Every transformer projection (the `nn::Linear` layers: q/k/v/o, gate/up/down) goes through
`linear3`. The LM head is the exception: it is the tied input embedding (no separate parameter), and
`tied_logits` projects `hidden @ embed_tokens.weightᵀ` directly in f32 (outside `linear3`), so the
full LM gradient flows back to the single shared matrix.

## The batch-safe Linear (`linear3`)

`nn::Linear::forward` on a 3-D input `[B, S, K]` lowers to a **broadcast batched matmul**:
`[B, S, K] @ [1, K, N]` (the weight is unsqueezed to `[1, K, N]`). On the CubeCL CUDA backend, for
some `(S, K, N)` shapes and autotune states, this returns **wrong values for batch > 1** — the rows
past the first are corrupted (finite, no error/NaN, so it's silent). It reproduces at common
transformer shapes, so batched training/inference can silently produce wrong gradients/logits.

`linear3` avoids it by flattening the leading dims and doing a plain **2-D GEMM**:

```rust
// [B, S, K] -> [B*S, K] @ [K, N] -> [B, S, N]   (mathematically identical; a Linear is per-token)
let x2 = x.reshape([batch * seq, d_in]);
let y2 = x2.matmul(weight);            // no broadcast batch dim => correct on every tested shape
y2.reshape([batch, seq, d_out])
```

`matmul_probe` demonstrates both: the 3-D broadcast path diverges (`|row0-row1|` up to several units
for identical input rows) while the 2-D path is exactly `0.0`.

## bf16 mixed precision (`Precision`)

`linear3` takes a `Precision` (`F32` default, or `Bf16`). With `Bf16` it casts both operands to bf16,
matmuls (the CUDA matmul accumulates in f32), and widens the output back to f32. Everything else —
RMSNorm, softmax, residual, the LM head, master weights, the optimizer — stays f32. Precision is
threaded down the forward chain from two fields on `Qwen3ForCausalLM`: `train_precision` (used by
`forward`) and `infer_precision` (used by `forward_with_cache` / `generate`, default `F32`), so bf16
training is decoupled from inference. Full design + verification: [BF16.md](BF16.md).

## Attention

Grouped-query attention: `num_attention_heads` query heads, fewer `num_key_value_heads` (expanded by
repetition). QK-norm (RMSNorm over `head_dim`) is applied to Q and K before RoPE. The implementation
uses the **reference** scaled-dot-product attention (not the fused kernel) because the fused kernel
mishandles a broadcast causal mask on this CUDA backend. Attention scores + softmax run in f32.
