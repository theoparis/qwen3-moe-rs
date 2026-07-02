## RECIPE (binding — from vLLM qwen3_next_mtp.py:118-145, llm_base_proposer.py:644-808)
Draft ONE token given target state at position t:
0. h_t = the target model tower output at the LAST position, POST model.norm — i.e. exactly the
   lm_head input. Repo tap: the return of self.model.forward_prec(...) at src/qwen3_5/mod.rs
   ~:825-828, BEFORE linear3(lm_head). [B,1,2048]
1. e = embed_tokens(tok_{t+1}) via the MAIN model's embedding (mtp_use_dedicated_embeddings=
   false; tok_{t+1} = the just-sampled/committed next token). [B,1,2048]
2. e_n = mtp.pre_fc_norm_embedding(e); h_n = mtp.pre_fc_norm_hidden(h_t) — each separately,
   BOTH are (1+gamma) RMSNorm (the loader's set_norm +1.0 fold is VALIDATED CORRECT for every
   mtp.* norm — no loader change).
3. x = cat([e_n, h_n], dim=-1) — EMBED FIRST. [B,1,4096]
4. x = mtp.fc(x) (bias-free linear 4096->2048; loaded transposed via set_linear — use linear3
   or matmul per repo convention on the f32 stream).
5. x = mtp.layers[0].forward_decoder_with_cache(x, position t, &mtp_kv_cache, prec) — the MTP
   layer is a STANDARD Qwen3_5FullAttnLayer (full-attn + 256-expert MoE + shared expert, NO
   GDN); it runs at the TARGET's position ids over the drafter's OWN KVCache (a separate
   KVCache, same T_max budget). The repo layer returns residual-added output.
6. x = mtp.norm(x) — plain RmsNorm.forward on the layer output ((1+gamma) already folded).
7. logits = linear3(shared lm_head, x). argmax = draft token d_1. [B,1,vocab]
K>1 chaining (for later 2d — implement the fn signature to support it now): step k+1 feeds
hidden = the MTP block's OWN post-mtp.norm output from step k (NOT a re-tap of the target),
embeds the last draft token, position t+k. Always layers[0].
Position-0 embedding masking: NOT applied (Qwen3-Next convention; DeepSeek-only).
