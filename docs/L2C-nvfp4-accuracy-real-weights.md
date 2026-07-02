# L2C — NVFP4 4-bit accuracy on REAL Qwen3.6-35B weights (D6 pre-gate finding)

CPU round-trip of the NVFP4 codec (E2M1 4-bit + per-16 E4M3 block scale + FP32 global scale, the
`src/nvfp4.rs` algorithm) on real checkpoint tensors, per-column / per-16-block, **naive amax scale
(no calibration)**:

| tensor | shape | cosine | max rel-err |
|---|---|---|---|
| L0 GDN out_proj | [2048,4096] | ~0.9955 | 0.111 |
| L3 attn q_proj | [8192,2048] | ~0.9955 | 0.064 |
| L3 attn o_proj | [2048,4096] | ~0.9956 | 0.086 |
| L0 shared_expert gate | [512,2048] | ~0.9956 | 0.142 |
| lm_head | [248320,2048] | ~0.9955 | 0.076 |

(numpy codec replication; validated ~0.994 vs the Rust probe's ~0.999 on random data, i.e. slightly
pessimistic — true per-tensor cosine ≈ 0.995–0.996.)

## Read (informs the L2C plan)
- Naive NVFP4-4bit preserves each tensor at **~0.995–0.996 cosine** — **borderline**, below the ~0.999
  clean-pass bar. Compounded over 40 layers this will likely **shift some greedy token picks**, so naive
  per-tensor NVFP4 is unlikely to pass the D6 **token-identity** gate on its own.
- ⇒ The plan's D6 design is validated: NVFP4 must be **calibration-gated** (AWQ/percentile scale selection
  beats raw amax, esp. for the higher-rel-err tensors like the shared gate at 0.14) **with a per-tensor FP8
  (w8a16) fallback** for tensors that fail token-parity. Do NOT ship blanket NVFP4.
- Practical next step for L2C: (1) implement the token-identity gate (run the model with per-tensor NVFP4
  vs bf16 on a fixed prompt set, keep NVFP4 only where the greedy ids match), (2) add calibration to the
  codec's scale selection, (3) keep sensitive tensors (lm_head? layernorms are already bf16) in FP8/bf16.
