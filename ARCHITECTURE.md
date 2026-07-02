# Architecture

A beginner-friendly tour of `qwen3-moe-rs` (crate name: `qwen3-burn`): what the pieces are and how a request flows
through them. Each section links its deep-dive plan under `docs/`.

`qwen3-moe-rs` is a Rust implementation of Alibaba's **Qwen3** large language models,
built on the **Burn** deep-learning framework instead of PyTorch. It began as a dense
text-generation engine and grew two large additions: a **GRPO** reinforcement-learning
trainer whose training loop runs natively in Rust/Burn, and **Mixture-of-Experts
(MoE)** inference for the larger Qwen3-30B and Qwen3.6-35B models, made to run on a
single NVIDIA GB10 (Grace-Blackwell, 128 GB unified memory). On top of that sit a
quantization stack (FP8 and NVFP4), CUDA-graph capture for faster decoding, and
`qwen-serve`, an OpenAI-compatible HTTP server. (Source of record:
`.release-notes/EXPLORATION.md`.)

Jargon used below, defined once:

- **MoE (Mixture of Experts):** a layer that holds many small feed-forward
  networks ("experts") but runs only a few per token, chosen by a small router.
  Qwen3-30B has 128 experts and runs the top-8 per token, so only ~3.3B of its
  ~30B parameters are active at each step.
- **Quantization:** storing weights in fewer bits (8- or 4-bit) instead of 16-bit,
  to fit a bigger model in the same memory and read it from memory faster.
- **SSE (Server-Sent Events):** a one-way HTTP stream the server uses to push
  generated tokens to the client as they are produced.

The MIT-licensed crate (imported as `qwen3_burn`) targets NVIDIA GPUs (sm_120 /
sm_121) through Burn's CubeCL backend, while the core model code stays
backend-generic and also runs on CPU.

---

## The big picture

### Serving a request (`qwen-serve`)

The server runs **one model per process** and decodes **one request at a time**
(FIFO). Because the Burn model type is `!Sync` (not safe to share across threads) and
its weights are the expensive resource, the model is loaded once inside a dedicated OS
thread and never leaves it; the async HTTP handlers only ever hold a cheap
`Send + Sync` channel handle (`src/serve/engine.rs`, `src/serve/handlers.rs`).

```
 HTTP client ── POST /v1/chat/completions | /v1/completions ; GET /v1/models /health
    │
    ▼  axum 0.8 handlers   (src/serve/handlers.rs, api.rs)  validate → render chat
    │                      template (minijinja+pycompat, byte-identical to HF) → parse
    ▼  engine channel      (bounded mpsc, cap 4) ───────────────── backpressure ──┐
    ▼  engine thread       (src/serve/engine.rs)  owns the !Sync model (loaded     │
    │                      once); fresh KV cache → prefill → per-token sample loop  │
    ▼  Burn/CubeCL CUDA backend → NVIDIA GPU (GB10, sm_121)  returns one token id   │
    ▼  detok + think-split (src/serve/detok.rs)  two holdbacks GATE emission        │
    ▼  frame channel       (bounded mpsc, cap 8) ──────────────────────────────────┘
    ▼  ReceiverStream → axum SSE → HTTP client  (streamed tokens)
```

Tokens flow forward to the GPU and back as SSE frames. Both channels are bounded, so
a slow client throttles the producer instead of growing memory, and a client
disconnect cascades down the same chain to stop decoding within one token (detailed
under "The serving layer"). Design of record: `docs/SERVE_PLAN.md`.

### The GRPO training loop

GRPO (Group Relative Policy Optimization, from DeepSeekMath) fine-tunes a model by
sampling several answers per prompt, scoring them with a reward function, and
nudging the model toward the better-than-average answers. The whole loop runs in
Rust/Burn (`src/grpo/`).

```
  prompts
    │
    ▼  rollout   (src/grpo/rollout.rs)  sample G answers/prompt on-device, record each token's log-prob
    ▼  reward    (src/grpo/reward.rs)   score each answer (Manim reward = Python subprocess; any error → 0.0)
    ▼  advantage (src/grpo/loss.rs)     group_norm: subtract the group mean, divide by the group std
    ▼  log-probs (src/grpo/logprob.rs)  policy + reference, gather − logsumexp (no [B,T,vocab] softmax)
    ▼  loss      (src/grpo/loss.rs)     clipped PPO surrogate + k3 KL to the reference
    ▼  optimizer (src/grpo/trainer.rs)  backward → AdamW step → updated policy ──┐
    │                                                                            │
    └──────────────────────────── next batch ────────────────────────────────────┘
```

Design of record: `docs/GRPO_PLAN.md`.

---

## The inference engines

There are **two concrete model families**, wired as two explicit code paths rather
than behind one generic abstraction. This is deliberate: the two architectures
differ enough that a shared trait would hide more than it saves, so the engine
thread simply branches on which model it loaded (`src/serve/engine.rs`).

- **Qwen3-30B-A3B** — a standard MoE transformer: 48 layers, 128 experts, top-8
  routed per token, no shared expert (`src/moe.rs`; `Qwen3MoeForCausalLM`). The
  server drives it with the fused static-decode path. Forward logits match HF
  transformers to **cosine > 0.9999** (`docs/MOE_PLAN.md`).
- **Qwen3.6-35B-A3B** — a hybrid tower that mixes **Gated-DeltaNet** layers with
  full-attention layers, plus multi-token prediction (`src/qwen3_5/mod.rs`;
  `Qwen3_5MoeForCausalLM`). The server drives it with a `forward_last_logits` loop.
  Design: `docs/QUANT_FLASH_SPEC_PLAN.md`, `docs/specs/L1*`.

The dense base model (`src/decoder.rs`; `Qwen3ForCausalLM`, `Qwen3Model`) underlies
both and is what the GRPO trainer fine-tunes; it uses Grouped-Query Attention, RoPE,
RMSNorm, SwiGLU, and QK-norm (`src/lib.rs`).

One low-level detail runs through everything: `src/linear2d.rs` (`linear3`) flattens
3-D input to a 2-D GEMM before every matmul, dodging a silent broadcast-batched-matmul
corruption bug on the sm_121 CubeCL backend, and carries bf16 mixed precision (`docs/BF16.md`).

---

## Quantization

Quantization shrinks the model and speeds up decode (bound by weight-read bandwidth). The release ships three precisions:

- **bf16** — 16-bit weights, the reference precision. 35B footprint ≈ 71 GB device
  memory (`docs/QUANT_FLASH_SPEC_PLAN.md`).
- **FP8 (W8A16, e4m3)** — 8-bit weights, 16-bit activations. Packed e4m3 bytes are
  read from GPU memory and dequantized inside the GEMM's load path, so the model is
  never expanded back to bf16 in memory. Kernel `src/w8a16.rs`, drop-in
  `src/w8a16_linear.rs` (`W8A16Linear`); validated vs a CPU oracle and OCP golden
  vectors to cosine > 0.999 (`docs/VLLM_KERNELS.md` §2). 35B footprint ≈ 40.3 GB.
- **NVFP4** — a 4-bit weight format. Each weight is an **E2M1** 4-bit value; every
  block of 16 weights shares an **E4M3** 8-bit block scale; and the whole tensor
  shares one **f32** global scale. The two-level scale is what lets 4-bit weights
  keep enough dynamic range to stay accurate. Host codec `src/nvfp4.rs`, drop-in
  `src/nvfp4_linear.rs` (`Nvfp4Linear`); golden vs the Python reference
  `scripts/nvfp4_reference_dequant.py` (`docs/specs/L2C-*`). 35B footprint ≈
  **22.5 GB** device in-use, and greedy decoding is **byte-identical to the bf16
  original** on the measured 16-token gate (`docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS).

The shared idea in FP8 and NVFP4 is **in-kernel dequant**: keep weights small in
memory and convert them to floats only in the GEMM's load path, where the cost hides
behind the memory read. The release can also load NVIDIA's own pre-quantized NVFP4
checkpoints — `src/nvidia_ckpt.rs` ingests the on-disk ModelOpt tensors for
`nvidia/Qwen3.6-35B-A3B-NVFP4`, bit-exact vs an external reference
(`docs/QUANT_FLASH_SPEC_PLAN.md` §M-B.5). Any quantization's accuracy is checked by
`src/quant_gate.rs`, which round-trips weights through the host codec and measures
reconstruction error on the model's normal path.

---

## CUDA-graph capture

At single-token decode, the GPU spends most of its time on **launch overhead** — the
fixed CPU-side cost of telling the GPU to start each operation — not on math. On the
30B MoE the naive decode ran at **0.73 tok/s**, ~16% of even the dense memory ceiling,
because it was launch-bound (`docs/PERF_80TOKS_PLAN.md` §1). A CUDA graph fixes this:
record the operations for one decode step once, then **replay** the whole recording as
a single launch on every later step.

`src/capture.rs` (`CapturedDecoder`) provides this harness. Capture requires a
fixed-shape, host-sync-free decode step, so it runs on the raw
`CubeBackend<CudaRuntime, f32, i32, u8>` below Burn Fusion, where the CUDA graph can
record the real launch list. Capture and replay are not in upstream CubeCL, so the
release vendors patched local copies: **`vendor/cubecl`** adds the capture/replay FFI
and **`vendor/cubek`** adds a device-seed RNG (so sampling inside the captured step
reads no host state). Both are redirected from their git pins via `[patch]` in
`Cargo.toml`, so the release is self-contained. Design: `docs/cudagraph/DESIGN.md`.
Measured: 30B reached **≈ 20.9 tok/s** captured (`docs/SERVE_PLAN.md`;
`docs/PERF_80TOKS_PLAN.md` reports the same run as 19.38 → 21.03 tok/s), and 35B
reached **11.78 tok/s** captured NVFP4 (`docs/QUANT_FLASH_SPEC_PLAN.md`). Capture is a
follow-up milestone: the v1 server serves the eager-static path, not these numbers.

---

## The serving layer

`qwen-serve` (bin `src/bin/qwen_serve.rs`, module `src/serve/`) exposes
`/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`. Five concerns
make up its host side, each solved to a strict correctness bar:

- **Byte-parity chat templates** (`src/serve/template.rs`). The prompt string a chat
  model actually sees is built by a Jinja template shipped with the model. The server
  reproduces HuggingFace's `apply_chat_template` **byte-for-byte** using minijinja
  plus a pycompat shim: it enables `trim_blocks`/`lstrip_blocks`, overrides `tojson`
  to match Python's `json.dumps(..., ensure_ascii=False)`, defines `raise_exception`,
  and — critically — preserves JSON object **key order** with an `OrderedJson` type,
  because this build's map types would otherwise sort keys and silently change the
  prompt. Gate: **12/12 byte-identical** vs HF transformers on both models
  (`tests/template_parity.rs`).
- **Incremental detokenization** (`src/serve/detok.rs`). Turning tokens back into
  text as they stream has two traps, and this module gates SSE emission behind both.
  A **UTF-8 holdback** holds a multi-byte character split across tokens until its
  final byte arrives (bounded: a real split completes within 4 tokens), so the client
  never sees a `U+FFFD` mojibake or half a character. A **stop-string holdback** holds
  any trailing text that could still grow into a stop string, so a stop sequence is
  never partially leaked before it matches. Both use a bounded-tail decode-and-diff:
  work is O(1) per token, never O(length²).
- **Think-tag splitting** (`src/serve/handlers.rs`). Qwen "thinking" output is wrapped
  in `<think>...</think>`. A small state machine routes text before the closing tag
  into the response's reasoning field and text after it into `delta.content`.
- **Cancel by channel drop** (`src/serve/engine.rs`, `handlers.rs`). There is no
  cancellation token. When the client disconnects, axum drops the SSE body; every
  stage's receiver drops in turn, the engine's next `blocking_send` fails, and
  decoding stops within one token. The engine also checks for a closed channel before
  tokenizing a queued request, so dead requests are skipped without work.
- **Backpressure** (`src/serve/handlers.rs`). The two bounded channels (frame cap 8,
  engine cap 4) mean a slow reader throttles the producer instead of growing memory.
  A 20-request sustained smoke test showed per-class throughput stable and RSS flat
  (`docs/SERVE_PLAN.md` GATE RESULTS).

Each request also gets a **fresh KV cache** dropped when it finishes, so no token from
one request can leak into the next, and `max_tokens` is always bounded by the process
`T_MAX` (`src/serve/engine.rs`).

---

## The GRPO trainer

GRPO the algorithm is **not ours** — it is DeepSeekMath's, and the math here
reproduces **OpenRLHF**'s implementation and is parity-tested against it. What is
built here is a native Rust/Burn implementation of the full loop (`src/grpo/`, diagram
above). Two details: device sampling in rollout (`src/sampling_device.rs`) copies back
only the `[N]` tokens and log-probs, not the `[N, vocab]` logits; and the Manim reward
is fail-safe — any spawn, exit, timeout, or parse error from the `a0/manim_reward.py`
subprocess (static-AST gate + staged partial credit) scores `0.0`. Parity is checked in
`tests/grpo_math.rs`, `grpo_rollout.rs`, `grpo_trainer.rs`, and `grpo_varprompt.rs`
against golden tensors in `tests/ref/grpo_expected.json`, from `a0/grpo_reference.py`.

**Honest boundaries.** The novelty claim is narrow: to the best of a mid-2026 search
(absence of evidence, not proof), this is the **first public GRPO LLM trainer whose
training loop runs natively in Rust/Burn** — rollout, forward and backward, advantage
and loss, KL, and the optimizer step. It is correct to say there is **no PyTorch in
the training loop**. It is **not** correct to say "zero Python" or "end-to-end Rust":
the Manim reward runs a Python subprocess, and only the `grpo_train` CPU convergence
smoke is Python-free. Manim-as-reward also has prior art (ManimTrainer). See
`docs/GRPO_PLAN.md`.

---

## Verification methodology

The distinguishing habit of this project is that almost every feature ships behind a
**parity gate** — a test that pins the Rust output to a trusted reference, often
**byte-for-byte or bit-for-bit**, not merely "close enough" — backed by adversarial
review batteries and runnable gate scripts. The gates on record:

- **GRPO math** — Rust vs Python golden JSON (`tests/ref/grpo_expected.json` from
  `a0/grpo_reference.py`); reproduces OpenRLHF within tolerance.
- **MoE forward** — logits and log-probs match HF transformers to **cosine > 0.9999**;
  18 lib tests cover routing parity, invariants, and determinism (`docs/MOE_PLAN.md`).
- **Chat template** — **12/12 byte-identical** vs HF transformers 5.12.1 on both
  models (`tests/template_parity.rs`).
- **Detokenization** — adversarial `U+FFFD` force-commit and stop-string boundary
  tests; emission is proven to be gated, never trailing, behind both holdbacks.
- **Quantization** — NVFP4 greedy decode **byte-identical to bf16**; host codec
  golden **bit-exact** vs the Python reference; FP8 kernel vs OCP golden vectors.
- **End-to-end server** — greedy output **byte-identical** to the `qwen35_generate`
  (35B) and `vllm_infer` (30B) examples over 16 tokens; non-stream equals
  streamed-concat; a 20-request sustained run shows memory flat
  (`docs/SERVE_PLAN.md` GATE RESULTS; `scripts/serve_gates/`).

There are **72 lib tests** in total (`docs/SERVE_PLAN.md`); none of the numbers above
were re-run while writing this document — each is quoted from the doc cited beside it.

---

## Module reference

One line per source module (`src/`). `(cuda)` / `(serve)` mark feature-gated modules.

| Module | What it is |
|--------|------------|
| `attention` | Grouped-Query Attention for Qwen3 (GQA, RoPE, QK-norm). |
| `cache` | KV cache and the MoE / hybrid decode-state caches for autoregressive generation. |
| `capture` | CUDA-graph capture/replay harness for fixed-shape static decode. |
| `cube_custom_op` *(cuda)* | Typed, safe wrapper around the Burn-Fusion custom-op bridge; foundation for the custom kernels. |
| `decoder` | The dense Qwen3 transformer: `Qwen3Config`, `Qwen3Model`, `Qwen3ForCausalLM`. |
| `flash_attn` *(cuda)* | Custom tiled online-softmax FlashAttention kernel (f32 accumulation). |
| `flash_decode` *(cuda)* | Split-K online-softmax flash-decode kernel, capture-ready. |
| `grpo` | The GRPO trainer: `rollout`, `reward`, `loss`, `logprob`, `trainer`. |
| `linear2d` | Batch-safe `Linear` (flatten-to-2D GEMM) + bf16 compute path; dodges the sm_121 batched-matmul bug. |
| `load` | Weight loading from HuggingFace safetensors (single-file and sharded). |
| `moe` | Qwen3-30B MoE model and sparse block (128 experts, top-8), the correctness oracle. |
| `moe_decode` | Capturable top-8 single-token MoE decode; reads only the 8 routed experts' weight slabs. |
| `moe_grouped` *(cuda)* | Dropless grouped-GEMM MoE fast path (vLLM `moe_align_block_size` layout). |
| `nvfp4` | Host NVFP4 codec: f32 ↔ E2M1 4-bit weight + E4M3 per-16 block scale + f32 global scale. |
| `nvfp4_linear` | Drop-in `Nvfp4Linear` weight-only quantized `Linear`. |
| `nvidia_ckpt` | Ingests NVIDIA ModelOpt NVFP4/FP8 checkpoints (`weight`, `weight_scale`, `weight_scale_2`). |
| `quant_gate` | Fake-quant PTQ accuracy gate: round-trips weights through the codec to measure error. |
| `qwen3_5` | Qwen3.6-35B hybrid MoE model (Gated-DeltaNet + full attention + MTP). |
| `rope` | Rotary Position Embeddings (theta = 1,000,000). |
| `sampling` | Host token sampling: temperature + top-k + top-p, then categorical draw. |
| `sampling_device` | Device-side sampling + raw log-prob for the GRPO rollout (copies back `[N]`, not `[N,V]`). |
| `serve` *(serve)* | The `qwen-serve` OpenAI server: `api`, `template`, `detok`, `engine`, `handlers`. |
| `tokenizer` | Qwen3 tokenizer wrapper over the `tokenizers` crate. |
| `w8a16` *(cuda)* | Fused FP8 W8A16 (e4m3 weight-only) GEMM: dequant in the GEMM load path. |
| `w8a16_linear` *(cuda)* | Drop-in `W8A16Linear` wrapping the `w8a16` kernel with a quantize-on-load path. |

---

## Models and getting started

Getting started, hardware, and feature flags: `README.md` and
`.release-notes/EXPLORATION.md` §5; each section above links its own deep-dive doc
under `docs/`. **Models ship separately** — the `models/` directory is empty; users
download checkpoints from HuggingFace (`Qwen/Qwen3-30B-A3B-Instruct-2507`,
`Qwen/Qwen3.6-35B-A3B`, `nvidia/Qwen3.6-35B-A3B-NVFP4`, `Qwen/Qwen3-0.6B` for GRPO) and
drop them in themselves.
