# Release Notes — qwen3-moe-rs v0.3.0

## What this release is

`qwen3-moe-rs` is a Rust implementation of the Qwen3 family of large language
models, built on the [Burn](https://burn.dev) deep-learning framework instead of
PyTorch. This release turns it from a single-model text generator into a small
stack: two Mixture-of-Experts inference engines, a quantization layer, fast
CUDA-graph decoding, a reinforcement-learning trainer whose training loop runs
natively in Rust, and `qwen-serve` — an OpenAI-compatible HTTP server. It was
developed and measured on a single NVIDIA GB10 (Grace-Blackwell, 128 GB unified
memory).

Jargon used below, explained once: a **Mixture-of-Experts (MoE)** model routes
each token to a few small "expert" sub-networks instead of one big network, so it
holds many parameters but only runs a fraction per token. **Quantization** stores
weights in fewer bits (here 8-bit FP8 and 4-bit NVFP4) to shrink memory and speed
up the math. A **CUDA graph** records a fixed sequence of GPU operations once and
replays it, cutting the per-step launch overhead. **GRPO** is a reinforcement-
learning algorithm for fine-tuning models. **tok/s** is decode tokens per second.
**SSE** (Server-Sent Events) is the streaming format the server uses to push
tokens to a client as they are generated.

## Headline features

### `qwen-serve` — OpenAI-compatible single-stream server (the milestone)

A new binary, `qwen-serve`, exposes one model per process over the OpenAI HTTP
API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`, with
both streaming (SSE) and non-streaming responses. It serves exactly the two proven
models — Qwen3-30B-A3B (bf16) and Qwen3.6-35B-A3B (NVFP4). Requests are handled
one at a time, first-in first-out (FIFO), with a bounded queue.

Every server gate passed on both models against the live binary on the GB10
(`docs/SERVE_PLAN.md`, GATE RESULTS, 2026-07-02):

- **72 library tests** pass in total.
- **Chat-template parity: 12/12 byte-identical** to Hugging Face `transformers`
  5.12.1 for both models. The template renderer (minijinja plus a small Python-
  compatibility shim) reproduces `apply_chat_template` byte-for-byte.
- **End-to-end greedy decoding is byte-identical** to the standalone example
  fixtures (`qwen35_generate` for 35B, `vllm_infer` for 30B) over 16 tokens, and
  the streamed output concatenates to exactly the non-streamed output.
- **Incremental detokenization** holds back partial UTF-8 and partial stop-strings
  before emitting each SSE chunk, so no chunk ever shows mojibake, and decoding
  cost does not grow with output length.
- **20 sequential mixed requests** kept per-class throughput stable with flat
  resident memory (no per-request leak).
- Error paths return correct HTTP codes (400 for unsupported length/tools/`n>1`/
  logprobs, 404 for unknown model, 429 when the queue is saturated), and FIFO
  order holds under two concurrent clients with mid-stream cancellation.

### The earlier pillars this builds on

- **Two MoE inference engines.** Qwen3-30B-A3B (48 layers, 128 experts, top-8;
  `src/moe.rs`) and the hybrid Qwen3.6-35B-A3B (a Gated-DeltaNet + full-attention
  tower with a Multi-Token-Prediction head; `src/qwen3_5/`). Forward logits match
  Hugging Face `transformers` with **cosine similarity > 0.9999**, and 18 library
  tests cover routing parity, invariants, and determinism (`docs/MOE_PLAN.md`).
- **FP8 and NVFP4 quantization.** An 8-bit weight-only path (FP8 / W8A16, e4m3;
  `src/w8a16.rs`, validated to cosine > 0.999 vs a CPU oracle and OCP golden
  vectors) and a 4-bit path (NVFP4: 4-bit weights, 8-bit per-16-element block
  scales, one f32 global scale; `src/nvfp4.rs`). The NVFP4 loader can also ingest
  NVIDIA's own pre-quantized ModelOpt checkpoint for `nvidia/Qwen3.6-35B-A3B-NVFP4`
  bit-exactly against an external reference (`docs/QUANT_FLASH_SPEC_PLAN.md`
  §M-B.5).
- **CUDA-graph capture decode.** `src/capture.rs` records and replays the decode
  step below Burn's fusion layer, using capture/replay FFI vendored in
  `vendor/cubecl` and a device-seed RNG in `vendor/cubek` (`docs/cudagraph/`).
  This is what drives the fastest measured numbers below.
- **Fused MoE and flash kernels.** A capturable top-8 single-token decode kernel
  that reads only the 8 routed experts' weights (`src/moe_decode.rs`), a dropless
  grouped-GEMM path (`src/moe_grouped.rs`), and tiled flash-attention / split-K
  flash-decode kernels (`src/flash_attn.rs`, `src/flash_decode.rs`).
- **MTP / speculative-decode probe.** The 35B engine carries a Multi-Token-
  Prediction head and can decode speculatively; this exists in the engine but is
  not yet wired into the server (`docs/QUANT_FLASH_SPEC_PLAN.md`).
- **Rust/Burn GRPO trainer.** The full training loop — rollout, forward and
  backward passes, group-normalized advantage, clipped PPO surrogate with a k3 KL
  penalty, and the AdamW optimizer step — runs natively in Burn with no PyTorch in
  the training loop (`src/grpo/`). It is parity-tested against golden tensors from
  a Python reference (`tests/grpo_*.rs`, `a0/grpo_reference.py`). To our knowledge
  this is the first public GRPO LLM trainer whose training loop runs natively in
  Rust/Burn — stated as absence of evidence from a mid-2026 search, not as proof.
  The optional Manim reward shells out to a Python subprocess behind a static-AST
  safety gate; any spawn, exit, timeout, or parse error scores `0.0`.

## Measured results

Every figure below is copied from a repo doc; the source is named in each row.
These were run on a single GB10; no numbers were re-run for these notes.

| Model / quant | Path | Throughput | Memory | Source |
|---|---|---|---|---|
| Qwen3-30B-A3B / bf16 | baseline (launch-bound) | 0.73 tok/s | — | `docs/PERF_80TOKS_PLAN.md` §1 |
| Qwen3-30B-A3B / bf16 | captured, fused gather-GEMV | 19.38→21.03 tok/s (SERVE_PLAN cites 20.9) | — | `docs/PERF_80TOKS_PLAN.md` §6; `docs/SERVE_PLAN.md` |
| Qwen3-30B-A3B / bf16 | bf16 decode roofline | ≈45 tok/s | — | `docs/PERF_80TOKS_PLAN.md` §0 |
| Qwen3-30B-A3B / bf16 | long context (858 tokens) | 5.85 tok/s (~171 ms/token) | — | `docs/perf-gap-vs-prod.md` |
| Qwen3.6-35B-A3B / bf16→fp8→nvfp4 | decode journey | 0.91 → 4.85 (fused fp8) → 8.96 (captured fp8) → 11.78 tok/s (captured NVFP4) | — | `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS |
| Qwen3.6-35B-A3B / NVFP4 | captured (1.32× over fp8 captured) | 11.78 tok/s (7.35 eager-static) | 22.5 GB device (vs 40.3 fp8 / 71 bf16); host HWM 22.7 GB; load ~4 min | `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS |
| Qwen3.6-35B-A3B / NVFP4 | served today (eager) | sampled ~7.5, greedy ~1.07 tok/s | — | `docs/SERVE_PLAN.md` GATE RESULTS |
| Qwen3-30B-A3B / bf16 | served today (eager) | greedy-short ~6.0, greedy-long ~9.9, sampled ~1.4 tok/s | — | `docs/SERVE_PLAN.md` GATE RESULTS |

Quality of the NVFP4 35B path: greedy decoding is **byte-identical to the bf16
original** over 16 tokens, and a teacher-forced check over 188 positions found
**top-1 agreement 89.9%**, **KL divergence 0.0374**, and **high-margin agreement
97.8%** (`docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS).

## Known limitations

We would rather under-promise. Please read this section before deploying.

- **Single-stream only.** The server decodes exactly one request at a time (FIFO).
  There is no batching, no continuous batching, no paged KV cache, and no prefix
  caching. A second request waits behind the first.
- **The server serves the eager path, not the captured path.** The fast captured
  numbers above (11.78 tok/s for 35B, ~20.9 for 30B) are the CUDA-graph capture
  follow-up milestone, not what the server hits today. Today's served throughput
  is the eager-static path in the results table (e.g. 35B NVFP4 ~7.5 tok/s
  sampled). CUDA-graph capture wiring is the designated first follow-up.
- **35B greedy is ~7× slower than sampled, and this is a known bug.** On the 35B
  NVFP4 eager path, greedy decoding runs at ~1.07 tok/s versus ~7.5 tok/s sampled —
  an unexpected inversion whose root cause (a candidate is per-token device argmax
  synchronization) is still pending (`docs/SERVE_PLAN.md`). Prefer the sampled path
  for now.
- **No model weights are included.** The `models/` directory ships empty (only a
  `.gitkeep`); `*.safetensors` are gitignored. Download checkpoints yourself from
  Hugging Face and place them under `models/`:
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` → `models/qwen3-30b-a3b-instruct-2507`
  - `Qwen/Qwen3.6-35B-A3B` (Apache-2.0) → `models/qwen3.6-35b-a3b`
  - `nvidia/Qwen3.6-35B-A3B-NVFP4` → `models/qwen3.6-35b-a3b-nvfp4`
  - `Qwen/Qwen3-0.6B` for the small dense / GRPO examples.
- **You need an NVIDIA GPU with a large memory pool.** The fast paths, quantization,
  capture, MoE, and the server require CUDA and were measured on a GB10 with 128 GB
  unified memory (sm_121). Only the core dense model and the GRPO math run on CPU.
  On aarch64 (GB10 / Grace) you must build with
  `RUSTFLAGS="-C target-feature=+fp16"`.
- **Vendored, pinned dependencies.** `Cargo.toml` pins exact Burn / CubeCL / cubek
  git revisions, and `vendor/` holds patched local copies wired via `[patch]`. The
  release is self-contained, but bumping Burn means re-running the precision probes
  (`docs/BF16.md`).
- **No auth, no TLS, no rate limiting** beyond the bounded queue. Do not expose the
  server directly to an untrusted network. Tool-call output is returned as text
  (not parsed into structured `tool_calls`); `logprobs`, `n>1`, and presence/
  frequency penalties return a 400 (`docs/SERVE_PLAN.md`, NOT in scope).
- **Not the largest model.** Qwen3-235B-A22B is spec'd but explicitly out of scope
  for a single GB10 (`docs/MOE_PLAN.md`); there is no 235B engine here.

## Prior art and credits

- The dense Qwen3 text engine descends from the MIT/Apache `holg/qwen3-burn`
  project.
- Built on **Burn**, **CubeCL**, and **cubek** — thanks to those teams. The vendored
  copies in `vendor/` add CUDA-graph capture FFI and a device-seed RNG.
- **GRPO the algorithm is not ours** — it is from DeepSeekMath. Our trainer's math
  reproduces and is parity-tested against **OpenRLHF**. Using **Manim** rendering as
  a reward signal has prior art (**ManimTrainer**); we do not claim either as novel.
  What is new here is that the training loop runs natively in Rust/Burn (see above),
  stated as absence of evidence rather than proof.
- The **NVFP4** loader interoperates with NVIDIA's **ModelOpt** checkpoint format.

## Roadmap

From the `docs/SERVE_PLAN.md` follow-up ladder, in order:

1. **Wire CUDA-graph capture into the server**, so served throughput reaches the
   captured numbers (expected ~7.3 → 11.78 tok/s on 35B NVFP4 greedy). This is the
   designated first follow-up.
2. **Root-cause the 35B greedy-eager gap** (the ~7× greedy/sampled inversion).
3. **Batching** — concurrent / continuous batching, and paged KV cache.
4. **Cache-pool reuse** across requests (today each request builds a fresh KV cache).

Further out: wiring the MTP speculative-decode path into the server, and structured
tool-call parsing of model output.
