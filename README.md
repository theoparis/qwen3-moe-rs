# qwen3-burn

> Rust crate: `qwen3-burn` (imported as `qwen3_burn`). MIT-licensed.

A Rust implementation of the [Qwen3](https://github.com/QwenLM/Qwen3) family of large language
models, built on the [Burn](https://burn.dev/) deep-learning framework instead of PyTorch. It runs
two kinds of workload: **inference** for the Mixture-of-Experts (MoE) chat models Qwen3-30B-A3B and
Qwen3.6-35B-A3B, and **reinforcement-learning training** through a GRPO trainer whose loop runs
natively in Rust/Burn. A *Mixture-of-Experts* model routes each token to a few of many expert
sub-networks, so it holds many parameters but activates only a fraction per token. *GRPO* (Group
Relative Policy Optimization) is the RL algorithm popularized by DeepSeekMath. The whole stack was
developed and measured on a single NVIDIA GB10 (Grace-Blackwell, 128 GB unified memory), but the core
model code stays backend-generic and also runs on CPU.

This is for people who want to run or post-train Qwen3 in Rust — on NVIDIA GPUs for the fast paths,
or on CPU for the smaller dense model and the GRPO math.

## Highlights

- **MoE inference engines.** Qwen3-30B-A3B (`src/moe.rs`: 48 layers, 128 experts, top-8) and the
  hybrid Qwen3.6-35B-A3B (`src/qwen3_5/`: a Gated-DeltaNet + full-attention tower). Forward logits
  match Hugging Face transformers at cosine > 0.9999 (`docs/MOE_PLAN.md`).
- **Quantization.** *Quantization* stores weights in fewer bits to save memory and bandwidth. Two
  schemes ship: FP8 weight-only (`src/w8a16.rs`) and NVFP4 4-bit (`src/nvfp4.rs`), plus a loader for
  NVIDIA's pre-quantized ModelOpt NVFP4 checkpoints (`src/nvidia_ckpt.rs`), bit-exact against an
  external reference (`docs/QUANT_FLASH_SPEC_PLAN.md`).
- **CUDA-graph captured decode.** A *CUDA graph* records a fixed sequence of GPU kernel launches once
  and replays it per token, removing per-launch host overhead. See `src/capture.rs` and
  `docs/cudagraph/DESIGN.md`.
- **`qwen-serve` — an OpenAI-compatible HTTP server.** One model per process, FIFO queue, endpoints
  `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, with true token-by-token
  streaming over SSE (*Server-Sent Events*, the streaming format the OpenAI API uses). Chat templates
  reproduce HF `apply_chat_template` byte-for-byte: 12/12 byte-identical on both models
  (`docs/SERVE_PLAN.md`).
- **Rust/Burn GRPO trainer.** To our knowledge this is the first public GRPO LLM trainer whose
  training loop runs natively in Rust/Burn — rollout, forward and backward, group-relative advantage,
  KL penalty, and the AdamW optimizer step (`src/grpo/`). This is an absence-of-evidence claim from a
  mid-2026 search of crates.io, GitHub, and the literature, not a proof. The mainstream GRPO stacks
  (HF TRL, verl, OpenRLHF, Unsloth) are Python on PyTorch; there is **no PyTorch in the training
  loop** here. The GRPO algorithm itself (DeepSeekMath) and using Manim as a reward (prior art:
  ManimTrainer) are not ours — the Rust-native loop is.

## Quick start (60 seconds)

Requires a recent Rust toolchain. On aarch64 (NVIDIA GB10 / Grace) every CUDA build also needs
`RUSTFLAGS="-C target-feature=+fp16"` to satisfy the half-precision intrinsics.

```bash
# 1. Text generation on CPU — no GPU, no features (download Qwen/Qwen3-0.6B into models/qwen3-0.6b first):
cargo run --release --example generate -- \
  --model models/qwen3-0.6b/model.safetensors --tokenizer models/qwen3-0.6b/tokenizer.json \
  --prompt "Hello, I am a language model and" --max-tokens 50

# 2. Start the OpenAI-compatible server (needs a downloaded checkpoint under models/, see below):
RUSTFLAGS="-C target-feature=+fp16" \
  MODEL=qwen3.6-35b QUANT=bf16 PORT=8000 \
  cargo run --release --features cuda,serve --bin qwen-serve
```

Then talk to it like any OpenAI endpoint:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-bf16","messages":[{"role":"user","content":"Hi"}]}'
```

The `"model"` field must match the id the server reports at `GET /v1/models` — for the 35B it
is quant-suffixed (`qwen3.6-35b-a3b-bf16` for the launch above, `-fp8`/`-nvfp4` likewise); the
30B reports `qwen3-30b-a3b`. The `MODEL` env value is a launch alias, not the API id.

Server config is via env: `MODEL={qwen3-30b|qwen3.6-35b}`, `QUANT={bf16|fp8|nvfp4}`, `MODEL_DIR`,
`HOST`, `PORT`, `T_MAX`, `QUEUE_DEPTH` (defaults `MODEL=qwen3.6-35b`, `QUANT=bf16`, `PORT=8000`,
`T_MAX=4096`). The full path — dependencies, weight download, all examples — is in
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Models

The `models/` directory ships **empty** (only a `.gitkeep`; weights are gitignored). Download the
checkpoints you want from Hugging Face and drop them in these directories:

| Directory | Hugging Face checkpoint | Notes |
|---|---|---|
| `models/qwen3-30b-a3b-instruct-2507` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30B MoE, 256K native context |
| `models/qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | 35B hybrid MoE (bf16 / fp8), Apache-2.0 |
| `models/qwen3.6-35b-a3b-nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | NVIDIA ModelOpt NVFP4 checkpoint |
| `models/` (small dense / GRPO examples) | `Qwen/Qwen3-0.6B` | for `generate`, `grpo_cuda` |

## Performance

Every number below is copied from a repo doc as written and was measured on a single NVIDIA GB10; the
source doc is named in each row. These are the figures reported by that doc, not re-run here.

| Metric | Value | Source |
|---|---|---|
| 30B bf16 decode — launch-bound baseline | 0.73 tok/s | `docs/PERF_80TOKS_PLAN.md` §1 |
| 30B bf16 decode — captured, fused gather-GEMV | 19.38 tok/s (re-run 21.03) | `docs/PERF_80TOKS_PLAN.md` §6 |
| 30B bf16 decode — roofline ceiling | ≈ 45 tok/s | `docs/PERF_80TOKS_PLAN.md` §0 |
| 35B decode journey (bf16 → fused fp8 → captured fp8 → captured NVFP4) | 0.91 → 4.85 → 8.96 → 11.78 tok/s | `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS |
| 35B NVFP4 device footprint (vs 40.3 fp8 / 71 bf16) | 22.5 GB in-use | `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS |
| 35B NVFP4 greedy output | byte-identical to the bf16 original (16 tok) | `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS |
| MoE forward logits vs HF transformers | cosine > 0.9999 | `docs/MOE_PLAN.md` |
| `qwen-serve` template parity vs HF transformers 5.12.1 | 12/12 byte-identical, both models | `docs/SERVE_PLAN.md` GATE RESULTS |
| `qwen-serve` eager throughput (server v1 path) | 35B NVFP4 sampled ~7.5 tok/s; 30B bf16 greedy-long ~9.9 tok/s | `docs/SERVE_PLAN.md` GATE RESULTS |

Note: the server v1 serves the **eager** decode path; the captured 11.78 / ~20 tok/s figures are the
designated follow-up milestone (CUDA-graph capture wiring), not what the server hits today
(`docs/SERVE_PLAN.md`).

## Repository map

| Directory | What is in it |
|---|---|
| `src/` | The crate: dense + MoE Qwen3 models, GRPO trainer (`grpo/`), quantization (`nvfp4`/`w8a16`), CUDA-graph capture, custom CubeCL kernels, the `qwen-serve` server (`serve/`, `bin/`) |
| `examples/` | ~60 runnable binaries: text gen, GRPO (CPU + CUDA), MoE generate, quantization gates, CUDA-graph / flash / MoE-kernel benches and probes |
| `tests/` | Integration tests (GRPO parity, template byte-parity, MoE/GDN capture) with `fixtures/` and the `ref/grpo_expected.json` golden |
| `docs/` | Engineering plans and research of record (ARCHITECTURE, BF16, GRPO_PLAN, MOE_PLAN, PERF_80TOKS_PLAN, QUANT_FLASH_SPEC_PLAN, SERVE_PLAN, plus `specs/`) |
| `a0/` | The Python reference-of-record harness (GRPO reference, Manim reward, SFT+GRPO script) that emits the golden tensors the Rust tests check against |
| `scripts/` | Helper scripts: chat-template fixture dumper, NVFP4 reference dequant, and `serve_gates/` (live-server smoke tests) |
| `models/` | Empty placeholder (`.gitkeep` only); drop downloaded HF checkpoints here |
| `vendor/` | Patched local copies of `cubecl` and `cubek` (CUDA-graph capture FFI + device-seed RNG), redirected from their git pins via `Cargo.toml` `[patch]` so the release is self-contained |

## Feature flags

Set in `Cargo.toml`; enable with `--features`.

- **`default = []`** — the core library builds on CPU with no extra dependencies.
- **`cuda`** — the CubeCL CUDA backend and all GPU kernels (MoE, quantization, capture, flash).
  Required for every fast path.
- **`train`** — autodiff + dataset API, needed for GRPO-on-CUDA and the bf16 throughput bench.
- **`serve`** — tokio / axum / minijinja / serde for the OpenAI server. Its host-side parts (API
  types, chat template, detokenizer) build and test without `cuda`; the engine and the `qwen-serve`
  binary need both `cuda` and `serve`.

`Cargo.toml` pins the exact Burn / CubeCL / cubek git revisions this was verified against, with the
patched copies in `vendor/`. Bumping Burn requires re-running the probes (`docs/BF16.md`).

## License and attribution

MIT (see [LICENSE](LICENSE)). Derived from [holg/qwen3-burn](https://github.com/holg/qwen3-burn)
(MIT/Apache-2.0), with the original copyright retained. Built on
[Burn](https://burn.dev/) and its [CubeCL](https://github.com/tracel-ai/cubecl) GPU compute layer.

- The GRPO stack reproduces the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) /
  [DeepSeekMath](https://arxiv.org/abs/2402.03300) GRPO math and is parity-tested against a Python
  reference; the algorithm is not ours.
- Using Manim as a GRPO reward is prior art
  ([ManimTrainer, arXiv:2604.18364](https://arxiv.org/abs/2604.18364)); the bundled reward is a
  fail-safe Python subprocess (any spawn / timeout / parse error scores 0.0). Because it invokes
  Python, the honest phrasing is **"no PyTorch in the training loop"**, not "zero Python".
