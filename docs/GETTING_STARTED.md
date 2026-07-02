# Getting Started

A walkthrough from zero to your first generated token. It assumes you have never
seen this repo before. Jargon is explained on first use.

This project is a Rust implementation of the **Qwen3** family of large language
models, built on the [Burn](https://burn.dev/) deep-learning framework instead of
PyTorch. It contains a dense text-generation engine, a Rust-native **GRPO**
trainer (Group Relative Policy Optimization — a reinforcement-learning method for
fine-tuning language models), **MoE** inference (Mixture-of-Experts: a model where
each token is routed to a few of many "expert" sub-networks), a quantization stack
(**quantization** = storing weights in fewer bits to save memory), CUDA-graph
capture for faster decoding, and `qwen-serve`, an OpenAI-compatible HTTP server.

## Prerequisites

- **Rust 1.85+** (the crate is `edition = "2024"`, `rust-version = "1.85"` in
  `Cargo.toml`). Install from <https://rustup.rs>.
- **An NVIDIA GPU** for the fast paths (quantization, CUDA-graph capture, MoE, and
  the server). This code was developed and measured on a single **NVIDIA GB10**
  (Grace-Blackwell, compute capability sm_121, 128 GB unified LPDDR5X memory,
  ~273 GB/s bandwidth). The dense model and the GRPO *math* also run on CPU with no
  GPU at all.
- **The CUDA toolkit** (required by Burn's CubeCL CUDA backend to compile kernels).
- **~60 GB of free disk per large model** you download (checkpoints are not shipped;
  see below).
- **aarch64 build flag.** On aarch64 hosts (the GB10 / Grace is aarch64) you must
  build with `RUSTFLAGS="-C target-feature=+fp16"` to satisfy the half-precision
  intrinsics (see the `cuda` feature comment in `Cargo.toml`). Every GPU build
  command below includes it.

## Downloading models

The `models/` directory ships **empty** — it holds only a `.gitkeep`, and
`*.safetensors` files are gitignored. You download the checkpoints you want from
Hugging Face yourself, into the directory names the code expects. Install the CLI
with `pip install -U "huggingface_hub[cli]"`, then:

```bash
# Small dense model for the CPU walkthrough and GRPO-on-GPU examples (~1.5 GB).
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir models/qwen3-0.6b

# Qwen3-30B-A3B MoE (the 30B examples + the qwen3-30b server model).
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir models/qwen3-30b-a3b-instruct-2507

# Qwen3.6-35B-A3B hybrid MoE, bf16 / fp8 (Apache-2.0).
huggingface-cli download Qwen/Qwen3.6-35B-A3B \
  --local-dir models/qwen3.6-35b-a3b

# NVIDIA's pre-quantized NVFP4 checkpoint of the 35B (~22.5 GB in use on device).
huggingface-cli download nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --local-dir models/qwen3.6-35b-a3b-nvfp4
```

The directory names above are the per-model defaults the server and examples look
for; you do not have to download all four. Rough device footprints for the 35B, as
measured in `docs/QUANT_FLASH_SPEC_PLAN.md`: bf16 **71 GB**, fp8 **40.3 GB**, NVFP4
**22.5 GB**.

## Build

The crate uses Cargo **feature flags** to keep the default build small. There are
three combinations:

```bash
# 1. Core library on CPU — no GPU, no extra deps. Builds the dense model + GRPO math.
cargo build --release

# 2. GPU engine: CubeCL CUDA backend + all kernels (MoE, quantization, capture).
RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda

# 3. Everything for the server: the CUDA engine plus the OpenAI-compatible HTTP stack.
RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda,serve
```

There is also a `train` feature (`--features cuda,train`) that adds autodiff and the
dataset API, used by the on-GPU GRPO example and the bf16 throughput bench.

The **first** build compiles the vendored `cubecl` and `cubek` crates from source
and expect it to take a while. These are patched local copies of the exact git
revisions Burn pins, checked out under `vendor/` and wired in via `[patch]` in
`Cargo.toml`, so the release is self-contained — Cargo builds them from `vendor/`,
not the network.

## First inference on CPU (no GPU needed)

The `generate` example runs the dense Qwen3-0.6B model on Burn's **NdArray** (CPU)
backend. It needs no GPU and no CUDA — just the downloaded 0.6B weights and
tokenizer:

```bash
cargo run --release --example generate -- \
  --model models/qwen3-0.6b/model.safetensors \
  --tokenizer models/qwen3-0.6b/tokenizer.json \
  --prompt "Hello, I am a language model and" \
  --max-tokens 50
```

It prints the loaded config, the input token ids, and the generated text. If the
`--model`/`--tokenizer` flags are omitted it falls back to `models/model.safetensors`
and `models/tokenizer.json`.

## First GPU inference

The GPU examples load the MoE checkpoints and decode on the fused, host-sync-free
fast path.

**30B (`vllm_infer`)** mirrors vLLM's `LLM(model).generate(prompts, SamplingParams)`:
the model is loaded once, then driven with per-request sampling. `--temperature 0`
gives greedy (deterministic) decoding:

```bash
RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda \
  --example vllm_infer -- \
  --dir models/qwen3-30b-a3b-instruct-2507 \
  --prompt "The capital of France is" --max-tokens 48 --temperature 0
```

**35B (`qwen35_generate`)** greedy-decodes a fixed prompt. The `QUANT` environment
variable selects the precision — `bf16` (default), `fp8`, or `nvfp4`. `QUANT=nvfp4`
loads the NVIDIA NVFP4 checkpoint from `models/qwen3.6-35b-a3b-nvfp4`; `bf16`/`fp8`
load from `models/qwen3.6-35b-a3b`:

```bash
RUSTFLAGS="-C target-feature=+fp16" QUANT=nvfp4 cargo run --release --features cuda \
  --example qwen35_generate
```

Override the checkpoint directory with `QWEN35_DIR=<path>` if you put it elsewhere.
The NVFP4 greedy output is byte-identical to the bf16 original, and NVFP4 decode was
measured at **11.78 tok/s** on the GB10 (`docs/QUANT_FLASH_SPEC_PLAN.md`).

## Running the server

`qwen-serve` is an OpenAI-compatible HTTP server: one model per process, one request
decoding at a time (FIFO). It is configured entirely through environment variables
(no command-line flags). The variables it reads, from `src/bin/qwen_serve.rs`:

| Variable      | Default                                   | Meaning                                      |
|---------------|-------------------------------------------|----------------------------------------------|
| `HOST`        | `0.0.0.0`                                  | bind host                                    |
| `PORT`        | `8000`                                     | bind port                                    |
| `MODEL`       | `qwen3.6-35b`                              | `qwen3-30b` or `qwen3.6-35b`                 |
| `QUANT`       | `bf16`                                     | `bf16`, `fp8`, or `nvfp4`                    |
| `MODEL_DIR`   | per-model default under `models/`          | checkpoint directory                         |
| `T_MAX`       | `4096`                                     | process context limit (tokens)               |
| `QUEUE_DEPTH` | `2`                                        | bounded submit-queue depth                   |

With `MODEL=qwen3-30b` the default directory is `models/qwen3-30b-a3b-instruct-2507`;
with `MODEL=qwen3.6-35b` it is `models/qwen3.6-35b-a3b` (or `models/qwen3.6-35b-a3b-nvfp4`
when `QUANT=nvfp4`). Start it:

```bash
RUSTFLAGS="-C target-feature=+fp16" MODEL=qwen3.6-35b QUANT=nvfp4 \
  cargo run --release --features cuda,serve --bin qwen-serve
```

On startup it loads the model **once** and blocks until ready. Loading the 35B NVFP4
checkpoint takes roughly **4 minutes** (`docs/QUANT_FLASH_SPEC_PLAN.md`); the smaller
models load faster. When it is ready it prints a banner ending in `[qwen-serve]
listening on http://0.0.0.0:8000`. The banner reports the resolved model id, quant,
directory, `t_max`, queue depth, sampling defaults, and the decode path
(`eager-static` — the CUDA-graph capture fast path is a follow-up milestone, so the
server today serves the eager-static path, not the captured 11.78/20.9 tok/s
numbers).

Once it is listening, in another terminal:

First run `curl http://127.0.0.1:8000/v1/models` and use the exact id it returns in
the `"model"` field below (for the `QUANT=nvfp4` launch above that id is
`qwen3.6-35b-a3b-nvfp4`; the 35B id is always quant-suffixed, and the 30B reports
`qwen3-30b-a3b`) — the `MODEL` env value is a launch alias, not the API id.

```bash
# Health check → 200 {"status":"ok"}
curl http://127.0.0.1:8000/health

# Chat completion (non-streaming)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-nvfp4","messages":[{"role":"user","content":"Why is the sky blue? One sentence."}],"temperature":0,"max_tokens":48}'

# Streaming (Server-Sent Events — the response is emitted token-by-token as `data:` lines)
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-nvfp4","messages":[{"role":"user","content":"Count to five."}],"stream":true,"max_tokens":48}'
```

The other routes are `GET /v1/models` and `POST /v1/completions` (legacy
text-completion). The server also works with the official Python `openai` SDK — pass
any non-empty `api_key`, since none is checked:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
model = client.models.list().data[0].id
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
    temperature=0, max_tokens=48,
)
print(r.choices[0].message.content)
```

## Running the GRPO training smoke

`grpo_train` is a convergence smoke for the GRPO trainer: a tiny random model with a
toy reward on the CPU (NdArray) backend. It needs no GPU, no weights, and no Python —
the whole training loop (rollout → reward → group-normalized advantage → policy/
reference log-probs → clipped surrogate + KL → AdamW step) runs natively in Rust/Burn.
This is the **first public GRPO LLM trainer whose training loop runs natively in
Rust/Burn** — an absence-of-evidence claim from a mid-2026 search, not a proof. Note
there is **no PyTorch in the training loop**; the GRPO algorithm itself (from
DeepSeekMath) and the bundled Manim-as-reward idea are prior art, not ours.

```bash
# The optional argument is the number of steps (default 30). mean_reward should trend up.
cargo run --release --example grpo_train 30
```

To train for real, three things change (the loop is unchanged): swap the backend to
`Autodiff<Cuda>`, load real weights via `Qwen3Config::qwen3_0_6b()` +
`load_weights`, and swap the toy reward for `ManimReward`. The on-GPU version is
`examples/grpo_cuda.rs` (`--features cuda,train`).

## Troubleshooting

- **Port already in use.** Another process holds `8000`. Set a different `PORT`, e.g.
  `PORT=8001 ... --bin qwen-serve`.
- **Out of memory.** Use a smaller model or a smaller precision — `QUANT=nvfp4`
  (22.5 GB) instead of `fp8` (40.3 GB) or `bf16` (71 GB) for the 35B. You can also
  lower `T_MAX` to shrink the KV cache.
- **`MODEL_DIR ... does not exist`.** The server refuses to start if the checkpoint
  directory is missing. Download the model (above) or point `MODEL_DIR` at where you
  put it.
- **Slow first tokens.** The server runs the eager decode path with per-shape warmup,
  so the first request of a new shape is slower while kernels compile; later requests
  of the same shape are fast.
- **Self-verify.** The gate scripts under `scripts/serve_gates/` exercise a live
  server through the real `openai` SDK (`openai_smoke.py`), SSE capture
  (`sse_capture.py`), greedy parity (`greedy_parity.py`), and a sustained
  memory-flat smoke (`sustained_and_errors.py`). Run one against your running server,
  for example: `python scripts/serve_gates/openai_smoke.py http://127.0.0.1:8000/v1`.

## Where to go next

- `README.md` — project overview and the GRPO novelty positioning.
- `docs/ARCHITECTURE.md` — the model and code layout.
- `docs/GRPO_PLAN.md`, `docs/MOE_PLAN.md`, `docs/QUANT_FLASH_SPEC_PLAN.md`,
  `docs/SERVE_PLAN.md` — the engineering plans and measured results of record.
