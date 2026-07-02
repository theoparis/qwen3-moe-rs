# EXPLORATION — grounding notes for the public release docs

> Source of record for the README / architecture / getting-started / release-notes rewrite.
> Every claim below is traced to a file in this repo copy (`/workspace/qwen3-burn-release`).
> No git history is present. Written by exploring the source + `docs/`; anything I could not
> verify is called out at the end.

> **IMPORTANT DRIFT NOTE for doc-writers:** `Cargo.toml` says `version = "0.3.0"` and the top-level
> `README.md`, `CHANGELOG.md`, and `docs/ARCHITECTURE.md` only describe work through **0.3.0 (GRPO
> Phase A) + 0.2 (bf16 / batch-safe Linear)**. The *code* has moved far past those docs: MoE
> inference (30B + 35B), FP8/NVFP4 quantization, CUDA-graph capture, custom CubeCL kernels, and the
> `qwen-serve` OpenAI server all exist in `src/` and are documented in `docs/*_PLAN.md`, not in the
> README/CHANGELOG. Treat the plan docs (`docs/MOE_PLAN.md`, `docs/PERF_80TOKS_PLAN.md`,
> `docs/QUANT_FLASH_SPEC_PLAN.md`, `docs/SERVE_PLAN.md`) as the current-state record for those areas.

## 1. What this project is

This is a Rust implementation of the **Qwen3** family of large language models, built on the
**Burn** deep-learning framework instead of PyTorch. It started as a dense Qwen3 text-generation
engine (derived from the MIT/Apache `holg/qwen3-burn` project) and grew two big additions: a
**Rust-native GRPO reinforcement-learning trainer** (the training loop — rollout, forward/backward,
advantage, KL, optimizer — runs in Rust/Burn), and **Mixture-of-Experts inference** for the larger
Qwen3-30B-A3B and Qwen3.6-35B-A3B models, made fast enough to run on a single NVIDIA GB10
(Grace-Blackwell, 128 GB unified memory). On top of that sit a quantization stack (FP8 and NVFP4,
including loading NVIDIA's pre-quantized checkpoints), CUDA-graph capture for fast decoding, and
`qwen-serve`, an OpenAI-compatible HTTP server. The crate is named `qwen3-burn` (imported as
`qwen3_burn`), is MIT-licensed, and targets NVIDIA GPUs (sm_120 / sm_121) via Burn's CubeCL backend,
while the core model code stays backend-generic and runs on CPU too.

## 2. The unique things built here

Each bullet: **the thing** — where it lives — evidence.

- **Rust/Burn GRPO trainer** — `src/grpo/` (`mod.rs`, `loss.rs`, `logprob.rs`, `rollout.rs`,
  `reward.rs`, `trainer.rs`); entry points `grpo_step` / `grpo_step_ragged`. The full loop (rollout →
  reward → group-norm advantage → policy/ref log-probs → clipped PPO surrogate + k3 KL → backward →
  AdamW) runs natively in Burn. Evidence: `docs/GRPO_PLAN.md`; parity-tested in `tests/grpo_math.rs`,
  `tests/grpo_rollout.rs`, `tests/grpo_trainer.rs`, `tests/grpo_varprompt.rs` against golden tensors
  `tests/ref/grpo_expected.json` emitted by the Python reference `a0/grpo_reference.py`.
- **Memory-safe per-token log-probs** — `src/grpo/logprob.rs` (`token_logprobs`): `gather − logsumexp`
  that never materializes the `[B,T,vocab]` softmax. Evidence: `CHANGELOG.md` 0.3.0, `README.md`.
- **Verifiable Manim reward (fail-safe)** — `src/grpo/reward.rs` (`RewardFn`, `ManimReward`); shells
  out to `a0/manim_reward.py` (static-AST safety gate + staged partial credit, off-by-default sandboxed
  `manim --dry_run`). Any spawn/exit/timeout/parse error scores `0.0`. Evidence: `README.md` Features,
  `a0/README.md`.
- **Qwen3-30B-A3B MoE inference engine** — `src/moe.rs` (`Qwen3MoeConfig`, `Qwen3MoeForCausalLM`,
  `Qwen3MoeSparseBlock`); 48 layers, 128 experts, top-8, no shared expert. Evidence: `docs/MOE_PLAN.md`
  (spec table + HF-literal routing); correctness gate cosine > 0.9999 vs HF; example `moe_generate`.
- **Qwen3.6-35B-A3B hybrid MoE engine** — `src/qwen3_5/mod.rs` (`Qwen3_5MoeConfig`,
  `Qwen3_5MoeForCausalLM`, `Qwen3_5LayerType`); a Gated-DeltaNet + full-attention hybrid tower with MTP.
  Evidence: `docs/QUANT_FLASH_SPEC_PLAN.md`, `docs/specs/L1-kickoff-gdn-port.md`,
  `docs/specs/L1.3-gdn-math.md`; examples `qwen35_generate`, `qwen35_model_smoke`.
- **FP8 (W8A16, e4m3) weight-only quantization** — kernel `src/w8a16.rs`, drop-in `src/w8a16_linear.rs`
  (`W8A16Linear`). Reads packed e4m3 bytes from HBM, dequants in the GEMM load path. Evidence:
  `docs/VLLM_KERNELS.md` §2; validated vs NdArray CPU oracle + OCP golden vectors in
  `examples/w8a16_spike.rs` (cosine > 0.999).
- **NVFP4 (E2M1 weight + E4M3 per-16 block scale + f32 global scale) quantization** — host codec
  `src/nvfp4.rs`, drop-in `src/nvfp4_linear.rs` (`Nvfp4Linear`). Evidence: `docs/specs/L2C-*.md`;
  golden vs Python reference `scripts/nvfp4_reference_dequant.py`; example `nvfp4_ckpt_golden`.
- **Loading NVIDIA ModelOpt (modelopt) NVFP4 checkpoints** — `src/nvidia_ckpt.rs` ingests the on-disk
  ModelOpt W4A16 tensors (`weight`, `weight_scale`, `weight_scale_2`) for `nvidia/Qwen3.6-35B-A3B-NVFP4`.
  Evidence: `docs/QUANT_FLASH_SPEC_PLAN.md` §M-B.5, `docs/specs/M-B.5-prior-art.md`; golden gate
  bit-exact vs an external reference on real bytes.
- **Fake-quant PTQ accuracy gate** — `src/quant_gate.rs` round-trips weights through the host codec to
  measure reconstruction error on the model's normal path. Evidence: examples `qwen35_nvfp4_gate`,
  `qwen35_fp8_deploy_gate`, `qwen35_experts_fp8_gate`.
- **CUDA-graph capture decode** — `src/capture.rs` (`CapturedDecoder`, raw `CubeBackend` below Burn
  Fusion); vendored capture/replay FFI in `vendor/cubecl` + device-seed RNG in `vendor/cubek` (both
  patched local copies of the pinned revs, wired via `[patch]` in `Cargo.toml`). Evidence:
  `docs/cudagraph/DESIGN.md`; phased benches `examples/cudagraph_p0_bench.rs` .. `cudagraph_pfinal_bench.rs`,
  `cudagraph_moe_decode_bench.rs`, `cudagraph_panic_recovery.rs`.
- **Fused MoE kernels** — capturable top-8 single-token decode `src/moe_decode.rs` (`MoeExpertCache`,
  `decode_topk`: reads only the 8 routed experts' weight slabs, no per-layer host sync); dropless
  grouped-GEMM `src/moe_grouped.rs` (vLLM `moe_align_block_size` layout, block-per-segment SwiGLU).
  Evidence: `docs/PERF_80TOKS_PLAN.md` §2-3 lever A, `docs/VLLM_KERNELS.md` §3; validated vs NdArray
  oracle in `examples/moe_grouped_spike.rs`.
- **Custom-op bridge + flash-attention/flash-decode kernels** — `src/cube_custom_op.rs` (typed
  Burn-Fusion custom-op wrapper), `src/flash_attn.rs` (tiled online-softmax), `src/flash_decode.rs`
  (L2A.2 split-K flash-decode, capture-ready). Evidence: `docs/VLLM_KERNELS.md` §1,
  `docs/specs/L2A.2-split-k-flash-decode-design.md`; examples `attn_kernel_spike`, `flash_decode_bench`.
- **Batch-safe / bf16 `Linear`** — `src/linear2d.rs` (`Precision`, `linear3`): flattens 3-D input to a
  2-D GEMM to dodge a silent CubeCL broadcast-batched-matmul corruption bug on sm_121, and carries the
  bf16-mixed-precision path. Evidence: `docs/ARCHITECTURE.md`, `docs/BF16.md`, `examples/matmul_probe.rs`.
- **`qwen-serve` — OpenAI-compatible single-stream server** — bin `src/bin/qwen_serve.rs`, module
  `src/serve/` (`api.rs`, `template.rs`, `detok.rs`, `engine.rs`, `handlers.rs`, `mod.rs`). One model per
  process (30B or 35B), FIFO, `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
  Evidence: `docs/SERVE_PLAN.md`.
  - **Byte-parity chat templates** — `src/serve/template.rs`: minijinja + pycompat reproducing HF
    `apply_chat_template` byte-for-byte (order-preserving `tojson`, `raise_exception`, `trim/lstrip_blocks`).
    Gate: 12/12 byte-identical vs HF transformers, both models (`tests/template_parity.rs`,
    `tests/fixtures/template/`).
  - **Incremental detokenization** — `src/serve/detok.rs`: bounded-tail decode-and-diff with UTF-8
    holdback + stop-string holdback that gates SSE emission; no O(len²) growth, no mojibake.
  - **True streaming** — `src/serve/engine.rs` + `handlers.rs`: engine thread owns the `!Sync` model,
    bounded mpsc backpressure, `ReceiverStream` feeds axum SSE; cancel = channel closure.
- **Gate / verification methodology** — byte-identical parity gates + adversarial batteries throughout:
  GRPO math vs Python golden JSON; MoE logits cosine > 0.9999 vs HF; template byte-parity; detok
  adversarial `U+FFFD` force-commit; NVFP4/greedy byte-identical to bf16; serve E2E greedy byte-identical
  to the example fixtures; 20-request sustained memory-flat smoke. Evidence: `docs/SERVE_PLAN.md` GATE
  RESULTS, `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS, `docs/MOE_PLAN.md` §8, `tests/`, `scripts/serve_gates/`.

## 3. Measured numbers (only figures written in the repo docs; each cited)

**Qwen3-30B-A3B (bf16 MoE decode, single GB10):**
- Baseline **0.73 tok/s** = launch-bound, ~16% of even the dense byte-ceiling. — `docs/PERF_80TOKS_PLAN.md` §1
- **19.38 tok/s** captured, fused gather-GEMV (lever c), ≈43% of bf16 peak; independently re-run
  **19.38 → 21.03 tok/s** (43% → 47% peak). — `docs/PERF_80TOKS_PLAN.md` §6 / line 192
- bf16 decode roofline ≈ **45 tok/s** (55-70 needs fp8). — `docs/PERF_80TOKS_PLAN.md` §0
- `docs/SERVE_PLAN.md` refers to the 30B captured number as **20.9 tok/s** (greedy-only).
- Long context: falls to **5.85 tok/s** (~171 ms/token) at 858 tokens. — `docs/perf-gap-vs-prod.md`
- Roofline table: f32 ≈ 21 / bf16 ≈ 41 / fp8 ≈ 80 tok/s over ~3.3B active params. — `docs/MOE_PLAN.md`

**Qwen3.6-35B-A3B decode journey (single GB10):**
- **0.91 → 4.85 (fused fp8) → 8.96 (captured fp8) → 11.78 tok/s (captured NVFP4).**
  — `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS (line 540)
- Captured NVFP4 **11.78 tok/s** / 7.35 eager-static; **1.32× over the 8.96 fp8 captured baseline**.
- NVFP4 footprint **22.5 GB device in-use** (vs 40.3 fp8 / 71 bf16); host HWM 22.7 GB; load ~4 min.
- NVFP4 greedy **byte-identical to the bf16 original** (16 tok); golden bit-exact vs the Python reference.
- Teacher-forced (188 pos): top1 **89.9%**, KL **0.0374**, high-margin **97.8%**.
  — all `docs/QUANT_FLASH_SPEC_PLAN.md` RESULTS table

**`qwen-serve` gates (both models, live binary on GB10) — `docs/SERVE_PLAN.md` GATE RESULTS:**
- All S.5 gates green on both models; **72 lib tests total**.
- Template parity **12/12 byte-identical vs HF transformers 5.12.1** (both models).
- E2E greedy **byte-identical** to `qwen35_generate` (35B) and `vllm_infer` (30B), 16 tokens;
  non-stream == streamed-concat.
- 20-request sustained: per-class tok/s stable, RSS plateaus (no per-request leak).
- Eager (per-token, sequential) throughput: 35B nvfp4 **sampled ~7.5 tok/s**, **greedy ~1.07 tok/s**
  (an unexpected greedy/device-argmax inversion, root-cause pending); 30B bf16 **greedy-short ~6.0**,
  **greedy-long ~9.9 tok/s** (fused static-decode), **sampled ~1.4 tok/s** (prefill-dominated).
  Note: the server v1 serves the *eager-static* path; the 11.78/20.9 captured numbers are the
  capture follow-up milestone, not what the server hits today.

**MoE correctness:** forward logits/log-probs match HF transformers **cosine > 0.9999**;
**18 lib tests pass** (routing parity, invariants, determinism). — `docs/MOE_PLAN.md`

## 4. HONESTY BOUNDARIES (copy these rules into every downstream doc; obey them)

- Claim: "the first public GRPO LLM trainer whose training loop runs natively in Rust/Burn (rollout,
  forward+backward, advantage/loss, KL, optimizer step) — stated as absence of evidence from a mid-2026
  search, not proof."
- Say "no PyTorch in the training loop" — NEVER "zero Python" or "end-to-end Rust" (the Manim reward
  runs a Python subprocess; only the grpo_train CPU smoke is Python-free).
- GRPO the algorithm is NOT ours (DeepSeekMath); the math reproduces OpenRLHF and is parity-tested.
  Manim-as-reward has prior art (ManimTrainer). Do not claim either as novel.
- Do not invent numbers; every figure must trace to a repo doc.

## 5. What a beginner needs to know to run it

- **Hardware.** An NVIDIA GPU is required for the fast paths, quantization, capture, MoE, and the
  server. Developed and measured on a single **NVIDIA GB10 (Grace-Blackwell, sm_121, 128 GB unified
  LPDDR5X, ~273 GB/s)**. The core dense model + the GRPO *math* run on CPU (NdArray) with no GPU.
- **aarch64 build flag.** On aarch64 (GB10 / Grace) you must build with
  `RUSTFLAGS="-C target-feature=+fp16"` to satisfy the half-precision intrinsics (README build notes,
  `Cargo.toml` `cuda` feature comment).
- **`models/` is EMPTY in this release** — it ships only `.gitkeep`; weights are gitignored
  (`.gitignore`: `/models`, `*.safetensors`). Users download checkpoints from Hugging Face themselves.
  Directory → HF repo mapping (from `src/bin/qwen_serve.rs`):
  - `models/qwen3-30b-a3b-instruct-2507` → **`Qwen/Qwen3-30B-A3B-Instruct-2507`** (256K native context;
    `docs/longctx-decode-findings.md`).
  - `models/qwen3.6-35b-a3b` (bf16 / fp8) → **`Qwen/Qwen3.6-35B-A3B`** (Apache-2.0;
    `docs/QUANT_FLASH_SPEC_PLAN.md` D3).
  - `models/qwen3.6-35b-a3b-nvfp4` → **`nvidia/Qwen3.6-35B-A3B-NVFP4`** (NVIDIA ModelOpt checkpoint;
    `docs/QUANT_FLASH_SPEC_PLAN.md` §M-B.5).
  - Small dense / GRPO examples: **`Qwen/Qwen3-0.6B`** (README quickstart + `examples/grpo_cuda.rs`).
- **Feature flags** (`Cargo.toml`): `default = []` (core lib builds on CPU with no extra deps);
  **`cuda`** (CubeCL CUDA backend + kernels); **`train`** (autodiff + dataset, bf16 throughput bench);
  **`serve`** (tokio/axum/minijinja/serde for the OpenAI server; its host-side parts build without
  `cuda`, the engine + `qwen-serve` bin need both `cuda` and `serve`).
- **Main entry points:**
  - Examples in `examples/` — CPU: `generate` (text gen), `grpo_train` (GRPO convergence smoke, no
    GPU/weights/Python). CUDA: `grpo_cuda`, `moe_generate`, `vllm_infer`, `qwen35_generate`, plus many
    benches/probes/gates. Run: `cargo run --release [--features cuda,train] --example <name>`.
  - The server binary: `cargo run --release --features cuda,serve --bin qwen-serve` (config via env:
    `MODEL={qwen3-30b|qwen3.6-35b}`, `QUANT={bf16|fp8|nvfp4}`, `MODEL_DIR`, `HOST`, `PORT`, `T_MAX`,
    `QUEUE_DEPTH`). Defaults: `MODEL=qwen3.6-35b`, `QUANT=bf16`, `PORT=8000`, `T_MAX=4096`.
- **Build pin caveat.** `Cargo.toml` pins exact Burn / CubeCL / cubek git revs; `vendor/` holds patched
  local copies wired via `[patch]`. Bumping Burn requires re-running the probes (`docs/BF16.md`).

## 6. Repo map (one line per top-level directory)

- `src/` — the crate: dense + MoE Qwen3 models, GRPO trainer (`grpo/`), quantization (nvfp4/w8a16),
  CUDA-graph capture, custom CubeCL kernels, the `qwen-serve` server (`serve/`, `bin/`).
- `examples/` — ~60 runnable binaries: text gen, GRPO (CPU + CUDA), MoE generate, quantization gates,
  CUDA-graph/flash/MoE-kernel benches and probes.
- `tests/` — integration tests (GRPO math/rollout/trainer/varprompt parity, template byte-parity,
  MoE/GDN capture, untied head) with `fixtures/` and the `ref/grpo_expected.json` golden.
- `docs/` — engineering plans + research of record (ARCHITECTURE, BF16, GRPO_PLAN, MOE_PLAN,
  PERF_80TOKS_PLAN, QUANT_FLASH_SPEC_PLAN, SERVE_PLAN, VLLM_KERNELS, plus `specs/` and `cudagraph/`).
- `a0/` — the Python "reference of record" harness (GRPO reference, Manim reward, SFT+GRPO script)
  that emits the golden tensors the Rust tests check against.
- `scripts/` — helper scripts: chat-template fixture dumper, NVFP4 reference dequant, and
  `serve_gates/` (Python OpenAI-SDK / SSE / parity / sustained smoke tests for the live server).
- `models/` — EMPTY placeholder (`.gitkeep` only); users drop downloaded HF checkpoints here.
- `vendor/` — patched local copies of `cubecl` and `cubek` (CUDA-graph capture FFI + device-seed RNG),
  redirected from their git pins via `Cargo.toml [patch]`.
- `.claude/`, `.superpowers/` — agent/tooling config, not part of the shipped library.

## What I could NOT verify

- **No numbers were run by me.** Every figure in §3 is copied from a repo doc as written; I did not
  build, test, or benchmark anything (no GPU used, no `cargo` invoked). The docs themselves say these
  were GPU-run on the GB10, but I only read them.
- **README / CHANGELOG / ARCHITECTURE are stale** relative to the code (see the drift note up top):
  they stop at 0.3.0 (GRPO) and do not mention MoE, 35B, quantization, capture, or the server. The
  crate `version` is still `0.3.0`. I could not determine the intended release version number from the
  repo — doc-writers should confirm it.
- **The 30B captured figure is quoted two ways** — `docs/PERF_80TOKS_PLAN.md` gives 19.38→21.03 tok/s
  while `docs/SERVE_PLAN.md` says "captured 20.9 tok/s". Both are in-range; I could not reconcile which
  is the canonical headline number.
- **`run_manim_cuda` expects an "untied 14B Manim finetune"** (`models/qwen3-manim-14b`) that has no
  public HF repo cited anywhere in the repo — it appears to be a user's own finetune, not a downloadable
  checkpoint. Do not invent an HF link for it.
- **235B-A22B is spec'd but out of scope** (`docs/MOE_PLAN.md` explicitly excludes it from single-GB10);
  there is no 235B engine to document.
- Some `docs/` files (splitk research, sglang/perf research, cudagraph research) I only skimmed or did
  not open; they are research background, not shipped-feature descriptions, so they do not change §2.
