# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.3.0] — GRPO reinforcement-learning trainer

GRPO Phase A: a full RL post-training loop that runs natively in Rust/Burn (no PyTorch in the
training loop), with the math parity-checked against a Python reference of record.

### Added
- **End-to-end GRPO trainer** (`grpo_step`, `grpo_step_ragged`, `GrpoTrainConfig`, `StepReport`):
  rollout → reward → group-norm advantage → policy/reference log-probs → clipped surrogate + k3 KL
  loss → backward → AdamW. The rollout and frozen reference run no-grad on the inner backend and are
  lifted into the autodiff graph as constants, so only the policy gradient pass is tracked.
- **GRPO math core** reproducing OpenRLHF v0.10.4 / DeepSeekMath: clipped PPO surrogate,
  group-normalized advantage (`GroupNorm` / `DrGrpo`), non-negative k3 KL, and `TokenGlobal` vs
  `SeqMean` reductions (`grpo_loss`, `group_norm_advantage`, `GrpoConfig`, `GrpoMetrics`). Verified
  against golden tensors in `tests/ref/grpo_expected.json` by `tests/grpo_math.rs`.
- **Memory-safe per-token log-probs** (`token_logprobs`) via `gather − logsumexp`, never materializing
  the `[B, T, vocab]` softmax.
- **Rollout engine** (`group_sample`, `group_sample_cached`, `group_sample_padded`, `RolloutConfig`,
  `Rollouts`): O(T) KV-cache decoding, per-sequence EOS masking, raw pre-warp old-logprob capture, and
  left-pad-invariant RoPE positions for ragged prompts.
- **Verifiable Manim reward** (`RewardFn`, `ManimReward`) shelling out to the tested
  `a0/manim_reward.py` harness — static-AST safety gate + dense staged partial credit, with an
  off-by-default sandboxed `manim --dry_run` render stage. Fail-safe: any spawn/exit/timeout/parse
  error scores `0.0`, never panicking or hanging training.
- **Untied output embeddings + sharded safetensors loading** (`load_weights_sharded`) with a strict
  union-coverage check that fails loudly on any weight missing from every shard — enables 8B/14B(/32B).
- **Examples**: `grpo_train` (CPU convergence smoke, no GPU/weights/Python), `grpo_cuda` (GRPO on the
  real Qwen3-0.6B, `Autodiff<Cuda>`), `run_manim_cuda` (untied 14B sharded load + Manim generation).
- **A0 Python reference of record** (`a0/`) and the GRPO integration tests (`grpo_math`,
  `grpo_rollout`, `grpo_trainer`, `grpo_varprompt`, `untied_head`).

### Fixed
- **`top_p` was silently ignored** on the generation path. Generation and GRPO rollouts now share a
  single unit-tested top-k/top-p sampler (`src/sampling.rs`).
- **Whole-batch-stops-on-seq0 rollout bug**: each sequence now tracks its own first EOS and gets an
  independent completion mask.

### Notes
- Phase A only. Phase B (reward worker pool, micro-batched policy pass, dataset + checkpoint/resume)
  is tracked in [docs/GRPO_PLAN.md](docs/GRPO_PLAN.md).
- GRPO-on-CUDA and bf16 are CUDA-validated; the GRPO math also runs and is tested on CPU (NdArray).
- No `qwen3_32b()` preset yet (the config is expressible by hand).

## [0.2.0] — bf16 mixed precision + batch-safe Linear

### Added
- **bf16 mixed-precision training.** A `Precision` enum and a `linear3` compute path run the seven
  Linear GEMMs (q/k/v/o, gate/up/down) in bf16 with f32 accumulation, while master weights, the
  optimizer, RMSNorm, softmax, residual adds, and the tied LM head stay in f32. Training and
  inference precision are decoupled on the model (`with_train_precision` / `with_infer_precision`,
  both default `F32`). See [docs/BF16.md](docs/BF16.md).
- **`matmul_probe` example** — reproduces the CubeCL broadcast batched-matmul correctness bug and
  verifies the 2-D workaround plus bf16 numerical parity (cosine, relative error, batch-safety).
- **`bench_bf16` example** — bf16-vs-f32 forward+backward throughput on Qwen3-0.6B (synthetic data,
  device-synced timing).

### Fixed
- **Batch-safe `Linear` on CubeCL CUDA.** On some `(M, K, N)` shapes the CUDA backend's *broadcast*
  batched matmul (`[B,S,K] @ [1,K,N]`, what `nn::Linear` lowers to for 3-D input) returned wrong
  values for batch > 1. `linear3` flattens to a 2-D GEMM (`[B*S,K] @ [K,N]`), which is mathematically
  identical and correct on every shape tested. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Notes
- bf16 **inference** (`with_infer_precision(Precision::Bf16)`) is not supported yet — it panics with a
  dtype mismatch in `RmsNorm` on the Fusion backend. The setter rejects it; inference runs in f32.
- bf16 is validated on the CubeCL **CUDA** backend (NVIDIA, sm_120/sm_121). Other backends default to
  f32; see the `Precision` docs.

## [0.1.0] — initial

- Qwen3 decoder-only transformer in Burn (GQA, RoPE, RMSNorm, SwiGLU, QK-norm), text generation with
  KV cache, Hugging Face safetensors loading, tokenizer support.
