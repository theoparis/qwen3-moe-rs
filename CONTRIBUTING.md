# Contributing

Thanks for your interest. This is a Qwen3 implementation in [Burn](https://burn.dev/) with a focus
on correct bf16 mixed-precision training on the CubeCL CUDA backend.

## Build & test

```bash
# CPU (NdArray) — the generate example works out of the box, no GPU needed:
cargo build --release --example generate

# CUDA (NVIDIA GPU). On aarch64 (e.g. NVIDIA GB10 / Grace) you MUST set the fp16 target feature:
RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example matmul_probe
RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda,train --example bench_bf16

# Lib check / unit tests:
cargo check --lib
cargo test
```

The project pins a specific Burn pre-release git rev (see `Cargo.toml`). The bf16 path depends on
Burn's runtime dtype and the CubeCL/cubek matmul accumulating bf16 inputs in f32. If you bump Burn,
re-run `matmul_probe` and confirm bf16 parity (cosine ~0.9999, batch-safety `|row0-row1| == 0`)
before trusting the numbers. See [docs/BF16.md](docs/BF16.md).

## Verifying the bf16 path

`matmul_probe` is the regression gate for the two things that matter most:

1. **Batch-safety** — identical input rows must produce identical output rows (`|r0-r1| == 0`). The
   3-D batched path fails this on some shapes; the 2-D `linear3` path does not.
2. **bf16 parity** — bf16 vs f32 matmul cosine and relative Frobenius error stay within bf16
   tolerance (cos ~0.9999, rel ~0.003).

## Style

- Match the surrounding code: comment density, naming, idiom.
- Keep `linear3` the single seam for Linear GEMMs — don't reintroduce the 3-D batched path.
- bf16 is for the compute-heavy matmuls only; norms, softmax, residual, the loss, master weights and
  optimizer state stay f32.

## Pull requests

Small, focused diffs. Include what you changed and how you verified it (probe output, parity numbers).
