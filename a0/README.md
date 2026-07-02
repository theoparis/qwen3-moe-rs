# A0 — Python validation (reference of record for the Burn GRPO port)

A0 de-risks the GRPO algorithm and the verifiable Manim reward in Python **before** the
Rust/Burn port (A1). Decided in the engineering review (`docs/GRPO_PLAN.md` §0b). A1 must
reproduce A0's numbers within tolerance.

## Status

| Artifact | Needs | Status |
|----------|-------|--------|
| `grpo_reference.py` — OpenRLHF GRPO math on fixed tensors → `../tests/ref/grpo_expected.json` | numpy only | **DONE, self-checks pass** |
| `manim_reward.py` — staged verifiable reward + safety pipeline | stdlib only | **DONE** |
| `test_reward.py` — reward unit tests (safety, variance) | numpy + stdlib | **DONE, all pass** |
| `run_sft_grpo.py` — SFT warm-start → GRPO on tiny Qwen (TRL) | torch/trl/datasets/manim + GPU + HF auth | **skeleton; needs prereqs** |

Verified here (no heavy deps):
```bash
python3 a0/grpo_reference.py     # writes tests/ref/grpo_expected.json, prints self-checks
cd a0 && python3 test_reward.py  # reward harness: safety gate + variance guarantee
```

## What each piece locks for A1 (the Burn port)

- **`tests/ref/grpo_expected.json`** — exact `logp`, `advantages`, `ratio`, per-token policy
  loss, k3 KL, and final `pol_loss`/`kl_loss`/`total_loss` on fixed inputs. The Rust
  `tests/grpo_math.rs` parity test diffs against this (cosine > 0.9999 / abs < 1e-5).
  Reproduces OpenRLHF v0.10.4: group_norm advantage `(r-mean)/(std+1e-9)` (sample std),
  token-global reduction, k3 KL in the loss, ratio clamp [-20,20], eps 0.2, beta 1e-3.
- **`manim_reward.py`** — the reward contract: static AST scoring (never executes model
  code), dense staged partial credit (parses → manim import → Scene → construct → anim calls
  → optional sandboxed render), hard-0 for forbidden imports/calls/dunder escapes. The Rust
  `reward.rs` mirrors this staging; the sandbox uses the same setsid + rlimit + killpg shape.

## Prerequisites for the full SFT+GRPO run (`run_sft_grpo.py`)

1. `pip install -r a0/requirements.txt` (pin versions — TRL's reduction/KL differ across releases).
2. `pip install manim` + system libs (cairo, pango, ffmpeg, latex) for the render reward stage.
3. Dataset is **gated**: accept terms at
   https://huggingface.co/datasets/BibbyResearch/3blue1brown-manim and
   `export HF_TOKEN=...` (else HTTP 401). Columns are confirmed after first access; the
   loader handles `prompt`/`code` (with fallbacks).
4. A GPU (the GB10 works for the 1.7B run).

Then:
```bash
python3 a0/run_sft_grpo.py --model Qwen/Qwen3-1.7B --steps 200 --group-size 8
```
Success criterion: mean reward rises over steps (convergence). Freeze the resulting
hyperparams + reward spec + metric curve; A1 must reproduce them.

## Gate to A1
Do not start the Burn loss/advantage port until `tests/ref/grpo_expected.json` exists
(done) and the convergence run shows reward rising (needs prereqs above).
