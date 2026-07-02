#!/usr/bin/env python3
"""
A0 reference-of-record training: SFT warm-start -> GRPO on a tiny Qwen, Manim verifiable
reward. This is the Python validation the Burn port (A1) must reproduce.

It is intentionally a thin wrapper over HuggingFace TRL (whose GRPOTrainer is the closest
maintained sibling of OpenRLHF's GRPO) so the algorithm is battle-tested, and it wires the
LOCAL reward harness (a0/manim_reward.py) verified by a0/test_reward.py.

PREREQUISITES (not installed in this environment — see a0/README.md):
  pip install -r a0/requirements.txt          # torch transformers trl datasets peft accelerate
  pip install manim                            # for the optional render reward stage
  huggingface-cli login                        # OR export HF_TOKEN=...   (dataset is GATED)
  # accept terms at https://huggingface.co/datasets/BibbyResearch/3blue1brown-manim

Run:
  python3 a0/run_sft_grpo.py --model Qwen/Qwen3-1.7B --steps 200 --group-size 8

What it locks for A1:
  * the reward distribution (via manim_reward.staged_reward)
  * GRPO hyperparams (G, eps, beta, kl estimator, token-global reduction)
  * the convergence curve (mean reward over steps) the Burn run must reproduce
  * the parity tensors are produced separately by a0/grpo_reference.py
"""
import argparse
import sys

from manim_reward import staged_reward

# Reward function in TRL's expected signature: (prompts, completions, **kw) -> list[float].
def manim_reward_func(prompts, completions, **kwargs):
    return [staged_reward(c if isinstance(c, str) else c[0]["content"], allow_render=False)[0]
            for c in completions]


def build_prompt(row):
    """Map a dataset row to a chat prompt. Column names verified after dataset access."""
    instruction = row.get("prompt") or row.get("instruction") or row.get("input") or ""
    return [{"role": "user",
             "content": f"Write Manim (Python) code for this animation.\n\n{instruction}\n\n"
                        f"Return only a single ```python``` code block with a Scene subclass."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dataset", default="BibbyResearch/3blue1brown-manim")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=512)  # verbosity guard (fix j)
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--out", default="a0/out")
    args = ap.parse_args()

    try:
        import torch  # noqa
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    except ImportError as e:
        sys.exit(f"[A0] missing dependency: {e}\n     pip install -r a0/requirements.txt "
                 f"(and `pip install manim`, and authenticate for the gated dataset).")

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(args.dataset, split="train")  # GATED: needs HF auth + accepted terms

    # ---- Stage 1: SFT warm-start on Manim code (so GRPO sees non-zero-advantage groups) ----
    def to_sft(row):
        code = row.get("code") or row.get("output") or row.get("completion") or ""
        msgs = build_prompt(row) + [{"role": "assistant", "content": f"```python\n{code}\n```"}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    sft_ds = ds.map(to_sft)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="bfloat16")
    SFTTrainer(model=model, args=SFTConfig(output_dir=f"{args.out}/sft",
               num_train_epochs=args.sft_epochs, per_device_train_batch_size=2),
               train_dataset=sft_ds).train()

    # ---- Stage 2: GRPO (reproduces OpenRLHF group_norm + k3 KL-in-loss + token-global) ----
    grpo_ds = ds.map(lambda r: {"prompt": tok.apply_chat_template(build_prompt(r), tokenize=False,
                                                                  add_generation_prompt=True)})
    cfg = GRPOConfig(
        output_dir=f"{args.out}/grpo",
        num_generations=args.group_size,           # G
        epsilon=args.eps, beta=args.beta,           # clip eps, KL coef
        max_completion_length=args.max_new_tokens,  # verbosity guard
        scale_rewards=True,                         # group_norm: divide by std (OpenRLHF default)
        max_steps=args.steps,
        logging_steps=1, per_device_train_batch_size=args.group_size,
        # NOTE for parity: TRL's default loss reduction differs across versions; we want
        # OpenRLHF token-global. Set loss_type accordingly and PIN the trl version (requirements).
    )
    trainer = GRPOTrainer(model=f"{args.out}/sft", reward_funcs=manim_reward_func,
                          args=cfg, train_dataset=grpo_ds)
    trainer.train()
    print(f"[A0] done. Inspect {args.out}/grpo for the reward curve A1 must reproduce.")


if __name__ == "__main__":
    main()
