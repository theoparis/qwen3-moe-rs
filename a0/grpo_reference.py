#!/usr/bin/env python3
"""
A0 GRPO numerical reference (reference of record for the Burn port).

Implements OpenRLHF v0.10.4 GRPO math EXACTLY on fixed tensors and emits the
expected intermediates + final loss to tests/ref/grpo_expected.json. The Rust
`tests/grpo_math.rs` parity test diffs its outputs against this file.

Dependencies: numpy only (no torch / no GPU). Deterministic (fixed arrays).

OpenRLHF contract reproduced here (see docs/GRPO_PLAN.md §2, §2b):
  advantage (group_norm):  A_i = (r_i - mean_G) / (std_G + 1e-9), std = sample std (ddof=1)
                           broadcast per-sequence to every response token; NO global whitening
  per-token logprob:       logp[n,t] = logit[n,t,target] - logsumexp(logit[n,t,:])   (gather - lse)
  ratio:                   rho = exp(clamp(logp_pi - logp_old, -20, 20))
  surrogate (per token):   L = -min(rho*A, clip(rho, 1-eps, 1+eps)*A),  eps=0.2
  KL (k3, in loss):        delta = logp_pi - logp_ref
                           KL = clamp(exp(-delta) - 1 + delta, -10, 10)        (>=0, unbiased)
  reduction:               token-level GLOBAL mean = sum(X*mask)/sum(mask)     (OpenRLHF-literal)
  total loss:              mean_tok(L) + beta*mean_tok(KL),  beta=1e-3
"""
import json
import os
import numpy as np


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def token_logprobs(logits, target_ids):
    """logp[n,t] = logit[n,t,target] - logsumexp(logit[n,t,:]). Shape [N,T]."""
    N, T, V = logits.shape
    gathered = np.take_along_axis(logits, target_ids[:, :, None], axis=2).squeeze(2)  # [N,T]
    return gathered - logsumexp(logits, axis=2)


def group_norm_advantage(rewards, P, G, std_norm=True, eps=1e-9):
    """OpenRLHF group_norm: (r - mean_G)/(std_G+eps) per prompt group. rewards [P*G]."""
    r = rewards.reshape(P, G)
    centered = r - r.mean(axis=1, keepdims=True)
    if std_norm:
        std = r.std(axis=1, ddof=1, keepdims=True)  # sample std (N-1), matches torch default
        adv = centered / (std + eps)
    else:  # dr_grpo / reinforce_baseline: mean-only
        adv = centered
    return adv.reshape(P * G)


def grpo_loss(logp_pi, logp_old, logp_ref, adv_seq, mask, eps_low=0.2, eps_high=0.2,
              beta=1e-3, kl_clip=10.0, ratio_logclip=20.0):
    """All [N,T] except adv_seq [N]. Returns dict of intermediates + scalars."""
    N, T = logp_pi.shape
    adv = np.broadcast_to(adv_seq[:, None], (N, T))                      # broadcast to tokens

    log_ratio = np.clip(logp_pi - logp_old, -ratio_logclip, ratio_logclip)
    ratio = np.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    l_pol = -np.minimum(surr1, surr2)                                   # per-token policy loss

    delta = logp_pi - logp_ref
    kl = np.clip(np.exp(-delta) - 1.0 + delta, -kl_clip, kl_clip)       # k3, >=0

    msum = mask.sum()
    pol_loss = float((l_pol * mask).sum() / msum)                       # token-global mean
    kl_loss = float((kl * mask).sum() / msum)
    total = pol_loss + beta * kl_loss

    clip_frac = float(((surr2 < surr1) * mask).sum() / msum)            # fraction clipped
    return dict(ratio=ratio, l_pol=l_pol, kl=kl, pol_loss=pol_loss,
                kl_loss=kl_loss, total_loss=total, clip_frac=clip_frac,
                mean_ratio=float((ratio * mask).sum() / msum))


def build_fixture():
    rng = np.random.default_rng(0)
    P, G, T, V = 2, 2, 5, 7          # 2 prompts x 2 completions = 4 seqs, len 5, vocab 7
    N = P * G
    logits_pi = rng.standard_normal((N, T, V)).astype(np.float64)
    logits_old = logits_pi + 0.05 * rng.standard_normal((N, T, V))      # old != pi (ratio != 1)
    logits_ref = logits_pi + 0.10 * rng.standard_normal((N, T, V))      # frozen ref
    target_ids = rng.integers(0, V, size=(N, T)).astype(np.int64)
    rewards = np.array([1.0, 0.0, 0.3, 0.9], dtype=np.float64)          # distinct within groups
    # completion mask: response region varies per sequence (tests masking + off-by-one)
    mask = np.zeros((N, T), dtype=np.float64)
    resp_start = [2, 1, 3, 2]
    for n, s in enumerate(resp_start):
        mask[n, s:] = 1.0
    return dict(P=P, G=G, T=T, V=V, N=N, logits_pi=logits_pi, logits_old=logits_old,
                logits_ref=logits_ref, target_ids=target_ids, rewards=rewards, mask=mask)


def main():
    fx = build_fixture()
    logp_pi = token_logprobs(fx["logits_pi"], fx["target_ids"])
    logp_old = token_logprobs(fx["logits_old"], fx["target_ids"])
    logp_ref = token_logprobs(fx["logits_ref"], fx["target_ids"])
    adv = group_norm_advantage(fx["rewards"], fx["P"], fx["G"], std_norm=True)
    out = grpo_loss(logp_pi, logp_old, logp_ref, adv, fx["mask"])

    # --- self-checks (independent recomputation) ---
    # advantage of group 0 = [1.0, 0.0]: mean .5, sample std = |1-0|/sqrt(2)=0.70710678
    a0 = (1.0 - 0.5) / (np.std([1.0, 0.0], ddof=1) + 1e-9)
    assert abs(adv[0] - a0) < 1e-9, (adv[0], a0)
    assert adv[0] == -adv[1], "group must be antisymmetric for G=2 distinct rewards"
    assert np.isfinite(out["total_loss"]), "loss must be finite"
    assert (out["kl"] >= -1e-12).all(), "k3 KL must be >= 0"
    # ratio at a token where logp_pi==logp_old would be exactly 1; check exp/clamp sane
    assert out["mean_ratio"] > 0
    print("self-checks PASSED")

    expected = {
        "_doc": "OpenRLHF v0.10.4 GRPO reference; Rust tests/grpo_math.rs must match within tol.",
        "config": {"P": fx["P"], "G": fx["G"], "T": fx["T"], "V": fx["V"],
                   "eps_low": 0.2, "eps_high": 0.2, "beta": 1e-3,
                   "advantage_estimator": "group_norm", "reduction": "token_global",
                   "kl_estimator": "k3", "kl_placement": "loss"},
        "inputs": {
            "logits_pi": fx["logits_pi"].tolist(),
            "logits_old": fx["logits_old"].tolist(),
            "logits_ref": fx["logits_ref"].tolist(),
            "target_ids": fx["target_ids"].tolist(),
            "rewards": fx["rewards"].tolist(),
            "mask": fx["mask"].tolist(),
        },
        "expected": {
            "logp_pi": logp_pi.tolist(),
            "logp_old": logp_old.tolist(),
            "logp_ref": logp_ref.tolist(),
            "advantages": adv.tolist(),
            "ratio": out["ratio"].tolist(),
            "l_pol": out["l_pol"].tolist(),
            "kl": out["kl"].tolist(),
            "pol_loss": out["pol_loss"],
            "kl_loss": out["kl_loss"],
            "total_loss": out["total_loss"],
            "clip_frac": out["clip_frac"],
            "mean_ratio": out["mean_ratio"],
        },
    }
    ref_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "ref")
    os.makedirs(ref_dir, exist_ok=True)
    path = os.path.join(ref_dir, "grpo_expected.json")
    with open(path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"wrote {os.path.normpath(path)}")
    print(f"  advantages = {np.round(adv, 6).tolist()}")
    print(f"  pol_loss   = {out['pol_loss']:.8f}")
    print(f"  kl_loss    = {out['kl_loss']:.8f}")
    print(f"  total_loss = {out['total_loss']:.8f}")
    print(f"  clip_frac  = {out['clip_frac']:.4f}  mean_ratio = {out['mean_ratio']:.6f}")


if __name__ == "__main__":
    main()
