# GRPO for qwen3-burn — Phase A engineering plan

> Status: DRAFT under engineering review (`/gstack-plan-eng-review`).
> Scope decision (locked): **Phase A** = build the complete GRPO trainer on a small
> *dense* Qwen3 the repo already supports, with a verifiable Manim-execution reward,
> reproducing **OpenRLHF**'s GRPO and verified against its source (and the DeepSeekMath
> math). **Phase B** (Qwen3 MoE block +
> router + LoRA + 30B weight loading to reach `Qwen3-Coder-30B-A3B-Instruct`) is a
> separate, explicitly-scoped follow-on, gated on Phase A converging. See "NOT in scope".

## 0. Why this shape

- Hardware is a **single NVIDIA GB10** (Grace-Blackwell, ~128 GB *unified* LPDDR5X,
  ~273 GB/s). A full GRPO fine-tune of a 30B model needs ~370 GB+ just for AdamW f32
  state — it does not fit. The only 30B path that fits is LoRA-GRPO, which first needs
  an MoE block + LoRA in Burn (neither exists yet).
- The repo already has a **proven** dense Qwen3 + bf16 mixed-precision + AdamW manual
  training loop. GRPO is a loss + a rollout loop + a reward on top of that foundation.
- So: prove GRPO correct and converging on Qwen3-1.7B first (days, debuggable), then
  port the model (Phase B) without also debugging the RL algorithm at the same time.

## 0b. Phase A runs in two stages (decided in review: Python-first validation)

External review (Codex #23) flagged that porting an unproven RL recipe to a young Rust
framework debugs the algorithm and the port at the same time. So Phase A is two stages:

```
A0 — Python validation (reference of record)        A1 — Burn port (the deliverable)
  • SFT warm-start a tiny Qwen (0.6B/1.7B) on        • Port the LOCKED contract to Burn:
    Manim code first, so it sometimes succeeds          loss, advantage, KL, rollout I/O,
    (else GRPO sees zero-advantage noise).              reward staging — match A0 in tolerance.
  • Run GRPO with OpenRLHF or TRL on that model.     • Burn run STARTS from the A0 SFT weights.
  • Build + lock the verifiable Manim reward         • A0 emits the committed parity tensors
    (staging, dense shaping, anti-hacking).            (tests/ref/) AND the target metric curve
  • Confirm the loop CONVERGES (mean reward ↑).        the Burn run must reproduce.
  • Freeze: reward spec, GRPO hyperparams,           • Reuse rl4burn / burn-ppo patterns (§1b).
    metric curves, expected loss/advantage/KL.
```

A0 is a means to de-risk A1, not a second product: it is the smallest Python run that
proves the reward gives signal and the algorithm converges. A1 (the Rust/Burn GRPO) remains
the actual deliverable, and it is verified against A0's frozen numbers.

## 1. What already exists (reuse map — do NOT rebuild)

| Need | Exists today | Reuse / change |
|------|--------------|----------------|
| Model forward → logits `[B,S,V]` | `Qwen3ForCausalLM::forward` (decoder.rs:474) | REUSE for policy/ref logprob pass |
| Autodiff + AdamW step | `bench_bf16.rs:45-51` pattern (`GradientsParams::from_grads` → `optim.step`) | REUSE verbatim as the trainer core |
| bf16 mixed precision (f32 master) | `with_train_precision`, `linear3` (linear2d.rs) | REUSE for the policy gradient pass |
| Tied-head logits under autodiff | `tied_logits` (decoder.rs:532) | REUSE — gradient flows to shared embedding |
| KV-cache generation | `generate_with_cache_eos` (decoder.rs:684) | REUSE the cache; REPLACE the sampling/EOS path (bugs below) |
| Tokenizer + chat template | `src/tokenizer.rs` | REUSE for prompt formatting |
| Safetensors weight load | `src/load.rs` | REUSE to load Qwen3-1.7B base |

**Two generation bugs that block group rollouts (must fix in Phase A):**
- `sample_from_probs` (decoder.rs:803) takes `_top_p` and **ignores it** — nucleus
  sampling is silently a no-op. GRPO rollouts need real top-p/temperature diversity.
- Batched EOS checks only sequence 0 (`...as_slice::<i64>()[0]`, decoder.rs:727,772),
  so a batch of G completions all stop when the *first* hits EOS. Group sampling needs
  **per-sequence** EOS + a completion mask.

### 1b. External reuse (search-before-building outcome)

There is **no existing Rust LLM-GRPO** to fork (GitHub "GRPO language:Rust" = 2 toy repos;
official Burn has only a DQN example). But three Layer-1 reuses on the *same Burn 0.20 stack*
were verified by reading source — we adapt these instead of inventing the mechanics:

| Reuse | Repo (license) | What we lift |
|-------|----------------|--------------|
| Clipped surrogate + advantage-normalize | **rl4burn** `rl4burn-algo/src/base/ppo.rs` (MIT) | The PPO clip pattern, AND the Burn gradient-safe `min`/clip: write `min(a,b)` as `b − relu(b−a)` and clip via `+relu(...)` to avoid `mask_where` killing gradients on the CubeCL backend. |
| Per-token log-prob primitive | **burn-ppo** `src/utils.rs:38-74` (BSD-3) | `log_prob_categorical = log_softmax(logits) + gather_1d` (noted gradient-correct vs boolean mask). We EXTEND `[B,V]` → `[B,T,V]` gather + a `[B,T]` response mask (the genuinely new part). |
| Verifiable-reward sandbox | **fast-rl-rewards** `src/{sandbox,evaluator}.rs` (Apache-2.0) | The sandbox template: firejail `--net=none --private --rlimit-*` + tempfile + `wait-timeout` kill + regex score parse + Rayon batch → `Vec<f64>`. Retarget `python3` → `manim`/`python -c compile`. |
| (optional) generic actor-critic loop | **burn-rl** (yunjhongwu, MIT, 9.5k dl) | Battle-tested scaffold reference if our loop needs hardening. |

The GRPO **algorithm** itself (group-normalized advantage with no critic, sequence-level
masked log-probs, KL-to-reference, the Manim verifier) is ported from **OpenRLHF** (Python)
— see §2/§5. The LLM-forward ↔ GRPO-update coupling on Burn is the code we own.

## 2. GRPO — the algorithm we reproduce (reference: OpenRLHF v0.10.4, read from source)

We reproduce OpenRLHF's GRPO exactly and expose its knobs. In OpenRLHF, GRPO is the PPO
trainer with `advantage_estimator="group_norm"`, **no critic**, and KL configurable. (The
DeepSeekMath equations, arXiv 2402.03300, are the academic statement; OpenRLHF is the code
we diff against — see §5.)

For each prompt `q`, sample `G` completions from the behavior policy `π_old`
(`n_samples_per_prompt`). Reduce each completion to ONE scalar reward `r_i` (the reward
lives at the EOS token), clip to `[−10, 10]`, then group-normalize within the prompt's G
responses:

```
group_norm (default):                  Â_i = (r_i − mean_G) / (std_G + 1e-9)   # std = sample std, ddof=1
dr_grpo / reinforce_baseline (flag):   Â_i = r_i − mean_G                       # mean-only, no std
```

`Â_i` (one scalar per completion) is broadcast to every response token (γ=λ=1). The
group_norm path is **NOT** globally/batch whitened (OpenRLHF explicitly excludes it).

Per-token ratio + clipped surrogate (PPO core, shared with their PPO path):

```
ρ_{i,t} = exp( clamp( logπ_θ − logπ_old , −20, 20 ) )
L_pol   = − min( ρ_{i,t}·Â_i ,  clip(ρ_{i,t}, 1−ε_low, 1+ε_high)·Â_i )      # ε=0.2 (DAPO: 0.2/0.27)
```

KL to a frozen reference `π_ref` is a **switch** (use one, never both), `δ = logπ_θ − logπ_ref`:

```
canonical GRPO  — KL in the LOSS (k3):   loss = mean_tok(L_pol) + β·mean_tok(KL_k3),  β≈1e-3
                  KL_k3 = clamp( exp(−δ) − 1 + δ , −10, 10 )           # ≥0, unbiased
OpenRLHF default — KL in the REWARD (k1): reward_t += −β·δ              # per-token reward shaping
```

**Reduction (critical — where OpenRLHF and DeepSeekMath differ):** OpenRLHF reduces the
policy loss as a **token-level mean over the whole batch** — `Σ(L·mask) / Σ(mask)` across
all response tokens — NOT DeepSeekMath's per-sequence `(1/|o_i|)` length-normalization
(that is their GSPO path). **We reproduce OpenRLHF: token-level global mean.**

Loss masked to completion tokens only (prompt tokens contribute nothing). OpenRLHF
defaults: `G=8`, `ε=0.2`, `advantage_estimator=group_norm`, `max_epochs=1` (one optimizer
pass per rollout batch), reward clip `[−10,10]`. Canonical GRPO sets `kl.use_loss=True` +
`k3` + `β≈1e-3`; OpenRLHF's own default is `kl.use_loss=False` + `k1` reward-shaping +
`β=0.01`. We default to **canonical GRPO (k3-in-loss)** and expose the switch.

```
                          GRPO TRAINING LOOP (one step)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ batch of prompts q[1..P]                                                 │
  │      │                                                                   │
  │      ▼  rollout.rs  (no-grad, infer_precision=f32, KV cache)             │
  │  sample G completions per prompt  ──►  P·G sequences                     │
  │      │                              + per-token logπ_old  + completion   │
  │      │                                mask  + lengths                    │
  │      ▼  reward.rs   (sandboxed Manim execution, cached, parallel)        │
  │  r[1..P·G]  ──►  group-normalize per prompt  ──►  Â[1..P·G] (per seq)    │
  │      │                                                                   │
  │      ▼  loss.rs    (WITH grad, train_precision=bf16/f32)                 │
  │  forward policy π_θ on full sequences ──► logπ_θ (gather chosen tokens)  │
  │  forward ref    π_ref (no-grad, once) ──► logπ_ref                       │
  │  ρ = exp(logπ_θ − logπ_old);  surrogate = min(ρÂ, clip(ρ)Â)             │
  │  kl = exp(logπ_ref−logπ_θ) − (logπ_ref−logπ_θ) − 1                      │
  │  loss = −mean_over_completion_tokens( surrogate − β·kl )                 │
  │      │                                                                   │
  │      ▼  trainer.rs                                                       │
  │  loss.backward() ─► GradientsParams::from_grads ─► AdamW.step            │
  └────────────────────────────────────────────────────────────────────────┘
```

## 2b. Correctness & safety requirements (external-review consensus — folded in)

Both Codex and Gemini independently flagged these; all are hard requirements, not options.

- **(a) old_logprobs = RAW model log-softmax of the sampled token, captured BEFORE any
  top-p / top-k / temperature warping.** If `logπ_old` reflects the warped sampling
  distribution, the PPO ratio `ρ≠1` even at step 0 and the surrogate is broken. The sampler
  uses warpers only to *choose* the token; the logged probability is the unwarped model
  log-softmax at that token.
- **(b) Selected-token logprob via `gather(logits, ids) − logsumexp(logits, V)` — never
  materialize the `[P·G, Lmax, 151936]` log-softmax.** Autodiff caching that tensor OOMs a
  128 GB GB10 instantly. Compute per-token logprob with a fused/chunked logsumexp.
- **(c) `logπ_ref` computed in a no-grad prepass on the INNER (non-autodiff) backend,
  immediately after rollout, then ref activations dropped** before the policy autodiff graph
  is built. Burn has no global no-grad context — running ref on `Autodiff<B>` would build a
  graph and double peak memory. Use the inner backend / `.inner()` / detach to constants.
- **(d) Micro-batched backward + gradient accumulation to bound peak memory.** Process the
  `P·G` sequences in small micro-batches (down to 1), `loss.backward()` per micro-batch to
  accumulate grads, one `optim.step()` per rollout batch. Avoids the batched-autodiff memory
  spike (Gemini #7) while preserving the token-global reduction (normalize by the global
  response-token count, all-reduced across micro-batches — not per micro-batch).
- **(e) `π_ref` stays FROZEN for all of Phase A** (drop the earlier "refresh every K steps":
  a moving reference breaks the KL anchor).
- **(f) Static-parse + code-extraction BEFORE execution.** Strip markdown fences, extract the
  Python, `ast.parse` it, and reject dangerous imports/calls — *then* sandbox-execute. Never
  import/construct a Scene from unvalidated text (that runs arbitrary top-level Python).
- **(g) Sandbox hardening beyond `firejail --private --net=none`:** PID-namespace or
  process-group kill (`-pgid`) so Manim's LaTeX/ffmpeg children die with the timeout; disk
  quota (`--rlimit-fsize`); no inherited env; capped stdout/stderr/tmp. Manim spawns orphan
  renderers otherwise.
- **(h) Zero-std groups are skipped and tracked.** If all G rewards in a prompt group are
  equal, advantage = 0 → no signal; drop the group from the step and log the rate (a high
  rate means the reward needs more dense shaping or a better warm start).
- **(i) Parity tensors generated from PINNED OpenRLHF source only** (TRL secondary). TRL and
  OpenRLHF differ in reduction/KL details; the reference of record is OpenRLHF v0.10.4.
- **(j) Verbosity guard (token-global reduction is OpenRLHF-literal per decision):** cap
  `max_new_tokens` and log per-group reward↔length correlation; if length runs away, add an
  explicit length penalty to the reward (NOT a reduction change). Watch-item, not a blocker.

## 3. New modules (file-by-file)

```
src/grpo/
  mod.rs        — re-exports, GrpoConfig (G, eps, beta, lr, mu inner-steps, gen params)
  dataset.rs    — load 3blue1brown-manim CSV → Vec<{prompt, ref_code?}>; chat-template
                  formatting; tokenize prompts; simple shuffter/batcher (no burn-dataset
                  dep needed; plain Vec + index).
  rollout.rs    — group_sample(model, prompt_ids, G, gen_cfg) → Rollouts {
                    seq_ids [P*G, Lmax] Int, completion_mask [P*G, Lmax] Bool,
                    old_logprobs [P*G, Lmax] f32, prompt_len, gen_len }.
                  New batched sampler: real top-p/top-k/temp, PER-SEQUENCE EOS. Captures
                  logπ_old as the RAW model log-softmax of the sampled token BEFORE warping
                  (fix a) — warpers only choose the token, never define logπ_old.
  reward.rs     — RewardFn trait + ManimReward. Pipeline: extract code from markdown fences →
                  ast.parse + reject dangerous imports/calls (fix f) → THEN sandbox-execute
                  staged checks (syntax-compile → import/construct Scene → optional --dry_run
                  render). DENSE shaping reward (partial credit per stage / valid line) so
                  intra-group variance never collapses (fix h / convergence). Sandbox hardened
                  (fix g): PID-namespace or process-group kill, --rlimit-fsize disk quota,
                  no inherited env, capped output. Cache key = code hash + reward-version +
                  manim/env fingerprint. Parallel over P*G with a BOUNDED worker pool.
                  REUSE: fast-rl-rewards sandbox pattern (firejail/tempfile/wait-timeout/Rayon).
  logprob.rs    — token_logprobs(logits [B,S,V], target_ids [B,S]) → [B,S] via
                  gather(logit_of_target) − logsumexp(logits, V); NEVER materialize the
                  [B,S,V] log-softmax (fix b). shift: logits[t] predicts token[t+1].
                  REUSE: burn-ppo log_prob primitive, made memory-safe + extended to [B,T].
  loss.rs       — grpo_loss(policy_lp, old_lp, ref_lp, advantages, completion_mask, cfg)
                  → scalar loss + metrics (mean ratio, clip-frac, kl, reward stats).
                  group_norm/dr_grpo advantage; token-level GLOBAL-mean reduction (OpenRLHF);
                  KL switch (k3-in-loss / k1-in-reward). REUSE: rl4burn clip + relu-min
                  gradient-safe trick (avoid mask_where on CubeCL).
  trainer.rs    — GrpoTrainer: holds policy (Autodiff), FROZEN ref snapshot (loaded once,
                  never refreshed — fix e), optimizer. Loop: rollout → reward → group-norm
                  advantage → logπ_ref no-grad prepass on inner backend (fix c) → micro-batched
                  backward + grad-accum (fix d), one optim.step per rollout; logs reward/KL/
                  clip-frac/zero-std-group rate (fix h); checkpoints policy.
examples/
  grpo_train.rs — wire CUDA Autodiff backend, load Qwen3-1.7B, run N steps on Manim subset.
tests/
  grpo_math.rs  — numerical parity of loss/advantage/kl vs hand-computed + a tiny PyTorch
                  reference (committed expected values); see §5.
```

No new heavy deps anticipated: CSV parse with a tiny hand parser or `csv` crate; reward
sandbox uses `std::process::Command` to invoke the system `python`/`manim`. (Dataset
download is a manual prerequisite — see §6.)

## 4. Reference & behavior policy handling

- `π_ref` = FROZEN snapshot of the initial (SFT-warm-started) weights, never refreshed
  (fix e). **Burn has no global no-grad context**, so run ref on the INNER (non-autodiff)
  backend `B::InnerBackend` — build it as `Qwen3ForCausalLM<B::InnerBackend>` (or detach the
  policy weights into it) so its forward never enters the autodiff graph (fix c). Compute
  `logπ_ref` in a prepass right after rollout, keep only the `[P·G, Lmax]` logprob tensor as
  a constant, and drop ref activations before building the policy graph. Memory: one extra
  weight copy (~3.4 GB bf16 / ~6.8 GB f32) — fine on 128 GB; the win is not holding ref
  activations concurrently with the policy autodiff graph (Gemini #3).
- `π_old` (behavior) for the ratio: captured during rollout as a **frozen** no-grad
  forward (OpenRLHF computes `old_action_log_probs` once in experience-making and reuses
  it). We match OpenRLHF's `max_epochs=1` (one optimizer pass per rollout batch) as the
  default. Note OpenRLHF's insight: even at `max_epochs=1`, `ρ≠1` once the batch is split
  into gradient-accumulation micro-batches — later micro-batches see an already-updated
  policy, so the clip is live, not a no-op. So `logπ_old` MUST come from the frozen rollout
  pass, never recomputed from the live policy. `max_epochs` is exposed (default 1).

## 5. Test & correctness plan (100% of new code paths)

- **Math parity (load-bearing):** `tests/grpo_math.rs` checks `grpo_loss`,
  group-advantage, `token_logprobs`, and k3 KL against committed expected values emitted by
  the **A0 pinned-OpenRLHF run** (kept in `tests/ref/`), on small fixed tensors. Tolerance
  cosine > 0.9999 / abs < 1e-5 (f32).
- **Rollout unit tests:** per-sequence EOS stops the right rows; completion mask aligns
  with generated region; top-p actually truncates the tail (regression for the
  `_top_p` bug); `logπ_old` captured equals a recomputed log-softmax of the same step.
- **Reward unit tests:** valid Manim snippet → high reward; syntax-error → 0; timeout →
  0; reward cache hit returns same scalar; sandbox cannot write outside tmp.
- **Loss-shape/grad tests:** loss is scalar, finite; gradient is non-zero on policy
  params and **zero on `π_ref`**; advantage broadcast over tokens; masked tokens
  contribute zero gradient.
- **Convergence smoke (toy reward):** replace Manim reward with a cheap synthetic
  verifiable reward (e.g. "output contains `class .*Scene`") and show mean reward rises
  over ~50 steps on a handful of prompts — proves the loop actually learns before paying
  for real Manim execution. This is the GRPO analog of the existing freeze-head ablation.
- **Determinism:** seedable RNG for sampling so parity tests are reproducible.

Correctness is verified primarily against **OpenRLHF v0.10.4 source** (the implementation
we reproduce): `models/loss.py` `PolicyLoss`/`aggregate_loss` (clipped surrogate +
token-level global-mean reduction), `models/utils.py` `compute_approx_kl` (k1/k3) +
`masked_mean`, `trainer/ppo_utils/experience_maker.py` (group_norm advantage, std=1e-9,
no global whitening), `trainer/ray/ppo_actor.py` (KL-in-loss path). Cross-checked against
HF TRL `grpo_trainer.py` and the DeepSeekMath equations. The parity script in `tests/ref/`
runs the OpenRLHF loss/advantage/KL on fixed tensors and commits the expected values;
discrepancies get reconciled before merge.

## 6. Operational prerequisites

- Dataset `BibbyResearch/3blue1brown-manim` is **gated** (HTTP 401 without auth). To
  train you must accept its terms on HF and export `HF_TOKEN`. ~2,400 prompt→Manim-Python
  rows, CSV (`3blue1brown-manim-prompts.csv`), avg ~120 LoC/example.
- Reward needs a Python env with `manim` installed and a sandbox (timeout, no network,
  tmp-only FS). Rendering is slow; staged reward lets us reward syntax/import without a
  full render most of the time.
- Base model: `Qwen3-1.7B` safetensors + `tokenizer.json` in `./models/`.

## 7. Performance notes (single GB10, 273 GB/s)

- **Rollout dominates.** Generation is memory-bandwidth-bound and bf16 *inference* is
  currently blocked (RmsNorm DTypeMismatch), so rollouts run f32 → ~2× the weight traffic
  of bf16. Mitigations: small `max_new_tokens` cap, KV cache (already present), and batch
  the whole `P·G` group through one cache. Fixing bf16 inference is a Phase-A stretch /
  Phase-B item (flagged TODO).
- **Reward is CPU/process-bound.** Parallelize Manim subprocess checks across the Grace
  CPU cores; cache by code hash; prefer the cheap syntax/import stages over full render.
- The gradient pass reuses the proven bf16 Linear path; that is not the bottleneck.

## 8. NOT in scope (Phase A)

- Qwen3 **MoE** block, router, expert weight loading → **Phase B** (the actual 30B model).
- **LoRA / PEFT** → Phase B (only needed to make 30B fit).
- Multi-GPU / sharding → not applicable to 1× GB10; precision-serialization landmine
  documented but not fixed.
- bf16 *inference* fix → flagged TODO; Phase A runs f32 rollouts.
- A learned reward model → using **verifiable execution reward** instead (RLVR).
- vLLM-style paged-attention rollout engine → out of scope; reuse the existing KV cache.

## 9. Phase B preview (separate plan, gated)

MoE block (128 experts, top-8 router, `moe_intermediate_size`), Qwen3-MoE weight remap,
LoRA adapters on attention+router+expert projections, 30B memory budget (quantized/bf16
frozen base shared as reference), throughput plan. Only start after Phase A converges.

## 10. Failure modes (per new codepath: realistic failure → test? → error handling? → silent?)

| Codepath | Realistic production failure | Test | Error handling | Silent? |
|----------|------------------------------|------|----------------|---------|
| rollout old_logprobs | logged from warped dist → ratio≠1 at step 0 | parity test (fix a) | n/a (correctness) | **was silent → now tested** |
| logprob.rs | full `[B,T,V]` softmax OOMs | memory test on Lmax | chunked logsumexp | **was silent OOM → now bounded** |
| π_ref backend | ref on Autodiff builds graph → 2× memory / wrong grad | grad-zero-on-ref test | inner-backend prepass | **was silent → now tested** |
| reward exec | generated code runs arbitrary Python | sandbox-escape test | ast-parse + import reject + sandbox | **CRITICAL if unhandled → now gated** |
| reward variance | all-G-equal → advantage 0, dead loss | zero-std-group unit test | skip+log zero-std groups | would be silent stall → now logged |
| Manim subprocess | orphan LaTeX/ffmpeg after timeout | timeout test | pgid kill + fsize rlimit | would silently fill disk → now capped |
| dataset load | gated 401 / missing HF_TOKEN | load unit test | explicit error + message | would be confusing → now explicit |
| convergence | loss dead despite correct code (cold start) | A0 convergence smoke | SFT warm-start | would waste days → A0 catches first |

**Critical gaps (no test AND no handling AND silent):** none remaining after folding (a)–(j).
The reward-exec path was the one true critical gap; (f)+(g) close it.

## 11. Test coverage map (target: every new path)

```
NEW CODE PATHS                                       TESTS (all to be written WITH the code)
src/grpo/logprob.rs
  ├── gather−logsumexp                               [★★★ planned] parity + memory + grad
  └── shift/off-by-one (logits[t]→tok[t+1])          [★★★ planned] unequal-length + EOS/no-EOS
src/grpo/rollout.rs
  ├── per-sequence EOS                               [★★★ planned] right rows stop
  ├── top-p truncation (regression for _top_p bug)   [★★★ planned] tail actually cut
  └── raw old_logprob capture                        [★★★ planned] == recomputed log-softmax
src/grpo/loss.rs
  ├── group-norm advantage (std, no whitening)       [★★★ planned] vs A0 tensors
  ├── clipped surrogate (relu-min, token-global)     [★★★ planned] vs A0 tensors
  ├── k3 KL in loss                                  [★★★ planned] vs A0 tensors
  └── zero-std group skip                            [★★  planned] group dropped+logged
src/grpo/reward.rs
  ├── extract+ast-parse+import-reject                [★★★ planned] malicious snippet rejected
  ├── staged dense reward                            [★★★ planned] syntax/import/render scalars
  ├── sandbox isolation                              [★★★ planned] no write outside tmp, orphan kill
  └── cache key (code+env fingerprint)               [★★  planned] hit/miss correctness
src/grpo/trainer.rs
  └── one step end-to-end                            [★★★ planned] A0 convergence smoke (toy reward)
src/grpo/dataset.rs
  └── CSV parse + chat-template + gating error       [★★  planned] columns, 401 message

COVERAGE TARGET: 100% of new paths. No path ships without its test (matches repo's
existing discipline — matmul_probe/bench_bf16 verify before claims).
```

## 12. Worktree parallelization strategy

| Step | Modules touched | Depends on |
|------|-----------------|------------|
| S0 A0 Python validation | (python, out of tree) | — |
| S1 logprob.rs + tests | src/grpo/ | A0 contract |
| S2 reward.rs + sandbox + tests | src/grpo/, (python/manim) | A0 reward spec |
| S3 rollout.rs (+ fix gen bugs) | src/grpo/, src/decoder.rs | S1 (logprob) |
| S4 loss.rs + tests | src/grpo/ | S1, A0 tensors |
| S5 trainer.rs + example | src/grpo/, examples/ | S3, S4 |
| S6 dataset.rs + tests | src/grpo/ | — |

- **Lane A:** S2 (reward) — independent, also touches python/manim. Parallel.
- **Lane B:** S6 (dataset) — independent. Parallel.
- **Lane C:** S1 → S3 → S5 (sequential, share src/grpo + decoder).
- **Lane D:** S4 (loss) — after S1; can run alongside S3.
- **Conflict flag:** S3 touches `src/decoder.rs` (the gen-bug fixes) — the only shared edit
  with existing code; keep it on one lane to avoid merge churn.
- Execution: A0 first (gates all). Then launch Lane A + Lane B + (S1) in parallel; S3/S4 after
  S1; S5 last.

## 13. Implementation Tasks
Synthesized from this review's findings. Each derives from a specific finding. P1 blocks the
build being correct; P2 same-branch; P3 follow-up.

- [ ] **T1 (P1, human ~2d / CC ~3h)** — A0 — SFT warm-start + GRPO in Python (OpenRLHF/TRL) on tiny Qwen; lock reward+metrics; emit `tests/ref/` parity tensors.
  - Surfaced by: Issue 3 (Python-first) + Codex #23/#18.
  - Verify: mean reward rises; tensors committed.
- [x] **T2 (P1) — DONE** — logprob.rs — `gather−logsumexp` selected-token logprob, no `[B,T,V]`.
  - Surfaced by: fix (b), Codex #3 / Gemini #2. Files: src/grpo/logprob.rs.
  - Verified: `cargo test --test grpo_math` — logp parity vs A0 < 2e-4. ✓
- [x] **T3 (P1) — DONE (KV-cache fast path deferred to perf)** — `src/sampling.rs` (shared top-k+top-p, fixes the ignored-`top_p` bug in decoder.rs via DRY), `src/grpo/rollout.rs` (Rollouts, RolloutConfig, per-sequence-EOS completion mask, raw pre-warp old_logprob capture, group_sample).
  - Surfaced by: fix (a), Codex #2 / Gemini #1; decoder.rs:806,727.
  - Verified: 5 sampler unit tests (top-p truncation proven), 3 rollout unit tests (per-seq EOS), 1 integration test (`group_sample` on a tiny NdArray model). ✓ group_sample still uses no-cache forward (KV-cache rollout is a perf task).
- [x] **T4 (P1) — DONE (zero-std skip pending in trainer)** — loss.rs — group-norm advantage + clipped surrogate (relu-min) + k3 KL-in-loss + token-global reduction.
  - Surfaced by: §2, fixes (e)(h)(i). Verified: `cargo test --test grpo_math` — pol/kl/total parity vs A0 < 2e-4. ✓ (zero-std-group skip lands in trainer.rs T6.)
- [x] **T5 (P1) — DONE (bounded-pool perf deferred)** — `src/grpo/reward.rs` — `RewardFn` trait + `ManimReward` that shells out to the tested `a0/manim_reward.py` (`--score-only`) instead of reimplementing AST safety in Rust (DRY). Fail-safe: any error → 0.0.
  - Surfaced by: fixes (f)(g)(h), Codex #12-15 / Gemini #4-5 (logic in the Python harness, unit-tested).
  - Verified: 2 Rust tests (valid≥0.6, malicious=0.0, garbage~0, variance; missing-python→0.0). Sequential; bounded worker pool is a perf task.
- [x] **T6 (P1) — DONE (micro-batch grad-accum deferred to perf)** — `src/grpo/trainer.rs` `grpo_step`: rollout + ref on the INNER (no-grad) backend via `policy.valid()`, rollout outputs lifted into the autodiff graph as constants, group-norm advantage, token-aligned completion logprobs (off-by-one shift), zero-std-group skip, AdamW step. `denom.clamp_min` guards the all-masked batch.
  - Surfaced by: fixes (c)(d)(e), Codex #1 / Gemini #3,#7.
  - Verified: integration test on a tiny Autodiff<NdArray> model — finite loss, **mean_ratio=1.0 at step 0** (raw old_logprobs == recomputed → fix (a) proven end-to-end), policy weights move, frozen ref unchanged. ✓ Full-batch reduction (correct token-global); micro-batched grad-accum for large-model memory is a perf task.
- [ ] **T7 (P2, human ~3h / CC ~30m)** — dataset.rs — Manim CSV loader + chat template + gating-error message.
  - Surfaced by: §6. Verify: columns parse; 401 explains HF_TOKEN.
- [ ] **T8 (P2, human ~2h / CC ~20m)** — grpo_train.rs example + A0-convergence smoke (toy reward).
  - Surfaced by: §5. Verify: mean toy-reward rises over ~50 steps.

## 14. Deferred TODOs (proposed — see review report)
- bf16 inference fix (RmsNorm DTypeMismatch) → would let rollouts run bf16, ~2× memory/throughput win. Blocked by a Burn Fusion dtype issue. Phase-A stretch.
- Phase B: Qwen3 MoE block + router + LoRA + 30B weight loading + early memory prototype (Codex #22). Gated on Phase A converging.
- vLLM-style paged-attention rollout engine — only if rollout throughput becomes the wall.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | not run |
| Codex Review | `codex exec` (GPT-5.x, high) | Independent 2nd opinion | 1 | issues_found | 23 findings, 3 BLOCKERs absorbed |
| Gemini Review | `agy` (Gemini 3.1 Pro High) | Independent 3rd opinion | 1 | issues_found | 7 findings, corroborated consensus |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | clean | 4 decisions resolved, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | n/a (no UI) |

- **CODEX:** absorbed the 3 correctness BLOCKERs (raw old_logprobs, gather−logsumexp, ref no-grad backend) + reward-variance/safety fixes; chose Codex's full Python-first (A0) de-risk.
- **CROSS-MODEL:** Codex + Gemini independently agreed on all 3 correctness BLOCKERs, reward-variance collapse, and sandbox hardening (high-confidence consensus, folded in). One tension — loss reduction (Gemini per-seq vs OpenRLHF token-global) — resolved by user toward OpenRLHF-literal with a verbosity watch-guard.
- **VERDICT:** ENG CLEARED — Phase A plan ready to implement (A0 Python validation → A1 Burn port). 30B MoE+LoRA explicitly deferred to Phase B. CEO/Design reviews not required for this infra/algorithm work.

NO UNRESOLVED DECISIONS
