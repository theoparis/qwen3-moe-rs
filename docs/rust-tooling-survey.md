# Rust tooling survey — fast LLM inference + GRPO/RL training (build-vs-adopt)

**Question.** Which Rust tools/frameworks/crates can get us to *published* single-stream efficiency on
GB10 / DGX Spark (Qwen3-30B-A3B: bf16 **~30–32**, fp8 **~50–55**, Q4_K_M **~57**, NVFP4 **~65** tok/s —
see [`perf-gap-vs-prod.md`](perf-gap-vs-prod.md), [`perf-research.md`](perf-research.md)), instead of /
alongside our hand-rolled **Burn + CubeCL** stack that runs at **13–47% of peak**? And — because we are the
*first public Rust/Burn GRPO trainer* — which options can **TRAIN** (forward+backward+optimizer), not just
infer?

> **Method.** Live web search via `THINK=HIGH /workspace/agy-direct.sh "<q> Search the web."` (Gemini 3.1
> Pro High + Google grounding), one focused query per tool. **All 7 queries returned grounded** (the
> `llama.cpp-bindings` and `burn/luminal` queries were initially HTTP-429-throttled by a burst-rate limit and
> succeeded on a backoff retry). Every exact star count and tok/s number is flagged with its provenance:
> **[grounded]** = returned by a web-grounded query (still community-reported, treat decimals as soft);
> **⚠[promo-tone]** = grounded but vendor/promotional-flavored and *in tension with our own measured data*
> (reconcile, don't take at face value); **⚠[likely-fabricated]** = product/figure with no independent
> corroboration. Trust the *ranking and the capability columns*, not any single decimal.

---

## TL;DR — the three answers

1. **Best Rust path to published single-stream inference efficiency:** **mistral.rs** (the "vLLM-in-Rust":
   PagedAttention + FlashAttn + fp8/GGUF/AWQ/ISQ + MoE + explicit Qwen3-30B-A3B, **candle-based**,
   embeddable as the `mistralrs` crate) — *for the rollout only, because it is **inference-only***. For the
   single most *proven* efficiency number, **llama.cpp Rust bindings** (`llama-cpp-2`) wrap the
   80–93%-of-peak C++ engine (the ~57 tok/s Q4_K_M GB10 figure is *its* number) — also inference-only.
2. **GRPO forces a training-capable framework for the POLICY.** Only **candle** and **burn** can do
   autograd+backward+optimizer in Rust. mistral.rs, llama.cpp-bindings, luminal, ratchet, tract, zml are
   **inference-only** — they can serve the *rollout* but can never hold the trainable policy. So we cannot
   "replace" our stack with mistral.rs; we can only *add* it as a rollout engine.
3. **Pragmatic architecture (ranked): port the trainable policy to *candle* and use *its own* (or
   mistral.rs's, same lineage) generate path for the rollout** — one framework, one weight space, no
   cross-engine sync, logprob parity by construction, mature hand-written CUDA kernels, **and** training +
   our novelty preserved. Second choice — newly stronger: stay on Burn but **adopt its now-shipping
   `burn_attention` (FlashAttention-v3) + `cubek` (Blackwell-TMA GEMM/MoE) kernels** instead of hand-rolling,
   then add the flash-decode O(pos) path (our #1 lever may be adopt-not-build). Third: keep Burn trainer +
   bolt mistral.rs/llama-cpp-2 on as the rollout engine (pays a weight-sync + fp8 parity tax).

---

## Per-tool table

| Tool | Stars / maturity | Train or infer? | CUDA | Gaps it closes (flash/paged · quant · cuda-graph · MoE/Qwen3) | Published single-stream efficiency |
|---|---|---|---|---|---|
| **mistral.rs** (EricLBuehler) | ~7.3k ⚠, v0.8.x, daily commits [grounded] | **Inference-only** (LoRA *load/merge* only, **no backprop**) | Yes (+Metal) | **PagedAttn ✓ · FlashAttn v2/v3 ✓** · GGUF/GPTQ/AWQ/**fp8**/HQQ/BNB + **ISQ** ✓ · cuda-graph ~ · **MoE ✓, Qwen3-30B-A3B ✓** (GGUF or ISQ) | ~70–88 tok/s for 30B-A3B on **M3 Max** ⚠[grounded-but-community]; **GB10 number not confirmed** |
| **candle** (HuggingFace) | ~20.6k ⚠, highly active [grounded] | **TRAINING ✓** (autograd, AdamW/SGD/RMSprop, backward) | Yes (`candle-kernels`, cuBLAS) | flash-attn via `candle-flash-attn` ✓ (Ampere/Ada/Hopper) · **GGUF ✓ GPTQ ✓**, fp8/NVFP4 partial/upstream ⚠ · cuda-graph ✗ (mitigated by no-GIL dispatch) · **MoE ✓, Qwen3 MoE GGUF ✓** | ~80–140 tok/s for **7B/8B** single-stream; llama.cpp keeps a slight edge (~178 vs ~140–150) [grounded] |
| **llama.cpp Rust bindings** (`llama-cpp-2` / utilityai; `llama_cpp`; `llm`=archived) | `llama-cpp-2` ~600★ v0.1.150 (Jun 2026), most-maintained; `llama_cpp` ~450★; `llm` (rustformers) **archived/no-GGUF** [grounded] | **Inference-only** (wraps C++) | Yes (CUDA build feature) | inherits llama.cpp: **flash-attn ✓ · GGUF ✓ · cuda-graph ✓ · MoE/Qwen3-30B-A3B ✓** (expert-offload + KV opt) | **~58–62 tok/s Q4_K_M on GB10** decode (prefill ~1.2–2.1k tok/s, 18.5 GB) [grounded] — corroborates our ~57; the most *proven* figure |
| **burn** (tracel-ai — *our stack*) | **~15.5k★**, v0.20–0.21 (2026), very active [grounded] | **TRAINING ✓** (burn-autodiff, AdamW) | Yes, **CubeCL** (cubecl-cuda) | framework now ships **`burn_attention` (FlashAttn v3)** + **`cubek`** TMA/Tensor-Core MoE GEMM (Blackwell) ⚠[promo-tone] — but **our** path is bf16, hand-rolled SDPA, cuda-graph in a bench only | claims "matches LibTorch / faster than candle" ⚠[promo-tone] vs **our measured 13–47% of peak** — the gap is *unadopted kernels*, not a missing framework |
| **luminal** (jafioti) | **~2,860★**, Luminal AI startup, steady [grounded] | **Inference-only** in practice (AOT graph *can* model autograd, but all tooling is inference) | Yes (`luminal_cuda` + cuBLAS) | AOT compiler reduces models to ~15 primitives, **search-derives FlashAttention**; edge/perf-research focus | claims Q8 Llama-3-8B at **~80% of H100 peak FLOPS** ⚠[promo-tone]; niche, fewer prebuilt models, **not a turnkey 30B-MoE GB10 path** |
| **ort** (pykeio, ONNX Runtime) | ~1–2k stars / **12M+ crates.io downloads**, battle-tested [grounded] | **Both** (ORT inference + training) | **Excellent** (CUDA/TensorRT/ROCm/CoreML) | great for embeddings/vision; **30B MoE export is painful**, no PagedAttn | gold-standard for embeddings/Whisper/YOLO, **bypassed for 30B LLMs** [grounded] |
| **zml** (Zig, not Rust) | ~3.7k ⚠ [grounded] | **Inference-only** | **Excellent** (OpenXLA→CUDA/ROCm/TPU) | FlashAttn3 ✓, 4-bit ✓, **Qwen3 ✓**, sharded big-MoE; **tech-preview** early 2026 | credible compiled vLLM-alt **soon**; Zig not Rust → off-path for us |
| **ratchet** (HuggingFace) | ~800 ⚠ [grounded] | Inference-only | **No CUDA** (WebGPU/CPU) | browser/edge SLMs only | **incapable of 30B MoE** — wrong tool |
| **tract** (Sonos) | ~3k ⚠ [grounded] | Inference-only | minimal/niche | edge/IoT ASR + vision | **no LLM** — out of scope |

---

## Per-tool detail

### 1. mistral.rs — the leading "vLLM-in-Rust" (rollout engine candidate) [grounded]
- **What it closes:** the *exact* set of our gaps — **PagedAttention** (O(pos) KV, our gap #1), **FlashAttention
  v2/v3** (our gap #5), **fp8 + GGUF/GPTQ/AWQ + ISQ** (our gap #3 — in-situ-quantize any HF model at load),
  continuous batching, and **explicit Qwen3-30B-A3B MoE** support (GGUF or ISQ-from-HF-weights).
- **Training:** **NO.** Inference-only; it can *load and merge* LoRA/X-LoRA adapters for inference but has no
  backprop/optimizer. → It can be our **rollout engine, never our trainer.**
- **Embeddable:** distributed as the `mistralrs` crate — you can `ModelBuilder`-load weights into VRAM and
  generate rollouts *in-process* (no HTTP), which is exactly the RL-rollout shape. **Built on candle**, so it
  shares candle's tensor/runtime lineage.
- **Efficiency caveat:** the **~70–88 tok/s** for 30B-A3B is community discussion on **M3 Max** [grounded but
  community] — **there is no confirmed mistral.rs GB10/Qwen3-30B-A3B single-stream number**; do not assume it
  hits the ~30–55 tok/s SGLang band on GB10 without measuring. ⚠

### 2. candle — the training-capable engine with mature CUDA kernels [grounded]
- **Training:** **YES** — eager autograd (PyTorch-like), `candle-nn` optimizers (**AdamW**, SGD, RMSprop),
  full backward. This is the key differentiator vs mistral.rs/llama.cpp.
- **Inference gaps it closes:** `candle-flash-attn` (custom CUDA FlashAttn v1/v2, Ampere/Ada/Hopper), native
  **GGUF + GPTQ**, **MoE + Qwen3-MoE GGUF**. Hand-written `candle-kernels` + cuBLAS bindings are **more mature
  than CubeCL** (no JIT/Fusion tax).
- **Weaknesses (flagged):** **fp8/NVFP4 native decode is partial** (largely upstream/cuBLASLt-wrapped as
  Blackwell scales — ⚠). **No exposed CUDA-graph capture** (it argues no-GIL Rust dispatch hides much of the
  launch tax — but for our 48-layer MoE that is *not* equivalent to a captured graph; verify before relying).
  Single-stream numbers are quoted for **7B/8B (~80–140 tok/s)**; no clean 30B-A3B GB10 figure surfaced.

### 3. llama.cpp Rust bindings — most *proven* efficiency, inference-only [grounded]
- **Crates:** **`llama-cpp-2`** (`utilityai/llama-cpp-rs`, ~600★, v0.1.150 Jun-2026) is the highest-activity
  FFI binding, tightly tracking upstream; the higher-level **`llama_cpp`** (~450★) updates slower; the
  pure-Rust **`llm`** crate (rustformers, 6.2k★ legacy) is **archived since mid-2024 and has no GGUF** — do
  not adopt it. (A *new* unrelated `llm` v1.x is a multi-provider API wrapper, not llama.cpp bindings.)
- **What it closes:** inherits the C++ engine wholesale — **flash-attn, GGUF (incl. Q4_K_M), CUDA-graph, MoE,
  Qwen3-30B-A3B** (the 2026 builds include the expert-offload + KV-cache logic for the 128-expert sparse
  arch) — at **80–93% of peak bandwidth** (our gap #8 ceiling). Grounded GB10 number: **~58–62 tok/s decode
  Q4_K_M**, prefill ~1.2–2.1k tok/s, 18.5 GB — corroborates the ~57 in our own docs. The most de-risked
  efficiency of any option.
- **Training:** **NONE** — "strictly inference-only"; the grounded answer itself points to **candle** as the
  Rust training alternative. Rollout-only.
- **Cost:** a C++/CUDA build dependency + an FFI seam + GGUF-only weights (you must export/quantize the policy
  to GGUF each sync — heavy for an RL loop where weights change every step).

### 4. burn + CubeCL — our stack: training-capable; the kernels now EXIST, we just haven't adopted them [grounded ⚠promo-tone]
- **Training:** **YES** — `burn-autodiff`, AdamW, eager-first dynamic graph; this is *why* our native-Rust
  GRPO trainer exists. Do not give this up lightly — it is the novelty. **~15.5k★, v0.20–0.21 (2026), very
  active** (distributed training, CPU, WASM upgrades).
- **The key new finding (reconcile carefully):** the grounded answer claims Burn has **closed the gap to
  LibTorch/candle** and now *ships* the building blocks we've been hand-rolling — a **`burn_attention` crate
  implementing FlashAttention v3 in CubeCL**, and **`cubek`** (CubeCL's kernel library) with **Tensor-Core /
  warp-level "Plane" ops and Blackwell/Hopper TMA** for fast GEMM/MoE. This is **⚠promotional-toned and in
  direct tension with our own measured 13–47% of peak** ([`perf-gap-vs-prod.md`](perf-gap-vs-prod.md) §8). The
  honest reconciliation: **the framework now has FA3 + TMA-GEMM kernels available — our slowness is that our
  *inference path* hasn't adopted them** (bf16, hand-rolled O(T_max) SDPA, physical GQA ×8, cuda-graph only in
  a bench). So the gap is **mostly "unadopted kernels," not "missing framework."**
- **Verdict:** the perf gap is **more fixable in-framework than we assumed** — `burn_attention`/`cubek` may
  give us FA3 + Blackwell GEMM to *adopt* rather than write from scratch — but (a) verify these crates'
  decode-path maturity and Qwen3-MoE fit before betting, and (b) a residual Fusion/JIT dispatch tax likely
  still caps us a constant factor below hand-tuned llama.cpp. (Ignore the "matches LibTorch / 4× / 8.2×"
  decimals — promotional; our measured numbers are load-bearing.)

### 5. luminal / ort / zml / ratchet / tract — surveyed, mostly off-path
- **luminal** (jafioti, ~2,860★, Luminal AI): AOT Rust→CUDA compiler that reduces models to ~15 primitives
  and *search-derives* FlashAttention; claims **Q8 Llama-3-8B at ~80% of H100 peak FLOPS** ⚠[promo-tone]. But
  it is **inference-only in practice** (the graph can model autograd AoT, but all tooling/benchmarks are
  inference) and niche/research-stage with few prebuilt models — **not a turnkey 30B-MoE GB10 path**, and it
  can't host the trainable policy.
- **ort** (pykeio): the *only other training-capable* option (ONNX Runtime training), **excellent CUDA/TensorRT**,
  battle-tested (12M+ downloads) — **but** exporting a 30B Qwen3-MoE to a static ONNX graph is "notoriously
  frustrating" and it lacks PagedAttention. Great for embeddings/vision, **bypassed for 30B generative LLMs**.
- **zml** (Zig): compiled, OpenXLA→CUDA, FlashAttn3, Qwen3, sharded big-MoE — credible vLLM-alt, but
  **tech-preview** and **written in Zig, not Rust** → off-path for a Rust/Burn codebase.
- **ratchet** (WebGPU, ~800★) and **tract** (Sonos edge, ~3k★): inference-only, **cannot run 30B MoE** — wrong
  tools for a GB10 backend.

---

## 6. The GRPO-needs-training constraint (confirms our novelty) [grounded]
A full GRPO/RLHF loop needs **rollout + policy/reference forward & backward + advantage/KL loss + AdamW**.
Grounded search confirms **no mature native-Rust GRPO/RLHF-for-LLMs exists**:
- **`burn-ppo`** (bhansconnect): PPO in Burn, but **discrete-action board games** (CartPole/Connect-Four) —
  no text rollouts, no KV-cache, no sequence-KL. Not an LLM trainer.
- **`Oxen-AI/GRPO-With-Cargo-Feedback`**: the training loop is **Python TRL `GRPOTrainer`**; Rust is only the
  reward (spawning `cargo test`). Matches our memory note.
- **`rlox`** (wojciechkpl): a Rust RL *data plane* (VecEnv/buffers/GAE/GRPO math) that bridges to **PyTorch via
  PyO3** — the model forward/backward is still PyTorch. Provides GRPO *math* in Rust, not LLM orchestration.
- **Mature path = PyTorch only:** HuggingFace **TRL**, **OpenRLHF**, **verl** (tightly coupled to vLLM for
  rollout + DeepSpeed/FSDP for the multi-model VRAM juggling).

→ **Our headline holds:** *first public GRPO LLM trainer whose training loop runs natively in Rust* (state as
absence-of-evidence, per [grpo-novelty-positioning]). **Implication:** since only **candle** and **burn** can
train in Rust, the *policy* must live on one of them — an inference engine can only ever do the rollout.

---

## 7. Ranked build-vs-adopt recommendation

**(a) Best Rust path to published single-stream INFERENCE efficiency** — for the *rollout*:
- **#1 mistral.rs** if we want one Rust crate that already has paged-attn + flash + fp8/ISQ + Qwen3-30B-A3B
  and is *candle-based* (so it shares lineage with the trainer). ⚠ but its GB10/30B-A3B tok/s is unconfirmed —
  **measure before betting**.
- **#1-proven llama.cpp `llama-cpp-2`** if we want the *de-risked* ~57 tok/s Q4_K_M / 80–93%-of-peak number
  today — at the cost of an FFI/GGUF seam.
- candle's own generate path is the dark-horse: less turnkey than mistral.rs, but *same framework as the trainer*.

**(b) Does GRPO force a training-capable framework? — YES.** The policy (forward+backward+AdamW) can only run
on **candle or burn**. No inference engine (mistral.rs / llama.cpp / luminal / zml) can hold it. So the real
choice is *which training framework* + *how to make the rollout fast*.

**(c) Pragmatic architecture — ranked:**

1. **PORT the trainable policy to candle; use candle's (or mistral.rs's) generate for the rollout.** *Best
   overall.* candle is **training-capable AND has mature hand-written CUDA kernels + flash-attn + GGUF** — it
   fixes the CubeCL-immaturity ceiling *and* keeps training in one framework, one weight space (no
   cross-engine weight sync, **logprob parity by construction** — critical for k3 KL). mistral.rs being
   candle-based means a candle policy can plausibly share a fast generate path. **Preserves the
   "native-Rust GRPO" novelty.** Cost: a real port; candle's 30B-A3B MoE + fp8/NVFP4 + decode-flash path is
   less proven than mistral.rs's and must be validated; candle has no exposed CUDA-graph.
2. **STAY on Burn; adopt `burn_attention` (FA3) + `cubek` (Blackwell-TMA GEMM/MoE) instead of hand-rolling,
   then add the flash-decode path + cuda-graph in `vllm_infer` + fp8.** The new grounded finding upgrades this
   option: the framework now *ships* FA3 and Tensor-Core/TMA GEMM kernels, so our #1 lever may be an
   **adopt-not-build** job rather than a from-scratch kernel. Keeps everything in one framework and the
   novelty. Cost: must verify those crates' *decode-path* maturity + Qwen3-MoE fit (they're attention/GEMM
   primitives, and decode-flash O(pos) for a static KV buffer may still need our own glue); a residual
   Fusion/JIT tax likely still caps below llama.cpp. This is the lower-disruption evolution of our PERF_80TOKS
   plan.
3. **KEEP Burn trainer + bolt mistral.rs (or llama-cpp-2) on as the rollout engine** (the TRL-style "fast
   engine in the loop" pattern). Cost: a **weight-sync seam** (policy weights change every GRPO step → reload
   into the rollout engine each step) **and an fp8/GGUF-rollout vs bf16-recompute parity tax** — our docs note
   fp8 rollout breaks GRPO recompute/k3-KL parity. Workable (it is exactly how verl/TRL use vLLM) but adds the
   most integration risk and a parity footgun for a single-stream Rust loop.

**Honest build-vs-adopt bottom line:** adopting a fast *inference* engine cannot replace our trainer (GRPO
needs autograd → candle/burn). The highest-leverage move that buys *both* speed and training is **#1 (port the
policy to candle)** — it swaps our weakest component (immature CubeCL kernels) for candle's mature ones while
keeping the novelty; **#2 (fix CubeCL in Burn)** is the lower-disruption, higher-effort, lower-ceiling
alternative we are already pursuing.

---

## Sources & verification flags

Grounded web search (Gemini 3.1 Pro High + Google Search), one query per tool — **all 7 returned grounded**
(q3 llama.cpp-bindings + q4 burn/luminal succeeded on a backoff retry after an initial HTTP-429). Raw logs in
`scratchpad/rust-survey/out{1..7}.md`.

- **candle [grounded]:** `huggingface/candle` ~20.6k★, training/autograd/AdamW ✓, `candle-flash-attn`,
  GGUF/GPTQ, Qwen3-MoE GGUF; 7B/8B ~80–140 tok/s, llama.cpp ~178 vs candle ~140–150. *(decimals soft.)*
- **mistral.rs [grounded]:** `EricLBuehler/mistral.rs` ~7.3k★ v0.8.x; PagedAttn/FlashAttn v2-v3/ISQ/fp8/
  GGUF/GPTQ/AWQ/MoE/Qwen3-30B-A3B; **inference-only** (LoRA load/merge); `mistralrs` crate embeddable for
  in-process rollouts; **~70–88 tok/s on M3 Max ⚠(community, NOT GB10)**.
- **ratchet/ort/tract/zml [grounded]:** ratchet ~800★ WebGPU inference-only; ort ~1–2k★/12M downloads ONNX
  **train+infer** CUDA/TensorRT but painful 30B-MoE; tract ~3k★ edge inference-only no-LLM; zml ~3.7k★ **Zig**
  inference-only OpenXLA→CUDA Qwen3 tech-preview.
- **GRPO-in-Rust [grounded]:** no mature native-Rust GRPO/RLHF; `burn-ppo` (board games), `Oxen-AI
  GRPO-With-Cargo-Feedback` (Python TRL + Rust reward), `rlox` (Rust data plane + PyO3→PyTorch); mature path =
  TRL/OpenRLHF/verl (Python). **Confirms our novelty.**
- **GB10 Rust benchmarks [grounded — but treat with heavy skepticism]:** the query surfaced an **"Atlas"
  engine (Avarok)** claiming **111–130 tok/s NVFP4+MTP on GB10** and models like "Qwen3.5-35B-A3B" / "Gemma 4"
  / "Nemotron-3 Nano", sourced to a **reddit.com** redirect. These product/model names and figures look
  **likely model-fabricated** — ⚠**do NOT cite as fact**; no independent corroboration. The only reliable GB10
  numbers remain the engine-agnostic roofline + the ~57 Q4_K_M / bf16~30 / fp8~50–55 / NVFP4~65 consensus from
  [`perf-research.md`](perf-research.md) / [`perf-gap-vs-prod.md`](perf-gap-vs-prod.md).
- **llama.cpp bindings [grounded]:** `llama-cpp-2` (utilityai/llama-cpp-rs) ~600★ v0.1.150 Jun-2026 = the
  maintained binding; `llama_cpp` ~450★; `llm` (rustformers) 6.2k★ **archived/no-GGUF**; **strictly
  inference-only** (the answer itself names candle as the Rust training alternative); GB10 Q4_K_M **~58–62
  tok/s decode** (prefill ~1.2–2.1k, 18.5 GB) — corroborates our ~57. Sources: github.com, crates.io,
  skiptodone.com, reddit.com (decimals soft).
- **burn/luminal [grounded, ⚠promo-tone]:** burn (tracel-ai) **~15.5k★** v0.20–0.21, training-capable, CubeCL
  backend; the answer claims Burn now **matches LibTorch** and ships **`burn_attention` (FA3)** + **`cubek`**
  (Tensor-Core/TMA GEMM for Blackwell) — **promotional and in tension with our measured 13–47% of peak**, so
  read it as *"the kernels exist to adopt"* not *"Burn is already fast for us."* luminal (jafioti) **~2,860★**,
  AOT Rust→CUDA, search-derives FlashAttention, ~80%-H100-FLOPS Q8-Llama3-8B claim ⚠, inference-only in
  practice. Sources: burn.dev, lib.rs, github.com, medium.com, reddit.com.
- **Our-side (measured, load-bearing):** [`perf-gap-vs-prod.md`](perf-gap-vs-prod.md) (13–47% of peak, the
  ranked CubeCL gaps), [`perf-research.md`](perf-research.md) (GB10 roofline + published tok/s),
  [grpo-novelty-positioning] (the novelty boundary).
</content>
</invoke>

---

## VERIFICATION RESULTS (spike + claim-check, this session)

### cubek-attention adoption spike → **NO-GO** (sm_121, our pinned revs)
Ran `examples/cubek_attn_spike.rs` on the real GB10 vs a CPU f32 SDPA oracle. Verdict: **cannot adopt cubek's
Tensor-Core kernels today.**
- The broadcast-mask bug class is sidestepped (cubek uses a comptime positional `causal` flag, not a
  `[1,1,s,s]` broadcast mask). Good.
- **BUT the BlackboxAccelerated (Tensor-Core) routine FAILS to compile on sm_121** for every dtype:
  `CmmaInstructionUnavailable` (f32/bf16), invalid `nvcuda::wmma` accumulator fragment (f16) — a cubecl/cubek
  codegen+feature gap at the pinned revs (cubecl `b19859ee`, cubek `1161040`).
- Only the **Unit/SIMT** routine runs, and it is **~93× SLOWER** than our `attention_fallback` (5856 vs 63
  µs on b1·h8·sq1·skv1024·d128) — adopting it would make decode worse.
- **No native GQA** (single `num_heads` from the query shape) — the ×8 repeat is not eliminated.
- **`cubek-matmul` is blocked the same way** (verified: it uses `CmmaMatmul`/`MmaMatmul`) — so the Tensor-Core
  MoE-GEMM hope is also blocked at our pins. `cubek-quant`'s per-element `dequantize_symmetric` (NOT
  Tensor-Core-dependent) may still be adoptable for an fp8 path. cubek-attention's `seq_kv` IS a free runtime
  dim (O(pos) feasible eager, but a new shape/step breaks fixed-shape capture).
- **Conclusion:** our hand-rolled `src/flash_attn.rs` (GQA-aware, decode-correct) stays the vehicle. cubek
  becomes adoptable only after a cubecl/cubek bump that registers working sm_121 Tensor-Core MMA (re-run the
  reusable spike then). Adopting cubek's Tensor Cores is NOT a near-term lever at our pins.

### "Atlas / Qwen3.5-35B / 111-130 tok/s" claim → **REAL, not fabricated** (the prior debunk was a
knowledge-cutoff blind spot — these post-date Jan 2026):
- **Qwen3.5-35B-A3B** is real (HF API 200: `qwen3_5_moe`, 35.95B, ~3B active, Feb 2026); the whole Qwen3.5
  line + **Qwen3.6-27B** exist. **Gemma 4** is real (gemma-4-12B/31B-it, May-Jun 2026).
- **Atlas** (`Avarok-Cybersecurity/atlas`) is a REAL open-source **Rust+CUDA inference engine for DGX
  Spark/GB10** — 539★, AGPL-3.0, active, public benchmark harness, NVIDIA forum threads, HF org, website.
- **111-130 tok/s** is its **NVFP4 (4-bit) + MTP K=2 spec-decode PEAK** on short gens; honest single-stream
  non-spec is **~70 tok/s** (NVFP4). Physically consistent (NVFP4 roofline ~165 for 3B-active) and reached via
  exactly the two levers our own research named: sub-fp8 quant + speculative decoding.
- **Implication:** Atlas is the existence proof that a Rust engine hits published efficiency on OUR exact
  hardware. Its ~70 tok/s NVFP4 single-stream is the real comparable to our ~45 bf16 roofline / ~21 captured.
  A real "adopt-or-study" candidate (caveat: AGPL-3.0 copyleft; inference-only, so GRPO still needs candle/Burn).
