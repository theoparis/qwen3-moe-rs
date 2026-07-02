# Perf research — single-stream MoE decode on bandwidth-limited unified memory

Grounding research for [PERF_80TOKS_PLAN.md](PERF_80TOKS_PLAN.md): prod-level patterns + real measured
numbers for single-stream (batch-1) decode of a top-8 MoE like **Qwen3-30B-A3B** (3.3B active params) on
**~273 GB/s unified memory** (NVIDIA GB10 / DGX Spark), and an honest verdict on whether **80 tok/s
single-stream** is reachable.

Method: web search via `THINK=HIGH /workspace/agy-direct.sh` (Gemini 3 Pro/Flash + live Google-grounding);
roofline computed locally from the [official Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B)
(H=2048, L=48, **E=128 experts, top-K=8**, moe_inter I=768, 32 q-heads / 4 kv-heads × head_dim 128,
untied lm_head [2048, 151936]). Source-domain citations are listed per section; treat the LLM-grounded
*numbers* as community-reported (corroborated by the independent roofline), not vendor-spec-sheet exact.

---

## TL;DR verdict (the honest ceiling)

- **GB10 / DGX Spark = 128 GB unified LPDDR5X @ 273 GB/s — confirmed.** Decode is memory-bandwidth bound;
  sparse MoE is the ideal workload for it.
- **Roofline for 30B-A3B (read active experts + attn + lm_head per token):** dense-all-128-experts ≈ **4.5
  tok/s**; **top-8 bf16 ≈ 45**; **top-8 fp8 ≈ 90** (theoretical, 100% BW). At a realistic **70–85%** DRAM
  efficiency the fp8 top-8 *sustained* single-stream number is **~62–76 tok/s**.
- **Published-best measured for 30B-A3B on DGX Spark today ≈ 57 tok/s (Q4_K_M / Ollama) to ~65 tok/s
  (NVFP4 4-bit on Blackwell).** bf16 ≈ 30 tok/s. **Nobody has publicly shown 80 tok/s single-stream on this
  class of hardware.**
- **Is 80 tok/s reachable pure single-stream autoregressive? Only at the optimistic edge** — it sits *above*
  the realistic sustained fp8 top-8 number and *above* every published measurement. fp8 + top-8 + device
  sampling + CUDA-graph gets you to **~62–72**, not a robust 80.
- **The robust path past ~72 is speculative decoding** — but for a *sparse* MoE the **union-of-experts
  penalty** dilutes it: a K-token verify pass routes to the *union* of the tokens' top-8 sets (~15 experts
  at K=2, ~29 at K=4 of 128), so it reads far more than 8 experts. Net EAGLE/MTP gain on this MoE is only
  **~1.2× (sweet-spot K=2)**, vs 2–4× on a dense model. That ~1.2× on top of ~72 → **~80–87**.
- **Recommended path to a robust 80:** fp8(or NVFP4) experts + real top-8 decode routing + device sampling +
  CUDA-graph → ~65–72 sustained, **then a light EAGLE-3/MTP draft (K=2)** to clear 80. fp8, top-8, and
  spec-decode stack (with the MoE caveat above).

---

## 1. GB10 / DGX Spark bandwidth + real 30B-A3B tok/s

**Hardware (confirmed):** DGX Spark = **GB10 Grace Blackwell superchip** (20-core ARM + Blackwell GPU over
NVLink-C2C), **128 GB coherent unified LPDDR5X**, **memory bandwidth = 273 GB/s**. LPDDR5X → bandwidth, not
compute, is the inference bottleneck, so low-active-param MoEs are the sweet spot. *(Sources: StorageReview
GB10 review; backend.ai GB10 deep-dive; nvidia.com.)*

**Measured single-stream decode for Qwen3-30B-A3B on DGX Spark (community-reported, early 2026):**

| Precision / engine | tok/s (single-stream decode) | note |
|---|---|---|
| **bf16** (vanilla vLLM/SGLang) | **~30–32** | reading 3.3B active × 2 B ≈ 6.6 GB/token saturates 273 GB/s |
| **4-bit AWQ / Marlin** | ~35–42 | |
| **Q4_K_M GGUF (Ollama)** | **~57.3** | "highly responsive" local-coding setup |
| **fp8 (vLLM / SGLang online fp8)** | **~44–55** | SGLang ~52–55 stable; fp8 halves bf16 bandwidth |
| **NVFP4 (Blackwell native FP4, FlashInfer/CUTLASS)** | **~65–66** | "currently the ceiling for this model on GB10" |

*(Sources: a Mar-2026 Medium benchmark "Qwen3-Coder-Next vs Qwen3:30B-A3B" — 57.3 tok/s Ollama; NVIDIA
Developer Forums + GitHub vLLM/SGLang threads tracking 31→55→65 tok/s; Creative Strategies Dec-2025 "~30
tok/s out-of-the-box". Domains: ifactoryapp.com, github.com, creativestrategies.com, nvidia.com,
medium.com.)*

**Cross-check — same regime, other unified memory (from §3):** llama.cpp on **AMD Strix Halo / Ryzen AI
Max 395** (128 GB LPDDR5X, ~256 GB/s theoretical, **~215 GB/s effective**) sustains **~25 tok/s** on a 100B
MoE (~8 GB active) and **~30 tok/s** on Qwen3-Next-80B Q4 — i.e. it saturates **~80–93% of effective
bandwidth**. This is the single most important calibration point: a *mature* engine hits high BW efficiency,
a naive one (and Apple MLX, ~40–50%) does not.

---

## 2. vLLM / SGLang / TensorRT-LLM — fused MoE DECODE at batch-1

**`fused_moe` + `moe_align_block_size` (vLLM/SGLang Triton; TRT-LLM = CUTLASS grouped-GEMM):** routing
assigns the token to top-K experts → `moe_align_block_size` sorts token-ids by expert and **pads each
expert's token count up to a block size (32/64)** so a single grouped-GEMM kernel can dispatch one block per
(expert, block) segment. *(Sources: vllm.ai docs; github.io blog; cefboud.com.)*

**Batch-1 decode reads ONLY the top-k active experts — confirmed and load-bearing.** Experts that receive 0
tokens get 0 blocks, so **their weights are never fetched from HBM**. The kernel computes offsets strictly
from the active `expert_ids`. At BS=1 the padding (1 real token → a 64-token block) wastes *FLOPs* but
**zero latency** — the step is bound by the bytes of the active-expert weights, identical for 1 or 64 tokens.
⇒ **batch-1 MoE decode is strictly memory-bandwidth bound on the active-expert read.** (This is exactly the
top-8 lever in PERF_80TOKS_PLAN §2(A).)

**fp8 for MoE — scale granularity:**
- *Per-tensor* (one scale / expert matrix): cheap, but degrades MoE accuracy (experts have very different
  weight distributions).
- *Per-channel / block-wise* (e.g. **1×128 blocks**, TRT-LLM DeepSeek-V3, vLLM `cutlass_scaled_mm`): scale
  vectors fetched alongside the fp8 blocks, dequant in registers before FP16/FP32 accumulate; preserves
  accuracy at a small extra scale-traffic cost. **This is the recommended granularity for a top-8 MoE.**

**W8A8 vs W8A16 for decode — W8A16 (weight-only) is the universal choice at batch-1.** At BS=1 the
activation is one token, so quantizing activations to fp8 saves negligible bandwidth but adds
dynamic-scaling/dequant overhead. **Keep weights fp8 (half the bandwidth), activations bf16/fp16.** W8A8 is a
large-batch *prefill* (compute-bound) lever. ⇒ confirms the project's W8A16 choice (`src/w8a16.rs`).

**CUDA-graph the MoE decode step — yes, all three:**
- **vLLM / SGLang:** capture with fixed-size `max_num_tokens_padded` buffers so the `fused_moe` grid is
  static; routing updates the `expert_ids`/token arrays *in place*; replay runs the static grid (dummy/`-1`
  experts contribute a zero-gated output).
- **TensorRT-LLM:** *Piecewise CUDA Graphs* + **Device Problem Sizes** — the CPU launches the grouped-GEMM
  with a static max grid; the GPU kernel reads the *actual* per-expert token counts from GPU memory
  (written by the GPU-side router), fully decoupling the CPU from data-dependent routing so the dynamic MoE
  step is captured with no host sync.

*(Cross-ref: this corroborates VLLM_KERNELS.md §3 (moe_align_block_size dropless layout) and §4 (CUDA-graph
needs fixed/bucketed shapes). Domains: vllm.ai, github.io, cefboud.com.)*

---

## 3. llama.cpp / ktransformers / MLX — MoE on unified memory

All three confirm: single-stream MoE decode reads **only the shared layers + the active experts** per token,
so the speed limit is `effective_BW / active_bytes_per_token`.

| Engine | Hardware (BW) | Model (quant) | active/token | tok/s | effective BW |
|---|---|---|---|---|---|
| **Apple MLX** | Mac Studio M3 Ultra (800 GB/s) | DeepSeek-V3/R1 671B (Q4, 37B active) | ~18.5 GB | **~16.5–18.4** | ~340 GB/s (~43%) |
| **Apple MLX** | M3 Ultra (800 GB/s) | Qwen3-235B-A22B (8-bit) | ~22 GB | **~18.8** | ~413 GB/s (~52%) |
| **llama.cpp** | Ryzen AI Max 395 / Strix Halo (~215 GB/s eff) | GLM-4.5-Air ~100B MoE (Q4) | ~8 GB | **~20–25** | ~200 GB/s (**~80–93%**) |
| **llama.cpp** | Strix Halo | Qwen3-Next-80B MoE (Q4) | — | **~30** | saturates the bus |
| **ktransformers** | dual-Xeon + RTX 4090 (CPU-offload) | DeepSeek-V3/R1 671B (INT4) | ~18.5 GB | **~14–28** | CPU-RAM bound |

**Tricks that matter for unified memory:**
- **llama.cpp:** builds the compute graph of *only the selected experts*, skipping dormant weights (Vulkan
  / ROCm-HIP backends). Gets ~80–93% of effective BW on Strix Halo — the proof that a lean engine *can*
  saturate ~273 GB/s-class memory.
- **ktransformers (the offload regime, less relevant to GB10's single pool but instructive):** (1)
  **compute experts on the CPU** (AMX/AVX-512 fp8 kernels) to avoid moving 18.5 GB of expert weights over
  PCIe — only the tiny hidden state crosses the bus; (2) **residual-based expert prefetch / PreSched** — a
  predictive scheduler reads early-layer residuals to predict which experts deeper layers will activate, and
  streams those weights into VRAM *before* the token arrives, hiding the I/O bubble. Lifts DeepSeek-V3 from
  ~4–5 → 14–28 tok/s.
- **MLX:** keeps the whole model in unified RAM, routes active experts straight to GPU cores (no PCIe). Note
  it only reaches **~40–50%** of theoretical BW — engine maturity, not the hardware, is the gap.

**Takeaway for GB10:** the *achievable* single-stream number is set by how close to 273 GB/s the engine
gets. Strix Halo/llama.cpp at ~85% says a fused, lean fp8 top-8 path on GB10 should reach **~0.8–0.9 × 90 ≈
72–80 tok/s**; a naive path lands at ~40–55. *(Sources: r/LocalLLaMA + GitHub benchmarks; ktransformers
Tsinghua arXiv [Chen et al. 2025]; Framework/Strix-Halo testing; MLX deployment posts. Domains:
localaimaster.com, reddit.com.)*

---

## 4. fp8 weight-only (W8A16) decode — multiplier + accuracy

**Standard practice.** fp8 weight storage with bf16/fp16 activations is the default bandwidth lever for
inference. Two vLLM/Neural-Magic kernels:
- **Marlin (FP8 Marlin):** for GPUs without native fp8 tensor cores (Ampere/Turing) — packs 4× fp8 into an
  int32, dequants in shared memory with bitwise SIMT ops, overlaps the HBM fetch with compute via
  `cuda::memcpy_async` + a circular shared-mem queue.
- **Machete:** the Hopper/**Blackwell** successor on CUTLASS 3.x — **warp-specialized** (producer warps move
  fp8 from HBM, consumer warps do the math), hides the fp8→bf16 upconvert entirely behind tensor-core math.

**Decode multiplier:** fp8 = 1 B/param vs bf16 2 B → **bandwidth strictly halved** → theoretical **2×** in
the bandwidth-bound linear layers. **Measured end-to-end: ~1.6–1.9×** (TPOT for the projections ≈ 2×); the
shortfall is because attention, layer-norms, and (un-quantized) KV-cache reads stay 16-bit. ⇒ for the
roofline, fp8 ≈ **1.8× over bf16** is the honest multiplier, not a clean 2×.

**Accuracy for a top-8 MoE (E4M3, the standard weight format — 4-exp/3-mantissa):**
- **>99% of bf16** recovered for ≥30B models; **MMLU drop < 0.5%** (often <0.3% with per-channel/block
  scaling).
- Sensitive math/reasoning (AIME, GPQA) on MoE variants: within **~1–2 pp** (≈ sampling noise).
- **Can be done "naively" without calibration** — E4M3's dynamic range natively covers pretrained MoE
  weights, unlike INT8/INT4 which need AWQ/GPTQ to protect outliers. ⇒ fp8 weight storage for the experts +
  attn + lm_head is safe for *inference generation* (the GRPO-parity concern in VLLM_KERNELS.md §2 is
  RL-training-specific, not an inference-accuracy issue).

*(Sources: vLLM GitHub + Neural Magic blogs (Marlin/Machete, 1.6–1.9×); HuggingFace + arXiv fp8-inference
studies; EmergentMind aggregation. Domains: github.com, huggingface.co, arxiv.org, emergentmind.com.)*

---

## 5. Speculative decoding for MoE — the only lever past the single-token ceiling

**Methods + dense-model acceptance (mean accepted tokens / forward pass):**

| Method | how | speedup (dense) | accept length |
|---|---|---|---|
| **n-gram / prompt-lookup** | model-free, copy substrings from prompt/output | ~1.2–1.5× | 1.0–1.4 |
| **Medusa** | extra heads on the last layer predict K future tokens | ~1.5–2.2× | 2.0–2.8 |
| **EAGLE / EAGLE-2** | drafter predicts target *hidden states*; EAGLE-2 dynamic tree-drafting | 2.0–3.5× | 2.5–3.5 |
| **EAGLE-3** (SOTA, 2026) | multi-layer feature fusion + training-time testing | **3.0–6.5× (dense)** / **1.5–2.0× (MoE)** | 3.5–5.0 |

**Production + Qwen3:** vLLM (`--speculative-model`, e.g. an `eagle3` speculator checkpoint) and SGLang
(RadixAttention + EAGLE-3 kernels) serve spec-decode for 30B-class MoEs. **Qwen3 ships Multi-Token
Prediction (MTP)** — an internal draft head, so the model itself proposes multiple tokens for self-
verification (no separate draft model needed).

**THE CRITICAL MoE CAVEAT — the "union-of-experts" penalty.** On a *dense* model, verifying K draft tokens
in one pass is free (you load all weights anyway). On a *sparse top-8 MoE*, each of the K tokens routes to
its **own** top-8 experts, so the verify pass must load the **union** of all their selected experts —
**more than 8 per layer.** Expected unique experts per layer (balls-in-bins, E=128, top-8):

| K (verify width) | unique experts/layer | fp8 GB/pass (experts+attn+lm_head) | accept (est.) | **tok/s @80% BW** | tok/s @70% BW |
|---|---|---|---|---|---|
| **1** (no spec) | 8.0 | 3.04 | 1.0 | **71.8** | 62.8 |
| **2** | 15.5 | 4.74 | ~1.9 | **87.5** | 76.6 |
| **3** | 22.5 | 6.33 | ~2.4 | **82.8** | 72.4 |
| **4** | 29.1 | 7.83 | ~2.8 | **78.1** | 68.4 |
| **5** | 35.3 | 9.23 | ~3.0 | **71.0** | 62.1 |

(Roofline-computed: per-expert 4.72M params × unique × 48 layers + 1.23B shared, fp8 1 B/param; accept
lengths are EAGLE-style estimates for this model. The shared attn+router+lm_head bytes are read once per
pass — that amortization is the only reason spec-decode still nets a win.)

**Reading the table:** the union penalty caps the win — beyond K≈2–3 the extra expert bytes outrun the extra
accepted tokens. **Sweet spot K=2, net ≈ 1.22× over single-token** (vs 2–4× on dense). At larger K
spec-decode can even go *slower* than autoregressive on a sparse MoE. *(Confirmed by "MoE-Spec: Expert
Budgeting" arXiv 2026 and "Cascade: Utility-Driven Speculative Decoding for MoEs" 2025, which both cap/adapt
K precisely because of this.)*

**Stacking:** spec-decode **does** stack with fp8 + top-8 — multiplicatively on the *throughput* side: MoE
sparsity (~10× over dense-total) × fp8 (~1.8×) × spec-decode (~1.2× on MoE, ~2–4× dense). fp8 also *softens*
the union penalty (you can afford to load 2× as many unique experts in the same time). The MoE caveat is why
the spec-decode factor is ~1.2×, not the ~2–3× a dense model would enjoy.

*(Sources: vLLM/SGLang docs; EAGLE-3 NeurIPS-2025; arXiv MoE-Spec & Cascade; PyTorch blog. Domains: vllm.ai,
github.com, huggingface.co, arxiv.org, pytorch.org, callsphere.ai.)*

---

## 6. The roofline — locally computed, the honest ceiling

Per-token DECODE weight read for 30B-A3B (active experts + attn + router + **untied lm_head**; input
embedding is a 1-row gather, not a full read):

```
attn/layer        = 18.87M   (q 2048×4096, k/v 2048×512, o 4096×2048)
1 expert          =  4.72M   (gate+up+down, each 2048×768)
top-8 experts/lyr = 37.75M ;  router/lyr = 0.26M
per-layer active  = 56.89M  → ×48 = 2.730B
lm_head           = 311.16M  (151936×2048, untied)
DECODE read       = 3.042B params / token
```

| precision / path | bytes/token | **@273 GB/s (100%)** | sustained @80% | @70% |
|---|---|---|---|---|
| **full-dense (all 128 experts)** — what runs now | 60.4 GB | **4.5** | 3.6 | 3.2 |
| **top-8, bf16** | 6.08 GB | **44.9** | 35.9 | 31.4 |
| **top-8, fp8 (W8A16)** | 3.04 GB | **89.7** | **71.8** | 62.8 |
| **top-8, NVFP4/int4 experts (~0.5 B/param exp)** | ~1.9 GB | ~140 | ~110 | ~96 |

Non-expert floor (attn+router+lm_head only) = 1.23B params → bf16 caps at ~111, fp8 at ~222 tok/s — i.e.
once experts are top-8'd, attn+lm_head are **not** the binding constraint at short context, but quantizing
them to fp8 still matters to reach the fp8 ceiling. **KV-cache read** (GQA, 4 kv-heads) adds 0.2 GB/token at
2k ctx (~0.7 ms, negligible) but **0.8 GB at 8k** and **3.2 GB at 32k** (~11.8 ms) — at long context the KV
read alone caps you near ~85 tok/s, so the 80 tok/s claim is implicitly **short-context**.

**Calibration:** measured bf16 ≈ 30 tok/s vs 45 roofline ⇒ ~67% BW efficiency in vanilla engines; measured
fp8 ≈ 52–55 vs 90 ⇒ ~58–61%; NVFP4 ≈ 65. So real engines on GB10 are landing at **~60–70% of the roofline
today**, and a fully-fused lean path (cf. llama.cpp on Strix Halo ~85%) is the target.

### Is 80 tok/s realistic single-stream?

- **fp8 + top-8 alone:** roofline 90, sustained **62–76** (70–85% eff). 80 is *above* the realistic
  sustained number and *above* every published measurement → **not a robust single-stream target by itself**;
  reachable only at the optimistic ≥85% BW-efficiency edge of a perfectly fused path.
- **+ device sampling + CUDA-graph** (PERF_80TOKS_PLAN C/D, both built): remove host launch/sync so execution
  approaches the bandwidth bound → pushes sustained toward the **65–72** top of that band, still not a clean
  80.
- **+ light speculative decoding (EAGLE-3 / Qwen3 MTP, K=2):** ~1.22× despite the union penalty → **~80–87**.
  **This is the lever that makes 80 robust.**
- **Alternative:** **NVFP4/int4 experts** (Blackwell-native, already ~65 measured) raises the roofline to
  ~140; with spec-decode on top, 80 is comfortable — at a small extra accuracy cost vs fp8.

**Bottom line:** 80 tok/s single-stream is **right at the top of the fp8+top-8 roofline and beyond today's
published best (~65)**. The plan's lever order is correct — **(A) real top-8 decode routing is the single
biggest win (0.73 → ~30–45)**, **(B) fp8 W8A16 → ~60–72**, **(C/D) device-sampling + CUDA-graph close to the
~72 sustained ceiling** — but **clearing 80 robustly needs a fifth lever: NVFP4 experts and/or EAGLE-3/MTP
speculative decoding (K=2)**, which the current plan lists only as an open question. Recommend promoting it
to a named Phase-5 lever.

---

## Sources (grounding-search domains, by section)

- **§1 GB10/measured:** storagereview.com, backend.ai, nvidia.com, medium.com (Qwen3:30B-A3B 57.3 tok/s
  Ollama), creativestrategies.com, github.com, ifactoryapp.com.
- **§2 fused-MoE:** vllm.ai (fused_moe / moe_align_block_size / cutlass_scaled_mm), github.io, cefboud.com;
  TRT-LLM Piecewise CUDA Graphs + Device Problem Sizes.
- **§3 unified-memory:** localaimaster.com, reddit.com (r/LocalLLaMA), ktransformers Tsinghua arXiv (Chen et
  al. 2025), Framework/Strix-Halo + MLX deployment reports.
- **§4 fp8 W8A16:** github.com (vLLM Marlin/Machete), huggingface.co, arxiv.org, emergentmind.com
  (Neural Magic 1.6–1.9×, MMLU <0.5%).
- **§5 spec-decode:** vllm.ai, github.com, huggingface.co, arxiv.org (EAGLE-3 NeurIPS-2025; MoE-Spec 2026;
  Cascade 2025), pytorch.org, callsphere.ai.
- **§6 roofline:** computed locally from huggingface.co/Qwen/Qwen3-30B-A3B config; cross-checked vs §1/§3
  measured tok/s.

_Cross-refs: [PERF_80TOKS_PLAN.md](PERF_80TOKS_PLAN.md) (lever plan), [VLLM_KERNELS.md](VLLM_KERNELS.md)
(fused-MoE / fp8 / CUDA-graph kernel designs), [VLLM_PARITY_PLAN.md](VLLM_PARITY_PLAN.md)._
