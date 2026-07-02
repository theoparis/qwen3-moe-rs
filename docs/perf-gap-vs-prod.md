# Perf gap vs prod — why our single-stream Qwen3-30B-A3B decode trails SGLang/vLLM/llama.cpp on GB10

Goal: explain why our **single-stream** Qwen3-30B-A3B decode on an **NVIDIA GB10 / DGX Spark** (128 GB
unified LPDDR5X, **273 GB/s**) is much slower than production engines, and give the **ranked** list of what
we're missing (biggest lever first, with a rough multiplier + the fix).

Method: web research via `THINK=HIGH /workspace/agy-direct.sh "<q> Search the web."` (Gemini 3.1 Pro High +
Google grounding), four focused queries; cross-checked against our own measured data and the prior docs
([`perf-research.md`](perf-research.md), [`PERF_80TOKS_PLAN.md`](PERF_80TOKS_PLAN.md),
[`sglang-engine-research.md`](sglang-engine-research.md),
[`longctx-decode-findings.md`](longctx-decode-findings.md)). **Source-quality caveat:** the GB10 per-engine
tok/s and the framework/spec-decode citations are community/forum-grounded and several returned citations are
clearly model-fabricated (flagged inline as ⚠); trust the *ranges* and the *independently-roofline-confirmed*
numbers, not any single decimal. The KV/GQA math (Q3) and our own measured numbers are solid.

---

## 0. Our measured reality (the thing to explain)

| regime | our tok/s | notes |
|---|---|---|
| short ctx, **eager** (`vllm_infer`) | **14–15** | bf16, fused split-K MoE GEMV, single-stream, NO graph capture |
| short ctx, **CUDA-graph captured** (separate bench) | **19–21** | capture only removes the launch tax; same kernels |
| **700–858 generated tokens** | **5.85–8.78** | the cliff — same path, just a larger `T_max` |

bf16 top-8 weight read ≈ **6.06 GB/token** → **22.2 ms** ideal → **~45 tok/s roofline**. So short-ctx eager
14 tok/s ≈ **31% of roofline** (71 ms/token; ~49 ms is overhead+attention over the 22 ms weight read);
captured 21 tok/s ≈ **47%**. At 858 tokens we fall to **5.85 tok/s = 171 ms/token** — i.e. **+100 ms/token**
appears, and it is **all in attention**, because `vllm_infer`'s decode reads the **full `T_max` KV buffer**
every step (`T_max = prompt_len + max_tokens`, `examples/vllm_infer.rs:167`), with a **physical GQA `repeat()`**
(4→32 heads) and a **non-fused reference SDPA** (sm_121 fused kernel mishandles the broadcast mask), run
**eagerly**. None of these are bandwidth on the weights — they are O(`T_max`) attention overhead that prod
engines do not pay.

---

## 1. Published single-stream 30B-A3B tok/s (confirm + per-engine breakdown)

### On DGX Spark / GB10 (273 GB/s) — batch-1 decode, short context

| engine | precision | tok/s | note |
|---|---|---|---|
| SGLang / vLLM | **bf16/fp16** | **~30–32** | 6.6 GB/token saturates 273 GB/s; matches our 45-roofline × ~67% eff |
| SGLang | **4-bit AWQ/Marlin** | ~35–42 | |
| SGLang | **fp8 (online)** | **~52–55** | |
| vLLM | **fp8** | **~50** | `Qwen3-30B-A3B-FP8`, sustained ⚠(forum-grounded) |
| llama.cpp / Ollama | **Q4_K_M GGUF** | **~57.3** (CLI logs 49–65) | the most-cited local number |
| vLLM | **NVFP4 (4-bit)** | **~56–75** | `RedHatAI/Qwen3-30B-A3B-NVFP4`, NVIDIA dev-forum range ⚠ |
| TensorRT-LLM | **NVFP4 (Triton backend)** | **~40** | **needs `TRTLLM_MOE_BACKEND=TRITON`** |
| TensorRT-LLM | **NVFP4 (CUTLASS default)** | **~4.8** | **real SM121 trap:** Spark has only **99 KiB** shared mem/block (vs 228 KiB on B200/SM100) → default CUTLASS MoE kernel won't compile, falls back to a slow path |
| MLX / TurboQuant | TQ3/TQ4 mixed | ~18 (1.84 w/o custom kernel) | MLX is tuned for Apple >400 GB/s, not GB10 ⚠ |

**Consensus published ceiling on GB10:** bf16 **~30**, fp8 **~50–55**, Q4_K_M **~57**, NVFP4 **~65** (vLLM
forum spread to 75). **Nobody has publicly shown ~80 tok/s single-stream on this hardware class** — confirms
[`perf-research.md`](perf-research.md). The SM121 99-KiB-shared-mem CUTLASS trap is a concrete, relevant GB10
gotcha (it bites prod engines too).

### Long-context decode cliff (GB10)

Community SGLang benchmarking reports a "performance cliff" — short-context ~**90–98 tok/s** (4-bit) collapsing
to **~4 tok/s at ~100K context** as the KV read dominates the 273 GB/s bus ⚠(forum-grounded, exact numbers
soft). Even prod engines fall hard at very long context — the difference is *where* the cliff starts: theirs
is set by the **real** KV bytes (O(pos)); **ours starts ~10–60× earlier** because we read the full padded
buffer × 8 (GQA) × a multi-pass SDPA.

### Datacenter GPUs (for scale — batch-1 decode)

| GPU | bf16 | fp8 | prefill (prompt processing) |
|---|---|---|---|
| B200 | ~180–220 | ~250–300 | ~20,000+ tok/s |
| H200 | ~140–165 | ~185–210 | ~8,000–15,000 tok/s |
| H100 | ~120–145 | ~160–190 | ~8,000–15,000 tok/s |
| A100-80G | ~60–75 | ~85–100 | ~2,000–4,000 tok/s |
| RTX 4090 | ~70–85 | ~90–110 | |

⚠(aggregated forum/blog numbers; treat as order-of-magnitude). They scale ≈ with HBM bandwidth (B200 ~8 TB/s
≈ 30× GB10), confirming decode is bandwidth-bound. **A 30-token prefill should take 5–50 ms** (TTFT
150–300 ms incl. serving stack) — see §2 gap #6 for why ours took 23.5 s.

---

## 2. The RANKED gaps (biggest lever first, for OUR single-stream case)

> Multipliers are for the **single-stream** path. "Long-ctx ×" = effect in the 700–858-token regime; "short
> ×" = effect at short context. The single biggest lever for the user's 700–858 regime is **#1**.

### #1 — O(`T_max`) full-buffer masked attention → O(pos) flash-decode  ⟶ THE long-ctx killer
- **What prod does:** paged KV + flash-decoding; the kernel's KV loop bound is the per-request `seq_len`
  **value** (carried in a buffer), so attention is **O(pos)**, not O(max). vLLM PagedAttention / SGLang
  FlashInfer / FlashAttention flash-decode (Dao 2023; Kwon/vLLM 2023). Captured graph stays static in *shape*,
  dynamic in `seq_lens` *values* — one graph serves every length (see [`sglang-engine-research.md`](sglang-engine-research.md) §3,5).
- **What we do:** `t_max = key.dims()[1]` = the FULL static buffer; dense QK^T over all `T_max` columns, mask
  the future with `idx>pos → -inf`. **Masking hides columns, it does not skip the work.** O(`T_max`) every
  step from step 1 (`src/attention.rs:494`).
- **Quantified KV read (in-kernel-broadcast minimum, bf16, 96 KiB/token):** 800 ctx = **78.6 MB/step (0.29 ms)**;
  4096 = **403 MB (1.5 ms)**; 32768 = **3.0 GiB (11.8 ms, caps ~84 tok/s)**. **Wasted-bandwidth at pos=800 in a
  32K buffer = 97.6%.** Our *realized* cost is far above these minima because of #2 (×8) + #5 (multi-pass) +
  eager launches — which is why an 858-token run already shows **+100 ms/token** (14→5.85).
- **Multiplier:** **long-ctx ~2–3× at 800–1K, growing to catastrophic + OOM at 32K/256K**; short-ctx ~1×.
- **Fix:** a from-scratch CubeCL **flash-decode kernel** — grid `(num_q_heads, num_kv_splits)` sized to the
  static `T_max` (capturable), inner loop bounded by the **device `pos` value**, **GQA-broadcast** the KV head
  (kills #2), **online-softmax** (kills #5). One kernel fixes #1 + #2 + #5. (Our cache already has the
  device-`pos` capture-safe write side; only the read kernel is missing — [`longctx-decode-findings.md`](longctx-decode-findings.md).)

### #2 — GQA: in-kernel broadcast vs our physical `repeat()`  ⟶ ×8 on attention traffic
- We materialize KV from 4→32 heads (n_rep=8) **every step** before SDPA → **8× the KV bytes** (786 KB/token
  vs 98 KB) + an alloc/round-trip of pure waste. Prod reads the 4 unique KV heads and broadcasts into
  registers/SRAM (multiplier **1.0**). Ainslie 2023 (GQA); vLLM/FlashInfer paged attention.
- **Multiplier:** up to **8× on the attention/KV term** (modest at short ctx where KV ≪ weights; large at long
  ctx where KV dominates). **Folded into the #1 flash-decode kernel** (one fix).

### #3 — bf16 → fp8 / NVFP4 weights  ⟶ ×1.6–1.9 (fp8), raises the short-ctx ceiling
- We're bf16 (6.06 GB/token, 45 roofline). fp8 W8A16 **halves weight bytes** → ~90 roofline, **measured
  1.6–1.9×** end-to-end (Marlin/Machete; attn/KV stay 16-bit so not a clean 2×); E4M3 keeps >99% accuracy.
  NVFP4 (Blackwell-native) → ~140 roofline, already the published GB10 ceiling (~65). **Does NOTHING for the
  long-ctx attention cliff** — it only shrinks the 22 ms weight read. fp8 also breaks GRPO rollout/recompute
  parity (inference-only lever — repo docs).
- **Multiplier:** **~1.6–1.9× at short ctx**; ~1× on the long-ctx cliff.

### #4 — CUDA-graph capture (launch tax)  ⟶ ×1.3–1.4 (ours), and `vllm_infer` is EAGER
- Eager per-kernel CPU dispatch ≈ 5–10 µs (our Fusion path ~44 µs incl. framework); a 48-layer MoE eager
  fragments into **thousands of launches/token** (our oracle ≈ 31k; routed still 48 host-syncs/layer). Capture
  replays as one graph → prod cites **1.65–2.7×**; **we measured 14→19–21 (~1.3–1.4×)**. **`vllm_infer` does
  not capture** (capture lives only in a separate bench), so the user's example pays the full launch tax today.
- **Multiplier:** **~1.3–1.4×** (already demonstrated). Cheap, orthogonal to #1.

### #5 — Fused SDPA / FlashAttention vs our non-fused reference SDPA  ⟶ folded into #1
- On sm_121 we deliberately use a **non-fused reference SDPA** (the fused kernel mishandles the broadcast mask)
  → a chain of separate matmul/mask/exp/sum/div/matmul kernels, each sweeping `[.,32,1,T_max]`, each a launch.
  A fused flash kernel does this in one online-softmax pass. **Subsumed by the #1 flash-decode kernel.**
- **Multiplier:** constant-factor on attention (several ×), realized as part of #1's long-ctx win.

### #6 — Prefill efficiency: 23.5 s for 31 tokens is a ONE-TIME warmup, not steady-state
- **Expected:** prefilling 31 tokens should be **<50 ms** (weights read once: ~22 ms routed / ~220 ms if the
  dense 128-expert oracle path runs; attention is trivial at T=31). 23.5 s is **~100–1000× too slow** ⇒
  **one-time JIT/autotune warmup**: CubeCL/Burn-Fusion compiles + autotunes every kernel on the **first
  forward**, and `vllm_infer` times the very first `forward_with_cache` (no warm-up pass). Prod sees the same
  (vLLM/SGLang Triton/CUDA-graph warmup = 30–60 s on first request, then near-instant). Secondary suspect:
  the prefill running the **dense all-experts** path (~38× the compute of top-8) — confirm `vllm_infer` prefill
  uses routed, not the oracle.
- **Impact:** a **TTFT / short-generation** killer, **not** a sustained-decode lever (amortized to ~0 over a
  long run). **Fix:** a throwaway warm-up forward at load (compile + autotune once), and ensure routed prefill.

### #7 — Continuous batching / chunked prefill / spec-decoding  ⟶ aggregate, not single-stream (except spec)
- Continuous batching, chunked prefill, RadixAttention are **multi-request throughput** levers — they do
  **nothing for single-stream latency**. The only single-stream member is **speculative decoding** (EAGLE-3 /
  Qwen3 MTP): **2.2–3.6× on dense**, but the **union-of-experts penalty** dilutes it to **~1.2–1.5× on this
  top-8 MoE** (a K-token verify routes to the union of top-8 sets — ~15 experts at K=2, ~29 at K=4 of 128, so
  it reads >8 experts; net win only with high acceptance + expert overlap, sweet-spot K=2).
- **Multiplier:** spec-decode **~1.2× single-stream** (and it's a large build). Chunked prefill matters only
  for very long prompts (bounds O(L²) activation / avoids OOM).

### #8 — Framework overhead: CubeCL/Burn-Fusion vs a hand-written CUDA engine  ⟶ the residual cap
- Mature lean engines (llama.cpp, TensorRT-LLM) saturate **80–93%** of effective bandwidth (llama.cpp on Strix
  Halo, comparable 256 GB/s unified); naive/abstracted frameworks land **15–50%**. **We measure 13–47%** —
  squarely in the naive band, consistent with a JIT/Fusion tax (dynamic dispatch, dequant widening, small-GEMV
  under-occupancy at N=8). This is not one fix; it's the cumulative ceiling the levers above chip away at, and
  the gap between our fused-kernel best and llama.cpp. ⚠(specific "Burn 8.2×" / "66–88% of llama.cpp" claims
  from Q4 are uncorroborated/likely fabricated — ignore the decimals, keep the band).

---

## 3. The #1 lever for OUR 700–858-token regime — ranked

For the user's medium-to-long-context single-stream case, ranked by payoff **in that regime**:

1. **The O(pos) flash-decode kernel (#1, +#2/#5 folded in).** This is the **only** lever that attacks the
   +100 ms/token attention cliff that takes us 14→5.85 at 858 tokens, and it's the one whose cost **grows**
   with context (2–3× at ~800–1K, catastrophic + OOM at 32K/256K). It simultaneously kills the 8× GQA repeat
   and the non-fused multi-pass SDPA. **Single biggest lever for this regime — build it first.**
2. **CUDA-graph capture in `vllm_infer` (#4).** Cheap, already-demonstrated **~1.3–1.4×** (14→19–21); the
   user's example is eager and leaves it on the table. Stacks on top of #1.
3. **fp8 weights (#3).** ~1.6–1.9×, but it raises the **short-ctx** ceiling (22 ms weight read) and does
   nothing for the attention cliff — so it pays off **after** #1 makes the path bandwidth-bound again.
4. **Prefill warm-up (#6).** Fixes TTFT (23.5 s → <0.1 s) but **does not change sustained tok/s** — do it for
   UX, not for the decode-rate number.

**Why not fp8/capture/prefill as #1:** fp8 shrinks the 22 ms weight read, but the cliff is the **+100 ms
attention**; capture removes the launch tax (a ~constant factor, already only ~1.3×); fixing prefill is a
one-time TTFT win amortized to zero over a long run. **Only the O(pos) flash-decode kernel removes the
position-dependent attention term that defines the 700–858 regime.** This matches our existing diagnosis
([`longctx-decode-findings.md`](longctx-decode-findings.md), [`sglang-engine-research.md`](sglang-engine-research.md) §3).

---

## Sources

Grounded web search (Gemini 3.1 Pro High + Google Search), 4 queries. Raw logs:
`scratchpad/q1_engines.txt` (per-engine GB10), `q2_datacenter_prefill.txt` (datacenter + prefill),
`q3_longctx_gqa.txt` (KV/GQA math), `q4_capture_spec_fw.txt` (capture/spec/framework).

- **GB10 per-engine tok/s:** NVIDIA Developer Forums (`spark-vllm-docker`, `Qwen3-30B-A3B-NVFP4`); Ollama/
  llama.cpp GGUF benchmarks (Q4_K_M ~57.3); TensorRT-LLM SM121 CUTLASS-vs-TRITON MoE backend issue (99 KiB
  shared-mem trap); SGLang long-context "cliff" threads. ⚠ several decimals/product names are
  model-fabricated — trust the bf16~30/fp8~52/Q4~57/NVFP4~65 consensus, not single figures.
- **Datacenter + prefill:** Millstone AI H200/H100 report; vLLM chunked-prefill slowdown issue #25677;
  JarvisLabs serving-stack comparison; LocalScore leaderboard. JIT/kernel-warmup + dense-all-experts as the
  classic causes of a 20+ s "prefill" on a 30-token prompt.
- **KV / GQA math (solid):** FlashAttention / Flash-Decoding (Dao et al. 2023); GQA (Ainslie et al. 2023);
  PagedAttention (Kwon et al., vLLM 2023). KV = 96 KiB/token; 800/4K/32K = 0.29/1.5/11.8 ms; GQA repeat = 8×.
- **Capture / spec-decode / framework:** CUDA-graph 1.65–2.7× decode; eager 5–10 µs/launch, MoE fragments to
  thousands of launches/token; EAGLE-3/Medusa/MTP 2.2–3.6× dense diluted to ~1.2–1.5× on top-8 MoE
  (union-of-experts); llama.cpp 80–93% vs naive-framework 15–50% of peak BW. ⚠ Q4's numbered arXiv/blog
  citations are largely fabricated (only `developer-tech.com` actually grounded) — directions corroborate our
  own measured 13–47%-of-peak and ~1.3× capture data; the specific Q4 figures are NOT load-bearing.
- **Our-side (measured, load-bearing):** [`perf-research.md`](perf-research.md),
  [`PERF_80TOKS_PLAN.md`](PERF_80TOKS_PLAN.md) (roofline + §5–7 measured 0.73→21 tok/s journey),
  [`sglang-engine-research.md`](sglang-engine-research.md) (paged O(pos) decode),
  [`longctx-decode-findings.md`](longctx-decode-findings.md) (the O(`T_max`) flaw + flash-decode fix),
  `examples/vllm_infer.rs:167` (`T_max = lp + max_tokens`, eager decode), `src/attention.rs:494`
  (`t_max = key.dims()[1]` full-buffer SDPA).
</content>
</invoke>
