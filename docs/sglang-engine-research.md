# SGLang inference-engine internals — what makes DECODE efficient (MoE + long context)

Research doc. Goal: understand how SGLang (sgl-project/sglang) keeps **decode** fast for
Qwen3-30B-A3B-class **MoE** models and **long context**, and contrast it with OUR current
static-decode design so we know exactly what to copy.

> Method: live web search via `THINK=HIGH /workspace/agy-direct.sh "<q> Search the web."`
> (Gemini 3.1 Pro High + Google Search grounding). Sources are listed per-section at the
> bottom. GitHub file paths are as cited by the grounded answers; treat line numbers as
> approximate (repo moves fast) and re-verify against the pinned ref before relying on them.

---

## TL;DR — the contrast that matters for us

OUR static decode (`src/attention.rs::forward_with_cache_static_pre_lp`, lines ~494–525):
each step we read the **FULL `T_max` KV buffer** `key.dims()[1] == T_max`, compute scores over
**all** `T_max` columns, then mask the future with `idx > pos` (-inf). Cost is **O(T_max)
per token regardless of position** — a 100-token prompt in a 32k buffer still pays the 32k
attention. Single CUDA stream, one captured graph shape.

SGLang does **NOT** do this. Decode attention is **paged + ragged**: each request's
`seq_lens[i]` drives the kernel so it reads only the **valid prefix `[0..pos]`** of that
request's KV (gathered through a `req_to_token` page table). Cost is **O(actual context
length)**, not O(max). That is the single biggest decode-efficiency gap between us and SGLang
at long context. Everything below is the machinery that makes that work while still being
CUDA-graph-captured and continuously batched.

---

## 1. Runtime architecture — Scheduler / ModelRunner / request lifecycle

SGLang Runtime (SRT) is a multi-process, asynchronous design that **decouples CPU request
coordination from GPU model execution** (IPC via ZeroMQ).

**Request lifecycle:**
1. **Tokenize** — `TokenizerManager` (`python/sglang/srt/managers/tokenizer_manager.py`)
   turns the prompt into `input_ids`, assigns a request id (RID), forwards to the Scheduler.
2. **Schedule** — `Scheduler` (`python/sglang/srt/managers/scheduler.py`) puts the `Req` in
   the **`waiting_queue`**; once KV-cache + sequence slots are free it admits it to the
   **`running_batch`**.
3. **Prefill / Extend** — compute-bound; `ForwardMode.EXTEND`. Long prompts can be split via
   **chunked prefill** (see §6).
4. **Decode** — `ForwardMode.DECODE`; auto-regressive, one token/step, **continuous
   batching**; memory-bandwidth (KV-read) bound.
5. **Detokenize** — `next_token_ids` ferried back over IPC; detokenized, stop-criteria
   checked, streamed to the client.

**Scheduler event loop** (`scheduler.py`): an infinite loop, `event_loop_overlap` overlaps
CPU scheduling with GPU compute. Each iteration: `recv_requests()` → fill `waiting_queue` →
evaluate against free KV slots → form a batch → `run_batch()` dispatch. **Continuous
batching**: when a request hits its stop token it is evicted from `running_batch` immediately,
freeing a slot for a waiting request to start prefilling the very next step (no head-of-line
blocking). Scheduling policy: FCFS or **LPM (Longest-Prefix-Match)** to maximize radix reuse.

**Execution layers:**
- **`TpModelWorker`** (`python/sglang/srt/managers/tp_worker.py`) — GPU-side executor
  (one per TP shard); entry `forward_batch_generation`.
- **`ModelRunner`** (`python/sglang/srt/model_executor/model_runner.py`) — owns weights +
  kernels, implements the **prefill/decode split**: `forward_extend()` (dense attention for
  prefill) vs `forward_decode()` (autoregressive, **CUDA-graph captured**, `init_device_graphs()`).

**Three batch representations** (separation of concerns):
- **`ScheduleBatch`** (`managers/schedule_batch.py`) — CPU-only, list of `Req`, memory
  tracking, `ForwardMode`.
- **`ModelWorkerBatch`** — lightweight IPC payload Scheduler → worker.
- **`ForwardBatch`** (`model_executor/forward_batch_info.py`) — GPU tensors the kernels
  consume: `input_ids`, **`req_pool_indices`** (pointers into paged KV), **`seq_lens`**
  (per-request lengths that drive attention), tagged `EXTEND` / `DECODE` / `MIXED`.

**Contrast:** we have no scheduler / continuous batching — a single static graph drives a
fixed batch. SGLang's `seq_lens` in `ForwardBatch` is the exact mechanism we lack: it tells
the kernel each request's real length so attention is O(real len), not O(T_max).

---

## 2. RadixAttention — prefix-cache KV reuse (radix tree)

SGLang treats the **global KV cache as a radix tree** (compressed trie), not a flat buffer
or a plain block hash-map. `python/sglang/srt/mem_cache/radix_cache.py` (`RadixCache`).

- **Nodes** = a token segment + its physical KV indices; **path root→node** = a prefix
  (system prompt / shared doc); KV is **retained after a request finishes**.
- **`match_prefix`** — traverse from root, longest-common-prefix match in **O(L)**; returns
  the cached KV indices for the shared part so only the *new* suffix is prefilled.
- **`insert`** — adds new tokens+KV; **node-split** at the divergence point so a shared
  prefix stays a single physical copy and branches fork from it.
- **Eviction** — reference-counted (`lock_ref`); active nodes are locked. **LRU on
  zero-ref leaves**, recursively up the tree, so hot ancestors (system prompts) are evicted
  last (`mem_cache/evict_policy.py`). Scheduling integrates via `managers/schedule_policy.py`
  (LPM ordering to maximize hits).

**Why it helps:** multi-turn chat (turn N reuses turns 1..N-1 KV), few-shot (shared examples
cached once), shared system prompts (root node), and **agent tree-of-thought / forking**
(branches share the common ancestor KV with zero recompute).

**vs vLLM PagedAttention APC:** vLLM does **block-level hashing** (hash of block content +
prev-block hash) in a global hash-map; SGLang does **tree traversal** for longest prefix.
The tree natively represents branching/forks and its leaf-LRU naturally tiers shared roots to
live longest; block-hashing matches exact block sequences and is less flexible across
variable-length gaps. Both eliminate fragmentation; the tree is tuned for branching agent
workloads.

**Contrast:** orthogonal to our work today (we don't reuse KV across requests), but it is the
reason SGLang's *effective* prefill cost collapses for shared-prefix workloads — and the
radix indices feed the same paged `req_to_token` map that makes decode ragged (§3).

---

## 3. KV cache + decode attention at LONG context — the core efficiency

**ANSWER (the contrast we care about): SGLang decode reads only the valid prefix `[0..pos]`,
cost O(seq_len_so_far) per request — NOT a padded `max_seq_len` buffer.** This is the synergy
of a **paged KV pool** + **flash-decoding** (FlashInfer or Triton).

**Two-tier memory (`mem_cache/memory_pool.py`):**
- **`token_to_kv_pool` (physical)** — one big pre-allocated pool of KV "slots"/pages holding
  the actual K/V tensors, **agnostic to which request owns a slot** (`TokenToKVPool` /
  `PagedTokenToKVPoolAllocator`). No per-request `[B, max_seq, d]` padded tensor exists.
- **`req_to_token` page table (`ReqToTokenPool`)** — the MMU: maps `(req_id, logical token
  pos) → physical slot index` in `token_to_kv_pool`. A request's KV may be scattered; the
  page table gathers exactly its valid tokens. **There is no "padding element" inside the
  cache.**

**`seq_lens` drives the kernel (ragged batch):** each decode step the scheduler builds a
batch whose requests have *different* lengths → a **ragged** batch. `ForwardBatch.seq_lens`
(e.g. `[15, 2048, 128]`) carries each request's exact current length.

**`flashinfer_backend.py` (`layers/attention/flashinfer_backend.py`):** configures
`BatchDecodeWithPagedKVCacheWrapper`. Before the forward, an **`update()`/`plan()`** call
feeds `req_pool_indices`, `paged_kv_indptr`, `paged_kv_indices`, and `seq_lens`; FlashInfer
sizes its **CUDA grid dynamically from the actual lengths**, not the max.

**Why it is O(pos) — flash-decoding mechanics:**
1. **Dynamic grid sizing** — number of CTAs is computed from `seq_lens[i]` (e.g. 120 valid
   tokens / block 64 → 2 CTAs). A length-10 request spawns **no** blocks for tokens 11..2048.
2. **Pointer-chasing via `req_to_token`** — the kernel's KV loop runs `O(seq_len_so_far)`
   steps; at step `j` it reads `req_to_token[req][j]` → physical slot → loads that one KV
   vector. (Split-K over the sequence = the "flash-decoding" parallelization.)
3. **No padded math** — there is **no dense QK^T over zeros + causal mask**. It computes
   `q·k` only for the valid fetched tokens, online-softmax reduces, writes out.

Triton decode path (`layers/attention/triton_ops/`) is the same shape: the outer KV loop
bound is literally `seq_len[batch_idx]`, so it does exact O(N) work for context N.

**>>> THE CONTRAST WITH US <<<** Our `forward_with_cache_static_pre_lp` does precisely the
thing SGLang avoids: `t_max = key.dims()[1]` is the **full buffer width**; we run dense
`attention(q, k_full, v_full, mask = idx>pos)` over **all `T_max` columns** and throw away the
future with -inf. So our decode is **O(T_max) every step**, position-independent — a short
context in a big buffer pays full price, and a long buffer is unaffordable. SGLang's
`req_to_token` + `seq_lens` + flash-decode make the *same* logical attention cost **O(pos)**.
To match it we need (a) a paged/compact KV (only valid slots), and (b) a decode kernel whose
KV loop is bounded by a per-request length, not the static `T_max`.

---

## 4. Fused MoE decode — fused_moe_triton / EPMoE

Two tracks: **local multi-expert** (`FusedMoE` / `fused_moe_triton`, all experts on one GPU)
vs **expert-parallel** (`EPMoE`, experts sharded across GPUs). Qwen3-30B-A3B (128 experts,
top-8, ~3B active) typically runs the local `fused_moe_triton` track on one GPU.

**`moe_align_block_size` (`sgl-kernel/csrc/moe/moe_align_kernel.cu`)** — the key that turns
scattered routing into one grouped GEMM:
1. **Histogram** tokens-per-expert; 2. **prefix-sum** → each expert's start offset; 3. write
`sorted_token_ids` grouped by expert, **padding each expert's run** up to a multiple of the
GEMM block size (16/64) with dummy tokens. For batch-1 decode, newer kernels use multiple
thread-blocks + SM-cooperative sync to do this in microseconds (was a single-block serial
bottleneck).

**`fused_moe_triton` (`layers/moe/fused_moe_triton/{layer,fused_moe}.py`)** — a **single
grouped GEMM** (GEMV at batch-1) instead of a per-expert loop of small GEMMs. Because
`moe_align_block_size` pre-sorted+padded the tokens, the Triton kernel reads contiguous blocks
per expert; each block multiplies its token rows by the expert's weights. At decode (small
batch) the MoE is **memory-bandwidth bound** (reading the active experts' weights), so the
kernel is tuned as a grouped GEMV: stream expert weights → SRAM, warp-level dot product,
no large layout shuffle. **This is exactly the "load-contiguous expert storage" our Wave-2
chose (b)** — the experts live in one `[E, ...]` buffer indexed by a sorted/aligned id map.

**`EPMoE` (`layers/moe/ep_moe/layer.py`) + DeepEP (`token_dispatcher/deepep.py`)** — for
models too big for one GPU: shard experts across GPUs; `dispatch()` does an All-to-All
(RDMA/NVLink) sending token hidden states to the GPU owning the target expert, the remote GPU
runs the local grouped GEMM (DeepGEMM/MegaMoE), `combine()` does the reverse All-to-All,
scaling by router weights and reducing back. **Not needed for single-GPU 30B-A3B**, but the
mechanism that scales to 200B+.

**CUDA-graph capture of MoE decode: YES.** The hard part is that routing is dynamic (expert
4 gets 5 tokens this step, 0 the next). SGLang makes it capturable by **static-max padding**:
`sorted_token_ids` / dispatch buffers are sized to a fixed `max_num_tokens_padded` bound, so
shapes are constant across steps and the graph can replay. EP adds DeepEP **`low_latency`
decode mode** (pre-allocated buffers + graph-compatible collectives, no CPU-GPU sync); a
"breakable CUDA graph" fallback exists if padding exceeds limits.

**Contrast:** SGLang captures the *whole* MoE decode (router + align + grouped GEMM) in a
graph by padding the routing to a static max — the same trick we need: a contiguous expert
store (our (b)) + a fixed-shape aligned token map so the gather-GEMV is capturable.

---

## 5. CUDA graphs for decode — CudaGraphRunner

`CudaGraphRunner` (`model_executor/cuda_graph_runner.py`) eliminates per-step launch overhead
(decode launches hundreds of tiny kernels).

**Batch-size bucketing (`capture_bs`)** — at init, capture a *separate* graph per batch-size
bucket `[1, 2, 4, 8, 16, …, max_cuda_graph_bs]`. At runtime, real `bs=13` → pad up to the
smallest bucket `16`.

**Padding + replay + slice:** the 13 real requests fill slots 0..12; the 3 pad slots get
*valid-but-harmless* dummy KV indices (so no illegal-memory-access). `graphs[16].replay()`
runs all 16; then slice `logits[:13]` and discard the dummies.
(`pad_up_to_next_captured_batch_size`.)

**Graph memory pool:** capture buckets in **reverse order (largest → smallest)** sharing one
`torch.cuda.graph_pool_handle()`; the largest reserves the max, smaller graphs reuse the same
pool for free. Outputs as weak refs to maximize reuse.

**Static buffers** (pre-allocated to `[capture_bs]`, filled by `copy_()` each step):
`input_ids`, `positions` (RoPE), `seq_lens`, `out_cache_loc` (where this step's new KV is
written in the pool).

**>>> THE KEY MOVE FOR LONG CONTEXT UNDER CAPTURE <<<** SGLang does **NOT** bake a fixed-max
KV layout into the graph. The graph is static in **shape** (query is `[bs, 1, H, d]`, always
one token; `seq_lens` is a fixed-size `[bs]` tensor) but **dynamic in the VALUES of
`seq_lens`**. FlashInfer's paged decode reads the *contents* of `seq_lens` + the page table:
the captured kernel's threads loop `seq_lens[i]` times over the paged KV, so control-flow
divergence (a 50-token request next to a 4000-token one) happens at the **thread/block level
on the GPU**, with no change to graph topology. Hooks:
`init_forward_metadata_capture_cuda_graph` (capture) and
`init_forward_metadata_replay_cuda_graph` (replay) in `flashinfer_backend.py` re-point
FlashInfer's page tables/indices into the static graph each step.

- **Captured once:** the whole model graph per bucket — kernel grid/block configs, arg
  pointers to the static buffers, attention metadata init.
- **Replayed each step:** `copy_()` new `input_ids/positions/seq_lens/out_cache_loc` →
  update FlashInfer metadata → `graph.replay()` (no Python, no launches).

**>>> CONTRAST WITH US <<<** We *also* capture a fixed-shape graph — but our shape includes
the **full `T_max` KV** and we get position-correctness from a `idx > pos` mask, so the
captured kernel always does `T_max` work. SGLang captures a graph whose **attention work
tracks `seq_lens` values, not a captured max** — same static graph, but O(pos) compute,
because the length lives in *buffer contents* consumed by a paged kernel, not in the captured
shape. This is the design change that lets one captured graph serve every context length
cheaply. (SGLang still buckets on **batch size**, like us — but NOT on context length.)

---

## 6. Long-context specifics — chunked prefill, RoPE/YaRN, decode scaling

**Chunked prefill (`--chunked-prefill-size`, default ~4096/8192;
`managers/schedule_batch.py` + `scheduler.py`):** split a 100K prompt into e.g. 8K chunks,
one chunk per scheduler iteration. Bounds peak activation memory to **O(chunk²)** instead of
**O(context²)** (the QK^T activation), avoiding prefill OOM, and lets the scheduler
**piggyback decode steps for other requests** between chunks (no head-of-line blocking).

**Max length + YaRN (`server_args.py`: `--max-model-len`, `--rope-scaling`):** YaRN extends
context with **piecewise frequency scaling** — preserve high-freq (local) dims, compress
low-freq (global) dims — vs naive linear interpolation that degrades both. Injected via
`--rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`.

**Decode cost = KV-read bound, O(N):** to make one token, you compute a single Q vector but
**must read K,V for all N prior tokens** from HBM → SRAM. Arithmetic intensity is tiny;
the wall is HBM **bandwidth (TB/s)**, and per-token latency scales **O(N)** in context length
(100K context = streaming GBs of KV per token). FLOPs are not the limit.

**Paged KV avoids padded slots:** VRAM is partitioned into fixed-size blocks (16/32 tokens);
a **block table** maps a request's logical sequence to scattered physical blocks, allocated
**on demand** as tokens arrive. The paged-attention kernel (FlashInfer/Triton) iterates **only
the block-table entries**, i.e. the N *populated* tokens — it doesn't know `--max-model-len`.
So decode read cost **tracks true context length, with zero bandwidth/compute on padding**,
and RadixAttention lets prefix-sharing requests reuse the same physical blocks.

**Contrast:** SGLang's decode is O(N_real) and **inherently bandwidth-bound on only the live
KV**; ours is O(T_max) — we pay the *padding* bandwidth too (reading masked-out KV columns
every step). At 32K T_max with a 1K real context, that is ~32× wasted KV-read bandwidth per
token. We also have no chunked prefill, so a long prompt is one giant O(L²) activation.

---

## 7. What SGLang does better for long-context decode (vs our static decode)

| Axis | OUR static decode (today) | SGLang | Cost gap |
|---|---|---|---|
| KV read per step | Full `T_max` buffer, masked `idx>pos` | Only valid `[0..pos]` via paged `req_to_token` + flash-decode | **O(T_max) → O(pos)** |
| KV layout | Dense `[B, T_max, …]` padded | Paged pool + page table (`token_to_kv_pool` / `req_to_token`), block size 16/32 | no padded reads |
| Length under CUDA graph | Baked into captured **shape** (`T_max`) | In **buffer contents** (`seq_lens` values); shape stays `[bs,1,…]` | one graph serves all lengths cheaply |
| Bucketing | (we'd bucket length too) | Bucket on **batch size only**, never context length | fewer graphs, length-free |
| MoE decode | contiguous `[E,…]` store (Wave-2 (b)), gather-GEMV | `moe_align_block_size` sort+pad → grouped GEMV; captured via static-max pad | same idea; theirs is captured |
| Prefill at long L | single O(L²) pass | chunked prefill, O(chunk²), interleaves decode | no OOM / no HOL block |
| Cross-request reuse | none | RadixAttention prefix tree | shared-prefix prefill ~free |

**The one change that matters most for us:** make decode attention **ragged/paged** so the KV
loop is bounded by a **per-request length carried in a buffer**, not by the captured `T_max`
shape. SGLang's recipe, concretely portable to our single-stream captured decode:

1. **Paged (or at least compacted) KV** — store only valid tokens; a `req_to_token`-style
   index gives the kernel the physical slots. Even single-request, a compact `[0..pos]` KV
   beats a `T_max` buffer.
2. **A decode attention kernel whose sequence loop bound is a runtime `seq_len` value**, read
   from a *static-shaped* buffer — not the tensor's static dim. This is the crux: it keeps the
   graph capturable (shapes constant) while making compute O(pos). We currently encode length
   in the **shape** (`t_max = key.dims()[1]`) and mask; SGLang encodes it in **contents**.
3. **Keep batch-size bucketing, drop any length bucketing.** One captured graph per bs bucket
   serves every context length because length is data, not shape.
4. **MoE:** our Wave-2 (b) contiguous expert store already matches SGLang's `FusedMoE`
   `w13/w2` layout; to capture the MoE decode, pad the routing/aligned-token map to a static
   max (`moe_align_block_size` style) so the grouped-GEMV is fixed-shape and replayable.
5. (Later/optional) chunked prefill to bound long-prompt activation memory; RadixAttention if
   we ever serve shared-prefix/multi-turn workloads (GRPO rollouts share a prompt prefix — a
   natural radix win).

**Caveat on these notes:** answers are LLM-grounded web search (Gemini 3.1 Pro High). Class
names and the O(pos) decode behavior are well-corroborated across GitHub + LMSYS/arXiv
sources; treat exact file line numbers as approximate and verify against a pinned SGLang ref
before coding. (E.g. `radix_attention.py` is the attention-layer wrapper; the page-table/pool
lives in `mem_cache/memory_pool.py`; the decode wrapper config in
`layers/attention/flashinfer_backend.py`.)

---

## Sources

Grounded web-search (Gemini 3.1 Pro High + Google Search). Key SGLang source paths cited:

- **Architecture / lifecycle:** `python/sglang/srt/managers/{tokenizer_manager,scheduler,
  tp_worker,schedule_batch,schedule_policy}.py`,
  `python/sglang/srt/model_executor/{model_runner,forward_batch_info}.py`.
- **RadixAttention:** `python/sglang/srt/mem_cache/{radix_cache,evict_policy}.py`,
  `python/sglang/srt/layers/radix_attention.py`. Refs: LMSYS RadixAttention blog, SGLang
  arXiv paper, sgl-project/sglang GitHub.
- **Paged KV / ragged decode:** `python/sglang/srt/mem_cache/memory_pool.py`
  (`ReqToTokenPool`, `TokenToKVPool`/`PagedTokenToKVPoolAllocator`),
  `python/sglang/srt/layers/attention/flashinfer_backend.py`
  (`BatchDecodeWithPagedKVCacheWrapper`, `init_forward_metadata_{capture,replay}_cuda_graph`),
  `python/sglang/srt/layers/attention/triton_ops/` (Triton decode).
- **Fused MoE:** `python/sglang/srt/layers/moe/fused_moe_triton/{layer,fused_moe}.py`,
  `python/sglang/srt/layers/moe/ep_moe/layer.py`,
  `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`,
  `sgl-kernel/csrc/moe/moe_align_kernel.cu`.
- **CUDA graphs:** `python/sglang/srt/model_executor/cuda_graph_runner.py`
  (`capture_one_batch_size`, `pad_up_to_next_captured_batch_size`, `replay`, graph pool).
- **Long context:** `python/sglang/srt/server_args.py` (`--chunked-prefill-size`,
  `--max-model-len`, `--rope-scaling`/YaRN); FlashInfer/flash-decoding for KV-bandwidth-bound
  O(N) decode. Refs: arXiv, FlashInfer/flash-decoding writeups.

Grounding redirect URLs (Google Vertex grounding) for each query are preserved in the raw
query logs under
`/tmp/.../scratchpad/q{1_arch,2_radix,3_kv,4_moe,5_cudagraph,6_longctx}.txt`.

OUR-SIDE refs (for the contrast): `src/attention.rs:445-534`
(`forward_with_cache_static_pre_lp`, full-`T_max` masked decode),
`docs/WAVE2_STATIC_DECODE.md` (contiguous expert store decision (b)),
`docs/PERF_80TOKS_PLAN.md`.
