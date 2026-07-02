# CUDA-Graph Capture + Graph-Private Memory Pool for Autoregressive Decode

**Goal:** understand exactly how PyTorch / vLLM / TensorRT-LLM / gpt-fast implement CUDA-graph
*capture + replay* and a *graph-private memory pool* for LLM decode, so we can design the
equivalent for CubeCL on top of `cudarc`'s graph FFI.

> Sourcing note: the framework/driver facts below were gathered via grounded web search (Gemini
> grounding over NVIDIA/PyTorch/vLLM/GitHub docs) and cross-checked against the canonical primary
> sources cited in §F. The `cudarc` facts (§D) were read directly from the crate source on disk at
> `…/cudarc-0.19.8/src/driver/` (`result.rs`, `safe/graph.rs`, `sys/mod.rs`).

---

## A. The canonical capture / replay lifecycle

A CUDA graph is a static DAG of GPU work (kernel nodes, memcpy nodes, memset nodes, child-graph
nodes, memory alloc/free nodes). You build one by **stream capture**: put a stream into capture mode,
issue the normal eager op stream, and the driver *records* the launches as graph nodes instead of
executing them. The lifecycle (driver-API names; runtime-API names in parentheses):

1. **Warm up on a side (non-default) stream.** Run the exact workload eager 3-11 times *before*
   capturing. This forces lazy one-time work to happen outside the graph: cuBLAS/cuDNN workspace
   allocation + algo selection, Triton/JIT compilation, NCCL channel setup, and the caching
   allocator reaching steady state. Must be a non-default stream because the legacy default (NULL)
   stream implicitly synchronizes with everything and cannot be captured cleanly.

2. **`cuStreamBeginCapture_v2(stream, mode)`** (`cudaStreamBeginCapture`). Stream enters capture
   mode. Subsequent async work on this stream — and on any stream that *joins* via a captured event
   (`cuEventRecord`/`cuStreamWaitEvent`) — is recorded as nodes rather than run. `mode` controls how
   "potentially unsafe" calls process-wide are policed (§A.1).

3. **Issue the workload.** Every kernel/memcpy/memset becomes a node; data dependencies are inferred
   from stream/event ordering. Allocations made with the stream-ordered allocator
   (`cuMemAllocAsync`) become **mem-alloc nodes** owned by the graph (§B).

4. **`cuStreamEndCapture(stream, &graph)`** (`cudaStreamEndCapture`) → a `CUgraph` *template*. If the
   capture was invalidated mid-flight, end-capture returns an error and a NULL graph.

5. **`cuGraphInstantiateWithFlags(&exec, graph, flags)`** (`cudaGraphInstantiate`) → a `CUgraphExec`
   *executable*. This is the "compile" step: the driver finalizes node ordering, bakes in the exact
   device pointers and kernel launch params, pre-resolves the topology, and optionally uploads.
   Flags include `UPLOAD`, `AUTO_FREE_ON_LAUNCH`, `USE_NODE_PRIORITY`, `DEVICE_LAUNCH`.

6. **`cuGraphUpload(exec, stream)`** (optional) — pre-stage resources so the first launch has no
   setup spike. **`cuGraphLaunch(exec, stream)`** (`cudaGraphLaunch`) — enqueue the whole DAG with a
   single CPU call; replay as many times as you like.

7. **Update without recapture (hot path for decode):** `cuGraphExecKernelNodeSetParams` /
   `cuGraphExecUpdate` surgically patch a node's scalar args/pointers in the *executable* graph in
   microseconds — no recapture. Used to swap a changing `current_pos` / KV offset each token (§E.5).

8. **Teardown:** `cuGraphExecDestroy(exec)` then `cuGraphDestroy(graph)`.

### A.1 Capture modes (the third arg to begin-capture)

- **`GLOBAL`** (default, strictest): while *any* thread is capturing, *every* thread in the process
  is forbidden from "potentially unsafe" calls (sync `cudaMalloc`, host syncs, etc.). Process-wide
  safety, worst concurrency.
- **`THREAD_LOCAL`** (`CU_STREAM_CAPTURE_MODE_THREAD`): the restriction applies only to the
  capturing thread; other threads may keep doing unsafe calls. The right default for a server with
  background threads.
- **`RELAXED`**: unsafe calls in the capturing thread are *not* errored — but they execute
  *immediately as side effects* and are **not** recorded into the graph. Dangerous: a one-time
  allocation that happened during capture won't be reproduced on replay.

### A.2 What BREAKS / invalidates an in-progress capture

The cardinal rule: **no host synchronization inside capture.** The driver is building a pure-GPU
DAG; anything that makes the CPU wait on the GPU, or makes the next GPU op depend on a CPU-observed
result, cannot be recorded. Concretely, these poison a capture:

- **Host syncs:** `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaEventSynchronize`,
  blocking D2H `cudaMemcpy` (non-async), `cudaEventQuery`/`cudaStreamQuery` on captured work.
- **Synchronous `cudaMalloc`/`cudaFree` to the default pool** — these are global sync points on the
  memory map; illegal under GLOBAL/THREAD_LOCAL. Use `cudaMallocAsync` (becomes a mem-alloc node).
- **Uncaptured library calls / default-stream interference** — a library that internally uses the
  NULL stream or hidden syncs (older cuBLAS/cuDNN without a provided workspace) invalidates capture.
- **Two error codes to know:** `cudaErrorStreamCaptureUnsupported` is returned *immediately* by the
  offending call (e.g. you called `cudaStreamSynchronize` mid-capture). `cudaErrorStreamCaptureInvalidated`
  means the capture is already "poisoned" — a prior error or cross-stream interference broke the
  dependency chain; you must still call end-capture to reset the stream, then start over.

---

## B. The private-mempool design (PyTorch) and how it maps to a caching allocator

CUDA graphs **bake exact virtual addresses** into every kernel node. So every tensor a replay
touches must live at a *fixed, exclusively-owned* address forever. Eager PyTorch's caching allocator
normally recycles a freed block's address into an unrelated tensor a few steps later — which would
make a graph replay silently clobber live memory. The fix is a **graph-private memory pool**.

**PyTorch mechanism (the model to copy):**

- `torch.cuda.CUDAGraph` wraps `cudaGraph_t`/`cudaGraphExec_t`; `graph.replay()` == `cudaGraphLaunch`.
- `graph.capture_begin(pool=…)` / `graph.capture_end()` wrap begin/end-capture.
  `torch.cuda.graph(g, pool=…)` is the context manager that also forces a side stream + the pre/post
  syncs. `torch.cuda.make_graphed_callables` automates warmup + static-IO wiring.
- `torch.cuda.graph_pool_handle()` returns an opaque **pool id**. Passing the same id to multiple
  captures lets graphs **share** one private pool.
- **During capture**, PyTorch redirects the `CUDACachingAllocator` so every allocation made by the
  captured region is served from an **isolated private pool** tied to this graph (it tags blocks
  with the capturing graph's id).
- **After capture**, even when the Python intermediates are GC'd, those blocks are **never returned
  to the general pool** — they stay locked to the graph. Therefore the allocator can never hand
  those addresses to any other tensor, and the baked-in pointers stay valid across unlimited
  replays.
- **Pool sharing semantics:** within one private pool, blocks freed by graph A's dead activations
  *can* be reused by graph B captured against the same `pool` id — **but only because the graphs are
  guaranteed never to run concurrently** (they replay serially). This recovers eager-style memory
  reuse without ever exposing those addresses to the rest of the program. This is exactly how vLLM
  keeps ~50 batch-size graphs at roughly the memory cost of the single largest one (§C).

**Static input / output tensors.** Because addresses are frozen, you allocate fixed input/output
buffers *once* and, between replays, mutate their *contents in place* — `static_in.copy_(new)` —
never rebind the Python variable to a fresh tensor (that points at a new address the graph doesn't
know about). Outputs likewise always land in the same buffer; you read them after `replay()`.

**How this maps to a caching-allocator design (for CubeCL):**

1. A pool = (a) a large **reserved virtual-address range** + (b) a free-list of sub-blocks carved
   from it. The driver-level primitives are the **stream-ordered allocator with explicit pools**
   (`cuMemPoolCreate` → `CUmemoryPool`, `cuMemAllocFromPoolAsync` / `cuMemAllocAsync`), and under the
   hood the low-level VMM API (`cuMemAddressReserve` reserves VA once, `cuMemCreate` + `cuMemMap`
   back it with physical pages) — VA reserved once ⇒ pointers are stable ⇒ the driver skips
   pointer-patching on replay.
2. Add a per-block **owning-graph tag**. While a graph G is capturing, route its allocations into
   G's pool and mark them `owned_by=G`.
3. On free *during capture* with `cuMemFreeAsync`, the block becomes a **mem-free node** but the
   address stays reserved to the pool — only re-issuable to a *later* allocation request *inside the
   same serial graph family*, never to general traffic.
4. Track the high-water mark with `cuDeviceGetGraphMemAttribute` (`RESERVED_MEM_HIGH` /
   `USED_MEM_HIGH`) so you can size the reservation and avoid replay-time OOM.
5. Capture the **largest** decode shape first so the pool is sized to the worst case; smaller-shape
   captures reuse that same pool.

---

## C. Dynamic decode shape: static+mask vs buckets — who uses what

The conflict: graphs need **fixed** launch shapes + addresses, but per token the KV length grows by
1 and (in a server) the active batch fluctuates. Three building blocks, mixed per framework:

- **(a) static max-seqlen KV buffer + length/position input + masking.** Allocate the KV cache at the
  full max context up front; the per-step Q/K/V and the attention launch are a **fixed shape**; a
  device-side length/`input_pos` and a causal mask make the kernel *ignore* the unused tail. Shape
  never changes ⇒ one graph.
- **(b) one graph per batch (or seqlen) BUCKET + pad up.** Capture a discrete set of sizes; at
  runtime route the real batch to the next-largest captured size and pad with dummy rows that
  attention metadata masks out. Trades wasted FLOPs on padding for a bounded number of graphs.
- **(c) `input_pos` / cache position as a DEVICE tensor.** Keep the changing scalar on-device and
  update it in place between replays, so the captured graph (and `torch.compile`) never sees a
  changing Python int that would force recapture.

| Framework | KV length growth | Batch-size variation | # graphs | Notes |
|---|---|---|---|---|
| **gpt-fast** | (a) static max-len buffer + (c) `input_pos` device tensor | fixed batch | ~**1** | `torch.compile(mode="reduce-overhead")`; no paging ⇒ reserves max context per seq (VRAM-wasteful) but zero pad waste. |
| **vLLM** | handled *inside* the PagedAttention kernel via block tables + `context_lens` (so the launch shape is `[batch, 1]`, independent of seqlen) | (b) batch-size buckets + dummy padding | **many** (e.g. `[1,2,4]+range(8,256,8)+range(256,max,16)`) | All graphs **share one** `graph_pool_handle` pool (§B). |
| **TensorRT-LLM** | paged KV; seqlen abstracted by the kernel | (b) batch buckets/opt-profiles + padding (seqlen bucketing mostly for prefill) | medium-many | AOT engine + inflight batching; padding waste hidden by fused C++ kernels. |

**Tradeoffs:** (a)+(c) = fewest graphs, simplest, but max-len KV reservation wastes VRAM and bakes a
fixed max context. (b) = paged/efficient KV and continuous batching, but N graphs to capture at
startup (minutes) + pad-FLOP waste + N× the *graph metadata* (mitigated by the shared pool).

---

## D. The cudarc 0.19.8 FFI surface we can build on

Read directly from `…/index.crates.io-*/cudarc-0.19.8/src/driver/`. Two layers:

**Safe layer** (`driver/safe/graph.rs`, re-exported as `cudarc::driver::CudaGraph`) — *thin* and the
only graph type exposed:

- `CudaStream::begin_capture(mode: sys::CUstreamCaptureMode) -> Result<()>` → `cuStreamBeginCapture_v2`.
- `CudaStream::end_capture(flags: sys::CUgraphInstantiate_flags) -> Result<Option<CudaGraph>>` →
  `cuStreamEndCapture` + **immediately** `cuGraphInstantiateWithFlags` (so end-capture both ends and
  instantiates; returns `None` if the captured graph is NULL).
- `CudaStream::capture_status() -> Result<sys::CUstreamCaptureStatus>` → `cuStreamIsCapturing`.
- `CudaGraph::launch()` → `cuGraphLaunch`; `CudaGraph::upload()` → `cuGraphUpload`;
  `cu_graph()` / `cu_graph_exec()` expose the raw handles; `Drop` calls `exec_destroy` then `destroy`.
- ⚠️ `CudaGraph` is explicitly **NOT thread-safe** (graph objects aren't internally synchronized;
  even `cudaGraphInstantiate`/`cudaGraphClone` must be externally serialized).

**`result::` layer** (`driver/result.rs`) — small wrappers:
- `result::stream::{begin_capture, end_capture, is_capturing}`.
- `result::graph::{instantiate, exec_destroy, destroy, launch, upload}` — **that's the whole
  graph module.** No node-param updates, no mem-alloc nodes, no pool selection at capture.
- `result::mem_pool::{create, destroy, trim_to, get_attribute, set_attribute, …}` → `cuMemPool*`.
- `result::malloc_async` → `cuMemAllocAsync` (stream-ordered alloc), plus `malloc_sync`.

**`sys::` layer** (`driver/sys/mod.rs`, 16k lines of bindgen) — **the full driver graph API is
present as raw FFI**, even though the safe layer doesn't wrap it. Confirmed symbols include:
`cuStreamBeginCapture_v2`, `cuStreamBeginCaptureToGraph`, `cuStreamEndCapture`,
`cuStreamGetCaptureInfo{,_v2,_v3}`, `cuStreamUpdateCaptureDependencies`,
`cuThreadExchangeStreamCaptureMode`, `cuGraphInstantiateWithFlags`/`WithParams`/`_v2`,
`cuGraphLaunch`, `cuGraphUpload`, `cuGraphExecKernelNodeSetParams{,_v2}`, `cuGraphExecUpdate{,_v2}`,
`cuGraphAddKernelNode`, `cuGraphAddMemAllocNode`, `cuGraphAddMemFreeNode`, `cuGraphAddMemcpyNode`,
`cuGraphAddMemsetNode`, `cuGraphAddChildGraphNode`, plus the `cuMemPool*` / `cuMemAllocAsync` async
allocator. Enums present: `CU_STREAM_CAPTURE_MODE_{GLOBAL,THREAD,RELAXED}`,
`CU_STREAM_CAPTURE_STATUS_{NONE,ACTIVE,INVALIDATED}`,
`CU_GRAPH_INSTANTIATE_FLAG_{AUTO_FREE_ON_LAUNCH,UPLOAD,DEVICE_LAUNCH,USE_NODE_PRIORITY}`.

**Implication for CubeCL:** begin/end-capture/launch are usable through the safe API today; the
**graph-private mempool + per-step node-param update** path requires dropping to `cudarc::driver::sys`
for `cuGraphExecKernelNodeSetParams`, `cuGraphAddMemAllocNode/FreeNode`, and `cuMemPoolCreate` +
`cuMemAllocFromPoolAsync` — all present, none wrapped. **Bigger blocker (per `src/grpo/rollout.rs`):**
`cubecl-cuda` itself exposes no capture/replay hook and launches eagerly through a *lazy* Fusion op
queue, so the recorded launch list isn't stable (autotune/plan-store warmup shifts it); and there is
no graph-aware allocator (per-step intermediates get recycled → replay corruption). Capturing decode
in CubeCL needs framework-level work in `cubecl-cuda`, not just calling these FFIs.

---

## E. The 5 hardest pitfalls

1. **Stale baked pointers after reallocation.** A graph hardcodes addresses; if any tensor it reads/
   writes is freed and reallocated between replays (incl. allocator recycling of a *different*
   tensor onto the same address), the replay corrupts memory. *Fix:* graph-private pool (§B) +
   static buffers updated in place (`copy_`/`fill_`), never rebind. This is precisely why CubeCL's
   recycling allocator currently makes capture unsafe.

2. **One-time cuBLAS/cuDNN workspace "ghost" allocations.** A GEMM that lazily `cudaMalloc`s its
   workspace on first call will either error the capture (sync malloc) or bake a one-shot setup into
   the graph. *Fix:* warm up first **and** pin a static workspace (`cublasSetWorkspace`) so the
   allocation is outside the captured region.

3. **Default-stream / cross-stream interference.** Under `GLOBAL` mode, unrelated work launched on
   the NULL stream by another thread can get pulled into your capture (or invalidate it). *Fix:*
   capture on a dedicated side stream, prefer `THREAD_LOCAL` mode, and join helper streams only via
   explicit captured events.

4. **NCCL / collectives inside capture.** Graph-capturing `ncclAllReduce` needs NCCL ≥2.9 + the
   communicator warmed up; capturing on the cold first iteration can hit init-time allocations or a
   "mixing multiple communicators" deadlock. *Fix:* one eager warmup pass; consider
   `NCCL_GRAPH_MIXING_SUPPORT` tuning for multi-GPU. (Relevant if GRPO ever shards/all-reduces.)

5. **Recapturing every token instead of updating in place.** Decode changes a scalar each step
   (`current_pos`, KV write offset, sampled-token feedback). Recapturing per token throws away the
   entire win. *Fix:* `cuGraphExecKernelNodeSetParams` / `cuGraphExecUpdate` to patch just those
   params on the executable graph, and keep `input_pos`/length as a **device** tensor mutated in
   place so the captured shape never changes. Plus the decode-specific traps: the attention shape
   grows each step (needs masked full-`T_max` attention + a device length counter to stay
   fixed-shape), and any host-seeded RNG (temperature sampling) bakes frozen noise into the graph →
   identical samples on every replay (only greedy is capture-safe without an on-device RNG).

---

## F. Sources

Primary docs (cross-checked against the grounded search results):

- PyTorch — "Accelerating PyTorch with CUDA Graphs": https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- PyTorch API — `torch.cuda.CUDAGraph`, `torch.cuda.graph`, `graph_pool_handle`, `make_graphed_callables`:
  https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html ·
  https://pytorch.org/docs/stable/generated/torch.cuda.graph.html ·
  https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- NVIDIA — "Getting Started with CUDA Graphs": https://developer.nvidia.com/blog/cuda-graphs/
- NVIDIA — CUDA C++ Programming Guide, "CUDA Graphs" / "Creating a Graph Using Stream Capture":
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- NVIDIA Driver API — Stream capture (`cuStreamBeginCapture_v2`, modes, `cuStreamEndCapture`):
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html
- NVIDIA Driver API — Graph management (`cuGraphInstantiateWithFlags`, `cuGraphLaunch`,
  `cuGraphExecKernelNodeSetParams`, `cuGraphAddMemAllocNode`):
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html
- NVIDIA Driver API — Stream-ordered allocator / memory pools (`cuMemAllocAsync`, `cuMemPoolCreate`):
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html · blog:
  https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
- NVIDIA — Graphs thread-safety: https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html
- vLLM — docs (CUDA graph / optimization, V1 `cudagraph_mode` FULL vs PIECEWISE, `enforce_eager`):
  https://docs.vllm.ai/ · source: `vllm/worker/model_runner.py` (`capture_model`, `CUDAGraphRunner`,
  `_BATCH_SIZES_TO_CAPTURE`) and the V1 compilation/cudagraph dispatcher.
- gpt-fast — https://github.com/pytorch-labs/gpt-fast (static KV buffer + `input_pos` device tensor).
- TensorRT-LLM — https://nvidia.github.io/TensorRT-LLM/ (inflight batching, paged KV, opt profiles).
- `cudarc` 0.19.8 — read on disk:
  `…/cudarc-0.19.8/src/driver/safe/graph.rs`, `…/src/driver/result.rs` (`graph`, `mem_pool`,
  `stream` mods), `…/src/driver/sys/mod.rs` (full FFI). Crate docs: https://docs.rs/cudarc/0.19.8/
- This repo — `src/grpo/rollout.rs` (`group_sample_cached_device_loop` doc-comment): the existing
  in-tree assessment of why CubeCL capture is currently blocked and the expected payoff.
