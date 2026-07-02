# CUDA-Graph Integration Map — cubecl @ `b19859ee` (+ burn, cubek-random)

Source-reading map of the exact change points to add CUDA-graph capture/replay to the
Burn → CubeCL → cubecl-cuda → cudarc stack used by this repo.

Versions actually resolved (from `qwen3-burn-manin-grpo/Cargo.lock`):
- cubecl rev `b19859ee693bb02a25e4da2ca53797bb164be140` (`/workspace/cubecl`)
- **cudarc `0.19.8`** — has `result::stream::{begin_capture,end_capture,is_capturing}` and
  `result::graph::{instantiate,launch,upload,exec_destroy,destroy}` at the *raw `result::` layer*
  cubecl already uses (`/usr/local/cargo/registry/src/.../cudarc-0.19.8/src/driver/result.rs:788,798,807,1494,1521,1531,1507,1514`).
  A safe `CudaStream::begin_capture/end_capture` + `CudaGraph` also exist
  (`.../cudarc-0.19.8/src/driver/safe/graph.rs`) but cubecl does **not** use safe stream objects.
- cubek rev `1161040` — `/usr/local/cargo/git/checkouts/cubek-21eb4731b65c1fbd/1161040/crates/cubek-random/`
  (NOT the `fb2ddf2` checkout, which differs: `fb2ddf2` already passes seeds as `seeds[0]` immediates
  to a `Vector`-based kernel; `1161040` wraps them in `ScalarArg::new(...)` — cite `1161040`).

---

## Component 1 — Capture FFI (cubecl-cuda)

### The eager kernel-launch → CUstream path
`CudaServer::launch` (`crates/cubecl-cuda/src/compute/server.rs:154-408`)
→ `command.kernel(...)` (`server.rs:396`)
→ `Command::kernel` (`crates/cubecl-cuda/src/compute/command.rs:433-466`): compiles on miss
   (`command.rs:444-446`), grabs `stream = self.streams.current()` (`command.rs:448`), then
→ `CudaContext::execute_task` (`crates/cubecl-cuda/src/compute/context.rs:269-314`): **the single eager
   submit point** — `cudarc::driver::result::launch_kernel(kernel.func, dispatch_count, cube_dim,
   shared_mem, stream.sys, &mut bindings)` at **`context.rs:297-306`, submitting onto `stream.sys`
   (`context.rs:304`)**.

The raw stream is `Stream.sys: cudarc::driver::sys::CUstream` (= `*mut CUstream_st`),
**a public field** (`crates/cubecl-cuda/src/compute/stream.rs:18`), created by
`CudaStreamBackend::create_stream` via `cudarc::driver::result::stream::create(NonBlocking)`
(`stream.rs:36-39`). Capture FFI takes exactly this `CUstream` by value.

### Host syncs (must be OUTSIDE the capture window)
- `Command::sync` — `Fence::new(current.sys)` + `fence.wait_sync()` (`command.rs:410-414`).
- `Command::read_async` — `Fence::new` + `wait_sync` in the returned future (`command.rs:182-202`);
  d→h memcpy in `write_to_cpu` (`command.rs:557-622`).
- `CudaServer::sync` (`server.rs:412-415`); `flush` is a **no-op** (`server.rs:410`).
- `start_profile`/`end_profile` — `block_on(self.sync())` (`server.rs:417-434`).
- `Fence` (`crates/cubecl-cuda/src/compute/sync/fence.rs`): `event::create`+`event::record` are
  capturable; `wait_sync` (`event::synchronize`, `fence.rs:42-59`) is a **host** sync — used by the
  cross-stream GC thread (`crates/cubecl-runtime/src/stream/event.rs:124-129,161-182`) and must not fire mid-capture.

### `CubeCount::Dynamic` — forces a host read, EXCLUDE from capture
`server.rs:177-191`: `future::block_on(command.read_async(...))` reads 3 ints off the buffer to get
grid dims. This is a device→host sync embedded in launch; a captured graph has frozen node params.
**Capture path must assert `CubeCount::Static`** and reject/realize-before `Dynamic`.

### Who owns the stream?
`CudaContext` (`context.rs:36-44`) owns only `context: *mut CUctx_st` (the **CUcontext**), module cache,
ptx cache, timestamps — **NOT the stream**. The `CUstream` lives in `Stream` (`stream.rs:17-21`),
owned by `MultiStream<CudaStreamBackend>` → `StreamPool` → `CudaServer.streams` (`server.rs:47`).
Each `Stream` also owns its own GPU + CPU `MemoryManagement` (`stream.rs:19-20`) — relevant to C2:
capture must target the same stream's allocator as the captured launches.
`ctx.unsafe_set_current()` (binds CUcontext to the thread) is already called by `command()`
(`server.rs:715`, `context.rs:90-92`) before every op.

### Smallest seam
Wrap at `CudaContext`/`Command` level around `execute_task`. Because every kernel already targets
`stream.sys`, *any* kernels issued between begin/end on that stream are captured with **zero changes to
`execute_task`**. Concretely:
1. `begin`: `result::stream::begin_capture(stream.sys, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)`.
2. run a closure that issues the per-step ops (the existing `launch`/`execute_task` path).
3. `end`: `result::stream::end_capture(stream.sys) -> CUgraph`; `result::graph::instantiate(graph, flags)
   -> CUgraphExec`; return a replayable `CudaGraph { graph, exec, pinned_handles }`.
4. replay: `result::graph::launch(exec, stream.sys)` (+ optional `graph::upload` to prime).

Thread it as new methods server→channel→client mirroring `allocation_mode`
(`ComputeServer::allocation_mode` `crates/cubecl-runtime/src/server.rs:393`;
`ComputeClient::allocation_mode` `crates/cubecl-runtime/src/client.rs:661-663`), e.g.
`begin_capture(stream_id)`, `end_capture(stream_id) -> GraphHandle`, `graph_launch(handle, stream_id)`.
Bracket under a device-wide lock like `memory_persistent_allocation` (`client.rs:671-691`,
which uses `self.context.lock_device()`).

### Hard problems (C1)
- **Allocation during capture**: `GpuStorage::alloc` = `malloc_async(stream,…)` (`crates/cubecl-cuda/src/compute/storage/gpu.rs:162`)
  and dealloc = `free_async` (`gpu.rs:71`). During capture these turn into graph-ordered alloc/free
  *nodes* (capture-stream only, fragile lifetime). Must route all allocs to a pre-reserved capture pool
  (C2) so **no malloc happens inside the window**.
- `CubeCount::Dynamic` block_on (`server.rs:177`) → reject in capture.
- Autotune probe-launch + sync (`crates/cubecl-runtime/src/tune/`) → must pre-warm + freeze.
- GC-thread host waits + `read_async` fences must not occur in the window (keep capture single-stream;
  no reads/profiles).
- Cross-stream `Fence::wait_async` (`server.rs:569,628,677`) is capturable only if both streams are in
  the capture — single-stream capture is the safe target.

---

## Component 2 — Graph-aware allocator (cubecl-runtime)

### How allocation works
`MemoryManagement::reserve` (`crates/cubecl-runtime/src/memory_management/memory_manage.rs:403-474`):
`alloc_reserve_count += 1` → `persistent.try_reserve` (free same-size slice) → if `mode==Persistent`
or `persistent.has_size` → `persistent.alloc` → else first dynamic pool with `accept(size)`
(`memory_manage.rs:448-455`) → `pool.try_reserve` else `pool.alloc`.

`SlicedPool::try_reserve` (`memory_pool/sliced_pool.rs:52-61`): for each page, `coalesce()` then
`page.try_reserve`. `MemoryPage::try_reserve` (`memory_pool/memory_page.rs:110-143`) is **first-fit**:
first free slice with `storage.utilization.size >= effective_size`, splitting if larger.

`MemoryAllocationMode` enum = `Auto | Persistent` (`memory_manage.rs:92-101`); a field on
`MemoryManagement` (`memory_manage.rs:110`), set by `mode()` (`memory_manage.rs:336-353`, gated by
config `PersistentMemory::{Enabled,Disabled,Enforced}`), consulted in `reserve` (`memory_manage.rs:421`).

### Address stability — the evidence
- A device address = (base ptr of a `StorageId`) + offset. `GpuStorage::get` (`gpu.rs:139-154`) looks up
  `(ptr,_)` by `handle.id`, returns `GpuResource{ ptr: ptr+offset }`; base ptr per `StorageId` is fixed
  at `malloc_async` (`gpu.rs:184`). **A live `SliceHandle` → fixed `StorageHandle(id+offset+size)` →
  STABLE device address**, as long as that slice stays alive and its page isn't coalesced/realloc'd.
- The kernel-arg indirection is *not* stable but doesn't matter: `PtrBindings::register`
  (`gpu.rs:117-129`, ring buffer `94-130`) writes the ptr value into a **new ring slot per `get`** and
  returns a pointer-to-slot as the arg. The *value* (device address) is recomputed from the stable base
  each launch; the indirection pointer changes — but graph replay bakes params at capture, so only the
  *value at capture time* matters.
- **Across iterations with drop+realloc the address is NOT guaranteed identical**: `SlicedPool` coalesces
  every `try_reserve` (`sliced_pool.rs:55`) and `coalesce` mints fresh `SliceHandle::new()` at new offsets
  (`memory_page.rs:199-206`), so the "same" logical buffer can move. `PersistentPool::try_reserve`
  (`persistent_pool.rs:72-87`) returns *a* free slice of the exact size bucket (stable `StorageId` per
  slice) but "first free of that size" → identity can swap between two same-size buffers.
  (Tests confirm reuse-when-freed but not address identity: `memory_manage.rs:642-671` `alloc_reuses_storage`.)

**Conclusion:** cubecl yields a stable address only while the handle is kept alive (no drop/realloc
between iterations). CUDA-graph replay requires exactly that — addresses baked at capture must be valid
every replay.

### Where a "capture pool" hooks in (PyTorch private-pool equivalent)
Minimal API:
1. Add `MemoryAllocationMode::Capture` variant (`memory_manage.rs:92-101`).
2. Add `capture: Option<CapturePool>` field to `MemoryManagement<Storage>` (`memory_manage.rs:104-113`):
   a pool that **never recycles a live slice and never frees** during capture, retaining every minted
   `SliceHandle` in a `Vec`.
3. In `reserve` (`memory_manage.rs:403`), top branch:
   `if matches!(self.mode, Capture) { return self.capture…alloc(&mut self.storage, size) }` — bypass
   `try_reserve` so each alloc gets a fresh pinned address (or hand back the Nth-by-order allocation
   deterministically across the warmup+capture passes for replay-stable addresses).
4. Switch via existing `mode()` (`memory_manage.rs:336`), threaded `Command::allocation_mode`
   (`command.rs:92-94`) → `CudaServer::allocation_mode` (`server.rs:459-462`) →
   `ComputeServer::allocation_mode` (`server.rs:393`) → `ComputeClient::allocation_mode`
   (`client.rs:661`). Add a `memory_capture_allocation` bracket mirroring `memory_persistent_allocation`
   (`client.rs:671-691`) that also drives C1 begin/end under `lock_device()`.
5. Exit: move the retained `Vec<SliceHandle>` into the returned `CudaGraph`; its `Arc` refcounts keep
   storage alive; dropping the graph drops them → only then is the arena reclaimable. Because handles
   aren't dropped during capture, **no `free_async` is queued** (`gpu.rs:71,192-194`).

**Pre-size the arena with a warmup pass** (PyTorch-style): one iteration in `Capture` mode populates the
pool; a second iteration then hits only `try_reserve` on the populated capture pool → **zero malloc**,
addresses identical → capture that one.

### Hard problems (C2)
- First-fit + coalescing → default pools don't guarantee identical addresses across iterations
  (`memory_page.rs:159,199-206`) → must bypass into an isolated arena.
- `malloc_async` mid-capture (`gpu.rs:162`) → eliminate via warmup/pre-reserve.
- `cleanup`/`dealloc_period` (`memory_manage.rs:356-368`, `sliced_pool.rs:102-125`) and storage
  `flush`→`free_async` (`gpu.rs:197-199`) must skip/inhibit the capture arena while a graph is live.
- Per-stream allocators (`stream.rs:19`) → capture the allocator of the same stream as the launches.

---

## Component 3 — Device-seed RNG (cubek-random `1161040`)

### Today (host immediates)
- `static SEED: Mutex<Option<StdRng>>` (`crates/cubek-random/src/base.rs:14`); `seed(u64)` (`base.rs:16-20`);
  `get_seeds() -> [u32;4]` draws 4 u32 from the host `StdRng` and stores it back (`base.rs:74-87`).
- **Single chokepoint** `random()` (`base.rs:23-63`): all distributions funnel here. Calls
  `get_seeds()` (`base.rs:29`), then `prng_kernel::launch::<F,R>(…, ScalarArg::new(seeds[0..3]), …)`
  (`base.rs:48-62`) — **4 seeds as host immediates** (`ScalarArg`).
- Per-element keying in `prng_kernel` (`base.rs:118-156`): `thread_seed = 1000000007u32 * ABSOLUTE_POS`
  (`base.rs:136`); `state_i = thread_seed + seed_i` (`base.rs:138-141`); TAUS88 + LCG advance in
  `inner_loop` (`uniform.rs:24-67`; `taus_step_0/1/2` + `lcg_step` `base.rs:158-188`).
- burn entry: `Tensor::random` → burn-cubecl `random_uniform/normal/bernoulli`
  (`/workspace/burn/crates/burn-cubecl/src/kernel/prng/uniform.rs:6-26`, also `ops/tensor.rs:45-51`,
  `ops/int_tensor.rs:563-569`) → `cubek::random::random_uniform` (`uniform.rs:16`) →
  `cubek::random::random` (`base.rs:23`). Seeding via `cubek::random::seed` from
  `/workspace/burn/crates/burn-cubecl/src/backend.rs:50`.

**Why it breaks under capture:** `get_seeds()` runs on the host at launch time and bakes the 4 u32 as
immediates into kernel params. A captured graph freezes them → every replay yields an **identical**
random tensor. GRPO sampling needs fresh randomness per replay.

### Minimal change → device buffer advanced per launch
1. `prng_kernel` signature (`base.rs:118-128`): drop `seed_0..3: u32`, add a device arg — e.g.
   `seeds: &Array<u32>` (len ≥4) and a per-launch `offset: &Array<u32>` counter (a small `TensorArg`/
   `linear_view` like `output`). Read in-kernel (replace `base.rs:138-141`):
   `state_i = thread_seed + seeds[i] + offset[0]` (add, don't xor, to preserve per-thread independence).
2. Launcher (`base.rs:48-62`): pass the seed/offset buffer binding instead of `ScalarArg::new(seeds[k])`.
3. `get_seeds()` (`base.rs:74-87`) becomes one-time device-buffer init (h→d copy of initial 4 u32),
   not per-launch.
4. Capture-safe advance: a tiny 1-thread "bump" kernel that applies `taus/lcg` to the offset buffer,
   **captured inside the graph** so each replay decorrelates. (Host-side advance between replays would
   break pure replay.)
5. Centralized: only `base.rs` `random()` + `prng_kernel` change; `inner_loop` in
   `uniform.rs`/`normal.rs`/`bernoulli.rs` is untouched (still consumes `state_*`).

### Hard problems (C3)
- Keep `thread_seed = C*ABSOLUTE_POS` + per-thread TAUS/LCG; only move the *global* seed source device-side.
- Without the captured device-side advance, replays repeat → must add the bump node.
- Seed/offset buffer must be a persistent/pinned alloc (C2 capture pool) for a stable address across replays.
- `address_type="dynamic"` (`base.rs:118`) is fine; an extra tensor arg is a normal `LaunchArg`.
- `SEED` mutex can go away per-client, but concurrent launches sharing one buffer rely on stream ordering.

---

## Cross-cutting — the Fusion layer (burn)

`Cuda = Fusion<CubeBackend<CudaRuntime>>` (`/workspace/burn/crates/burn-cuda/src/lib.rs:13`). Ops are
**lazy** — enqueued as `OperationIr` in a per-stream `OperationQueue`, processed lazily.

### Lazy execution boundary + the drain hook
- `MultiStream::register` → `enqueue_operation` (`/workspace/burn/crates/burn-fusion/src/stream/multi.rs:191-212`):
  `s.queue.add(...)` then `processor.process(..., ExecutionMode::Lazy)` (may fuse+execute or hold).
- **The force-flush hook: `MultiStream::drain` (`multi.rs:244-258`)** — `processor.process(...,
  ExecutionMode::Sync)` runs ALL pending ops. Capture must `drain` before `begin_capture` so the
  captured sequence == the per-step ops.
- Plan cache: `optimizations: ExecutionPlanStore<R::Optimization>` (`multi.rs:110`) — must be frozen
  (no new fusion-plan compilation) during capture so the kernel sequence is stable.
- Execution descends `Segment::execute` → `queue.execute` (`multi.rs:279-281`) → `CubeBackend` →
  `client.launch` (`/workspace/cubecl/crates/cubecl-runtime/src/client.rs:578`) → `CudaServer::launch`.

### What capture needs from fusion
1. Run warmup iterations so `ExecutionPlanStore` is populated and stable; `drain` once to flush; then
   bracket the next iteration's register/drain between `begin_capture`/`end_capture`. The captured pass
   must hit **cached plans only** (no recompile, no autotune).
2. Freeze autotune (`/workspace/cubecl/crates/cubecl-runtime/src/tune/{tuner.rs,tune_cache.rs}`) — probe
   kernels + syncs would otherwise land in the graph.
3. Keep it single-stream: avoid cross-stream shared-view drains (`tag_shared_view` `multi.rs:168-188`)
   and the `id.executes` reentrancy guard (`multi.rs:245`).

### Hard problems (X)
- Different inputs (shape/address) can change the fused plan → capture needs shape- and address-stable
  steps (ties to C2).
- Autotune cache miss mid-capture launches benchmark kernels into the graph → pre-warm.
- Cross-stream sharing injects extra drains/syncs → forbid during capture.

---

## Change-point table

| # | Concern | File:line | Change |
|---|---------|-----------|--------|
| C1 | Eager submit onto CUstream | `cubecl-cuda/src/compute/context.rs:297-306` (`stream.sys` @304) | leave as-is; bracket with begin/end capture |
| C1 | Raw CUstream handle | `cubecl-cuda/src/compute/stream.rs:18` (`pub sys: CUstream`) | feed to `result::stream::begin/end_capture` |
| C1 | New capture seam | `cubecl-cuda/src/compute/context.rs` / `command.rs:433` | add `begin_capture/end_capture/graph_launch` on `CudaContext`/`Command` |
| C1 | Dynamic count host read | `cubecl-cuda/src/compute/server.rs:177-191` | assert `Static`; reject in capture |
| C1 | Host syncs to exclude | `command.rs:410-414,182-202`; `server.rs:412-434`; `sync/fence.rs:42-59` | keep outside window |
| C1 | cudarc FFI (raw) | `cudarc-0.19.8/.../result.rs:788,798,807,1494,1521,1531,1507,1514` | `stream::{begin,end}_capture`, `graph::{instantiate,launch,upload,exec_destroy,destroy}` |
| C1/C2 | Server→client wiring | `cubecl-runtime/src/server.rs:393`; `client.rs:661-691` | mirror `allocation_mode`/`memory_persistent_allocation` |
| C2 | Alloc mode enum | `cubecl-runtime/.../memory_manage.rs:92-101` | add `Capture` variant |
| C2 | MemoryManagement struct | `memory_manage.rs:104-113` | add `capture: Option<CapturePool>` |
| C2 | reserve() entry | `memory_manage.rs:403-421` | top-branch to capture arena (no recycle) |
| C2 | mode() switch | `memory_manage.rs:336-353` | handle `Capture` arm |
| C2 | Address (in)stability | `gpu.rs:139-154,184`; `memory_page.rs:110-143,199-206`; `persistent_pool.rs:72-87` | evidence: stable only while handle live |
| C2 | malloc/free during capture | `gpu.rs:162,71,197-199` | inhibit; pre-size via warmup |
| C3 | Seed source (host) | `cubek-random/.../base.rs:74-87,29,48-62` | init device buffer once; pass buffer binding |
| C3 | Kernel seed args | `base.rs:118-128,136-141` | `u32` immediates → device `Array<u32>` + `offset` |
| C3 | Device advance | `base.rs` (new) | captured 1-thread bump kernel on offset buffer |
| C3 | burn entry | `burn-cubecl/src/kernel/prng/uniform.rs:16`; `backend.rs:50` | unchanged surface; flows through `random()` |
| X | Fusion force-flush | `burn-fusion/src/stream/multi.rs:244-258` (`drain`) | flush before capture; bracket one iter |
| X | Fusion plan cache | `multi.rs:110` (`ExecutionPlanStore`) | freeze during capture (pre-warm) |
| X | Autotune freeze | `cubecl-runtime/src/tune/{tuner.rs,tune_cache.rs}` | warm + lock before capture |
