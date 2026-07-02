# CUDA-Graph-Safe Device RNG — Research (for a CubeCL device-seed/counter redesign)

Status: research / design input. Goal: stop CubeCL's `cubek-random` from drawing RNG
seeds on the **host** and baking them into each launch as immediate `ScalarArg`s — which
**freeze under CUDA-graph capture** and make every replay reuse identical noise. We study
how PyTorch (and JAX/cuRAND/Random123) solve this and map the solution onto CubeCL.

> One-line thesis: keep the RNG **seed + offset in DEVICE memory** (a stable pointer the
> kernel dereferences), use a **counter-based** generation scheme (RNG = pure function of
> `key, counter, thread_id`), and **advance the offset by a known increment per launch** so
> a captured graph re-reads a fresh counter on every replay. This is exactly PyTorch's
> "CUDA Graph-safe RNG states" (`PhiloxCudaState` + `seed_extragraph`/`offset_extragraph`).

---

## A. Why host-immediate seeds break under CUDA-graph capture

### A.1 What CUDA graphs do to kernel arguments
A CUDA graph captures a *topology of kernel launches* once and replays it many times with
near-zero CPU dispatch overhead. A kernel launch is recorded as a **kernel node** whose
parameters — the `void* kernelParams[]` passed to `cuLaunchKernel` — are **snapshotted into
the node at capture time** and then **frozen** for the life of the instantiated graph
(`cudaGraphExec_t`). On `cudaGraphLaunch`, every node re-executes with the *exact same*
by-value arguments it was captured with. (NVIDIA, *Getting Started with CUDA Graphs*;
*CUDA Programming Guide* §"Graph Memory Nodes" / kernel-node params.)

Consequence for RNG: if a seed/offset is passed **by value** as a kernel argument, the
literal seed bytes present at capture time become a graph-node constant. Every replay
restarts the PRNG from the *same* `(seed, offset)` → identical "random" output each replay.
For stochastic sampling (GRPO rollout sampling) or dropout this is fatal: the policy emits
the *same* tokens / the same dropout mask every captured step. (Confirmed by the
device-offset research below and PyTorch's `Note [CUDA Graph-safe RNG states]`.)

### A.2 How `cubek-random` hits exactly this trap
`cubek-random` (CubeCL) draws its seeds on the **host** from a process-global mutex RNG and
passes them as **four immediate `ScalarArg<u32>`** baked into the launch:

`/usr/local/cargo/git/checkouts/cubek-21eb4731b65c1fbd/1161040/crates/cubek-random/src/base.rs`
```rust
static SEED: Mutex<Option<StdRng>> = Mutex::new(None);          // HOST state

pub(crate) fn get_seeds() -> [u32; 4] {                         // drawn on the HOST
    let mut rng = ...; let mut seeds = Vec::with_capacity(4);
    for _ in 0..4 { seeds.push(rng.random()); } ...
}

prng_kernel::launch::<F, R>(client, cube_count, cube_dim, address_type, output,
    ScalarArg::new(seeds[0]), ScalarArg::new(seeds[1]),         // <-- IMMEDIATES
    ScalarArg::new(seeds[2]), ScalarArg::new(seeds[3]),
    args, N_VALUES_PER_THREAD, output_line_size, dtype)
```
Inside the kernel each thread mixes the host seed with its absolute position and then runs a
**stateful** TAUS88 + LCG recurrence per element:
```rust
let thread_seed = 1000000007u32 * ABSOLUTE_POS as u32;         // position-keyed
let mut state_0 = thread_seed + seed_0; ... state_3 = thread_seed + seed_3;
// inner_loop: taus_step_0/1/2 + lcg_step per value, int = s0^s1^s2^s3
```
The *only* thing that varies launch-to-launch is `seed_0..3`. Under graph capture those four
words freeze into the kernel node → every replay reuses them → identical noise. (See §E for
exactly how CubeCL lowers `ScalarArg` to a frozen kernel param.)

---

## B. PyTorch's capturable-Philox mechanism (the canonical solution)

PyTorch's design lives in `aten/src/ATen/cuda/` and is documented in
`CUDAGeneratorImpl.h` → **`Note [CUDA Graph-safe RNG states]`**. Core pieces:

- **`CUDAGeneratorImpl`** — backend of `torch.Generator(device='cuda')`; owns the live
  `seed_` and `philox_offset_per_thread_` (the running 64-bit Philox counter base).
- **`PhiloxCudaState`** (`ATen/cuda/PhiloxCudaState.h`) — the *tiny struct actually passed to
  kernels*. It is a **tagged union**: in eager mode it carries the raw `(seed.val,
  offset.val)`; in capture mode it carries **device pointers** `seed.ptr`, `offset.ptr` plus a
  per-op constant `offset_intragraph_`, and `captured_ = true`.
- **`philox_cuda_state(uint64_t increment)`** (`CUDAGeneratorImpl.cpp`) — the host-side state
  arbiter every RNG op calls. It (a) returns a `PhiloxCudaState` for the kernel, and (b)
  **advances the generator's offset by `increment`** so the next op gets a disjoint slice of
  the counter space. It 4-aligns the increment (`increment = ((increment + 3) / 4) * 4;`).

### B.1 The capturable path (what makes it graph-safe)
1. **Register at capture start.** `torch.cuda.graph()` registers the generator and allocates
   two one-element `int64` **device tensors**: `seed_extragraph` and `offset_extragraph`.
2. **Each RNG op during capture** calls `philox_cuda_state(increment)`. Instead of a raw
   offset, the returned `PhiloxCudaState` holds the **device pointer** to `offset_extragraph`
   and an `offset_intragraph_` = the *cumulative* counter consumed by prior RNG ops *within
   this graph* (a compile-time-known constant for that node). `captured_ = true`.
3. **The graph records the launch with the pointer** (a stable device address), not a value —
   so the node param does not need to change between replays.
4. **Inside the kernel**, `at::cuda::philox::unpack(PhiloxCudaState)` resolves it
   (`ATen/cuda/detail/UnpackRaw.cuh`):
   ```cpp
   __host__ __device__ std::tuple<uint64_t,uint64_t> unpack(PhiloxCudaState arg) {
     if (arg.captured_) {                                   // capture/replay path
       return { static_cast<uint64_t>(*arg.seed_.ptr),
                static_cast<uint64_t>(*arg.offset_.ptr + arg.offset_intragraph_) };
     } else {                                               // eager path
       return { arg.seed_.val, arg.offset_.val };
     }
   }
   ```
   i.e. it **dereferences the captured device pointer** to read the *current* base offset, then
   adds the per-op intragraph constant. Result feeds `curand_init(seed, thread_id, offset, &st)`.
5. **Before each replay (prologue)** the host writes the generator's *current* seed/offset into
   the captured `seed_extragraph`/`offset_extragraph` tensors (a cheap `.fill_()` /
   `cudaMemcpyAsync` to the *same addresses*). **After replay (epilogue)** the host advances the
   CPU generator by the graph's total `wholegraph_increment`, so eager ops *after* the graph
   don't collide with what the graph consumed.

Net effect: the **seed lives in device memory**, the **kernel reads it via a captured
pointer**, and **each replay sees a freshly-written base offset** → fresh, non-overlapping
noise every replay, while the host generator stays in sync. (Sources: PyTorch
`CUDAGeneratorImpl.h/.cpp`, `PhiloxCudaState.h`, `detail/UnpackRaw.cuh`,
`native/cuda/Dropout.cu`; cross-checked via web search, citations §F.)

### B.2 The increment, concretely (Dropout.cu)
`Dropout.cu` shows how an op sizes its `increment` so that **two consecutive launches never
overlap in counter space**: it rounds the per-thread element count up to whole `curand4`
draws (UNROLL = 4):
```cpp
uint64_t counter_offset = ((nelem - 1)/(block_size*grid.x*UNROLL) + 1) * UNROLL;
PhiloxCudaState rng_engine_inputs = gen->philox_cuda_state(counter_offset);
```
`(nelem-1)/(threads*UNROLL)+1` is `ceil(nelem / (threads*UNROLL))` = max loop iterations any
thread runs; `* UNROLL` accounts for each `curand_uniform4` consuming 4 counter steps. This is
the GPU analogue of "how many random numbers did I just consume" — and it's exactly the
quantity a CubeCL redesign must track to advance its device offset.

---

## C. Counter-based RNG — the right primitive (why it's graph-friendly)

**Random123 / Philox** (Salmon, Moraes, Dror, Shaw, *"Parallel Random Numbers: As Easy as
1, 2, 3"*, SC11) replaced the classic stateful recurrence `S_{n+1}=f(S_n)` with a **stateless
keyed bijection**: `R_n = bijection(key, counter)` — essentially a reduced-round block
cipher. Philox4x32-10 = 10 Feistel-ish rounds over a 128-bit counter + key using fast wide
32-bit multiplies; outputs four 32-bit words per evaluation.

Why this is the correct primitive for GPUs **and** for graphs:
- **Stateless / pure.** No per-thread RNG state in global memory to load/store/mutate. A
  thread computes `f(key, counter)` where `counter` is derived from `(thread_id, offset)`. With
  XORWOW you'd carry ~48 B of state per thread (hundreds of MB read+written *per launch* at
  scale); Philox keeps state in registers. This statelessness is also what makes it
  **replay-safe**: nothing in mutable device memory has to be reset between graph replays —
  only the small `(seed, offset)` inputs change.
- **O(1) seekable.** You can jump to the N-th number directly (`counter = N`), so disjoint
  streams are assigned by partitioning counter space — no jump-ahead matrix tricks. cuRAND's
  Philox does this in `curand_init(seed, subsequence, offset, &state)`: `seed` = key;
  `subsequence` shifts the 128-bit counter by 2^67 (one private block per thread, typically the
  global thread id); `offset` shifts by 1 (the per-op base). The next op just bumps `offset`.
- **Trivially parallel & deterministic.** Same `(key, counter)` ⇒ same output on any device,
  any thread order ⇒ bit-reproducible, race-free even with concurrent graph executions.

**JAX** makes the same bet with **threefry2x32** (the other Random123 cipher) and a
*splittable, functional* key model: `jax.random.PRNGKey(seed)` is a 2×u32 value you must pass
explicitly; the same key ⇒ the same draw (pure). You derive independent streams with
`split(key) -> subkeys` (deterministic hash) and bind external data (step, device id) with
`fold_in(key, data)`. Because randomness is a pure function of an explicit key, JAX RNG is
**replayable and `jit`/`vmap`/`pmap`-safe by construction** — the functional analogue of "keep
the counter explicit so a graph can re-read it." (Sources §F: Random123 site/paper, cuRAND
docs, JAX PRNG design notes.)

> Takeaway: counter-based = the seed/offset are *inputs*, not hidden mutable state. That is
> precisely the property a CUDA graph needs: the only thing that must change per replay is a
> couple of small input words you can keep in a device buffer and advance.

---

## D. Advancing the offset each replay — concrete options & tradeoffs

How does a *captured* graph see a different counter on each replay? Three mechanisms:

| Option | Mechanism | Host work / replay | Notes |
|---|---|---|---|
| **(a) Device buffer + in-graph increment kernel** | `(seed, offset)` live in a device buffer; RNG kernels read it via a stable pointer; a tiny kernel **inside the graph** adds the per-graph increment to `offset` each replay | **Zero** (fully GPU-autonomous after instantiation) | Cleanest for "fire-and-forget" replay loops. Downside: a host-side eager generator no longer knows how far the device advanced unless you read it back. |
| **(b) Host patches node params** via `cudaGraphExecKernelNodeSetParams` before each replay | Keep passing seed/offset by value, but rewrite the kernel-node params on the host before every `cudaGraphLaunch` | **High** — host walks nodes & calls the API per replay; reintroduces the CPU dispatch overhead graphs exist to remove | Acceptable only when RNG ops are few and you already touch the host between replays; does not scale to many RNG nodes. |
| **(c) Host writes a device buffer the kernels read (PyTorch's choice)** | Stable captured **pointer** to `seed_extragraph`/`offset_extragraph`; host updates the *contents* (one small `memcpy`/`fill`) before replay; per-op `offset_intragraph_` constants disambiguate ops within the graph | **Tiny, O(1)** — one device write regardless of how many RNG nodes | Best balance: O(1) host work, keeps the host generator in sync, no per-node patching. This is what PyTorch actually ships. |

PyTorch deliberately uses **(c)**, *not* a pure in-graph increment **(a)**: it wants the
host-side `torch.Generator` to stay authoritative so a later *eager* `torch.randn` continues
the same stream without overlap. The one device write per replay is negligible vs. the launch
savings. **(a)** is attractive if you never mix graphed and eager RNG and want literally zero
host work; **(b)** is the thing to avoid at scale. (Sources §F.)

---

## E. Mapping to CubeCL / `cubek-random` — what must change

### E.1 How CubeCL freezes the seed today (the mechanism, confirmed in source)
`ScalarArg` → `ScalarBinding` (`cubecl-runtime/src/server.rs`: `ScalarBinding { ty, length,
data: Vec<u64> }`) → lowered in `cubecl-cuda/src/compute/server.rs`. There are **two** paths
and *both* freeze the host seed under capture:

`/workspace/cubecl/crates/cubecl-cuda/src/compute/server.rs` (~L193–229):
- **grid-constants path:** scalar bytes go **straight into the `void* kernelParams[]` array**
  passed to `cuLaunchKernel` (`scalars.push(binding.data.as_ptr() ... as *mut c_void)`), i.e.
  passed **by value** (CUDA `__grid_constant__`). Under capture these literal bytes are recorded
  into the kernel node and frozen.
- **non-grid-constants path:** each scalar is uploaded to a **fresh per-launch device buffer**
  via `command.create_with_data(binding.data())`. Under capture the H2D copy *and* that specific
  allocation are baked in; the captured pointer points at a buffer holding the **capture-time**
  seeds. (Fresh-allocation-per-launch is itself capture-hostile.)

Either way the host-drawn seeds are snapshotted at capture → frozen. This is the CubeCL
embodiment of §A.1.

### E.2 Is CubeCL's scheme already "counter-based"? Partly.
- **Counter-like aspect:** the per-thread *init* is keyed by position: `thread_seed =
  1000000007 * ABSOLUTE_POS`, `state_i = thread_seed + seed_i`. `ABSOLUTE_POS` plays the role of
  Philox's `subsequence`/thread-id, giving seekable, race-free per-thread streams within a
  launch. Good.
- **NOT counter-based within a thread:** the per-element generation is a **stateful** TAUS88 +
  LCG recurrence (`taus_step_*`, `lcg_step` in `base.rs`), advanced element by element — i.e.
  `S_{n+1}=f(S_n)`, not `f(key, counter)`. There is **no explicit offset/counter input** that the
  host could advance per launch; the only run-to-run variation is the 4 host immediate seeds.

So CubeCL is "position-keyed" but its **only** per-launch entropy is a host immediate — exactly
the thing that freezes. The minimal fix does not even require swapping the bit-mixer.

### E.3 The redesign (smallest viable, then the clean version)

**Change 1 — seeds: `ScalarArg` immediates → a device buffer input (the load-bearing change).**
Replace the four `ScalarArg::new(seeds[i])` with a small **device tensor/array** input
(e.g. `[u32; 8]` holding `seed[0..4]` + a 64-bit `offset`). It is bound as a normal buffer
(handle/pointer) like `output` already is (`linear_view(...)`), so under capture the kernel
records a **stable pointer**, not the values. Inside `prng_kernel`, read the seed words and the
offset from this buffer instead of from `seed_0..3` params:
```text
// pseudo-CubeCL
seed_i  = rng_state[i];            // read from device buffer (captured pointer)
offset  = u64(rng_state[4], rng_state[5]);
thread_seed = 1000000007 * ABSOLUTE_POS;
state_i = thread_seed + seed_i + mix(offset);   // fold the per-launch offset into init
```
This alone is enough to make replays differ **iff** the offset (or seed) changes per launch.

**Change 2 — advance the offset per launch (pick a §D option).**
- Host-write (PyTorch-style, recommended): before each replay the host writes a fresh
  `(seed, offset)` into the *same* device buffer (one tiny `create_with_data`-into-existing /
  `memcpy`); after the graph, advance the host counter by the launch's increment =
  `ceil(numel / (threads * N_VALUES_PER_THREAD))`-style accounting (the CubeCL analogue of
  Dropout.cu's `counter_offset`). Keep a host-side `offset` so eager and graphed draws don't
  overlap.
- Or in-graph increment kernel (zero host work) if you never interleave eager RNG.
- Avoid per-node `cudaGraphExecKernelNodeSetParams` patching.

**Change 3 (optional, cleaner) — adopt a true counter-based core (Philox4x32 / threefry2x32).**
Generate `f(key=seed, counter = base_offset + ABSOLUTE_POS * N_VALUES_PER_THREAD + i)` per
element. Benefits: exact, overlap-free offset accounting (like cuRAND/PyTorch), no stateful
recurrence, identical results regardless of vectorization/`line_size` (today's TODO at
`base.rs:36` warns vectorization can correlate). This is the same primitive PyTorch and JAX
standardized on and is the most defensible long-term design.

### E.4 Net diff summary
| Concern | Today (`cubek-random`) | Graph-safe redesign |
|---|---|---|
| Seed origin | host `static Mutex<StdRng>` | host seed **written into a device buffer** |
| Passed to kernel as | 4× `ScalarArg<u32>` **immediates** (frozen on capture) | **device buffer pointer** (stable across replays) |
| Per-launch entropy | only the 4 frozen seeds | a **device `offset`/counter** read in-kernel |
| Advance per replay | none (immediates re-used) | host writes / in-graph kernel bumps `offset` by a known increment (§D) |
| Within-thread RNG | stateful TAUS88+LCG, no counter input | (min) fold device `offset` into init; (clean) Philox/threefry `f(seed, counter)` |
| Capture behavior | identical noise every replay | fresh, non-overlapping noise every replay |

This mirrors PyTorch's `PhiloxCudaState` (`ScalarArg`→pointer), `seed_extragraph`/
`offset_extragraph` (the device buffer), and `philox_cuda_state(increment)` (the per-launch
offset advance) — adapted to CubeCL's launch model.

---

## F. Sources (web-grounded via agy-direct / Gemini-3 + repo source)

PyTorch graph-safe RNG:
- PyTorch `aten/src/ATen/cuda/CUDAGeneratorImpl.h` — `Note [CUDA Graph-safe RNG states]`;
  `CUDAGeneratorImpl.cpp` — `philox_cuda_state(uint64_t increment)` (4-aligns increment,
  advances `philox_offset_per_thread_`).
- PyTorch `aten/src/ATen/cuda/PhiloxCudaState.h` — tagged union (`val` vs `ptr`, `captured_`,
  `offset_intragraph_`).
- PyTorch `aten/src/ATen/cuda/detail/UnpackRaw.cuh` — `at::cuda::philox::unpack(...)`:
  `captured_` ⇒ `{ *seed.ptr, *offset.ptr + offset_intragraph_ }`, else `{ seed.val, offset.val }`.
- PyTorch `aten/src/ATen/native/cuda/Dropout.cu` — `counter_offset =
  ((nelem-1)/(block_size*grid.x*UNROLL)+1)*UNROLL;` then `gen->philox_cuda_state(counter_offset)`.

Counter-based RNG:
- Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3," SC11 —
  Philox / Threefry, Random123 (random123.com / DESRES).
- NVIDIA cuRAND — Philox4x32-10 device API `curand_init(seed, subsequence, offset, &state)`
  (subsequence = +2^67 per thread; offset = +1).
- JAX PRNG design — `threefry2x32`, `jax.random.PRNGKey` / `split` / `fold_in`
  (functional, splittable, replayable).

CUDA graphs & the freeze mechanism:
- NVIDIA Developer Blog, "Getting Started with CUDA Graphs"; CUDA C++ Programming Guide
  (graph kernel-node params snapshotted at capture; `cudaGraphExecKernelNodeSetParams` to
  re-patch a node; per-replay device-buffer update is the scalable alternative).

CubeCL / cubek-random source (this machine):
- `/usr/local/cargo/git/checkouts/cubek-21eb4731b65c1fbd/1161040/crates/cubek-random/src/base.rs`
  — `static SEED: Mutex<Option<StdRng>>`, `get_seeds()`, `ScalarArg::new(seeds[i])` launch,
  TAUS88/LCG (`taus_step_*`, `lcg_step`), `thread_seed = 1000000007 * ABSOLUTE_POS`.
- `.../cubek-random/src/{uniform,bernoulli,normal}.rs` — `inner_loop` per-element recurrence.
- `/workspace/cubecl/crates/cubecl-cuda/src/compute/server.rs` (~L193–229) — scalar→kernelParams
  (grid-constants, by-value) vs scalar→`create_with_data` device buffer paths; both freeze the
  host seed under capture.
- `/workspace/cubecl/crates/cubecl-runtime/src/server.rs` — `ScalarBinding`, `MetadataBinding`.
