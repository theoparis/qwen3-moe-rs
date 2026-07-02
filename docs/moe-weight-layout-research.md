# MoE expert-weight storage in production inference engines — research

**Question.** How do well-tested production MoE inference engines store the expert weights for a
top-k MoE like **Qwen3-30B-A3B** (128 experts, top-8) so single-token decode reads only the *k*
routed experts **without holding a duplicate copy** of the ~58 GB of expert weights?

**Why we asked.** Our Burn/Rust impl keeps experts as `experts: Vec<Qwen3MLP<B>>` (128 owned
per-expert `Linear` weights ≈ 58 GB) **and** builds a per-layer pre-stacked contiguous `[E,H,I]`
view with `Tensor::cat` (`src/moe.rs` `stacked_experts()` @410, `forward_fast` @340) — a **second**
~58 GB copy. On a 119 GB unified-memory box (NVIDIA GB10 / DGX-Spark) that 2× = OOM.

> **TL;DR / canonical answer.** Prod engines use pattern **(b) LOAD-CONTIGUOUS**: they
> pre-allocate **one** contiguous `[num_experts, …]` buffer (`w13_weight`, `w2_weight`) and the
> `weight_loader` writes each checkpoint shard **directly into slot `expert_id`**
> (`param.data[expert_id].copy_(loaded_weight)`). The per-expert tensors **never exist** as
> separate parameters, so there is **never a 2× copy** — not even transiently. The single buffer
> also gives a **stable base pointer** required for CUDA-graph capture, and batch-1 decode reads
> only the *k* slabs via **grouped-GEMM with stride/pointer arithmetic** over that one buffer — no
> copy of selected experts. **Recommendation for us: adopt (b); if the loader rewrite is too big,
> use (a) strictly per-layer (build the stack for one layer, free that layer's `Vec` before the
> next) so the transient peak is ~1.2 GB, not 58 GB.** Never `cat` per decode step.

---

## 0. Memory math — why we OOM, and that holding-once is enough

| Precision | bytes/param | Qwen3-30B-A3B weights | Fits in 119 GB? |
|---|---|---|---|
| **bf16** | 2 | **~61 GB total** (experts ~58 GB, non-expert ~3 GB) | **Yes, once** (~50–58 GB headroom for KV+activations) |
| **bf16, duplicated** (our bug) | 2 | experts 58 × 2 = **116 GB** + ~3 GB + fp32 router casts | **No → OOM** (matches observed) |
| fp8 | 1 | ~30.5 GB | trivially |
| int4 (Q4_K_M / NVFP4) | ~0.5 | ~18.6 GB | trivially |

Holding the experts **exactly once** (≈58 GB) + ~3 GB non-expert + KV/activations leaves
~50 GB headroom on a 119 GB box → **no quantization or offload needed to fit + decode**. The OOM is
purely the duplicate. (DGX-Spark is unified LPDDR5X, NVLink-C2C coherent CPU+GPU; weights live in
the one pool, no host→device copy, so a single resident copy is the whole cost.)

---

## 1. vLLM `FusedMoE` — contiguous `[E,…]` param + load-into-slot (PRIMARY SOURCE)

Storage and loader are split across two files in `vllm/model_executor/layers/fused_moe/` (the layer
package was refactored; `layer.py` is now a thin shell).

**Storage — one contiguous buffer per fused projection.**
`unquantized_fused_moe_method.py`, `UnquantizedFusedMoEMethod.create_weights()` (def @88):
```python
w13_up_dim = 2 * intermediate_size_per_partition            # @98  (gate+up fused)
w13_weight = torch.nn.Parameter(torch.empty(
    num_experts, w13_up_dim, hidden_size, dtype=params_dtype))   # @102–110  -> [E, 2I, H]
layer.register_parameter("w13_weight", w13_weight)          # @111
set_weight_attrs(w13_weight, extra_weight_attrs)            # @112  (attaches weight_loader)
w2_weight = torch.nn.Parameter(torch.empty(
    num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype))  # @121–129 -> [E, H, I]
layer.register_parameter("w2_weight", w2_weight)            # @130
```
So `w13_weight = [E, 2I, H]` and `w2_weight = [E, H, I]` are **single pre-allocated empty 3-D
Parameters**. Nothing per-expert is registered.

**Loader — writes each expert's shard into its slot, no second copy.**
`routed_experts.py`, `weight_loader()` (overloads @558/569/579):
```python
global_expert_id = expert_id
expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)  # @601–602 (EP map)
...
expert_data = param.data[expert_id]          # @646  VIEW into the contiguous buffer's slot
if <full per-expert tensor>:
    expert_data.copy_(loaded_weight)         # @659  in-place write, NO new allocation
else:
    self._load_w13(... expert_data=expert_data ...)   # gate/up halves
```
- `_load_w13()` (def @442) narrows the slot to the gate or up half and copies:
  `expert_data.narrow(shard_dim, 0/shard_size, shard_size)` (@479/483) then `.copy_()`.
- `_load_w2()` (def @493) copies the down half into the slot.
- `_load_model_weight_or_group_weight_scale()` (@319) does the actual `expert_data.copy_(loaded_weight)` (@375).
- Fused/full checkpoints: `expert_data = param.data if full_load else param.data[expert_id]` (@686).
- `_load_single_value`: `param_data[expert_id] = loaded_weight` (@536).

**EP sharding / `expert_map`.** `_map_global_expert_id_to_local_expert_id()` (@270) maps a global
expert id to the rank-local slot; `expert_map` (property @220, built by `ExpertMapManager`) tells the
rank which global experts it owns, so the loader **skips** non-local experts and writes the owned
ones into compact local slots. `num_local_experts` = experts on this rank.

➡ **Pattern (b).** The pre-allocated `w13_weight`/`w2_weight` is the *only* copy; the on-disk shard
is freed right after `copy_`. There is no per-expert `nn.Parameter` and no concatenation step.

---

## 2. SGLang `FusedMoE` / `EPMoE` — same pattern (PRIMARY SOURCE)

`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` (SGLang's MoE is adapted from vLLM):

- **Storage**: `self.quant_method.create_weights(...)` (@302) → contiguous `w13_weight` / `w2_weight`
  `[num_local_experts, …]`. Checkpoint→stacked mapping list:
  `("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13")` and the `w2` row (@1202–1236).
- **Loader**: `weight_loader()` (def @613) →
  `expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)` (@641/698) →
  `expert_data = param.data[expert_id]` (@839) → `_load_w13`/`_load_w2` do
  `expert_data.copy_(loaded_weight)` (@431/503/573). Fused path `weight_loader_fused()` (@997) uses
  `expert_data = param.data` (@1050).
- **EPMoE** (`python/sglang/srt/layers/moe/ep_moe/layer.py`): expert-parallel range-shards experts
  across ranks (rank *i* owns `[i·L, (i+1)·L)`); same per-slot loader; tokens are dispatched to the
  owning rank via **DeepEP** all-to-all. After dispatch, **grouped-GEMM** (Triton runner / DeepGEMM)
  runs all local experts in one launch, and **`moe_align_block_size`**
  (`sgl-kernel/csrc/moe/moe_align_kernel.cu`) sorts tokens by expert id + builds the cumsum block
  offsets that tell the grouped-GEMM which slab of the one buffer to read per expert.

➡ **Pattern (b)** for storage, **(c)** for compute — same as vLLM.

---

## 3. llama.cpp & ktransformers (unified-memory / offload targets)

**llama.cpp / GGUF — one stacked 3-D tensor per layer, indexed by `mul_mat_id`.**
Per-expert HF FFN tensors are **merged at convert time** into a single stacked tensor per projection
(`src/llama-arch.cpp` @391–394):
```
blk.%d.ffn_gate_exps    blk.%d.ffn_up_exps    blk.%d.ffn_down_exps   // each [n_embd, n_ff, n_expert]
```
Routing uses `ggml_mul_mat_id(ctx, as, b, ids)` (`ggml/include/ggml.h` @1435): `as` = the stacked
weight, `ids` = selected expert ids; it indexes the active slices **in place** and multiplies only
those — no copy, no duplicate. So GGUF stores experts **once** (stacked) and gathers by id. (This is
essentially storage-(b) + compute-(c) baked into the format.) On unified memory / `--cpu-moe` the
same single tensor is read by the CPU or GPU kernel; no second copy.

**ktransformers — CPU-resident routed experts, GPU never duplicates them.**
Keeps attention + shared experts on GPU and the bulk **routed** experts **on CPU** (computed with
AMX/AVX-512 `llamafile` kernels); activations are sent to the CPU instead of streaming weights to
GPU (avoids PCIe-bound prefetch). "Expert deferral" overlaps CPU expert compute with the next
layer's GPU attention; `--kt-num-gpu-experts` pins the hottest experts to GPU if VRAM allows. Net:
expert weights exist **once**, on CPU, never duplicated on GPU. (Ref: KTransformers SOSP'25,
"Arithmetic-Intensity-Guided Offloading", kvcache-ai/ktransformers.)

---

## 4. The canonical (a)/(b)/(c) decision — with prod evidence

| Option | What it is | Peak expert RAM | Who does it |
|---|---|---|---|
| **(a) MOVE** | build the `[E,…]` stack from per-expert tensors, then free the originals | **2× transiently** (both live until freed) | nobody, globally — too risky on a tight box |
| **(b) LOAD-CONTIGUOUS** | pre-allocate `[E,…]` first; loader writes each shard into slot `expert_id`; per-expert tensors never exist | **1×** (+ one small shard staging buffer) | **vLLM, SGLang** (`param.data[expert_id].copy_`) |
| **(c) keep-per-expert + gather-by-pointer** | no contiguous stack; gather selected experts by pointer at compute time | depends | **compute-side** of every engine (grouped-GEMM stride math); llama.cpp `mul_mat_id`. But storage is still ONE buffer — nobody keeps 128 separate *parameters* for the fast path |

**Verdict:** **(b) is the production storage pattern.** (c) is the *compute* pattern that runs **on
top of** a (b) buffer (batch-1 decode reads only the *k* slabs via `base_ptr + expert_id*stride` —
no copy of selected experts). (a) is what our code accidentally does (Vec + `cat`), and its 2×
transient peak is exactly the OOM; no prod engine relies on it.

### CUDA-graph implication (matters a lot for our Lever A)
A captured graph **bakes in tensor base pointers**. The single contiguous `w13_weight`/`w2_weight`
is allocated once → `data_ptr()` is **stable across replays** → capturable. The dynamic routing
stays inside the graph: the kernel reads `expert_ids` (a fixed-address tensor) on-GPU and computes
`weight_ptr = base_ptr + expert_id*stride`. A **per-step `Tensor::cat`** (our `stacked_experts()`)
allocates a **fresh buffer every decode step** → moving pointer + CPU-GPU sync → **uncapturable**.
This is precisely why `docs/PERF_80TOKS_PLAN.md` Lever A wants a *post-load, persistent*
pre-stacked cache, not a per-call stack.

---

## 5. Recommendation for our Burn codebase

**Adopt (b).** Concretely:

1. **Loader (the real fix).** Replace `experts: Vec<Qwen3MLP<B>>` with persistent stacked module
   fields `gate_stack [E,H,I]`, `up_stack [E,H,I]`, `down_stack [E,I,H]`. Pre-allocate them once,
   then in the safetensors loader map each `mlp.experts.{j}.{gate,up,down}_proj.weight` **directly
   into slot `j`** of the corresponding stacked buffer via `slice_assign`/narrow+copy — the Burn
   analogue of vLLM's `param.data[expert_id].copy_(loaded_weight)`. The per-expert `Linear` weights
   then **never materialize** → 1× memory, fits. (This fights Burn's positional-`Vec` auto-key
   mapping, so it needs a custom param + manual load step — same shape as a vLLM `weight_loader`.
   Ties into MOE_PLAN §7 loader changes and the §3 "128-key→1-tensor pack at load time" note.)

2. **Pragmatic interim if the loader rewrite is deferred — (a) STRICTLY PER LAYER.** Keep the
   existing `Vec` loader, but right after a layer loads, build that layer's `[E,…]` stack and
   **drop that layer's `Vec` experts before moving to the next layer**. Transient peak overhead is
   **one layer's experts (~1.2 GB)**, not the whole model (58 GB). Store only the stacked buffers.
   This avoids the global 2× without touching the key-mapping logic.

3. **Either way: delete the per-call stacking.** `stacked_experts()` (@410) and `forward_fast`'s
   in-body `cat` (@340) must not run on the 30B per decode step — they re-duplicate 58 GB *and*
   break CUDA-graph capture. Expose the persistent stacked buffers (cf. `stacked_experts_pub` @171)
   and feed the grouped-GEMM / gather kernel directly, indexing by `expert_id` (pattern (c) over the
   one buffer).

---

## 6. Sources

**Primary (direct GitHub fetch, line numbers verified against `main`/`master` at research time):**
- vLLM expert storage — `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
  `create_weights` @88, `w13_weight` @102–112, `w2_weight` @121–130.
- vLLM loader — `vllm/model_executor/layers/fused_moe/routed_experts.py`
  `weight_loader` @558–605, `_map_global_expert_id_to_local_expert_id` @270/602,
  `expert_data = param.data[expert_id]` @646, `expert_data.copy_()` @659/375,
  `_load_w13` @442, `_load_w2` @493, `_load_single_value`/`param_data[expert_id]=` @536,
  full-load `param.data if full_load else param.data[expert_id]` @686, `expert_map` @220.
- SGLang — `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  `create_weights` call @302, `weight_loader` @613, `_map_global…` @641/698,
  `expert_data = param.data[expert_id]` @839, `expert_data.copy_()` @431/503/573,
  `weight_loader_fused` @997 / `expert_data = param.data` @1050, ckpt→stacked map @1202–1236;
  EPMoE `python/sglang/srt/layers/moe/ep_moe/layer.py`; `sgl-kernel/csrc/moe/moe_align_kernel.cu`.
- llama.cpp — `src/llama-arch.cpp` @391–394 (`ffn_gate/up/down_exps`); `ggml/include/ggml.h` @1435
  (`ggml_mul_mat_id`).

**Corroborating web search (Gemini 3.1 Pro High via `agy-direct.sh`, grounded):**
- vLLM `create_weights`/`weight_loader`, `param.data[local_expert_id].copy_(loaded_weight)`,
  `w13_weight [E,2I,H]` / `w2_weight [E,H,I]`, `expert_map`/EP sharding.
- SGLang `FusedMoE`/`EPMoE`, `_load_w13`/`_load_w2`, DeepEP all-to-all, grouped-GEMM,
  `moe_align_block_size`.
- llama.cpp stacked `ffn_*_exps` + `ggml_mul_mat_id` (no copy); ktransformers CPU-resident routed
  experts + expert deferral (KTransformers SOSP'25).
- CUDA-graph pointer stability (single contiguous buffer → stable `data_ptr()`; per-step pack breaks
  capture); batch-1 decode = grouped-GEMM `base_ptr + expert_id*stride`, no expert copy.
- Memory: bf16 ~61 GB total / ~58 GB experts; once-resident fits 119 GB with ~50 GB headroom;
  fp8 ~30.5 GB; int4 ~18.6 GB.
