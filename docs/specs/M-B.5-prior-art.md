# M-B.5 Prior Art: modelopt NVFP4/FP8 formats + kernels (research findings)

Research date: 2026-07-02. Sources: vLLM `main` (`vllm/model_executor/layers/quantization/modelopt.py`, 2542 lines; `utils/marlin_utils_fp4.py`, 802 lines), NVIDIA TensorRT-Model-Optimizer `main` (`modelopt/torch/export/quant_utils.py`, 1603 lines; `modelopt/torch/quantization/qtensor/nvfp4_tensor.py`, 407 lines), plus the **local partially-downloaded checkpoint** at `/workspace/qwen3-burn-manin-grpo/models/qwen3.6-35b-a3b-nvfp4/` (shard 3 safetensors header parsed directly — ground truth for shapes/dtypes).

---

## 1. modelopt NVFP4 on-disk layout (GROUND-TRUTHED)

### 1.1 Shapes/dtypes — read directly from the local checkpoint's safetensors header

From `model-00003-of-00003.safetensors` (header JSON, no tensor data loaded):

```
model.language_model.layers.37.mlp.experts.0.gate_proj.weight          U8      [512, 1024]    # logical [512, 2048]
model.language_model.layers.37.mlp.experts.0.gate_proj.weight_scale    F8_E4M3 [512, 128]     # = [out, in/16]
model.language_model.layers.37.mlp.experts.0.gate_proj.weight_scale_2  F32     []             # scalar (per-tensor)
model.language_model.layers.37.mlp.experts.0.gate_proj.input_scale     F32     []             # scalar (per-tensor)

model.language_model.layers.37.mlp.experts.0.down_proj.weight          U8      [2048, 256]    # logical [2048, 512]
model.language_model.layers.37.mlp.experts.0.down_proj.weight_scale    F8_E4M3 [2048, 32]

model.language_model.layers.37.mlp.shared_expert.gate_proj.*           same pattern as expert gate_proj

lm_head.weight        U8      [248320, 1024]
lm_head.weight_scale  F8_E4M3 [248320, 128]
lm_head.weight_scale_2 F32 [] ; lm_head.input_scale F32 []
```

So for a Linear of logical shape `[out_features, in_features]`:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `weight` | uint8 | `[out, in/2]` | **packed along the INPUT (last) axis**, 2 e2m1 nibbles/byte, row-major |
| `weight_scale` | float8_e4m3fn | `[out, in/16]` | per-16-element block scale along input dim, row-major |
| `weight_scale_2` | float32 | scalar | per-tensor global scale = `global_amax / (6.0 * 448.0)` |
| `input_scale` | float32 | scalar | per-tensor activation global scale (see §2 — unused in W4A16) |

### 1.2 Which nibble is element 0? **LOW nibble = even element (element 0)**

The pack, from `NVFP4QTensor.quantize` in TensorRT-Model-Optimizer `modelopt/torch/quantization/qtensor/nvfp4_tensor.py` (line 337):

```python
# Cast weights to fp4
q_weight = cls._cast_fp4(scaled_weight)
# Pack weights
packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
```

And its exact inverse, `NVFP4QTensor.dequantize._unpack_tensor` (lines 355-356):

```python
unpacked[..., 1::2] = input >> 4
unpacked[..., 0::2] = input & 0x0F
```

So byte `b` at packed position `j` holds logical elements `2j` (low nibble, `b & 0x0F`) and `2j+1` (high nibble, `b >> 4`). Packing is along the **last (input/K) axis** of the `[out, in]` weight — confirmed both by the code (`[..., ::2]` on the last dim) and by the header shapes (`in` halved, `out` unchanged).

vLLM agrees — `ModelOptNvFp4W4A16LinearMethod` docstring + `create_weights` ([modelopt.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py) ~line 1246, 1306):

```python
    weight          uint8     packed NVFP4 (2 nibbles/byte along input dim)
    weight_scale    fp8-e4m3  per 16-elem group along input dim
    weight_scale_2  fp32      per-tensor global scale = amax / (6.0 * 448.0)
...
        # Packed NVFP4 weights: uint8, 2 nibbles per byte along the input dim.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1, output_dim=0, ...)
...
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e4m3fn,
            ), ...)
```

**Beware of a red herring**: `pack_int4_in_uint8` in modelopt's `export/quant_utils.py` (line 786) packs along the **output** dim — but it is used ONLY for `QUANTIZATION_INT4_AWQ / W4A8_AWQ` (`to_quantized_weight`, line 905-906). The NVFP4 path (line 908-924) goes through `NVFP4QTensor.quantize`, i.e. the input-axis / low-nibble-first pack above. Do not mix them up.

### 1.3 The e2m1 code itself

`nvfp4_tensor.py` lines 26-27 — the 4-bit code is `(sign<<3) | magnitude_ordinal`, and the 16-entry decode LUT indexed by the raw uint4 is:

```python
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
```

Encoding (`_cast_fp4`, lines 230-250): `sign_bit = (w<0)`, `ord = searchsorted(e2m1_bounds, |w|)`, ties at the odd bounds [0.75, 1.75, 3.5] round UP (`+ equals_odd_bounds`), result `= (sign_bit << 3) + ord`. Note code `0b1000` is negative zero (decodes to 0).

URLs:
- https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
- https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/export/quant_utils.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py

**Confidence: HIGH.** Shapes ground-truthed against the actual checkpoint header; pack/unpack quoted from the producing code (modelopt v0.44 lineage) and cross-confirmed by the consuming code (vLLM).

---

## 2. The dequant math + the `input_scale` verdict for W4A16

### 2.1 Reconstruction formula

`reconstructed = e2m1_value * float32(e4m3_block_scale) * weight_scale_2`. The block scale is **not** global-folded; both factors multiply. Reference dequant, `NVFP4QTensor.dequantize` (nvfp4_tensor.py lines 390-407):

```python
q_per_block_scale = kwarg["scale"].to(torch.float32)          # the e4m3 weight_scale
per_block_quant_scale = kwarg["double_scale"]                  # weight_scale_2
per_block_scale = q_per_block_scale * per_block_quant_scale
deq_data = _unpack_tensor(self._quantized_data)               # e2m1 -> values via LUT
deq_data = deq_data.view((*shape[:-1], -1, block_size)) * per_block_scale.unsqueeze(-1)
```

Scale provenance (nvfp4_tensor.py):
- `weight_scale_2 = reduce_amax(W).float() / (E2M1_MAX * E4M3_MAX)` = `global_amax / (6.0 * 448.0)` = `amax / 2688` (line 207).
- `weight_scale` (stored e4m3) = `per_block_amax / (6.0 * weight_scale_2)`, clamped to e4m3 range (lines 192-195).

### 2.2 Is `input_scale` used in W4A16 serving? **NO — vLLM deletes it unread.**

`ModelOptNvFp4W4A16LinearMethod.process_weights_after_loading` (vLLM modelopt.py lines 1354-1378), verbatim:

```python
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Discard the input_scale placeholder. Whether it carries values
        # (W4A4 ckpt loaded as W4A16) or is uninitialized (native W4A16
        # ckpt), W4A16 mode does not quantize activations, so this is unused.
        if hasattr(layer, "input_scale"):
            del layer.input_scale

        if torch.unique(layer.weight_scale_2).numel() != 1:
            logger.warning_once(
                "In W4A16_NVFP4 linear, the global weight scale "
                "(weight_scale_2) differs across fused parallel layers "
                "(e.g. q/k/v_proj). This will likely reduce accuracy. ...")

        # Rename weight_scale_2 -> weight_global_scale. NO reciprocation:
        # ModelOpt already stores amax/2688, which is exactly what Marlin
        # consumes via nvfp4_marlin_process_global_scale (called inside the
        # Marlin adapter's process_weights_after_loading).
        layer.weight_global_scale = Parameter(
            layer.weight_scale_2.max().to(torch.float32), requires_grad=False
        )
        del layer.weight_scale_2
        self.kernel.process_weights_after_loading(layer)
```

And `create_weights` registers `input_scale` only as a loader placeholder (lines 1341-1352): "Placeholder input_scale param so W4A4-shaped checkpoints can be loaded under this method without KeyError ... Discarded in process_weights_after_loading; **never read by the kernel**."

The class docstring is equally explicit (lines 1253-1256): "**No activation quantization.** Marlin expects the global scale in the same form ModelOpt stores (amax/2688), so we rename weight_scale_2 -> weight_global_scale **without reciprocation**".

### 2.3 Is any activation-side factor folded into `weight_scale_2` at export? **No.**

`weight_scale_2` is derived purely from the weight amax (`get_weights_scaling_factor_2 = reduce_amax(weight) / (6*448)`, nvfp4_tensor.py line 205-207). The activation scale is exported separately via `get_activation_scaling_factor` (line 216-221: `amax / (quantizer.maxbound * E4M3_MAX)`) into `input_scale`. In W4A16 mode the `input_scale` tensors on disk are vestigial calibration artifacts (this checkpoint has them because modelopt exports them regardless); the serving math for W4A16 is exactly:

```
y = (bf16 activation) @ dequant(weight)^T,   dequant = e2m1 * e4m3_block_scale * weight_scale_2
```

(For W4A4/`ModelOptNvFp4LinearMethod`, by contrast, `input_scale` IS used: lines 1215-1225 keep `input_global_scale` and precompute `alpha = input_global_scale * weight_global_scale` for the fused FP4 GEMM's output rescale.)

**Confidence: HIGH** — all statements are verbatim from current vLLM main and modelopt main. One caveat: quotes are from `main` as of 2026-07-02; the exact class was reworked over time (older vLLM used cutlass-only W4A4), so pin behavior to a vLLM version if citing in docs.

---

## 3. FP8 dense layers: weight_scale granularity — **PER-TENSOR scalar**

### 3.1 Ground truth from the checkpoint

```
model.language_model.layers.39.self_attn.q_proj.weight        F8_E4M3  [8192, 2048]
model.language_model.layers.39.self_attn.q_proj.weight_scale  F32      []      # scalar
model.language_model.layers.39.self_attn.q_proj.input_scale   F32      []      # scalar
# (no weight_scale_2 for FP8 layers)
```

### 3.2 modelopt export code (quant_utils.py `to_quantized_weight`, lines 848-860)

```python
    if quantization == QUANTIZATION_FP8:
        ...
        if weight.dim() == 3:
            # for MOE stacked weights
            return (weight / weights_scaling_factor.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return (weight / weights_scaling_factor).to(torch.float8_e4m3fn)
```

For plain `FP8` the scaling factor divides the whole tensor (scalar broadcast). modelopt has a *separate* algo `FP8_PC_PT` (per-channel weight, per-token activation, lines 873-903) — this checkpoint's `hf_quant_config.json` says plain `"quant_algo": "FP8"` for all dense layers, i.e. per-tensor.

### 3.3 vLLM consumer confirms (`ModelOptFp8LinearMethod`, modelopt.py lines 492-527)

```python
            # WEIGHT SCALE
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
...
    def process_weights_after_loading(self, layer):
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = requantize_with_max_scale(
                layer.weight, layer.weight_scale, layer.logical_widths)
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
```

(The `len(output_partition_sizes)` entries exist only for fused q/k/v shards; they collapse to one scalar via `max()` + requantize.) vLLM's per-channel variant is a different class, `ModelOptFp8PcPtLinearMethod` ("weight_scale: fp32, shape [out] (per-output-channel); no input_scale"), not used for `quant_algo: FP8`.

**Adapter direction for our engine** (per-output-channel FP8): broadcast the scalar `weight_scale` to all `out` channels at load: `per_channel_scale[o] = weight_scale` for all `o`. Lossless, no requantization needed. `input_scale` (scalar) is the static activation scale if we ever do FP8 activations; for our current bf16-activation FP8-weight path it can be ignored, matching vLLM W16A16-style handling.

**Confidence: HIGH** (checkpoint header + producer + consumer all agree).

---

## 4. Kernel prior art: w4a16 NVFP4 on non-tensor-core paths, and the fastest e2m1 unpack

### 4.1 Marlin is THE W4A16 NVFP4 path in vLLM, and it supports MoE

- `ModelOptNvFp4W4A16LinearMethod.__init__` pins `self.kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())` (line 1277) — "For W4A16 there is exactly one valid kernel, so we pin it."
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` contains both linear (`prepare_fp4_layer_for_marlin`, `apply_fp4_marlin_linear` with `b_q_type=scalar_types.float4_e2m1f`) and **MoE** processing (`prepare_moe_fp4_layer_for_marlin` handling `w13_`/`w2_` `_weight_scale_2` per-expert tensors, lines ~419-553: "All experts share one global_scale, so compute the max ... `g_scales = nvfp4_marlin_process_global_scale(g_scales, param_dtype)`").
- The nvidia model card's `--moe-backend marlin` recommendation is consistent: on sm_121 the FP4 MoE path runs Marlin W4A16 (dequant-to-bf16-in-registers GEMM), i.e. exactly our engine's strategy. Marlin is an fp16/bf16-CORE (non-FP4-tensor-core) kernel — it uses regular mma on dequantized fragments, so its dequant tricks transfer to SIMT.

### 4.2 The fastest SIMT-friendly e2m1-pair -> 2xf32 recipe (bit-math, no LUT)

Marlin's reference unpack (`rand_marlin_weight_nvfp4_like`, marlin_utils_fp4.py lines 699-709) shows the trick vLLM uses to decode e2m1 **as an fp8-e4m3 bit-pattern**:

```python
    fp4_weight_part_1 = (fp4_weight & 0b10000000) | ((fp4_weight & 0b01110000) >> 2)
    fp4_weight_part_1 = fp4_weight_part_1.view(torch.float8_e4m3fn)
    fp4_weight_part_1 = fp4_weight_part_1.to(weight.dtype) * (2**6)

    fp4_weight2 = fp4_weight << 4          # bring low nibble to the top
    fp4_weight_part_2 = (fp4_weight2 & 0b10000000) | ((fp4_weight2 & 0b01110000) >> 2)
    fp4_weight_part_2 = fp4_weight_part_2.view(torch.float8_e4m3fn)
    fp4_weight_part_2 = fp4_weight_part_2.to(weight.dtype) * (2**6)
```

Recipe: place the 4-bit code in the top nibble of a byte, keep the sign bit (bit 7), shift the 3 magnitude bits (exp2+man1) right by 2 so they land in the e4m3 exponent/mantissa field, reinterpret as e4m3, convert, multiply by `2^6` (= difference of exponent biases: e4m3 bias 7 vs e2m1 bias 1). Two nibbles per byte = two of these, one on `b` and one on `b << 4`. All ops are AND/SHIFT/OR + one cvt + one mul — no LUT, no branches; on CubeCL/SIMT this is a handful of integer ops per pair and vectorizes over u32 (4 bytes = 8 elements).

An equivalent exponent-fold appears in the actual kernel path: `nvfp4_marlin_process_global_scale` (lines 142-154) pre-multiplies the global scale by `2^(exponent_bias - 7)` (`exponent_bias` = 14 for fp16, 126 for bf16) — i.e. Marlin decodes the e2m1 bits directly into the *activation dtype's* exponent field and folds ALL bias correction into the per-tensor scale, so the inner loop has zero multiplies for bias fixup. Our engine can do the same: fold `2^6` (or the full bias delta) into `weight_scale_2` at load.

Alternative recipe (what modelopt itself uses on host, §1.3): 16-entry LUT `e2m1_values[code]`. On GPU SIMT a 16xf32 LUT in shared memory or 8 registers with `select` chains also works but the e4m3-reinterpret trick above is fewer ops and is what production Marlin uses.

Also relevant: `nvfp4_marlin_process_scales` (lines 55-126) — Marlin additionally rescales the e4m3 block scales ("Marlin will convert the scales from FP8-E4M3 format to FP8-S0E5M3 format to speedup the dequantization", compensating via `global_scale / scale_factor`). That is a Marlin-internal layout optimization, not part of the on-disk format.

### 4.3 Tensor-core paths (not directly usable for us, for context)

- TRT-LLM / CUTLASS NVFP4 GEMM (`torch.ops.trtllm.fp4_quantize`, referenced from nvfp4_tensor.py line 301) targets FP4 tensor cores (sm100/sm120-class); it also uses a different ("cutlass swizzled") scale layout — see `cutlass_fp4_scale_to_modelopt_fp4_scale` conversion helper (nvfp4_tensor.py lines 364-377). Irrelevant to a SIMT port except as a warning that *two* scale layouts exist in the wild; the safetensors checkpoint uses the plain row-major modelopt layout (§1).
- vLLM native SM120/121 CUTLASS NVFP4 GEMM landed in PR #40082 (per search results, merged 2026-05); used for dense layers on GB10 while MoE stays on Marlin.

URLs:
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py

**Confidence: HIGH on the unpack recipe and Marlin-MoE existence (code quoted). MEDIUM on the PR #40082 detail and the "dense=CUTLASS / MoE=Marlin on sm121" split (from search summaries of community docs, not read line-by-line). llama.cpp's exact NVFP4 register-unpack sequence: UNRESOLVED (not fetched); its MXFP4 path historically uses a LUT-in-registers approach.**

---

## 5. NVFP4 on DGX Spark / GB10 (sm_121) — who has it running

Findings from web search (2026-07):

- **vLLM works on GB10** with NVFP4: stock upstream vLLM >= v0.19.0 builds working sm_121 NVFP4 kernels; a native SM120/121 CUTLASS NVFP4 GEMM shipped around PR #40082. On sm121, FP4 **MoE must use `--moe-backend marlin`** (Marlin W4A16 fallback), dense can use flashinfer/cutlass. Community end-to-end example: [bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4](https://github.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4) (Docker build compiling FlashInfer for SM121 + vLLM nightly + NVFP4 patches). NVIDIA's own playbook: https://build.nvidia.com/spark/vllm
- **Known broken/slow**: [NVIDIA/dgx-spark-playbooks#79](https://github.com/NVIDIA/dgx-spark-playbooks/issues/79) — Llama-3.3-70B-NVFP4 at ~4 tok/s on a single Spark under vLLM 26.02 container (dense-70B is bandwidth-starved; not MoE-relevant but shows perf expectations are unsettled). [vllm#37883](https://github.com/vllm-project/vllm/issues/37883) — UVA CPU offload completely broken with NVFP4 MoE (Qwen3.5-35B-A3B): Marlin GEMM requires all tensors on GPU ("RuntimeError: b_scales is not on GPU"). Multi-node NVFP4 over Ray: open usage issue [vllm#30163](https://github.com/vllm-project/vllm/issues/30163).
- **llama.cpp**: NVFP4 is GPU-accelerated on GB10; the "121a" CUDA-arch builds accelerate MXFP4/NVFP4 microscaling formats via the hardware FP4 path, and as of 2026-05 upstream llama.cpp surpassed the community fork ([croll83/llama.cpp-dgx](https://github.com/croll83/llama.cpp-dgx)) for NVFP4+GB10 workloads. Discussion: [ggml-org/llama.cpp#16578](https://github.com/ggml-org/llama.cpp/discussions/16578), NVIDIA forum thread ["Llama.cpp - NVFP4 native support on Blackwell"](https://forums.developer.nvidia.com/t/llama-cpp-nvfp4-native-support-on-blackwell/368430), guides: [Sggin1/DGX-SPARK nvfp4-guide](https://github.com/Sggin1/DGX-SPARK/tree/main/nvfp4-guide), [vlaicu.io DGX Spark playbooks](https://vlaicu.io/posts/dgx-vllm/).
- **mlx**: no GB10-relevant NVFP4 findings (mlx is Apple-silicon). UNRESOLVED / not applicable.

**Confidence: MEDIUM** — based on search-result summaries of issues/forums; individual claims (e.g. which vLLM version first worked, whether sm_121 lacks FP4 tensor-core MMA vs merely lacking kernels) were not verified against primary sources line-by-line. The consistent, load-bearing takeaway (multiple independent sources): **the production MoE path for NVFP4 on GB10 is Marlin-style W4A16 dequant-in-kernel — the same architecture as our planned SIMT kernels.**

---

## 6. Qwen3.6-35B-A3B-NVFP4 specifics

### 6.1 Checkpoint facts (local `config.json` + `hf_quant_config.json` + safetensors index)

- `architectures: ["Qwen3_5MoeForConditionalGeneration"]` (multimodal wrapper; language model under `model.language_model.*`). text_config: `hidden_size 2048`, `num_hidden_layers 40`, `num_experts 256`, `num_experts_per_tok 8`, `moe_intermediate_size 512`, `shared_expert_intermediate_size 512`, `vocab_size 248320`.
- `hf_quant_config.json`: `producer modelopt 0.44.0`, `quant_algo: MIXED_PRECISION`, `kv_cache_quant_algo: FP8`. Per layer: `mlp.experts` + `mlp.shared_expert.{gate,up,down}_proj` + `lm_head` are `W4A16_NVFP4, group_size 16`; attention projections are `FP8` — hybrid pattern: layers 3,7,11,...,39 have `self_attn.{q,k,v,o}_proj` (full attention every 4th layer), all others have `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (Gated-DeltaNet). `exclude_modules: ["mtp.layers.0*", "mtp*"]` — the MTP head is UNQUANTIZED.
- **Experts are per-expert split tensors, gate/up NOT fused**: `...experts.M.gate_proj.{weight,weight_scale,weight_scale_2,input_scale}`, same for `up_proj`, `down_proj`, M = 0..255 — 4 tensors x 3 projections x 256 experts x 40 layers. No `w13`/fused ordering on disk (vLLM fuses gate+up into `w13` at load time itself; loaders must handle the split->fused mapping, gate first then up in vLLM convention).
- Router (`mlp.gate`) and `mlp.shared_expert_gate` are absent from `quantized_layers` — they stay bf16. Norms/embeddings unquantized.
- Expert shapes ground-truthed (§1.1): gate/up `[512, 2048]` logical -> `[512,1024]` u8 + `[512,128]` e4m3; down `[2048, 512]` logical -> `[2048,256]` u8 + `[2048,32]` e4m3. Every expert tensor carries its OWN scalar `weight_scale_2` and `input_scale` (per-expert per-projection global scales — vLLM's Marlin MoE path collapses `weight_scale_2` across experts via `max()`, see §4.1; a parity-exact engine should keep them per-expert).

### 6.2 vLLM serving of this exact model — issues & recommended config

- Official/NVIDIA-recommended config (from model-card discussions/search): `--moe-backend marlin` plus MTP speculative decoding `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`; vLLM >= 0.19.0 required for Qwen3.6 NVFP4 artifacts. Recipe page: https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B
- Open problems people hit with this exact checkpoint:
  - [HF discussion #10 on nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/discussions/10) — DGX Spark not running it on NVIDIA's own vLLM image.
  - [NVIDIA forum: "Qwen3.6-35B-A3B-NVFP4 hangs after attention backend selection across 3 vLLM images, including NVIDIA's own official recipe"](https://forums.developer.nvidia.com/t/qwen3-6-35b-a3b-nvfp4-hangs-after-attention-backend-selection-across-3-vllm-images-including-nvidias-own-official-recipe/373274) — hang before weight loading on GB10.
  - `moe_backend='marlin'` errors out for *unquantized* MoE ("not supported for unquantized MoE") — relevant because the MTP module is unquantized, hence the recommended split config (marlin for main model, triton for MTP).
  - [vllm#37883](https://github.com/vllm-project/vllm/issues/37883) (sibling Qwen3.5-35B-A3B NVFP4): CPU offload incompatible with Marlin MoE.
- Benchmarks exist: [NVIDIA forum benchmark report on DGX Spark / Jetson Thor / Blackwell 6000 Pro](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810). Also [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) mirror.

**Gotchas checklist for our port** (derived from §1-§6): (a) low-nibble-first, input-axis packing; (b) dequant = `e2m1 * e4m3_scale * weight_scale_2`, keep per-expert `weight_scale_2` (do NOT max-collapse for parity work); (c) ignore `input_scale` everywhere in W4A16; (d) FP8 dense = per-tensor scalar -> broadcast to our per-channel format; (e) router + shared_expert_gate + norms + MTP are unquantized bf16; (f) gate/up are separate on-disk (no fused-ordering ambiguity); (g) lm_head IS NVFP4 (vocab 248320 x 2048 — the biggest single W4A16 GEMV in the model).

**Confidence: HIGH for 6.1 (all read from the local checkpoint). MEDIUM for 6.2 (search-level; specific issue threads found but bodies not fully read). No vLLM PR specifically about this model family's NVFP4 MoE loader was identified beyond the general modelopt/marlin machinery — UNRESOLVED whether qwen3_next-specific loader patches exist upstream.**
