# QUANT_FLASH_SPEC_PLAN execution — consolidated state map (2026-07-01)

Ground truth from 4 read-only subagent investigations + the L2C design doc. Drives the
remaining implementation. File:line refs are the handoff currency for coding subagents.

## What is DONE (in git, verified)
- **Lane 1 (L1.1–L1.6):** Qwen3.6-35B `qwen3_5_moe` hybrid port — greedy parity PASSED (`src/qwen3_5/mod.rs`, `src/load.rs`; example `examples/qwen35_generate.rs`).
- **P0.1–P0.5:** probes done (`docs/*P0*.md`).
- **Lane 2A kernels:** `flash_decode_raw` split-K online-softmax (`src/flash_decode.rs:161`), `flash_attention_raw` (`src/flash_attn.rs:240`). CPU-oracle verified. bf16-KV OK. **Not wired into attention.**
- **Lane 2C kernels:** NVFP4 host codec `quantize_nvfp4`/`dequant_nvfp4` (`src/nvfp4.rs:148/209`, **naive amax, no calibration**) + SIMT decode-GEMV `nvfp4_decode_gemv` (`src/nvfp4.rs:77`, manual nibble unpack) + `Nvfp4GemvBackend` trait w/ Cuda(Fusion) + raw impls (`src/nvfp4.rs:428`) + capture-safe `nvfp4_gemv_raw` (`src/nvfp4.rs:389`). GPU numerics verified. **Not wired into the model.**
- **Capture harness:** `src/capture.rs` `CapturedDecoder`/`DecodeState` — bench-validated (`examples/cudagraph_moe_decode_bench.rs`), **greedy only**, P0.2-harden (RNG seed, streaming) NOT done.
- **FP8 fallback:** `src/w8a16.rs` + `src/w8a16_linear.rs` (validated eager; NOT capture-safe — Fusion bridge).
- **D6 pre-gate:** naive NVFP4 ≈ 0.995 cosine per-tensor (`docs/L2C-nvfp4-accuracy-real-weights.md`) → borderline, needs calibration + FP8 fallback.

## Execution order (dependency-driven)
```
M-B (D6 NVFP4 gate)  -- accuracy-MACHINERY lever (NOT the central perf lever; that's the MoE-expert
      |                 NVFP4 in M-B.5). Runs EAGER on the RAW CubeBackend (per Opus F4, not Fusion).
      |                 Independent of 2A/2B. 3-voice plan: docs/specs/M-B-nvfp4-gate-plan.md.
M-D (Lane 2B capture) -- provides raw CaptureBackend below Fusion + device sampler
      |
M-C (Lane 2A flash-decode wire-in) -- rides on M-D raw backend (A4 also calls CaptureBackend-typed kernel)
      |
M-E (MTP.1 n-gram probe) -- GDN snapshot/restore + KV filled-rewind, KV+GDN rewind to SAME pos (Opus F1)
      |
M-F (Phase-2 converge on 35B + MTP.2 full block) -- last; device-pos KV write must OVERWRITE not Add (F2)
```

## M-B — D6 NVFP4 token-identity gate (START HERE)
Runs eager on `Cuda=Fusion` via `Nvfp4GemvBackend::nvfp4_gemv` Cuda impl (`src/nvfp4.rs:444`). No capture needed. Steps:
- **B-1 CREATE `src/nvfp4_linear.rs`** mirroring `src/w8a16_linear.rs` (struct@51, from_weight@73, from_linear@100, forward@142, forward3@162):
  `Nvfp4Linear<B>{ qw:Tensor<B,2,Int>(I8 [N,K/2]), bs:Tensor<B,2,Int>(I8 [N,K/16]), gscale:Tensor<B,1>(F32 [1]), bias, k, n, m_max }`;
  `from_weight` = `weight.cast(F32).into_data().to_vec::<f32>()` → `nvfp4::quantize_nvfp4(&w,k,n)` → upload I8/F32 tensors (NOTE: NVFP4 stores column-major `[N,K/2]`, transpose is inside the codec — do NOT `.transpose()` again);
  `from_linear`; Cuda `forward`/`forward3` calling `Nvfp4GemvBackend::nvfp4_gemv`; `enum QuantLinear<B>{Nvfp4,Fp8,Bf16}` common `forward3`. Tests vs bf16 Linear (cosine + rel-max + argmax-margin) mirror `w8a16_linear.rs:220`.
- **B-2 calibration** in `quantize_nvfp4` (`src/nvfp4.rs:160-176`): replace global `amax` + per-block `bamax` with a calibrated statistic (percentile-clip first — cheapest; AWQ/SmoothQuant later); **re-derive gscale from post-calibration amax** (design §1 "kills A3"). Structure unchanged.
- **B-3 load hook** `src/load.rs` `set_linear@718` → add `set_quant_linear` (manifest-keyed); dispatch by tensor name in `load_full_attention@672`, `load_gdn_attention@649`, `load_mlp@689`, lm_head `load_qwen35_tensor@435`. Route the ~11 `linear3` call sites (`src/qwen3_5/mod.rs` q/k/v/o@917-919,988; GDN@768-780,880; router gate@557; shared_expert@629-633; lm_head@505) through `QuantLinear`.
- **B-4 gate harness** `examples/qwen35_nvfp4_gate.rs` (mirror `qwen35_generate.rs`): run greedy twice — bf16 baseline vs per-tensor-NVFP4 — token-identity → emit per-tensor `{nvfp4|fp8|bf16}` manifest. Default tiers (design §5): **router-gate→bf16** (tiny, most brittle; a top-8-of-256 flip reroutes experts), **lm_head→FP8** (safer argmax on the 248K head), promote to NVFP4 per-tensor only if it passes. Known-good 30B greedy string `vllm_infer.rs:19-21`.
- **SCOPE:** MoE routed experts are fused rank-3 `Param<Tensor<B,3>>` (`src/qwen3_5/mod.rs:465`, applied via `matmul_out_in@638`) — NVFP4-ing them is a SEPARATE `moe_grouped.rs` extension, NOT a gate blocker. Dense linears first.
- Decode/inference-only; NEVER in GRPO grad recompute (parity break, `w8a16.rs:27`).

## M-D — Lane 2B capture (vllm_infer)
`examples/vllm_infer.rs` runs `Cuda=Fusion` (wrong for capture). Poison: per-step D2H sample (`:132-151,183`), host EOS branch (`:184`), per-step `from_data`/`full`/host `pos+=1` (`:192-196`).
- **B1** switch to raw `CaptureBackend` (mirror `cudagraph_moe_decode_bench.rs:49`), add `--capture` mode.
- **B2** device sampler: greedy `device_select_tokens(logits,0.0)` ok; temp>0 needs capture-safe seeded RNG (port `seeded_gumbel_select` `cudagraph_pfinal_bench.rs:285` into `src/sampling_device.rs`) + top-k/top-p is a real new device kernel (or scope to greedy+unfiltered first).
- **B3** reuse `CapturedDecoder::build`+`decode_n` with vllm_infer prefill/step closures (mirror bench `:154-231`).
- **P0.2-harden** in `src/capture.rs`: fix `decode_n` cumulative-pos overrun@302, device RNG seed field, VA `pad`, logits-history + new-tokens-only + EOS early-stop, warmup≥8.
- Do NOT re-multiply capture speedup (already in the 21 tok/s baseline).

## M-C — Lane 2A flash-decode wire-in
- **A4 (drop-in, no kernel change):** dynamic path `attention.rs forward_with_cache` seq_len==1 (`:334-341`): drop GQA repeat@265-278, call `flash_decode_raw(q.movedim(1,2), k.movedim(1,2), v.movedim(1,2), scale, n_splits)`. Maskless-correct (cache.update returns `0..filled` visible prefix). Needs CaptureBackend specialization or Fusion bridge (couples to M-D). 35B full-attn is a parallel edit `qwen3_5/mod.rs:947-980` (head_dim 256, keep `sigmoid(output_gate)@983`).
- **A5 (kernel change, for capturable static path):** add device `pos`/`lo` bound to `gpu::flash_decode_split@36` (loop bound `n_keys=pos[b]+1`, `start=max(g*split_len, lo[b])`), thread through `flash_decode_raw@161`; wire `forward_with_cache_static_pre_lp@465` (device pos/lo already exist@468,473; drop software pos_mask@509-516). Persistent scratch for capture (currently per-call@193-195).
- head_dim 128 (30B) + 256 (35B) both `d%32==0`. Test: NdArray f32 oracle + GRPO ragged parity (CRITICAL).

## M-E — MTP.1 (n-gram probe machinery)
**Correction from map:** the HYBRID (MTP target) uses `KVCache::update` (slice_assign@`cache.rs:83` + `filled` prefix), NOT `select_assign(Add)`. So KV rollback = **rewind `filled`** (add setter; `filled` private@`cache.rs:31`), NO zeroing needed (reads are `0..filled` prefix, writes are assignment). The `Add`-accumulate hazard only affects the OLD dense model (`update_static@cache.rs:142`).
- **GDN state snapshot/restore is the real mandatory work** (order-dependent, unmaskable): `GdnStateCache@cache.rs:207` — `state:[B,32,128,128]` f32 (@214) + `conv:[B,3,8192]` bf16 (@216). Add `snapshot`/`restore` (clone/write-back the two `Option<Tensor>`), fan out via `Qwen3_5HybridCache` (mirror @426-451).
- `forward_prec` already supports M>1 verify batches (`mod.rs:1096`; full-attn causal mask@1051, GDN loops token-by-token@885). No `Qwen3_5MtpBlock::forward` exists yet (struct@477, weights loaded@load.rs:471).
- CREATE `src/spec_decode.rs` + `examples/qwen35_mtp_generate.rs`: n-gram draft → M-token forward → accept
  longest prefix → rollback. **CORRECTED (Opus F1):** KV and GDN must rewind to the SAME position (GDN state
  is mutated per-token in the verify batch — `set_state@mod.rs:869`, `push_conv@799` — so snapshot@pos +
  KV@pos+acc silently diverges). Use (A) rewind BOTH to `pos` + re-forward accepted(+bonus) tokens (≤K
  steps), OR (B) rewind BOTH to `pos+acc` with GDN checkpointed PER verify step. Gate: token-identical-to-
  greedy + bit-exact RECURRENT-STATE equality (not just KV), test with acc>0.

## Cross-cutting hazards (mandated guards)
- linear3 F32 path silently mis-handles f32×bf16 matmul → cast weight to f32 (`linear2d.rs:68`).
- NVFP4 codec scale floor (zero-block silent-zeroing) already in codec; keep golden-vector tests (all-zero + outlier).
- capture: no per-step from_data/arange/full inside region; VA-stable persistent buffers.
- GPU runs SERIALIZED (30B ~60GB, box 119GB).
