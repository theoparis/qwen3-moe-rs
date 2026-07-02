# L2C NVFP4 decode-GEMV — cubecl codegen blocker (dequant-in-load cast) — RESOLVED

**RESOLVED 2026-07-01:** the GEMV now passes the numerics-identity gate (GPU == codec-dequant matmul,
cosine 1.0, max_abs ~1e-6 at K=256/512, M=1/4) via the **manual u8 nibble-unpack** workaround (option 1
below): the kernel takes `qw: &Tensor<u8>`, reads packed bytes, extracts low/high nibbles, and decodes E2M1
in-kernel with the branch-only `gpu::e2m1_decode` (no `Line::<f32>::cast_from`, no runtime-indexed table).
The history below is kept for the record.

**Status (historical):** codec DONE + CPU-verified (`src/nvfp4.rs` quantize/dequant, `examples/nvfp4_codec_probe.rs`,
cosine 0.9994). GEMV kernel WRITTEN per the vetted design (`docs/specs/L2C-nvfp4-decode-gemv-design.md` §2,
subagent-coded by Codex) + type-fixed + compiled Rust-side, but **failed CUDA codegen at launch** until the fix.

## The blocker (reproduce: `cargo run --release --features cuda --example nvfp4_gemv_probe`)
```
[Compilation Error]
  default_program(150): error: identifier "__half2_8" is undefined
  default_program(160): error: identifier "__half_16" is undefined
```
The kernel does `let vals = Line::<f32>::cast_from(line);` where `line: Line<e2m1x2>` width 8 (→ 16 f32).
On the active cubecl rev (`/workspace/cubecl` patch), that e2m1x2→f32 widening lowers through a **16-wide half
vector intermediate** (`__half_16` / `__half2_8`) whose C++ type definition is **never emitted** → nvrtc fails.
It is a cubecl codegen gap for this cast+width combo (analogous to the P0.3c e4m3 scale-lane fix), NOT a logic
bug: the codec, the GEMV math, the plane_sum reduction, the from_raw_parts e2m1x2/e4m3 upload, and the grid are
all correct per the design + the working `nvfp4_gemm_probe.rs` upload pattern.

## Fix options (next task — bounded)
1. **Manual nibble unpack (preferred, avoids the cast entirely):** pass the packed weight as `&Tensor<u8>`
   (raw bytes, not `Line<e2m1x2>`); in-kernel read byte `b`, `lo = b & 0x0F`, `hi = (b >> 4) & 0x0F`, decode
   each 4-bit code to f32 via the E2M1 value set `{0,±.5,±1,±1.5,±2,±3,±4,±6}` (the host `e2m1_bits_to_f32`
   logic, as a comptime lookup / branch). Loses the vectorized load but sidesteps the broken cast; correctness
   first, then re-vectorize.
2. **Two narrower casts:** cast the width-8 e2m1x2 line as two width-4 sub-lines → two 8-wide f32 (`__half_8`),
   if that width IS emitted. Cheap to test (does `Line::<f32>::cast_from` at width-4 compile?).
3. **cubecl codegen fix:** emit the missing `__half_16`/`__half2_8` vector typedefs for the minifloat-widening
   path (upstream-style, like P0.3c). Larger; do only if (1)/(2) can't hit the bandwidth target.

## Verification gate (unchanged, ready)
`examples/nvfp4_gemv_probe.rs` already encodes the numerics-identity gate: GPU kernel output == host
`dequant_nvfp4(w)` then f32 matmul (same codec ⇒ near-exact), cosine > 0.999. It currently reproduces the
codegen blocker; once the cast is worked around it becomes the pass/fail gate. Then wire the D6
calibration/token-identity gate (design §3) + the FP8 fallback (§4).
