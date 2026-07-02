#!/usr/bin/env python3
"""Named external reference dequant for NVIDIA ModelOpt NVFP4/FP8 safetensors.

This script intentionally does not call Rust code or model helper paths. It reads safetensors bytes
directly, then applies the formulas documented in docs/specs/M-B.5-prior-art.md:

  NVFP4: fp32 = e2m1(nibble) * e4m3(weight_scale[out,k/16]) * weight_scale_2
  FP8:   fp32 = e4m3(weight[out,k]) * weight_scale

`input_scale` is not read; W4A16 serving keeps higher-precision activations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def e4m3_to_f32(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if exp == 0:
        if mant == 0:
            return math.copysign(0.0, sign)
        return f32(sign * mant * (2.0 ** -9))
    if exp == 0x0F and mant == 0x07:
        return float("nan")
    return f32(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))


def assert_format_specs() -> None:
    assert e4m3_to_f32(0x38) == 1.0
    assert e4m3_to_f32(0x7E) == 448.0
    assert e4m3_to_f32(0x78) == 256.0
    for mant in range(1, 8):
        assert e4m3_to_f32(mant) == f32(mant * (2.0 ** -9))
        assert e4m3_to_f32(0x80 | mant) == f32(-mant * (2.0 ** -9))
    for byte in range(256):
        assert math.isnan(e4m3_to_f32(byte)) == (byte in (0x7F, 0xFF))

    modelopt_e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    assert E2M1 == modelopt_e2m1
    assert math.copysign(1.0, E2M1[8]) < 0.0


class SafeTensorReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}

    def _header(self, shard: str) -> tuple[int, dict]:
        if shard in self._headers:
            return self._headers[shard]
        path = self.model_dir / shard
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        self._headers[shard] = (8 + n, header)
        return self._headers[shard]

    def tensor(self, name: str) -> tuple[str, list[int], bytes]:
        shard = self.weight_map[name]
        data0, header = self._header(shard)
        meta = header[name]
        start, end = meta["data_offsets"]
        with open(self.model_dir / shard, "rb") as f:
            f.seek(data0 + start)
            data = f.read(end - start)
        return meta["dtype"], meta["shape"], data


def f32_scalar(reader: SafeTensorReader, name: str) -> float:
    dtype, shape, data = reader.tensor(name)
    if dtype != "F32" or shape != [] or len(data) != 4:
        raise ValueError(f"{name}: expected F32 scalar, got {dtype} {shape} {len(data)} bytes")
    return struct.unpack("<f", data)[0]


def dequant_nvfp4(
    reader: SafeTensorReader,
    base: str,
    n_start: int = 0,
    n_limit: int | None = None,
) -> tuple[int, int, int, list[float]]:
    wdtype, wshape, w = reader.tensor(base + ".weight")
    sdtype, sshape, s = reader.tensor(base + ".weight_scale")
    g = f32_scalar(reader, base + ".weight_scale_2")
    if wdtype != "U8" or len(wshape) != 2:
        raise ValueError(f"{base}.weight: expected U8 [N,K/2], got {wdtype} {wshape}")
    n = wshape[0]
    k = wshape[1] * 2
    if sdtype != "F8_E4M3" or sshape != [n, k // 16]:
        raise ValueError(f"{base}.weight_scale: expected F8_E4M3 [{n},{k//16}], got {sdtype} {sshape}")
    n_out = min(n - n_start, n_limit) if n_limit is not None else n - n_start
    if n_start < 0 or n_out < 0 or n_start + n_out > n:
        raise ValueError(f"{base}: invalid N window start={n_start} count={n_out} for N={n}")
    out = [0.0] * (k * n_out)
    packed_per_col = k // 2
    blocks_per_col = k // 16
    for nn in range(n_out):
        src_n = n_start + nn
        for block in range(blocks_per_col):
            scale = f32(e4m3_to_f32(s[src_n * blocks_per_col + block]) * g)
            for pair in range(8):
                byte = w[src_n * packed_per_col + block * 8 + pair]
                kk = block * 16 + pair * 2
                out[kk * n_out + nn] = f32(E2M1[byte & 0x0F] * scale)
                out[(kk + 1) * n_out + nn] = f32(E2M1[(byte >> 4) & 0x0F] * scale)
    return k, n_out, n_start, out


def dequant_fp8(reader: SafeTensorReader, base: str) -> tuple[int, int, int, list[float]]:
    wdtype, wshape, w = reader.tensor(base + ".weight")
    scale = f32_scalar(reader, base + ".weight_scale")
    if wdtype != "F8_E4M3" or len(wshape) != 2:
        raise ValueError(f"{base}.weight: expected F8_E4M3 [N,K], got {wdtype} {wshape}")
    n, k = wshape
    out = [0.0] * (k * n)
    for nn in range(n):
        for kk in range(k):
            out[kk * n + nn] = f32(e4m3_to_f32(w[nn * k + kk]) * scale)
    return k, n, 0, out


def write_bin(path: Path, values: list[float]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<" + "f" * len(values), *values))


def default_samples() -> list[tuple[str, str, int, int | None]]:
    samples: list[tuple[str, str, int, int | None]] = []
    for layer in (0, 3):
        for expert in (0, 1):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                samples.append(("nvfp4", f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}", 0, None))
    for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
        samples.append(("fp8", f"model.language_model.layers.0.linear_attn.{proj}", 0, None))
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        samples.append(("fp8", f"model.language_model.layers.3.self_attn.{proj}", 0, None))
    samples.append(("nvfp4", "lm_head", 0, 128))
    samples.append(("nvfp4", "lm_head", 151632, 32))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        samples.append(("nvfp4", f"model.language_model.layers.0.mlp.shared_expert.{proj}", 0, None))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/qwen3.6-35b-a3b-nvfp4")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_format_specs()
    reader = SafeTensorReader(model_dir)

    manifest = []
    for idx, (kind, base, n_start, n_limit) in enumerate(default_samples()):
        if kind == "fp8":
            k, n, n0, values = dequant_fp8(reader, base)
        else:
            k, n, n0, values = dequant_nvfp4(reader, base, n_start, n_limit)
        file_name = f"{idx:03d}.bin"
        write_bin(out_dir / file_name, values)
        manifest.append(f"{kind}\t{k}\t{n}\t{file_name}\t{base}\t{n0}\n")

    with open(out_dir / "manifest.tsv", "w", encoding="utf-8") as f:
        f.writelines(manifest)


if __name__ == "__main__":
    main()
