#!/usr/bin/env python3
"""Mechanical Burn 0.21 Tensor<B, D> -> 0.22 Tensor<D> migration (src/examples/tests only)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "src", ROOT / "examples", ROOT / "tests"]

MODULE_TYPES = [
    "Linear",
    "Embedding",
    "RmsNorm",
    "Dropout",
    "Qwen3Attention",
    "Qwen3AttentionConfig",
    "Qwen3Model",
    "Qwen3ForCausalLM",
    "Qwen3DecoderLayer",
    "Qwen3MLP",
    "Qwen3MoeConfig",
    "Qwen3MoeModel",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeSparseBlock",
    "MoeExpertCache",
    "MoeStaticDecode",
    "KVCache",
    "ModelCache",
    "GdnStateCache",
    "GdnModelCache",
    "Qwen3_5HybridCache",
    "Qwen3_5HybridLayerCache",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5Model",
    "Qwen3_5DecoderLayer",
    "Qwen3_5FullAttention",
    "Qwen3_5FullAttnLayer",
    "Qwen3_5GdnAttention",
    "Qwen3_5GdnLayer",
    "Qwen3_5SharedMoeBlock",
    "Qwen3_5LayerType",
    "W8A16Linear",
    "Nvfp4Linear",
    "QuantLinear",
    "ExpertNvfp4",
    "ExpertNvfp4Sidecar",
    "ExpertQuantSidecar",
    "QuantSidecar",
    "Rollouts",
    "Param",
    "FusedSwigluBackend",
]

# Longer names first so we don't leave leftovers.
MODULE_TYPES.sort(key=len, reverse=True)

TENSOR_BACKENDS = r"(?:B(?:::InnerBackend)?|Backend|IB|Inner|NdArray(?:<[^>]+>)?|Cuda(?:<[^>]+>)?|Autodiff<[^>]+>|CaptureBackend)"


def rewrite_tensors(src: str) -> str:
    # Tensor::<B, D, K> and Tensor<B, D, K>
    src = re.sub(
        rf"Tensor::<{TENSOR_BACKENDS},\s*(\d+)\s*,\s*(Int|Bool)>",
        r"Tensor::<\1, \2>",
        src,
    )
    src = re.sub(
        rf"Tensor<{TENSOR_BACKENDS},\s*(\d+)\s*,\s*(Int|Bool)>",
        r"Tensor<\1, \2>",
        src,
    )
    src = re.sub(
        rf"Tensor::<{TENSOR_BACKENDS},\s*(\d+)>",
        r"Tensor::<\1>",
        src,
    )
    src = re.sub(
        rf"Tensor<{TENSOR_BACKENDS},\s*(\d+)>",
        r"Tensor<\1>",
        src,
    )
    # Tensor<B, D> with const generic D
    src = re.sub(
        rf"Tensor::<{TENSOR_BACKENDS},\s*(D)>",
        r"Tensor::<\1>",
        src,
    )
    src = re.sub(
        rf"Tensor<{TENSOR_BACKENDS},\s*(D)>",
        r"Tensor<\1>",
        src,
    )
    return src


def rewrite_modules(src: str) -> str:
    for name in MODULE_TYPES:
        src = re.sub(rf"\b{name}<{TENSOR_BACKENDS}>", name, src)
        src = re.sub(rf"\b{name}<{TENSOR_BACKENDS},\s*", name + "<", src)
    return src


def strip_backend_generics(src: str) -> str:
    # struct / enum Foo<B: Backend>
    src = re.sub(
        r"\b(struct|enum|union)\s+(\w+)\s*<B:\s*(?:Backend|AutodiffBackend)\s*>",
        r"\1 \2",
        src,
    )
    src = re.sub(
        r"\b(struct|enum|union)\s+(\w+)\s*<B:\s*(?:Backend|AutodiffBackend),\s*",
        r"\1 \2<",
        src,
    )
    # impl<B: Backend> Foo<B>
    src = re.sub(
        r"\bimpl\s*<B:\s*(?:Backend|AutodiffBackend)\s*>\s+(\w+)\s*<B>",
        r"impl \1",
        src,
    )
    src = re.sub(
        r"\bimpl\s*<B:\s*(?:Backend|AutodiffBackend)\s*>\s+",
        r"impl ",
        src,
    )
    src = re.sub(
        r"\bimpl\s*<B:\s*(?:Backend|AutodiffBackend),\s*",
        r"impl<",
        src,
    )
    # Foo<B> leftover after impl
    src = re.sub(r"\bimpl\s+(\w+)\s*<B>", r"impl \1", src)

    # fn foo<B: Backend>(
    src = re.sub(
        r"\bfn\s+(\w+)\s*<B:\s*(?:Backend|AutodiffBackend)\s*>\s*\(",
        r"fn \1(",
        src,
    )
    src = re.sub(
        r"\bfn\s+(\w+)\s*<B:\s*(?:Backend|AutodiffBackend),\s*",
        r"fn \1<",
        src,
    )
    # fn foo<B, O, R>(  — drop leading B, if where clause has B: AutodiffBackend
    src = re.sub(r"\bfn\s+(\w+)\s*<B,\s*", r"fn \1<", src)

    # trailing Foo<B> in signatures
    src = re.sub(r"\b([A-Z][A-Za-z0-9_]*)<B>", r"\1", src)
    src = re.sub(r"\b([A-Z][A-Za-z0-9_]*)<B::InnerBackend>", r"\1", src)

    src = src.replace("B::Device", "Device")
    src = src.replace("<B as Backend>::Device", "Device")
    src = src.replace("<B as burn::prelude::Backend>::Device", "Device")

    # where B: ...
    src = re.sub(r"\n\s*B:\s*AutodiffBackend,?\n", "\n", src)
    src = re.sub(r"\n\s*B:\s*Backend,?\n", "\n", src)
    src = re.sub(r"where\s*\n\s*O:", "where\n    O:", src)
    src = re.sub(r"O:\s*Optimizer<([^>]+),\s*B>", r"O: Optimizer<\1>", src)
    src = src.replace("B::InnerBackend", "/* InnerBackend removed */")
    return src


def rewrite_ignored(src: str) -> str:
    src = re.sub(r"Ignored<([^>]+)>", r"\1", src)
    src = re.sub(r"Ignored\(", "(", src)
    # fix extra parens from Ignored(x) -> (x)  — leave as (x) which is fine
    # *self.field for former Ignored Copy fields: only head_dim / rope_theta / similar
    src = re.sub(r"\*self\.(head_dim|rope_theta|num_heads|num_kv_heads|num_experts|top_k|norm_topk_prob|hidden_size|moe_intermediate_size|epsilon)\b", r"self.\1", src)
    return src


def rewrite_devices(src: str) -> str:
    src = src.replace("CudaDevice::default()", "Device::cuda(0)")
    src = src.replace("NdArrayDevice::default()", "Device::ndarray()")
    src = re.sub(
        r"use burn::backend::cuda::\{Cuda, CudaDevice\};",
        "use burn::prelude::Device;",
        src,
    )
    src = re.sub(
        r"use burn::backend::cuda::\{CudaDevice, Cuda\};",
        "use burn::prelude::Device;",
        src,
    )
    src = src.replace("backend::cuda::{Cuda, CudaDevice}", "prelude::Device")
    src = src.replace("use burn::backend::NdArray;", "use burn::prelude::Device;")
    src = src.replace("use burn::backend::{Autodiff, NdArray};", "use burn::prelude::Device;")
    src = src.replace("use burn::backend::Autodiff;", "")
    return src


def cleanup_imports(src: str) -> str:
    # Ensure Device is imported if used
    if "Device" in src and "use burn::tensor::{" in src and "Device" not in re.search(r"use burn::tensor::\{[^}]+\}", src).group(0) if re.search(r"use burn::tensor::\{[^}]+\}", src) else "":
        src = re.sub(
            r"use burn::tensor::\{",
            "use burn::tensor::{Device, ",
            src,
            count=1,
        )
    if re.search(r"\bDevice\b", src) and "use burn::prelude::Device" not in src and "tensor::{Device" not in src and "prelude::*" not in src:
        # add after first use burn
        if "use burn::{" in src and "Device" not in src.split("use burn::{", 1)[1].split("}", 1)[0]:
            src = src.replace("use burn::{", "use burn::{Device, ", 1)
    src = src.replace("prelude::Backend", "prelude::Device")
    src = src.replace("use burn::prelude::Backend;", "use burn::tensor::Device;")
    return src


def process(text: str) -> str:
    text = rewrite_tensors(text)
    text = rewrite_modules(text)
    text = strip_backend_generics(text)
    text = rewrite_ignored(text)
    text = rewrite_devices(text)
    text = cleanup_imports(text)
    return text


def main() -> None:
    n = 0
    for d in DIRS:
        for path in d.rglob("*.rs"):
            old = path.read_text()
            new = process(old)
            if new != old:
                path.write_text(new)
                n += 1
                print(f"updated {path.relative_to(ROOT)}")
    print(f"rewrote {n} files")


if __name__ == "__main__":
    main()
