"""Weight-only execution (W8A16 / W4A16) from an existing quantized safetensors, natively in ComfyUI.

    python_embeded\\python.exe -s tools\\quant_weight_only.py --input <quantized.safetensors> [--dry-run]

ComfyUI's per-layer quant config accepts `"full_precision_matrix_mult": true`: the layer keeps its
quantized weight in memory, and `comfy/ops.py` dequantizes it (comfy-kitchen's CUDA kernel for that
layout) right before a plain compute-dtype matmul. Activations are never quantized. So:

    int8 ConvRot (int8_tensorwise+convrot)  ->  W8A16, same bytes, BF16 activations
    W4A8 (asym_w4a8_int8)                   ->  W4A16, same bytes, BF16 activations

The tensors are copied byte for byte; only `_quantization_metadata` changes. The output goes beside
the source as `<stem>_a16.safetensors` (+ `.quant.json`), written to `.partial` and renamed.
Checkpoints that carry inline `.comfy_quant` tensors instead of the metadata are refused.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402

NOMES = {"int8_tensorwise": "W8A16", "asym_w4a8_int8": "W4A16", "convrot_w4a4": "W4A16"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source = args.input.resolve()
    output = source.with_name(f"{source.stem}_a16.safetensors")
    conv = C.Conversion(source, output, output.with_suffix(".quant.json"))
    conv.refuse_unsafe(allow_quantized_source=True)
    header, metadata = conv.header, conv.metadata
    if any(name.endswith(".comfy_quant") for name in header):
        raise SystemExit("Refusing: inline '.comfy_quant' layer configs are not handled, only _quantization_metadata")
    if "_quantization_metadata" not in metadata:
        raise SystemExit("Refusing: source has no _quantization_metadata, so it is not a quantized checkpoint")

    quant = json.loads(metadata["_quantization_metadata"])
    layers = quant["layers"]
    formatos = Counter(conf["format"] for conf in layers.values())
    desconhecidos = set(formatos) - set(NOMES)
    if desconhecidos:
        raise SystemExit(f"Refusing: no weight-only mapping for formats {sorted(desconhecidos)}")
    if any(conf.get("full_precision_matrix_mult") for conf in layers.values()):
        raise SystemExit("Refusing: source already has full_precision_matrix_mult layers")
    execucao = "+".join(sorted({NOMES[f] for f in formatos}))
    print(f"Source: {source}\nLayers: {dict(formatos)} -> {execucao} (BF16 activations)\nOutput: {output}")
    if args.dry_run:
        return 0

    started = time.perf_counter()
    for conf in layers.values():
        conf["full_precision_matrix_mult"] = True
    novo = dict(metadata)
    novo["_quantization_metadata"] = json.dumps(quant, separators=(",", ":"))
    novo["quantization"] = f"{metadata.get('quantization', '')} weight-only ({execucao})".strip()
    entradas = [C.plan_copy(name, info) for name, info in header.items()]
    conv.guard(conv.planned_size(entradas, novo))
    conv.commit(entradas, novo, sidecar=lambda: {
        "source": str(source), "source_size": source.stat().st_size,
        "output": str(output), "output_size": conv.output_size,
        "execution": execucao, "layers": dict(formatos),
        "change": "full_precision_matrix_mult=true on every quantized layer; tensors copied byte for byte",
        "conversion_seconds": round(time.perf_counter() - started, 3),
    })
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
