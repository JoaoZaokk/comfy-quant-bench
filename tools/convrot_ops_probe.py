"""Probe whether comfy's MixedPrecisionOps dispatches ConvRot W4A4 to the native CUDA kernel.

Builds a real MixedPrecisionOps.Linear, loads a real quantized layer out of a converted
checkpoint, and runs it under both the text-encoder settings and the diffusion-model
settings so the two paths can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="a converted convrot_w4a4 safetensors file")
    parser.add_argument("--layer", help="layer name; defaults to the first quantized layer")
    return parser.parse_args()


def read_header(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    return raw, metadata


def load_tensor(path: Path, info: dict) -> torch.Tensor:
    dtypes = {"I8": torch.int8, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    start, end = info["data_offsets"]
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_size + start)
        raw = bytearray(end - start)
        view = memoryview(raw)
        position = 0
        while position < len(raw):
            read = handle.readinto(view[position:])
            if not read:
                raise EOFError(f"short read on {path}")
            position += read
    return torch.frombuffer(raw, dtype=dtypes[info["dtype"]]).reshape(info["shape"]).cuda()


def instrument():
    import comfy_kitchen.tensor.convrot_w4a4 as convrot
    from comfy_kitchen.registry import registry

    counters = {"linear": 0, "dequant": 0, "impls": set()}
    original_linear = convrot.convrot_w4a4_linear
    original_dequant = convrot.TensorCoreConvRotW4A4Layout.dequantize.__func__

    def counting_linear(x, qweight, wscales, bias=None, **kwargs):
        counters["linear"] += 1
        impl = registry.get_implementation(
            "convrot_w4a4_linear",
            kwargs={"x": x, "qweight": qweight, "wscales": wscales, "bias": bias, **kwargs},
        )
        counters["impls"].add(f"{impl.__module__}.{impl.__name__}")
        return original_linear(x, qweight, wscales, bias=bias, **kwargs)

    def counting_dequant(cls, qdata, params):
        counters["dequant"] += 1
        return original_dequant(cls, qdata, params)

    convrot.convrot_w4a4_linear = counting_linear
    convrot.TensorCoreConvRotW4A4Layout.dequantize = classmethod(counting_dequant)
    return counters


def run_case(name, ops_kwargs, layer, weight_info, scale_info, model, counters, force_cast, input_dtype):
    import comfy.ops

    rows, packed_columns = weight_info["shape"]
    operations = comfy.ops.mixed_precision_ops(**ops_kwargs)
    module = operations.Linear(packed_columns * 2, rows, bias=False, device="cuda")

    layer_conf = json.dumps({"format": "convrot_w4a4", "convrot_groupsize": 256}).encode("utf-8")
    state_dict = {
        "weight": load_tensor(model, weight_info),
        "weight_scale": load_tensor(model, scale_info),
        "comfy_quant": torch.tensor(list(layer_conf), dtype=torch.uint8),
    }
    missing, unexpected, errors = [], [], []
    module._load_from_state_dict(state_dict, "", {}, False, missing, unexpected, errors)
    module.comfy_force_cast_weights = force_cast

    counters["linear"] = 0
    counters["dequant"] = 0
    counters["impls"] = set()
    x = torch.randn((4, packed_columns * 2), device="cuda", dtype=input_dtype)
    with torch.no_grad():
        output = module(x)
    torch.cuda.synchronize()

    return {
        "case": name,
        "layer": layer,
        "full_precision_mm": ops_kwargs.get("full_precision_mm", False),
        "comfy_force_cast_weights": force_cast,
        "input_dtype": str(input_dtype),
        "compute_dtype": str(ops_kwargs["compute_dtype"]),
        "output_dtype": str(output.dtype),
        "native_linear_calls": counters["linear"],
        "weight_dequant_calls": counters["dequant"],
        "impls": sorted(counters["impls"]),
        "native": counters["linear"] > 0 and counters["dequant"] == 0,
    }


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    header, metadata = read_header(model)
    layers = json.loads(metadata["_quantization_metadata"])["layers"]
    layer = args.layer or next(iter(layers))
    weight_info = header[f"{layer}.weight"]
    scale_info = header[f"{layer}.weight_scale"]

    counters = instrument()
    results = [
        run_case(
            # Line numbers were from ComfyUI 0.29.0. Re-located against the installed 0.33.0
            # on 2026-08-18: the call sites are sd.py:269, sd1_clip.py:213 and ops.py:431-434.
            "text encoder path (sd1_clip.py:213 + sd.py:269, 0.33.0)",
            {"quant_config": {}, "compute_dtype": torch.float32, "full_precision_mm": True},
            layer, weight_info, scale_info, model, counters, force_cast=True, input_dtype=torch.float32,
        ),
        run_case(
            "text encoder path, force_cast cleared",
            {"quant_config": {}, "compute_dtype": torch.float32, "full_precision_mm": True},
            layer, weight_info, scale_info, model, counters, force_cast=False, input_dtype=torch.float32,
        ),
        run_case(
            "diffusion path (pick_operations defaults)",
            {"quant_config": {}, "compute_dtype": torch.bfloat16, "full_precision_mm": False},
            layer, weight_info, scale_info, model, counters, force_cast=False, input_dtype=torch.bfloat16,
        ),
        run_case(
            "observed LTXAV gemma: bf16 weights, fp32 activations",
            {"quant_config": {}, "compute_dtype": torch.bfloat16, "full_precision_mm": True},
            layer, weight_info, scale_info, model, counters, force_cast=True, input_dtype=torch.float32,
        ),
        run_case(
            "same, dtype mismatch removed only",
            {"quant_config": {}, "compute_dtype": torch.bfloat16, "full_precision_mm": True},
            layer, weight_info, scale_info, model, counters, force_cast=True, input_dtype=torch.bfloat16,
        ),
    ]
    print(json.dumps(results, indent=2))
    return 0 if any(r["native"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
