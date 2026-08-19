"""Check whether UNETLoader's weight_dtype changes anything for a convrot_w4a4 checkpoint.

Loads the same quantized model twice, once the way weight_dtype="default" does and once the way
weight_dtype="fp8_e4m3fn" does, and reports the dtype of a quantized Linear and of a preserved
(non-quantized) parameter in each case.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="file name inside models/diffusion_models")
    return parser.parse_args()


def describe(label, model_options):
    import comfy.sd
    import folder_paths
    from comfy.quant_ops import QuantizedTensor

    path = folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    patcher = comfy.sd.load_diffusion_model(path, model_options=dict(model_options))
    model = patcher.model
    diffusion_model = model.diffusion_model

    quantized = None
    preserved = None
    for name, module in diffusion_model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if quantized is None and isinstance(weight, QuantizedTensor):
            quantized = (name, str(weight.dtype), str(weight._qdata.dtype))
        if preserved is None and not isinstance(weight, QuantizedTensor) and weight.ndim >= 1:
            preserved = (name, str(weight.dtype))
        if quantized and preserved:
            break

    result = {
        "case": label,
        "model_options": {k: str(v) for k, v in model_options.items()},
        "unet_config_dtype": str(model.model_config.unet_config.get("dtype")),
        "manual_cast_dtype": str(model.model_config.manual_cast_dtype),
        "ops_compute_dtype": str(getattr(type(diffusion_model.double_blocks[0].img_attn.qkv), "_compute_dtype", None))
        if hasattr(diffusion_model, "double_blocks") else None,
        "quantized_linear": {"module": quantized[0], "logical_dtype": quantized[1], "packed_dtype": quantized[2]}
        if quantized else None,
        "preserved_param": {"module": preserved[0], "dtype": preserved[1]} if preserved else None,
    }

    del patcher, model, diffusion_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    args = parse_args()
    results = [
        describe("weight_dtype=default", {}),
        describe("weight_dtype=fp8_e4m3fn", {"dtype": torch.float8_e4m3fn}),
    ]
    print(json.dumps(results, indent=2))
