"""Walk the diffusion-model load one stage at a time, flushing progress to disk.

A 0xC0000005 access violation kills the interpreter without a traceback, so the only way to
find the failing stage is to record each step as it completes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="file name inside models/diffusion_models")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--stop-after", type=int, default=99)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    handle = args.log.open("w", encoding="utf-8")

    def step(number, message):
        handle.write(f"[{number}] {message}\n")
        handle.flush()
        os.fsync(handle.fileno())
        print(f"[{number}] {message}", flush=True)

    step(0, "interpreter up")

    import comfy.model_detection
    import comfy.sd
    import comfy.utils
    import folder_paths
    step(1, "comfy imported")

    path = folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    step(2, f"resolved {path}")
    if args.stop_after < 3:
        return 0

    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    step(3, f"load_torch_file ok: {len(state_dict)} tensors, metadata keys {list(metadata or {})}")
    if args.stop_after < 4:
        return 0

    state_dict, metadata = comfy.utils.convert_old_quants(state_dict, "", metadata=metadata)
    comfy_quant_keys = [k for k in state_dict if k.endswith(".comfy_quant")]
    step(4, f"convert_old_quants ok: {len(comfy_quant_keys)} comfy_quant keys injected")
    if args.stop_after < 5:
        return 0

    model_config = comfy.model_detection.model_config_from_unet(state_dict, "", metadata=metadata)
    step(5, f"model_config {type(model_config).__name__}, quant_config {model_config.quant_config}")
    if args.stop_after < 6:
        return 0

    missing = [k for k in comfy_quant_keys if k.replace(".comfy_quant", ".weight") not in state_dict]
    step(6, f"comfy_quant keys without a matching weight: {len(missing)} {missing[:5]}")
    if args.stop_after < 7:
        return 0

    processed = model_config.process_unet_state_dict(state_dict)
    processed_quant = [k for k in processed if k.endswith(".comfy_quant")]
    step(7, f"process_unet_state_dict ok: {len(processed)} keys, {len(processed_quant)} comfy_quant")
    step(7.1, f"sample remapped quant key: {sorted(processed_quant)[0] if processed_quant else 'NONE'}")
    if args.stop_after < 8:
        return 0

    unet_dtype = torch.bfloat16
    model_config.set_inference_dtype(unet_dtype, None)
    step(8, "set_inference_dtype ok")
    if args.stop_after < 9:
        return 0

    model = model_config.get_model(processed, "")
    step(9, f"get_model ok: {type(model.diffusion_model).__name__}")
    if args.stop_after < 10:
        return 0

    linear_names = {
        name for name, module in model.diffusion_model.named_modules() if hasattr(module, "quant_format")
    }
    quant_module_names = {k[: -len(".comfy_quant")] for k in processed_quant}
    unmatched = sorted(quant_module_names - linear_names)
    step(10, f"quant keys targeting a real module: {len(quant_module_names) - len(unmatched)}/"
             f"{len(quant_module_names)}; unmatched {len(unmatched)} {unmatched[:5]}")
    if args.stop_after < 11:
        return 0

    model.load_model_weights(processed, "")
    step(11, "load_model_weights ok")
    if args.stop_after < 12:
        return 0

    from comfy.quant_ops import QuantizedTensor

    quantized = [
        (name, module)
        for name, module in model.diffusion_model.named_modules()
        if getattr(module, "quant_format", None) is not None
        and isinstance(getattr(module, "weight", None), QuantizedTensor)
    ]
    step(12, f"quantized modules after load: {len(quantized)}; first {quantized[0][0] if quantized else 'NONE'}")
    if args.stop_after < 13 or not quantized:
        return 0

    import comfy.model_management

    offload_device = comfy.model_management.unet_offload_device()
    model.to(offload_device)
    step(13, f"model.to({offload_device}) ok")
    if args.stop_after < 14:
        return 0

    device = comfy.model_management.get_torch_device()
    name, module = quantized[0]
    module.to(device)
    step(14, f"moved {name} to {device}; weight dtype {module.weight.dtype}")
    if args.stop_after < 15:
        return 0

    counters = instrument()
    x = torch.randn((1, 64, module.in_features), device=device, dtype=module.weight.dtype)
    with torch.no_grad():
        out = module(x)
    torch.cuda.synchronize()
    step(15, f"forward ok: {tuple(out.shape)} {out.dtype}; native={counters['linear']} "
             f"dequant={counters['dequant']} impls={sorted(counters['impls'])}")
    if args.stop_after < 16:
        return 0

    import comfy.model_patcher

    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=offload_device)
    step(16, "patcher built; calling load_models_gpu(force_full_load=True) -- this is the suspected crash")
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    step(17, f"load_models_gpu ok; vram allocated {torch.cuda.memory_allocated()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
