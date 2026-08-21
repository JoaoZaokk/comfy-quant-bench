"""Load a quantized diffusion model through ComfyUI and prove its Linears dispatch natively.

A full HunyuanVideo forward needs a whole pipeline, so instead this loads the model exactly the
way the UI does, then drives one real quantized Linear from the loaded graph with a correctly
shaped activation and counts which ConvRot implementation ran.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _native_probe import instrument  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="file name inside models/diffusion_models")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=256, help="sequence length for the probe activation")
    return parser.parse_args()


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def main() -> int:
    args = parse_args()
    capture = LogCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    counters = instrument()

    import comfy.model_management
    import comfy.sd
    import folder_paths
    from comfy.quant_ops import QuantizedTensor

    # extra_model_paths.yaml is read by main.py at boot, not by importing folder_paths, so a
    # script that skips main.py sees only ComfyUI/models. Register it here or every model on the
    # D: mount is invisible.
    extra_paths = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra_paths.is_file():
        from utils.extra_config import load_extra_path_config

        load_extra_path_config(str(extra_paths))

    candidate = Path(args.model)
    if candidate.is_file():
        model_path = str(candidate)
    else:
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", args.model)

    started = time.perf_counter()
    patcher = comfy.sd.load_diffusion_model(model_path)
    load_seconds = time.perf_counter() - started

    model_config = patcher.model.model_config
    diffusion_model = patcher.model.diffusion_model

    formats, layouts, groupsizes = {}, {}, {}
    quantized_modules = []
    for name, module in diffusion_model.named_modules():
        quant_format = getattr(module, "quant_format", None)
        if quant_format is None:
            continue
        formats[quant_format] = formats.get(quant_format, 0) + 1
        layouts[str(getattr(module, "layout_type", None))] = layouts.get(str(getattr(module, "layout_type", None)), 0) + 1
        weight = getattr(module, "weight", None)
        if isinstance(weight, QuantizedTensor):
            size = getattr(getattr(weight, "_params", None), "convrot_groupsize", None)
            groupsizes[str(size)] = groupsizes.get(str(size), 0) + 1
            quantized_modules.append((name, module, weight))

    report = {
        "model": args.model,
        "load_seconds": round(load_seconds, 3),
        "model_config_class": type(model_config).__name__,
        "quant_config_detected": model_config.quant_config,
        "quantized_module_count": len(quantized_modules),
        "formats": formats,
        "layouts": layouts,
        "convrot_groupsizes": groupsizes,
    }

    if not quantized_modules:
        report["error"] = "no quantized modules found after loading"
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    device = patcher.load_device
    compute_dtype = quantized_modules[0][2].dtype

    probes = []
    for name, module, weight in quantized_modules[:4]:
        counters["native_calls"] = 0
        counters["dequant_calls"] = 0
        counters["impls"] = set()
        x = torch.randn((1, args.tokens, module.in_features), device=device, dtype=compute_dtype)
        with torch.no_grad():
            output = module(x)
        torch.cuda.synchronize()
        probes.append({
            "module": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "input_dtype": str(x.dtype),
            "weight_dtype": str(weight.dtype),
            "output_dtype": str(output.dtype),
            "native_linear_calls": counters["native_calls"],
            "weight_dequant_calls": counters["dequant_calls"],
            "impls": sorted(counters["impls"]),
            # Was `counters["native_calls"] > 0 and counters["dequant_calls"] == 0` -- a dispatcher
            # call count, which does not distinguish CUDA from eager (both route through the same
            # dispatcher). Read back which implementation actually resolved instead.
            "native": any(".backends.cuda" in impl for impl in counters["impls"]),
        })

    report["probes"] = probes
    report["all_native"] = all(p["native"] for p in probes)
    report["vram_after_load_bytes"] = torch.cuda.memory_allocated()
    report["log_warnings"] = [m for level, m in capture.records if level in ("WARNING", "ERROR")]

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "log_warnings"}, indent=2))
    print(f"warnings: {len(report['log_warnings'])}")
    for message in report["log_warnings"][:20]:
        print(f"  WARNING: {message}")
    print(f"Wrote {args.report}")
    return 0 if report["all_native"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
