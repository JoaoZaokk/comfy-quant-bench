"""Load an LTX-AV text encoder through the normal ComfyUI loader and encode one prompt.

Reports loader warnings, quantized module counts, which ConvRot backend actually ran,
VRAM/timing, and writes the conditioning tensor so two runs can be compared.
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
    parser.add_argument("--text-encoder", required=True, help="file name inside models/text_encoders")
    parser.add_argument("--ckpt", required=True, help="file supplying text_embedding_projection")
    parser.add_argument(
        "--ckpt-folder",
        default="checkpoints",
        help="folder_paths category holding --ckpt; use text_encoders for the split projection file",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--save-cond", type=Path)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Capture the forward-time gate values of one quantized Linear.",
    )
    parser.add_argument(
        "--second-pass-native",
        action="store_true",
        help="Re-encode after retyping the quantized weights to the text encoder's compute dtype, so "
             "cast_bias_weight stops dequantizing them. Runtime-only; no ComfyUI file is modified.",
    )
    parser.add_argument(
        "--native-dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="dtype to retype the quantized weights to; must match the activations reaching the Linear",
    )
    args = parser.parse_args()
    args.native_dtype = getattr(torch, args.native_dtype)
    return args


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def quant_module_summary(model) -> dict:
    from comfy.quant_ops import QuantizedTensor

    formats = {}
    layouts = {}
    groupsizes = {}
    quantized_weights = 0
    for module in model.modules():
        quant_format = getattr(module, "quant_format", None)
        if quant_format is None:
            continue
        formats[quant_format] = formats.get(quant_format, 0) + 1
        layout = getattr(module, "layout_type", None)
        layouts[str(layout)] = layouts.get(str(layout), 0) + 1
        weight = getattr(module, "weight", None)
        if isinstance(weight, QuantizedTensor):
            quantized_weights += 1
            params = getattr(weight, "_params", None)
            size = getattr(params, "convrot_groupsize", None)
            groupsizes[str(size)] = groupsizes.get(str(size), 0) + 1
    return {
        "formats": formats,
        "layouts": layouts,
        "convrot_groupsizes": groupsizes,
        "quantized_weight_tensors": quantized_weights,
    }


def main() -> int:
    args = parse_args()
    capture = LogCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    counters = instrument()

    import comfy.sd
    import folder_paths

    te_path = folder_paths.get_full_path_or_raise("text_encoders", args.text_encoder)
    ckpt_path = folder_paths.get_full_path_or_raise(args.ckpt_folder, args.ckpt)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[te_path, ckpt_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.LTXV,
    )
    load_seconds = time.perf_counter() - started
    load_peak_vram = torch.cuda.max_memory_allocated()

    model = clip.cond_stage_model
    summary = quant_module_summary(model)
    summary["model_class"] = type(model).__name__
    summary["text_projection_type"] = getattr(model, "text_projection_type", None)

    def encode_once():
        counters["native_calls"] = 0
        counters["dequant_calls"] = 0
        counters["impls"] = set()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        tokens = clip.tokenize(args.prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        torch.cuda.synchronize()
        return conditioning[0][0], {
            "encode_seconds": round(time.perf_counter() - started, 3),
            "encode_peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "convrot_linear_calls": counters["native_calls"],
            "convrot_dequant_calls": counters["dequant_calls"],
            "convrot_impls": sorted(counters["impls"]),
        }

    cond_tensor, pass_stats = encode_once()
    encode_seconds = pass_stats["encode_seconds"]
    encode_peak_vram = pass_stats["encode_peak_vram_bytes"]

    gates = None
    if args.diagnose:
        from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor

        target = next(m for m in model.modules() if getattr(m, "quant_format", None) == "convrot_w4a4")
        original_forward = type(target).forward
        captured = {}

        def probing_forward(self, input, *rest, **kwargs):
            if self is target and not captured:
                weight = self.weight
                params = getattr(weight, "_params", None)
                captured.update({
                    "input_dtype": str(input.dtype),
                    "weight_is_quantized_tensor": isinstance(weight, QuantizedTensor),
                    "weight_dtype": str(weight.dtype),
                    "weight_orig_dtype": str(getattr(params, "orig_dtype", None)),
                    "weight_device": str(weight.device),
                    "input_device": str(input.device),
                    "layout_type": getattr(self, "layout_type", None),
                    "_full_precision_mm": self._full_precision_mm,
                    "comfy_force_cast_weights": getattr(self, "comfy_force_cast_weights", False),
                    "comfy_cast_weights": getattr(self, "comfy_cast_weights", None),
                    "weight_function_count": len(self.weight_function),
                    "bias_function_count": len(self.bias_function),
                    "transposed_param": getattr(params, "transposed", None),
                    "quantize_input": QUANT_ALGOS.get(self.quant_format, {}).get("quantize_input", True),
                })
                captured["_use_quantized"] = (
                    captured["layout_type"] is not None
                    and not isinstance(input, QuantizedTensor)
                    and not captured["_full_precision_mm"]
                    and not captured["comfy_force_cast_weights"]
                    and captured["weight_function_count"] == 0
                    and captured["bias_function_count"] == 0
                )
                captured["dtype_mismatch_forces_dequant"] = (
                    captured["weight_dtype"] != captured["input_dtype"]
                )
            return original_forward(self, input, *rest, **kwargs)

        type(target).forward = probing_forward
        clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
        type(target).forward = original_forward
        gates = captured

    native_pass = None
    if args.second_pass_native:
        from comfy.quant_ops import QuantizedTensor

        retyped = 0
        for module in model.modules():
            if getattr(module, "quant_format", None) != "convrot_w4a4":
                continue
            weight = module.weight
            if not isinstance(weight, QuantizedTensor) or weight.dtype == args.native_dtype:
                continue
            # QuantizedTensor.to(dtype=...) only rewrites params.orig_dtype; it does not dequantize.
            module.weight = torch.nn.Parameter(weight.to(dtype=args.native_dtype), requires_grad=False)
            retyped += 1
        native_cond, native_pass = encode_once()
        native_pass["modules_retyped"] = retyped
        native_pass["target_dtype"] = str(args.native_dtype)
        difference = native_cond.float() - cond_tensor.float()
        native_pass["relative_rmse_vs_emulated"] = (
            difference.square().mean().sqrt() / cond_tensor.float().square().mean().sqrt()
        ).item()
        native_pass["max_abs_error_vs_emulated"] = difference.abs().max().item()
    report = {
        "text_encoder": args.text_encoder,
        "ckpt": args.ckpt,
        "ckpt_folder": args.ckpt_folder,
        "prompt": args.prompt,
        "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(encode_seconds, 3),
        "load_peak_vram_bytes": load_peak_vram,
        "encode_peak_vram_bytes": encode_peak_vram,
        "cond_shape": list(cond_tensor.shape),
        "cond_dtype": str(cond_tensor.dtype),
        "cond_mean": cond_tensor.float().mean().item(),
        "cond_std": cond_tensor.float().std().item(),
        "cond_absmax": cond_tensor.float().abs().max().item(),
        "convrot_linear_calls": pass_stats["convrot_linear_calls"],
        "convrot_dequant_calls": pass_stats["convrot_dequant_calls"],
        "convrot_impls": pass_stats["convrot_impls"],
        "native_second_pass": native_pass,
        "forward_gates": gates,
        **summary,
        "log_warnings": [message for level, message in capture.records if level in ("WARNING", "ERROR")],
        "log_info": [message for level, message in capture.records if level == "INFO"],
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.save_cond:
        torch.save(cond_tensor.cpu(), args.save_cond)

    print(json.dumps({k: v for k, v in report.items() if k not in ("log_warnings", "log_info")}, indent=2))
    print(f"warnings: {len(report['log_warnings'])}  info: {len(report['log_info'])}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
