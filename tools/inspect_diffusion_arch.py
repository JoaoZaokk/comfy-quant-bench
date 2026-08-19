"""Ask ComfyUI itself which architecture a diffusion checkpoint is and which Linears it has.

Builds a meta-device state dict from the safetensors header (no data is read), runs ComfyUI's
own model detection, instantiates the model on the meta device, and reports every Linear module
with the state-dict key that feeds it. Use this to derive a quantization profile from the real
loader instead of guessing from key names.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I8": torch.int8, "U8": torch.uint8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
CONVROT_GROUP_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def meta_state_dict(path: Path) -> tuple[dict, dict, dict]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    header = {k: v for k, v in raw.items()}
    state_dict = {
        name: torch.empty(info["shape"], dtype=DTYPES[info["dtype"]], device="meta")
        for name, info in header.items()
    }
    return state_dict, header, metadata


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    state_dict, header, metadata = meta_state_dict(model_path)

    import comfy.model_detection

    # unet_prefix_from_state_dict guesses; a bare diffusion_models file usually has no prefix at
    # all, so fall back the way load_diffusion_model_state_dict does.
    model_config = None
    for prefix in (comfy.model_detection.unet_prefix_from_state_dict(state_dict), ""):
        unet_config = comfy.model_detection.detect_unet_config(state_dict, prefix, metadata=metadata)
        if unet_config is None:
            continue
        model_config = comfy.model_detection.model_config_from_unet(state_dict, prefix, metadata=metadata)
        if model_config is not None:
            break
    if model_config is None:
        raise SystemExit("ComfyUI could not detect this checkpoint; do not guess a profile for it")

    report = {
        "model": str(model_path),
        "unet_prefix": prefix,
        "model_config_class": type(model_config).__name__ if model_config else None,
        "image_model": (unet_config or {}).get("image_model"),
        "unet_config": {k: v for k, v in (unet_config or {}).items() if not isinstance(v, torch.Tensor)},
        "existing_quant_config": getattr(model_config, "quant_config", None),
    }

    processed = model_config.process_unet_state_dict(state_dict)
    report["keys_after_process"] = len(processed)
    report["key_remapping_applied"] = sorted(set(state_dict) - set(processed))[:10]

    with torch.device("meta"):
        model = model_config.get_model(processed, "")
    diffusion_model = model.diffusion_model

    import comfy.ops

    linears = []
    for name, module in diffusion_model.named_modules():
        if not isinstance(module, torch.nn.Linear) and not hasattr(module, "in_features"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        key = f"{name}.weight"
        info = header.get(key)
        linears.append({
            "module": name,
            "class": type(module).__name__,
            "shape": list(weight.shape),
            "in_state_dict": info is not None,
            "state_dict_dtype": info["dtype"] if info else None,
            "convrot_eligible": bool(
                info
                and info["dtype"] in {"BF16", "F16", "F32"}
                and len(info["shape"]) == 2
                and info["shape"][1] % CONVROT_GROUP_SIZE == 0
            ),
        })

    report["linear_module_count"] = len(linears)
    report["linear_in_state_dict"] = sum(1 for entry in linears if entry["in_state_dict"])
    report["convrot_eligible_count"] = sum(1 for entry in linears if entry["convrot_eligible"])

    grouped = Counter()
    eligible_grouped = Counter()
    for entry in linears:
        pattern = re.sub(r"\.\d+\.", ".N.", entry["module"])
        grouped[pattern] += 1
        if entry["convrot_eligible"]:
            eligible_grouped[pattern] += 1
    report["linear_patterns"] = [
        {
            "pattern": pattern,
            "count": count,
            "convrot_eligible": eligible_grouped.get(pattern, 0),
            "example_shape": next(e["shape"] for e in linears if re.sub(r"\.\d+\.", ".N.", e["module"]) == pattern),
        }
        for pattern, count in sorted(grouped.items())
    ]

    print(f"model_config      : {report['model_config_class']}")
    print(f"image_model       : {report['image_model']}")
    print(f"unet prefix       : {prefix!r}")
    print(f"Linear modules    : {report['linear_module_count']} "
          f"({report['linear_in_state_dict']} present in file, {report['convrot_eligible_count']} ConvRot-eligible)")
    print()
    print(f"{'count':>5} {'elig':>5}  {'example shape':<22} pattern")
    for entry in report["linear_patterns"]:
        print(f"{entry['count']:5d} {entry['convrot_eligible']:5d}  {str(entry['example_shape']):<22} {entry['pattern']}")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
