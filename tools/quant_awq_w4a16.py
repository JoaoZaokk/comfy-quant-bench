"""Native W4A16 (Q4_1 codes in comfy-kitchen's AWQ W4A16 layout) from a BF16 safetensors DiT.

    python_embeded\\python.exe -s tools\\quant_awq_w4a16.py --input <bf16.safetensors> [--profile qwen_image21] [--dry-run]

Each Linear weight of the profile is quantized from FP32 with gguf-py's Q4_1 (uint4, one fp16 scale d and min m per
32 weights, the same codes as a GGUF Q4_1) and stored as ComfyUI format `awq_w4a16`:

    weight        int8 [N, K/2]   two uint4 per byte, low nibble = even column
    weight_scale  bf16 [K/32, N]  d
    weight_zeros  bf16 [K/32, N]  m + 8 d      so that W = (q - 8) * scale + zeros

Activations stay BF16. Needs the local ComfyUI patch that registers `awq_w4a16` and, for speed at DiT batch sizes,
the comfy-kitchen patch with the fused Triton dequant. Output goes beside the source as `<stem>_q4_1_awq.safetensors`
(+ `.quant.json`), written to `.partial` and renamed together.

Streams since 2026-09-29: each layer is quantized inside the write loop (`_formats.AwqW4A16`), one
tensor in memory at a time, instead of the whole quantized model held in a dict before writing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402
from _profiles import PROFILE_PATTERNS, detect_profile, select_layers  # noqa: E402

G = 32
q4_1_awq = F.q4_1_awq  # re-exportado: o nome antigo continua importavel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    output = source.with_name(f"{source.stem}_q4_1_awq.safetensors")
    conv = C.Conversion(source, output, output.with_suffix(".quant.json"))
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata
    profile = detect_profile(source, list(header)) if args.profile == "auto" else args.profile
    fmt = F.AwqW4A16(G)
    selected = select_layers(header, profile, fmt.accepts)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")
    print(f"Source: {source}\nProfile: {profile}  Q4_1 -> awq_w4a16 (group {G})  quantized={len(selected)}  "
          f"kept={len(header) - len(selected)}\nOutput: {output}")
    if args.dry_run:
        return 0

    started = time.perf_counter()
    formats = {name: fmt for name in selected}
    novo = F.quant_metadata(metadata, F.layer_configs(formats), "awq_w4a16 (Q4_1 codes, group 32)")
    erros: dict[str, float] = {}
    feitas = {"n": 0}

    def anota(name: str, extra: dict) -> None:
        erros[name] = extra["rel"]
        feitas["n"] += 1
        if feitas["n"] % 48 == 0 or feitas["n"] == len(selected):
            print(f"[{feitas['n']}/{len(selected)}] quantized", flush=True)

    with conv.tensors() as fonte:
        entradas = F.plan_model(header, formats, lambda name: fonte[name], None, on_quantized=anota)
        conv.guard(conv.planned_size(entradas, novo), accumulated=F.streaming_peak(header, selected),
                   label="AWQ W4A16 conversion")

        def manifesto() -> dict:
            rel = sorted(erros.values())
            return {
                "source": str(source), "source_size": source.stat().st_size,
                "output": str(output), "output_size": conv.output_size,
                "architecture": profile, "quantization": "awq_w4a16 from gguf-py Q4_1 codes", "group_size": G,
                "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
                "weight_rel_rmse_mean": sum(rel) / len(rel), "weight_rel_rmse_max": rel[-1],
                "gguf_version": importlib.metadata.version("gguf"), "torch_version": torch.__version__,
                "conversion_seconds": round(time.perf_counter() - started, 3),
            }

        conv.commit(entradas, novo, sidecar=manifesto)
    rel = sorted(erros.values())
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}); weight rel-RMSE mean {sum(rel) / len(rel):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
