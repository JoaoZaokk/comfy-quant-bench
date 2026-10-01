"""Create a ComfyUI asym_w4a8_int8 checkpoint from a high-precision source.

W4A8 keeps 4-bit weights but runs the matmul through the INT8 path, which on Ampere is a far more
mature tensor-core route than INT4. comfy-kitchen also applies a ConvRot rotation and a Lloyd-Max
codebook internally, so this is strictly richer than plain ConvRot W4A4.

Per quantized layer the checkpoint carries four tensors instead of two:

    <layer>.weight            int8 container, packed int4        [N, K // 2]
    <layer>.weight_s_rel      per-group scale, fp8 stored as u8  [N, K // group_size]
    <layer>.weight_s_channel  per-channel scale                  [N]
    <layer>.weight_codebook   Lloyd-Max levels                   [16]

`comfy/ops.py` reads exactly those names and ignores the optional asymmetric `correction` tensor,
so this converter only ever quantizes symmetrically -- an asymmetric weight would have its
correction silently dropped and decode wrong.

Same safety rules as quant_w4a4.py: streaming writes, atomic replace, refuses to overwrite a
source or an existing output, and refuses to run when normal ComfyUI would not pick the CUDA backend.

Transmite desde 2026-09-29: ate ali este conversor quantizava o modelo inteiro num dict antes de
escrever ("as formas saem do kernel"), o que exigia uma guarda de RAM propria -- medida em 10,777
GiB acumulados contra 2,375 GiB pedidos pela heuristica de streaming. As formas sao analiticas
(`_formats.AsymW4A8.tensors`); agora o pico e um tensor por vez.
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
from _native_probe import native_backend_ready  # noqa: E402
from _profiles import (  # noqa: E402,F401  (re-exportados para scripts antigos)
    HIGH_PRECISION_DTYPES,
    PROFILE_PATTERNS,
    detect_profile,
    is_qwen_image21,
    select_layers,
)

QUANTIZATION = "asym_w4a8_int8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--no-codebook", action="store_true", help="skip the Lloyd-Max codebook")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def w4a8_probe_ops(args: argparse.Namespace) -> dict:
    """The op this converter dispatches, under the flags this run is about to use.

    One op, not two, and deliberately unchanged: this file has only ever preflighted
    `quantize_w4a8_int8_weight`, and widening it to `w4a8_int8_linear` -- the op ComfyUI will
    execute the resulting checkpoint with -- is a new refusal, outside ticket 05's closing
    criterion, which is about *which configuration* each probe asks about and not about which ops.
    Flagged rather than done. `quant_mixed.py` does check all four, so the gap is visible.

    What did change: `group_size`, `convrot_groupsize` and `codebook` now come from the caller.
    The old private copy hardcoded 16 / 256 / True, so `--group-size 32`, `--convrot-groupsize 64`
    and `--no-codebook` were all preflighted as though they had not been passed.
    """
    return {
        "quantize_w4a8_int8_weight": {
            "group_size": args.group_size,
            "convrot_groupsize": args.convrot_groupsize,
            "codebook": not args.no_codebook,
        },
    }


def selected_layers(header: dict, profile: str, group_size: int, convrot_groupsize: int) -> list[str]:
    return select_layers(header, profile, F.AsymW4A8(group_size, convrot_groupsize).accepts)


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    portable_root = Path(__file__).resolve().parent.parent
    output = (args.output or source.with_name(f"{source.stem}_w4a8.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    conv = C.Conversion(source, output, sidecar)
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata
    profile = detect_profile(source, list(header)) if args.profile == "auto" else args.profile
    selected = selected_layers(header, profile, args.group_size, args.convrot_groupsize)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")

    print(f"Source: {source}")
    print(f"Profile: {profile}   group_size={args.group_size}  convrot_groupsize={args.convrot_groupsize}")
    print(f"Selected Linear weights: {len(selected)}")
    print(f"Output: {output}")
    if args.dry_run:
        for name in selected[:10]:
            print(f"  {name}: {header[name]['shape']} {header[name]['dtype']}")
        if len(selected) > 10:
            print(f"  ... {len(selected) - 10} more")
        return 0

    backend = native_backend_ready(portable_root, w4a8_probe_ops(args))
    if not backend["native_ready"]:
        raise SystemExit("Refusing: normal ComfyUI resolves W4A8 to "
                         f"{backend['resolved']['quantize_w4a8_int8_weight']}, "
                         "not a CUDA backend")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    import comfy_kitchen as ck

    started = time.perf_counter()
    fmt = F.AsymW4A8(args.group_size, args.convrot_groupsize, codebook=not args.no_codebook)
    formats = {name: fmt for name in selected}
    out_meta = F.quant_metadata(metadata, F.layer_configs(formats), QUANTIZATION)

    with conv.tensors() as fonte:
        def peso(name: str) -> torch.Tensor:
            # FP32 na entrada (26/09): com BF16 2,1% dos codigos saem diferentes dos da NidAll, que
            # quantiza do FP32 (99,9998% iguais com FP32). Os dtypes de saida nao mudam.
            return fonte[name].to(device="cuda", dtype=torch.float32)

        entradas = F.plan_model(header, formats, peso, ck)
        conv.guard(conv.planned_size(entradas, out_meta),
                   accumulated=F.streaming_peak(header, selected), label="W4A8 conversion")

        def progresso(indice: int, total: int, chave: str) -> None:
            if chave in formats and (indice % 48 == 0 or indice == total):
                print(f"[{indice}/{total}] quantized", flush=True)

        def manifesto() -> dict:
            return {
                "source": str(source), "source_size": source.stat().st_size,
                "output": str(output), "output_size": conv.output_size,
                "architecture": profile, "quantization": QUANTIZATION,
                "layout": "AsymW4A8Int8Layout",
                "backend": backend["resolved"]["quantize_w4a8_int8_weight"],
                "group_size": args.group_size, "convrot_groupsize": args.convrot_groupsize,
                "symmetric": True, "codebook": not args.no_codebook,
                "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
                "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "conversion_seconds": round(time.perf_counter() - started, 3),
            }

        conv.commit(entradas, out_meta, progress=progresso, sidecar=manifesto)

    elapsed = time.perf_counter() - started
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}) in {elapsed:.1f} s")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
