
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
from _profiles import HIGH_PRECISION_DTYPES, PROFILE_PATTERNS, detect_profile, select_layers  # noqa: E402,F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    convrot = parser.add_mutually_exclusive_group()
    convrot.add_argument("--convrot", dest="convrot", action="store_true", default=True,
                         help="rotate before quantizing, as Lightricks does (default)")
    convrot.add_argument("--no-convrot", dest="convrot", action="store_false",
                         help="plain row-wise int8, no rotation")
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_layers(header: dict, profile: str, convrot: bool, groupsize: int) -> list[str]:
    return select_layers(header, profile, F.Int8Tensorwise(convrot, groupsize).accepts)


def quantize(weight: torch.Tensor, convrot: bool, groupsize: int):
    """(qdata, scale) crus do comfy-kitchen, como sempre -- `quant_misto_w4a8_int8` importa isto.

    O caminho de escrita deste arquivo usa `_formats.Int8Tensorwise`, que chama exatamente o mesmo.
    """
    from comfy_kitchen.backends.eager import quantization as eager
    import comfy_kitchen as ck

    if convrot:
        implementation = ck.registry.get_implementation(
            "quantize_int8_convrot_weight",
            kwargs={"weight": weight, "group_size": groupsize})
        return implementation(weight, group_size=groupsize)
    return eager.quantize_int8_rowwise(weight)


def int8_probe_ops(groupsize: int) -> dict:
    """The one op this converter dispatches through `ck.registry`, at its real rotation groupsize.

    Not the W4A4 pair and not W4A8's quantizer: this converter's registry-dispatched op is
    `quantize_int8_convrot_weight` (see quantize() above), and resolving anyone else's op would
    pass or fail for the wrong reason. That distinction is why `native_backend_ready` takes the op
    names instead of assuming them -- of the six private probes it replaced, this was the only one
    already passing the caller's own groupsize through rather than a hardcoded literal.

    Still scoped to the `--convrot` path, and the caller below is what enforces that:
    `--no-convrot` calls `eager.quantize_int8_rowwise` directly, bypassing `ck.registry` entirely,
    so there is no native backend for it to resolve to on any device. That is this file's declared
    CPU-first design (module docstring), not a silent fallback a check should catch.
    """
    return {"quantize_int8_convrot_weight": {"convrot_groupsize": groupsize}}


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    portable_root = Path(__file__).resolve().parent.parent
    suffix = "int8_convrot" if args.convrot else "int8"
    output = (args.output or source.with_name(f"{source.stem}_{suffix}.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    conv = C.Conversion(source, output, sidecar)
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata

    profile = detect_profile(source, list(header)) if args.profile == "auto" else args.profile
    selected = selected_layers(header, profile, args.convrot, args.convrot_groupsize)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")

    print(f"Source: {source}")
    print(f"Profile: {profile}   convrot={args.convrot} "
          f"groupsize={args.convrot_groupsize}   device={args.device}")
    print(f"Selected Linear weights: {len(selected)}")
    print(f"Output: {output}")
    if args.dry_run:
        for name in selected[:8]:
            print(f"  {name}: {header[name]['shape']} {header[name]['dtype']}")
        if len(selected) > 8:
            print(f"  ... {len(selected) - 8} more")
        return 0

    # `--device cuda` used to only confirm the card exists (torch.cuda.is_available(), above),
    # not that comfy_kitchen's registry actually resolves the quantizer to a CUDA backend --
    # a card being present says nothing about whether normal ComfyUI would run this kernel
    # natively or fall back silently. `--convrot` is the only mode that goes through the
    # registry at all (see quantize() and int8_probe_ops() above), so that is what gets
    # checked; `--no-convrot` is unconditionally eager by this file's own design and prints why
    # instead of pretending to preflight a path that never touches the registry.
    backend = None
    if args.device == "cuda":
        if args.convrot:
            backend = native_backend_ready(portable_root,
                                           int8_probe_ops(args.convrot_groupsize))
            if not backend["native_ready"]:
                raise SystemExit(
                    "Refusing: normal ComfyUI resolves int8 ConvRot quantization to "
                    f"{backend['resolved']['quantize_int8_convrot_weight']}, "
                    "not a CUDA backend")
        else:
            print("--no-convrot always calls comfy-kitchen's eager backend directly, on any "
                  "device (quantize() bypasses ck.registry for this path) -- no native-backend "
                  "preflight applies here.")

    started = time.perf_counter()
    fmt = F.Int8Tensorwise(args.convrot, args.convrot_groupsize)
    formats = {name: fmt for name in selected}
    out_meta = F.quant_metadata(metadata, F.layer_configs(formats),
                                f"int8_tensorwise{'+convrot' if args.convrot else ''}")

    with conv.tensors() as fonte:
        def peso(name: str) -> torch.Tensor:
            # FP32 na entrada (26/09): em BF16 a rotacao muda ~8% dos codigos; a Comfy-Org quantiza
            # do FP32 (99,99999% dos codigos iguais aos dela com FP32, 91% com BF16).
            return fonte[name].to(args.device, torch.float32)

        # Transmite desde 2026-09-29. Antes: duas passadas, com a guarda de RAM medida em 2,375
        # GiB pedidos contra 19,144 GiB acumulados no LTX-2.5 (8,1x).
        entradas = F.plan_model(header, formats, peso, None)
        conv.guard(conv.planned_size(entradas, out_meta),
                   accumulated=F.streaming_peak(header, selected), label="INT8 conversion")

        def progresso(indice: int, total: int, chave: str) -> None:
            if chave in formats and (indice % 120 == 0 or indice == total):
                print(f"[{indice}/{total}] quantized", flush=True)

        def manifesto() -> dict:
            return {
                "source": str(source), "source_size": source.stat().st_size,
                "output": str(output), "output_size": conv.output_size,
                "architecture": profile,
                "quantization": f"int8_tensorwise{'+convrot' if args.convrot else ''}",
                "convrot": args.convrot, "convrot_groupsize": args.convrot_groupsize,
                "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
                "quantized_on": args.device,
                # None on the CPU path and on --no-convrot, both of which skip the probe by design.
                "backend": backend["resolved"]["quantize_int8_convrot_weight"] if backend else None,
                "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
                "torch_version": torch.__version__,
                "conversion_seconds": round(time.perf_counter() - started, 3),
            }

        conv.commit(entradas, out_meta, progress=progresso, sidecar=manifesto)

    elapsed = time.perf_counter() - started
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}) in {elapsed / 60:.1f} min")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
