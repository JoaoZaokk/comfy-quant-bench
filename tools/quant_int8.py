"""Create a ComfyUI int8_tensorwise checkpoint, with or without the ConvRot rotation.

Two formats from one converter, because they differ by exactly one step:

    --convrot     rotate with Hadamard(256), then row-wise int8   == what Lightricks ships as
                                                                     "comfy-int8-convrot"
    --no-convrot  row-wise int8 only                              == plain W8A8

Both write `<layer>.weight` as I8 plus `<layer>.weight_scale`, and both are read by ComfyUI's
`int8_tensorwise` layout. The `convrot` flag in the per-layer metadata is what makes `comfy/ops.py`
apply the inverse rotation at load (`ops.py:1267`), so it must match how the weight was produced.

Unlike quant_w4a8.py this does not require CUDA. Every quantizer here resolves to comfy-kitchen's
eager backend on CPU, measured at ~50 ms for a 2048x2048 layer, so a 1440-layer model costs about a
minute of compute and is dominated by reading and writing the file. Pass `--device cuda` to use the
GPU when it is free.

Note that producing a checkpoint on CPU says nothing about whether it will *run* natively -- that
is decided at load time by the GPU's capabilities. `int8_tensorwise` is not in the disabled set on
SM 8.6, so it runs natively on Ampere; nvfp4 and mxfp8 do not.

Same safety rules as the other converters: streaming writes, atomic replace, refuses to overwrite a
source, an existing output, a stale partial, or to requantize an already-quantized checkpoint.

    python tools/quant_int8.py --input ltx-2.5-...-bf16.safetensors --convrot --dry-run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402

from _native_probe import native_backend_ready  # noqa: E402
from quant_w4a8 import (  # noqa: E402
    HIGH_PRECISION_DTYPES,
    PROFILE_PATTERNS,
    SAFETENSORS_DTYPE,
    copy_range,
    detect_profile,
    human_size,
    read_header,
    read_tensor,
)


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
    pattern = PROFILE_PATTERNS[profile]
    out = []
    for name, info in header.items():
        shape = info["shape"]
        if not (pattern.fullmatch(name) and info["dtype"] in HIGH_PRECISION_DTYPES
                and len(shape) == 2):
            continue
        # Rotation needs K divisible by the Hadamard size; plain int8 has no such constraint.
        if convrot and shape[1] % groupsize:
            continue
        out.append(name)
    return out


def quantize(weight: torch.Tensor, convrot: bool, groupsize: int):
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
    # instead of pretending to preflight a path that never touches the registry. Both branches
    # below are unchanged by the ticket-05 consolidation: the exemption is deliberate and stays
    # an exemption, and the `--convrot` branch keeps refusing on exactly the same condition.
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

    selected_set = set(selected)
    # Two-pass design, not streaming. The old `largest * 3 + 2 GiB` came from quant_w4a4.py and
    # is the worst offender here: measured at 2.375 GiB asked against 19.144 GiB accumulated on
    # LTX-2.5, an 8.1x understatement. A guard that passes and then thrashes is worse than none.
    from _ram_guard import int8_bytes

    accumulated = sum(int8_bytes(*header[n]["shape"]) for n in selected)
    conv.guard(source.stat().st_size, accumulated=accumulated, label="INT8 conversion")

    quantized: dict[str, dict] = {}
    with source.open("rb") as handle:
        data_start = 8 + struct.unpack("<Q", handle.read(8))[0]
        for index, name in enumerate(selected, 1):
            info = header[name]
            start, end = info["data_offsets"]
            weight = read_tensor(handle, data_start + start, end - start,
                                 info["dtype"], info["shape"]).to(args.device)
            qdata, scale = quantize(weight, args.convrot, args.convrot_groupsize)
            quantized[name] = {"qdata": qdata.cpu().contiguous(),
                               "scale": scale.cpu().contiguous().float()}
            del weight, qdata, scale
            if args.device == "cuda":
                torch.cuda.empty_cache()
            if index % 120 == 0 or index == len(selected):
                print(f"[{index}/{len(selected)}] quantized", flush=True)

    layers = {name.removesuffix(".weight"): {"format": "int8_tensorwise",
                                             **({"convrot": True,
                                                 "convrot_groupsize": args.convrot_groupsize}
                                                if args.convrot else {})}
              for name in selected}
    output_metadata = dict(metadata)
    output_metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": layers}, separators=(",", ":"))
    output_metadata["quantization"] = f"int8_tensorwise{'+convrot' if args.convrot else ''}"

    # A chave da escala aqui e `{name}_scale`, com sublinhado e SEM ponto -- diferente do
    # `.weight_scale` do w4a4 e do `.weight_s_rel` do w4a8. Preservada exatamente como estava.
    entradas = []
    for name, info in header.items():
        if name not in selected_set:
            entradas.append(C.plan_copy(name, info))
            continue
        entry = quantized[name]
        entradas.append(C.plan_write(name, entry["qdata"]))
        entradas.append(C.plan_write(f"{name}_scale", entry["scale"]))

    conv.commit(entradas, output_metadata)

    elapsed = time.perf_counter() - started
    sidecar.write_text(json.dumps({
        "source": str(source), "source_size": source.stat().st_size,
        "output": str(output), "output_size": output.stat().st_size,
        "architecture": profile,
        "quantization": f"int8_tensorwise{'+convrot' if args.convrot else ''}",
        "convrot": args.convrot, "convrot_groupsize": args.convrot_groupsize,
        "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
        "quantized_on": args.device,
        # None on the CPU path and on --no-convrot, both of which skip the probe by design.
        "backend": backend["resolved"]["quantize_int8_convrot_weight"] if backend else None,
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__,
        "conversion_seconds": round(elapsed, 3),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({human_size(output.stat().st_size)}) in {elapsed / 60:.1f} min")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
