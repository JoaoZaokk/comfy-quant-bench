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
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _native_probe import native_backend_ready  # noqa: E402

PROFILE_PATTERNS = {
    # Derived from Lightricks' own ltx-2.5-...-comfy-int8-convrot checkpoint rather than guessed:
    # this pattern selects exactly the 1440 Linears they quantized, with no false positives and no
    # false negatives against their shipped `comfy_quant` markers. What they leave out is as
    # informative as what they include -- 304 `to_gate_logits`, every adaLN/timestep embedder, and
    # the patchify/proj_out pair stay in full precision, the same principle as this file's
    # hunyuan_video_15 profile preserving `*_mod.linear`.
    #
    # Note they ship int8 weights, not int4. Converting the same layers to W4A8 is strictly more
    # aggressive than what Lightricks considered safe, so A/B it before trusting the output.
    "ltx_2_5": re.compile(
        r"^(?:model\.diffusion_model\.)?"
        r"(?:"
        r"transformer_blocks\.\d+\."
        r"(?:"
        r"(?:audio_)?attn\d+\.(?:to_[qkv]|to_out\.\d+)"
        r"|(?:audio_to_video|video_to_audio)_attn\.(?:to_[qkv]|to_out\.\d+)"
        r"|(?:audio_)?ff\.net\.\d+(?:\.proj)?"
        r")"
        r"|(?:audio|video)_embeddings_connector\.transformer_\d+d_blocks\.\d+\."
        r"(?:attn\d+\.(?:to_[qkv]|to_out\.\d+)|ff\.net\.\d+(?:\.proj)?)"
        r")\.weight$"
    ),
    "hunyuan_video_15": re.compile(
        r"^double_blocks\.\d+\.(?:(?:img|txt)_attn_(?:qkv|proj)|(?:img|txt)_mlp\.fc[12])\.weight$"
    ),
    "gemma": re.compile(
        r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    "qwen": re.compile(
        r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
}
HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32"}
TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


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


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError


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


def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(header_size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    header = {
        name: {"dtype": i["dtype"], "shape": list(i["shape"]), "data_offsets": list(i["data_offsets"])}
        for name, i in raw.items()
    }
    return header, metadata


def detect_profile(path: Path, names: list[str]) -> str:
    name_set = set(names)
    # LTX-2.5 is the only architecture here with paired audio<->video cross-attention next to a
    # separate embeddings connector stack, so those two keys identify it without the file name.
    for prefix in ("", "model.diffusion_model."):
        if {f"{prefix}transformer_blocks.0.audio_to_video_attn.to_q.weight",
                f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"} <= name_set:
            return "ltx_2_5"
    if {"txt_in.individual_token_refiner.blocks.0.norm1.weight",
            "double_blocks.0.img_attn_qkv.weight"} <= name_set:
        return "hunyuan_video_15"
    lowered = path.name.lower()
    for profile in ("gemma", "qwen"):
        if profile in lowered and any(n.startswith("model.layers.") for n in names):
            return profile
    raise ValueError("auto-detection found no supported profile; pass --profile explicitly")


def selected_layers(header: dict, profile: str, group_size: int, convrot_groupsize: int) -> list[str]:
    pattern = PROFILE_PATTERNS[profile]
    out = []
    for name, info in header.items():
        shape = info["shape"]
        if not (pattern.fullmatch(name) and info["dtype"] in HIGH_PRECISION_DTYPES and len(shape) == 2):
            continue
        k = shape[1]
        # mirrors AsymW4A8Int8Layout.Params._validate_tensor_fields
        if k % 16 or k % group_size or k % convrot_groupsize:
            continue
        if group_size < 4 or (16 % group_size and group_size % 16):
            continue
        out.append(name)
    return out


def read_tensor(handle, start: int, size: int, dtype: str, shape: list[int]) -> torch.Tensor:
    handle.seek(start)
    raw = bytearray(size)
    view = memoryview(raw)
    position = 0
    while position < size:
        count = handle.readinto(view[position:])
        if not count:
            raise EOFError(f"unexpected end of source, {size - position} bytes short")
        position += count
    return torch.frombuffer(raw, dtype=TORCH_DTYPES[dtype]).reshape(shape)


def header_dtype(tensor: torch.Tensor) -> str:
    """Safetensors header dtype name for `tensor`.

    Both fp8 formats travel as raw U8 -- matching how comfy/ops.py reads weight_s_rel and
    weight_codebook back (module docstring above: "fp8 stored as u8"). Before this fix only
    weight_s_rel's float8_e4m3fn got that treatment in the write loop below; a float8_e5m2
    tensor would have hit `SAFETENSORS_DTYPE[torch.float8_e5m2]` -> KeyError, the exact gap
    as_bytes() existed to close and was never wired in to actually close (ticket 24).
    """
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return "U8"
    return SAFETENSORS_DTYPE[tensor.dtype]


def as_bytes(tensor: torch.Tensor) -> memoryview:
    """Raw byte view of `tensor`, ready for a file's .write().

    torch.Tensor.numpy() raises TypeError for bfloat16 and for both float8 dtypes --
    EXECUTED against this embedded torch (2026-08-21, python_embeded, CPU, no GPU touched):
    `torch.zeros(4, dtype=torch.bfloat16).numpy()` raises "Got unsupported ScalarType
    BFloat16", and float8_e5m2 fails the same way; viewing through int16 (bf16) or uint8
    (fp8) first, as done below, makes numpy() succeed. Every other dtype this converter
    writes (I8, U8, I16, F32, F16) already supports numpy() directly. Matches header_dtype()'s
    labelling above: fp8 -> uint8 bytes, bfloat16 keeps its own header entry but travels as
    int16 bytes.
    """
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.view(torch.uint8)
    elif tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return memoryview(tensor.numpy()).cast("B")


SAFETENSORS_DTYPE = {
    torch.int8: "I8", torch.uint8: "U8", torch.int16: "I16",
    torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
}


def copy_range(source_handle, output_handle, start: int, size: int) -> None:
    source_handle.seek(start)
    remaining = size
    while remaining:
        chunk = source_handle.read(min(16 * 1024**2, remaining))
        if not chunk:
            raise EOFError(f"unexpected end of source, {remaining} bytes short")
        output_handle.write(chunk)
        remaining -= len(chunk)


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    portable_root = Path(__file__).resolve().parent.parent
    output = (args.output or source.with_name(f"{source.stem}_w4a8.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    if output == source:
        raise SystemExit("Refusing to overwrite the source model")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"Refusing to overwrite existing output or sidecar: {output}")

    header, metadata = read_header(source)
    if metadata.get("_quantization_metadata"):
        raise SystemExit("Refusing to requantize a checkpoint that already has quantization metadata")
    # Comfy-Org and Lightricks ship per-layer `.comfy_quant` tensors with no `__metadata__` at all,
    # so the check above misses them entirely. Only the high-precision dtype filter in
    # selected_layers would catch it, and that is luck rather than a guard.
    inline_quant = sum(1 for name in header if name.endswith(".comfy_quant"))
    if inline_quant:
        raise SystemExit(f"Refusing to requantize: source carries {inline_quant} inline "
                         "'.comfy_quant' markers, so it is already quantized")
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
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise SystemExit(f"Refusing to overwrite stale partial output: {partial}")

    # Pass one: quantize every selected layer, holding only the small scale tensors in memory.
    selected_set = set(selected)
    quantized: dict[str, dict] = {}
    # Two-pass design: pass one fills `quantized` with every layer, pass two writes. The old
    # `largest * 3 + 2 GiB` is the streaming estimate from quant_w4a4.py and understates this by
    # the layer count -- measured at 2.375 GiB asked against 10.777 GiB accumulated.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _ram_guard import check, w4a8_bytes

    accumulated = sum(w4a8_bytes(*header[n]["shape"], args.group_size,
                                 codebook=not args.no_codebook) for n in selected)
    refusal = check(psutil.virtual_memory().available, accumulated, label="W4A8 conversion")
    if refusal:
        raise SystemExit(refusal)
    if shutil.disk_usage(output.parent).free < source.stat().st_size:
        raise SystemExit("Insufficient disk space")

    with source.open("rb") as source_handle:
        source_header_size = struct.unpack("<Q", source_handle.read(8))[0]
        data_start = 8 + source_header_size
        for index, name in enumerate(selected, 1):
            info = header[name]
            start, end = info["data_offsets"]
            weight = read_tensor(source_handle, data_start + start, end - start,
                                 info["dtype"], info["shape"]).to(device="cuda")
            qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
                weight,
                group_size=args.group_size,
                convrot_groupsize=args.convrot_groupsize,
                symmetric=True,
                scale_dtype=torch.float8_e4m3fn,
                codebook=not args.no_codebook,
                codebook_tensor=None,
                stochastic_rounding=0,
            )
            if correction is not None:
                raise SystemExit("symmetric=True returned a correction tensor; ComfyUI would drop it")
            quantized[name] = {
                "qdata": qdata.cpu().contiguous(),
                "s_rel": s_rel.cpu().contiguous(),
                "s_channel": s_channel.cpu().contiguous(),
                "codebook": None if codebook is None else codebook.cpu().contiguous(),
            }
            del weight, qdata, s_rel, s_channel
            torch.cuda.empty_cache()
            if index % 48 == 0 or index == len(selected):
                print(f"[{index}/{len(selected)}] quantized", flush=True)

    # Pass two: lay out the output header, then stream the file.
    layers = {
        name.removesuffix(".weight"): {
            "format": "asym_w4a8_int8",
            "group_size": args.group_size,
            "convrot_groupsize": args.convrot_groupsize,
        }
        for name in selected
    }
    output_metadata = dict(metadata)
    output_metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": layers}, separators=(",", ":"))
    output_metadata["quantization"] = "asym_w4a8_int8"

    target = {"__metadata__": output_metadata}
    offset = 0
    plan = []
    for name, info in header.items():
        if name in selected_set:
            base = name.removesuffix(".weight")
            entry = quantized[name]
            pieces = [(f"{name}", entry["qdata"]),
                      (f"{base}.weight_s_rel", entry["s_rel"]),
                      (f"{base}.weight_s_channel", entry["s_channel"])]
            if entry["codebook"] is not None:
                pieces.append((f"{base}.weight_codebook", entry["codebook"]))
            for key, tensor in pieces:
                nbytes = tensor.numel() * tensor.element_size()
                target[key] = {"dtype": header_dtype(tensor),
                               "shape": list(tensor.shape),
                               "data_offsets": [offset, offset + nbytes]}
                plan.append(("write", tensor))
                offset += nbytes
        else:
            start, end = info["data_offsets"]
            size = end - start
            target[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + size]}
            plan.append(("copy", (start, size)))
            offset += size

    payload = json.dumps(target, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * (-len(payload) % 8)

    try:
        with source.open("rb") as source_handle, partial.open("xb") as out_handle:
            source_header_size = struct.unpack("<Q", source_handle.read(8))[0]
            data_start = 8 + source_header_size
            out_handle.write(struct.pack("<Q", len(payload)))
            out_handle.write(payload)
            body_start = out_handle.tell()
            for kind, item in plan:
                if kind == "write":
                    out_handle.write(as_bytes(item))
                else:
                    start, size = item
                    copy_range(source_handle, out_handle, data_start + start, size)
            written = out_handle.tell() - body_start
            if written != offset:
                raise RuntimeError(f"length mismatch: wrote {written}, planned {offset}")
            out_handle.flush()
            os.fsync(out_handle.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()

    elapsed = time.perf_counter() - started
    manifest = {
        "source": str(source), "source_size": source.stat().st_size,
        "output": str(output), "output_size": output.stat().st_size,
        "architecture": profile, "quantization": "asym_w4a8_int8",
        "layout": "AsymW4A8Int8Layout",
        "backend": backend["resolved"]["quantize_w4a8_int8_weight"],
        "group_size": args.group_size, "convrot_groupsize": args.convrot_groupsize,
        "symmetric": True, "codebook": not args.no_codebook,
        "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "conversion_seconds": round(elapsed, 3),
    }
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({human_size(output.stat().st_size)}) in {elapsed:.1f} s")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
