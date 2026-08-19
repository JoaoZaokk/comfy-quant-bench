"""Rewrite a diffusers-named checkpoint under the names ComfyUI's modules actually carry.

Quantization cannot survive ComfyUI's diffusers conversion, and the reason is mechanical rather
than numerical. `comfy/model_detection.py:1498 convert_diffusers_mmdit` maps
`layers.N.attention.to_{q,k,v}.weight` into one fused `layers.N.attention.qkv.weight` by writing
each into a row range of the target. Per-row quantization survives that concatenation perfectly
well -- rows keep their own scales, and ConvRot rotates along columns. What does not survive is
everything *beside* the weight: `weight_scale`, `weight_s_rel`, `weight_s_channel` and the
`comfy_quant` marker are not in the map, and the z-image branch passes unmapped keys through
unchanged (`if k not in sd_map: sd_map[k] = k`). They would land under `...to_q.weight_scale`
while the module needing them is `...attention.qkv`, so the layer loads with no scale at all.

Converting the names first removes the problem instead of working around it: a natively-named
checkpoint is detected by `model_detection.py:555` directly and no remap runs.

Doing this as its own step, rather than folding it into the converter, is what makes it possible
to tell a remap bug from a quantization bug: the output here is still BF16, so it must generate
the *same image* as the source for the same seed. If it does not, nothing downstream is worth
measuring.

The plan is not hand-written. It is derived from `comfy.utils.z_image_to_diffusers` -- the same
map ComfyUI loads with -- and then checked against ComfyUI's own `convert_diffusers_mmdit` run on
meta tensors, which costs no memory and gives the exact key set and shapes the real loader would
produce. A single mismatch aborts.

    python_embeded\\python.exe -s tools/to_native.py \\
        --input ComfyUI/models/diffusion_models/beyond-reality-zimage-v2_bf16.safetensors \\
        --arch zimage --output D:/ComfyUI-Models/diffusion_models/beyond-reality-zimage-v2_native.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                "I8": torch.int8, "U8": torch.uint8}


def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    metadata = header.pop("__metadata__", {}) or {}
    return header, metadata


def build_map(arch: str, header: dict) -> dict:
    """Return {diffusers_key: native_key | (native_key, offset)} using ComfyUI's own builder."""
    import comfy.utils

    if arch == "zimage":
        layers = 0
        while f"layers.{layers}.attention.to_q.weight" in header:
            layers += 1
        dim = header["noise_refiner.0.attention.to_k.weight"]["shape"][0]
        sd_map = comfy.utils.z_image_to_diffusers({"n_layers": layers, "dim": dim},
                                                  output_prefix="")
        # convert_diffusers_mmdit does exactly this for the z-image branch, so anything the map
        # does not name keeps its name.
        for key in header:
            sd_map.setdefault(key, key)
        return sd_map
    raise SystemExit(f"No verified native remap for architecture {arch!r}")


def oracle_shapes(header: dict) -> dict[str, list[int]]:
    """What ComfyUI's own converter produces, computed on meta tensors at zero memory cost."""
    import comfy.model_detection

    fake = {name: torch.empty(info["shape"], dtype=torch.bfloat16, device="meta")
            for name, info in header.items()}
    produced = comfy.model_detection.convert_diffusers_mmdit(fake)
    if produced is None:
        raise SystemExit("ComfyUI's convert_diffusers_mmdit did not recognise this checkpoint, "
                         "so it is probably already in native naming")
    return {name: list(tensor.shape) for name, tensor in produced.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arch", choices=["zimage"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"No such file: {source}")
    if output == source:
        raise SystemExit("Refusing to overwrite the source model")
    if not args.dry_run and output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    header, metadata = read_header(source)
    if metadata.get("_quantization_metadata") or any(k.endswith(".comfy_quant") for k in header):
        raise SystemExit("Refusing: remap the high-precision checkpoint, not a quantized one")

    sd_map = build_map(args.arch, header)

    # native_key -> [(row_offset, source_key), ...] in the order they are laid out
    plan: dict[str, list[tuple[int | None, str]]] = {}
    for src, target in sd_map.items():
        if src not in header:
            continue
        if isinstance(target, str):
            plan.setdefault(target, []).append((None, src))
        else:
            if len(target) > 2 and target[2] is not None:
                raise SystemExit(f"{src}: the map carries a transform function, which this "
                                 "streaming remap cannot apply")
            native, offset = target[0], target[1]
            if offset[0] != 0:
                raise SystemExit(f"{src}: concatenation on axis {offset[0]}, not rows; "
                                 "the streaming writer only handles row concatenation")
            plan.setdefault(native, []).append((offset[1], src))

    resolved: dict[str, dict] = {}
    for native, pieces in plan.items():
        if len(pieces) == 1 and pieces[0][0] is None:
            info = header[pieces[0][1]]
            resolved[native] = {"dtype": info["dtype"], "shape": list(info["shape"]),
                                "sources": [pieces[0][1]]}
            continue
        if any(offset is None for offset, _ in pieces):
            raise SystemExit(f"{native}: mixes offset and whole-tensor sources")
        pieces.sort()
        dtypes = {header[src]["dtype"] for _, src in pieces}
        if len(dtypes) != 1:
            raise SystemExit(f"{native}: sources disagree on dtype {dtypes}")
        rest = header[pieces[0][1]]["shape"][1:]
        if any(header[src]["shape"][1:] != rest for _, src in pieces):
            raise SystemExit(f"{native}: sources disagree on trailing shape")
        rows = 0
        for offset, src in pieces:
            if offset != rows:
                raise SystemExit(f"{native}: source {src} starts at row {offset}, expected "
                                 f"{rows}; the layout is not contiguous")
            rows += header[src]["shape"][0]
        resolved[native] = {"dtype": dtypes.pop(), "shape": [rows, *rest],
                            "sources": [src for _, src in pieces]}

    expected = oracle_shapes(header)
    mine = {name: entry["shape"] for name, entry in resolved.items()}
    only_oracle = sorted(set(expected) - set(mine))
    only_mine = sorted(set(mine) - set(expected))
    wrong = sorted(n for n in mine if n in expected and mine[n] != expected[n])
    print(f"source keys {len(header)} -> native keys {len(resolved)} "
          f"(ComfyUI's own converter: {len(expected)})")
    if only_oracle or only_mine or wrong:
        for label, items in (("missing", only_oracle), ("extra", only_mine),
                             ("wrong shape", wrong)):
            for item in items[:5]:
                detail = f" {mine.get(item)} vs {expected.get(item)}" if label == "wrong shape" \
                    else ""
                print(f"  {label}: {item}{detail}")
        raise SystemExit("Refusing to write a remap that does not match ComfyUI's own conversion")
    print("plan matches ComfyUI's convert_diffusers_mmdit exactly")

    fused = {n: e for n, e in resolved.items() if len(e["sources"]) > 1}
    print(f"fused targets: {len(fused)}")
    for name in sorted(fused)[:3]:
        print(f"  {name} {resolved[name]['shape']} <- {fused[name]['sources']}")
    if args.dry_run:
        return 0

    target = {"__metadata__": dict(metadata)} if metadata else {}
    offset = 0
    order = []
    for name in sorted(resolved):
        entry = resolved[name]
        size = sum(header[src]["data_offsets"][1] - header[src]["data_offsets"][0]
                   for src in entry["sources"])
        target[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                        "data_offsets": [offset, offset + size]}
        order.append(name)
        offset += size

    payload = json.dumps(target, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise SystemExit(f"Refusing to overwrite stale partial output: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)

    chunk = 16 * 1024 * 1024
    try:
        with source.open("rb") as handle, partial.open("xb") as out:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            data_start = 8 + header_size
            out.write(struct.pack("<Q", len(payload)))
            out.write(payload)
            body_start = out.tell()
            for index, name in enumerate(order, 1):
                for src in resolved[name]["sources"]:
                    start, end = header[src]["data_offsets"]
                    handle.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        block = handle.read(min(chunk, remaining))
                        if not block:
                            raise RuntimeError(f"unexpected end of source reading {src}")
                        out.write(block)
                        remaining -= len(block)
                if index % 100 == 0 or index == len(order):
                    print(f"[{index}/{len(order)}] written", flush=True)
            written = out.tell() - body_start
            if written != offset:
                raise RuntimeError(f"length mismatch: wrote {written}, planned {offset}")
            out.flush()
            os.fsync(out.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()

    print(f"wrote {output} ({output.stat().st_size / 2**30:.2f} GiB)")
    print("This is still BF16. Generate with it at a fixed seed and compare against the source "
          "before quantizing anything from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
