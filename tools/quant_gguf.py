"""Weight-only GGUF (W8A16 / W4A16) from a BF16 safetensors DiT, for ComfyUI-GGUF's UnetLoaderGGUF.

    python_embeded\\python.exe -s tools\\quant_gguf.py --input <bf16.safetensors> --type Q8_0 [--profile qwen_image21]

    Q8_0  int8 weight, one fp16 scale per 32 values       == W8A16 (activations stay BF16)
    Q4_1  uint4 weight, fp16 scale + min per 32 values    == W4A16 (asymmetric)
    Q4_0  int4 weight, fp16 scale per 32 values           == W4A16 (symmetric)
    Q5_0  int5 weight, fp16 scale per 32 values

ComfyUI-GGUF dequantizes each weight to the compute dtype right before the matmul, so these are true
weight-only formats: less VRAM than BF16, no int8/int4 tensor-core math. Only the Linear weights of
the profile are quantized (the same 192 as our other Qwen-Image-2.1 builds); every other tensor is
kept in its source dtype. Quantization is done from FP32 by gguf-py (the reference numpy
implementation of llama.cpp's formats); K-quants are not implemented there and are refused.

Output goes beside the source as `<stem>_<TYPE>.gguf`, written to `.partial` and renamed; existing
outputs, sidecars and partials are refused.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402
from quant_w4a8 import PROFILE_PATTERNS, HIGH_PRECISION_DTYPES, detect_profile, read_header, read_tensor  # noqa: E402

TIPOS = {"Q8_0": (34, 32), "Q4_1": (20, 32), "Q4_0": (18, 32), "Q5_0": (22, 32)}  # bytes per block, block size
ARCH = {"qwen_image21": "qwen_image"}  # ComfyUI-GGUF's IMG_ARCH_LIST; the model itself is detected from the keys


def main() -> int:
    import gguf
    from gguf import GGMLQuantizationType as T

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--type", required=True, choices=list(TIPOS))
    ap.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    output = source.with_name(f"{source.stem}_{args.type}.gguf")
    sidecar = output.with_suffix(".quant.json")
    partial = output.with_name(output.name + ".partial")
    C.refuse_unsafe_output(source, output, sidecar)
    C.refuse_stale_partial(partial)

    header, metadata = read_header(source)
    C.refuse_already_quantized(header, metadata)
    profile = detect_profile(source, list(header)) if args.profile == "auto" else args.profile
    if profile not in ARCH:
        raise SystemExit(f"No GGUF architecture mapping for profile {profile!r}")
    pattern = PROFILE_PATTERNS[profile]
    bloco_bytes, bloco = TIPOS[args.type]
    selected = [n for n, i in header.items()
                if pattern.fullmatch(n) and i["dtype"] in HIGH_PRECISION_DTYPES and len(i["shape"]) == 2
                and i["shape"][1] % bloco == 0]
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")

    quant_bytes = sum(i["shape"][0] * i["shape"][1] // bloco * bloco_bytes for n, i in header.items() if n in selected)
    rest_bytes = sum(i["data_offsets"][1] - i["data_offsets"][0] for n, i in header.items() if n not in selected)
    print(f"Source: {source}\nProfile: {profile}  type={args.type}  quantized={len(selected)}  "
          f"kept={len(header) - len(selected)}\nOutput: {output} (~{C.human_size(quant_bytes + rest_bytes)})")
    if args.dry_run:
        return 0
    C.guard_disk(output.parent, quant_bytes + rest_bytes)
    # GGUFWriter holds every tensor until write_tensors_to_file.
    C.guard_ram(quant_bytes + rest_bytes, label="GGUF conversion")

    started = time.perf_counter()
    qtype = getattr(T, args.type)
    writer = gguf.GGUFWriter(path=None, arch=ARCH[profile])
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type({"Q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "Q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
                          "Q4_0": gguf.LlamaFileType.MOSTLY_Q4_0, "Q5_0": gguf.LlamaFileType.MOSTLY_Q5_0}[args.type])
    erros = {}
    selected_set = set(selected)
    with source.open("rb") as handle:
        start0 = 8 + struct.unpack("<Q", handle.read(8))[0]
        for index, (name, info) in enumerate(header.items(), 1):
            a, b = info["data_offsets"]
            t = read_tensor(handle, start0 + a, b - a, info["dtype"], info["shape"])
            if name in selected_set:
                x = t.to(torch.float32).numpy()
                q = gguf.quants.quantize(x, qtype)
                d = gguf.quants.dequantize(q, qtype)
                erros[name] = float(np.sqrt(((d - x) ** 2).mean() / max((x ** 2).mean(), 1e-30)))
                writer.add_tensor(name, q, raw_dtype=qtype)
                del x, d
            elif info["dtype"] == "BF16":
                writer.add_tensor(name, gguf.quants.quantize(t.to(torch.float32).numpy(), T.BF16), raw_dtype=T.BF16)
            elif info["dtype"] in ("F32", "F16"):
                writer.add_tensor(name, t.numpy())
            else:
                raise SystemExit(f"Unsupported source dtype {info['dtype']} for {name}")
            if index % 100 == 0 or index == len(header):
                print(f"[{index}/{len(header)}] tensors", flush=True)

    writer.write_header_to_file(path=partial)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    os.replace(partial, output)
    elapsed = time.perf_counter() - started
    rel = sorted(erros.values())
    sidecar.write_text(json.dumps({
        "source": str(source), "source_size": source.stat().st_size,
        "output": str(output), "output_size": output.stat().st_size,
        "architecture": profile, "gguf_arch": ARCH[profile],
        "quantization": f"gguf {args.type} (weight-only)",
        "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
        "weight_rel_rmse_mean": sum(rel) / len(rel), "weight_rel_rmse_max": rel[-1],
        "gguf_version": importlib.metadata.version("gguf"), "torch_version": torch.__version__,
        "conversion_seconds": round(elapsed, 3),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}) in {elapsed / 60:.1f} min; "
          f"weight rel-RMSE mean {sum(rel) / len(rel):.4f} max {rel[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
