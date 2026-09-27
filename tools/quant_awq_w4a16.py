"""Native W4A16 (Q4_1 codes in comfy-kitchen's AWQ W4A16 layout) from a BF16 safetensors DiT.

    python_embeded\\python.exe -s tools\\quant_awq_w4a16.py --input <bf16.safetensors> [--profile qwen_image21] [--dry-run]

Each Linear weight of the profile is quantized from FP32 with gguf-py's Q4_1 (uint4, one fp16 scale d and min m per
32 weights, the same codes as a GGUF Q4_1) and stored as ComfyUI format `awq_w4a16`:

    weight        int8 [N, K/2]   two uint4 per byte, low nibble = even column
    weight_scale  bf16 [K/32, N]  d
    weight_zeros  bf16 [K/32, N]  m + 8 d      so that W = (q - 8) * scale + zeros

Activations stay BF16. Needs the local ComfyUI patch that registers `awq_w4a16` and, for speed at DiT batch sizes,
the comfy-kitchen patch with the fused Triton dequant. Output goes beside the source as `<stem>_q4_1_awq.safetensors`
(+ `.quant.json`), written to `.partial` and renamed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402
from quant_w4a8 import HIGH_PRECISION_DTYPES, PROFILE_PATTERNS, detect_profile, read_tensor  # noqa: E402

G = 32


def q4_1_awq(weight: torch.Tensor):
    import gguf

    n, k = weight.shape
    x = weight.to(torch.float32).numpy()
    blocos = gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q4_1).reshape(n, k // G, 20)
    d = blocos[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    m = blocos[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    qs = blocos[..., 4:20]
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(n, k)  # ggml: low nibbles hold j, high hold j + 16
    packed = (vals[:, 0::2] | (vals[:, 1::2] << 4)).astype(np.uint8)
    scale = torch.from_numpy(d.T.copy()).to(torch.bfloat16)
    zeros = torch.from_numpy((m + 8.0 * d).T.copy()).to(torch.bfloat16)
    deq = ((torch.from_numpy(vals).view(n, k // G, G).float() - 8.0) * scale.float().t().unsqueeze(-1)
           + zeros.float().t().unsqueeze(-1)).view(n, k)
    rel = float((deq - weight.float()).norm() / weight.float().norm())
    return torch.from_numpy(packed.view(np.int8)), scale, zeros, rel


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
    pattern = PROFILE_PATTERNS[profile]
    selected = [n for n, i in header.items()
                if pattern.fullmatch(n) and i["dtype"] in HIGH_PRECISION_DTYPES and len(i["shape"]) == 2
                and i["shape"][1] % G == 0]
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")
    quant_bytes = sum(header[n]["shape"][0] * header[n]["shape"][1] * (1 / 2 + 4 / G) for n in selected)
    print(f"Source: {source}\nProfile: {profile}  Q4_1 -> awq_w4a16 (group {G})  quantized={len(selected)}  "
          f"kept={len(header) - len(selected)}\nOutput: {output}")
    if args.dry_run:
        return 0
    conv.guard(int(quant_bytes) + source.stat().st_size // 4, accumulated=int(quant_bytes), label="AWQ W4A16 conversion")

    started = time.perf_counter()
    feitos, erros = {}, {}
    with source.open("rb") as handle:
        start0 = 8 + struct.unpack("<Q", handle.read(8))[0]
        for index, name in enumerate(selected, 1):
            info = header[name]
            a, b = info["data_offsets"]
            weight = read_tensor(handle, start0 + a, b - a, info["dtype"], info["shape"])
            q, scale, zeros, erros[name] = q4_1_awq(weight)
            feitos[name] = (q, scale, zeros)
            if index % 48 == 0 or index == len(selected):
                print(f"[{index}/{len(selected)}] quantized", flush=True)

    entradas = []
    for name, info in header.items():
        if name not in feitos:
            entradas.append(C.plan_copy(name, info))
            continue
        q, scale, zeros = feitos[name]
        base = name.removesuffix(".weight")
        entradas += [C.plan_write(name, q), C.plan_write(f"{base}.weight_scale", scale),
                     C.plan_write(f"{base}.weight_zeros", zeros)]
    layers = {n.removesuffix(".weight"): {"format": "awq_w4a16", "group_size": G} for n in selected}
    novo = dict(metadata)
    novo["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": layers}, separators=(",", ":"))
    novo["quantization"] = "awq_w4a16 (Q4_1 codes, group 32)"
    conv.commit(entradas, novo)
    rel = sorted(erros.values())
    conv.write_sidecar({
        "source": str(source), "source_size": source.stat().st_size,
        "output": str(output), "output_size": output.stat().st_size,
        "architecture": profile, "quantization": "awq_w4a16 from gguf-py Q4_1 codes", "group_size": G,
        "quantized_tensors": len(selected), "preserved_tensors": len(header) - len(selected),
        "weight_rel_rmse_mean": sum(rel) / len(rel), "weight_rel_rmse_max": rel[-1],
        "gguf_version": importlib.metadata.version("gguf"), "torch_version": torch.__version__,
        "conversion_seconds": round(time.perf_counter() - started, 3),
    })
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}); weight rel-RMSE mean {sum(rel) / len(rel):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
