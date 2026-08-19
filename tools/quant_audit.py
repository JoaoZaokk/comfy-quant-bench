"""Inventory model files without materializing their tensor data."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


MODEL_EXTENSIONS = {
    ".safetensors", ".gguf", ".ckpt", ".bin", ".pt", ".pth",
    ".onnx", ".engine", ".tflite", ".pkl", ".pickle",
}
HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32", "FLOAT16", "FLOAT32", "BFLOAT16"}
LOW_PRECISION_DTYPES = {"F8_E4M3", "F8_E5M2", "I8", "U8", "INT8", "UINT8"}
ROLE_BY_DIRECTORY = {
    "diffusion_models": "diffusion",
    "diffusion_models_gguf": "diffusion",
    "unet": "diffusion",
    "checkpoints": "checkpoint",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "vae": "vae",
    "controlnet": "controlnet",
    "loras": "lora",
    "embeddings": "embedding",
    "latent_upscale_models": "latent_upscaler",
    "clip_vision": "vision_encoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("ComfyUI/models"))
    parser.add_argument("--json", type=Path, default=Path("quantization_inventory.json"))
    parser.add_argument("--markdown", type=Path, default=Path("quantization_inventory.md"))
    return parser.parse_args()


def human_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError


def tensor_bytes(shape: list[int] | tuple[int, ...], bytes_per_element: float) -> int:
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return int(elements * bytes_per_element)


def infer_role(relative_path: Path) -> str:
    first = relative_path.parts[0].lower() if relative_path.parts else "unknown"
    if first in ROLE_BY_DIRECTORY:
        return ROLE_BY_DIRECTORY[first]
    if "vae" in first:
        return "vae"
    if "lora" in first:
        return "lora"
    if "text" in first or first == "llm":
        return "text_encoder"
    return first


def infer_architecture(path: Path, keys: list[str], metadata: dict[str, str]) -> str:
    # Structural checks first. HunyuanVideo also uses double_blocks, so the "flux" needle below
    # would otherwise claim it; this mirrors the HunyuanVideo branch of comfy.model_detection.
    key_set = set(keys)
    if "txt_in.individual_token_refiner.blocks.0.norm1.weight" in key_set:
        return "hunyuan_video"

    text = " ".join([path.name, *keys[:200], *map(str, metadata.values())]).lower()
    patterns = (
        ("ltx", ("ltx", "ltxv")),
        ("wan", ("wan2", "wan.")),
        ("z-image", ("z_image", "z-image", "zimage")),
        ("qwen", ("qwen",)),
        ("gemma", ("gemma",)),
        ("t5", ("t5xxl", "umt5", "t5.")),
        ("hidream", ("hidream",)),
        ("ace-step", ("ace_step", "ace-step")),
        ("void", ("void_pass",)),
        ("hunyuan", ("hunyuan",)),
        ("seedvr2", ("seedvr2",)),
        ("flux", ("flux", "double_blocks")),
        ("stable-diffusion-xl", ("sd_xl", "sdxl", "conditioner.embedders.1")),
        ("stable-diffusion", ("diffusion_model.input_blocks", "model.diffusion_model")),
    )
    for architecture, needles in patterns:
        if any(needle in text for needle in needles):
            return architecture
    return "unknown"


# How many bits one weight really occupies in each format this inventory can recognise. The
# distinction that matters: `convrot_w4a4`, `asym_w4a8_int8` and SVDQuant all store 4-bit weights
# inside an INT8 container, so the *dtype* on disk is I8 and the *precision* is 4. Reading only
# the dtype -- which is what this file used to do for anything but its own outputs -- reports
# double the real bits and makes an already-int4 checkpoint look like a compression candidate.
FORMAT_WEIGHT_BITS = {
    "convrot_w4a4": 4,
    "asym_w4a8_int8": 4,
    "svdquant_w4a4": 4,
    "svdquant_w4a8": 4,
    "svdquant_w4a16": 4,
    "svdquant_w4": 4,
    "int8_tensorwise": 8,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "nvfp4": 4,
    "mxfp8": 8,
}


def read_quant_dialects(raw_metadata: object) -> tuple[list[str], dict[str, str]]:
    """Read every metadata dialect that declares quantization, not just ComfyUI's.

    Three are in use on this machine and only the first was being read:

      `_quantization_metadata`  ComfyUI / this project's converters. JSON, per-layer `format`.
      `quantization_config`     nunchaku SVDQuant. JSON, and it states the widths outright:
                                {"method":"svdquant","weight":{"dtype":"int4",...},
                                 "activation":{"dtype":"int4"},"rank":32}
      `quantization.bits`       modelspec/DaSiWa free text, e.g. "INT8 Row-wise ConvRot
                                (runtime)". Not parseable with any confidence, so it is recorded
                                verbatim and never turned into a format.

    Returns (formats, declared) where `declared` carries what was found but not interpreted.
    """
    if not isinstance(raw_metadata, dict):
        return [], {}
    formats: list[str] = []
    declared: dict[str, str] = {}

    raw_quant = raw_metadata.get("_quantization_metadata")
    if raw_quant:
        try:
            quant_meta = json.loads(raw_quant)
            formats += [
                str(layer.get("format"))
                for layer in quant_meta.get("layers", {}).values()
                if isinstance(layer, dict) and layer.get("format")
            ]
        except (TypeError, json.JSONDecodeError):
            formats.append("unparsed_quantization_metadata")

    raw_config = raw_metadata.get("quantization_config")
    if raw_config:
        try:
            config = json.loads(raw_config)
            method = str(config.get("method", "")).lower()
            weight_dtype = str((config.get("weight") or {}).get("dtype", "")).lower()
            act_dtype = str((config.get("activation") or {}).get("dtype", "")).lower()
            if method and weight_dtype:
                # Name it WxAy so it lines up with FORMAT_WEIGHT_BITS. Calling an int4-weight /
                # bf16-activation file "w4a4" would overstate what it does, and naming it after
                # the weight dtype alone ("svdquant_int4") produced a key the bits table did not
                # have, so effective_weight_bits came back None for exactly the files this
                # dialect exists to describe.
                bits = {"int4": 4, "int8": 8, "fp8": 8, "float8": 8,
                        "bf16": 16, "bfloat16": 16, "fp16": 16, "float16": 16}
                w_bits = bits.get(weight_dtype)
                a_bits = bits.get(act_dtype)
                if w_bits and a_bits:
                    formats.append(f"{method}_w{w_bits}a{a_bits}")
                elif w_bits:
                    formats.append(f"{method}_w{w_bits}")
                else:
                    formats.append(f"{method}_{weight_dtype}")
            elif method:
                formats.append(method)
            declared["quantization_config"] = raw_config[:200]
        except (TypeError, json.JSONDecodeError):
            formats.append("unparsed_quantization_config")

    for key in ("quantization.bits", "quantization.tool"):
        value = raw_metadata.get(key)
        if value:
            declared[key] = str(value)[:200]

    return sorted(set(formats)), declared


def effective_weight_bits(quant_formats: list[str], dtype_bytes: dict[str, int]) -> int | None:
    """Bits per weight for the quantized part, or None when nothing declared it.

    Deliberately derived from the declared format and never from the dtype: the dtype is the
    container. `dtype_counts` and `dtype_bytes` stay untouched -- they are true statements about
    the bytes on disk, and `dtype_bytes` sums to the file size, so falsifying them would break
    that accounting.
    """
    # No format declared and no low-precision container in the file: nothing is packed, so the
    # dtype IS the weight width and reporting it is safe. This is the narrow case where deriving
    # bits from the dtype cannot be wrong -- there is no container to be fooled by. Everything
    # else falls through to the declared format, or to None.
    if not quant_formats:
        present = {d for d, b in (dtype_bytes or {}).items() if b}
        # I8 and U8 are the two dtypes used as *containers* here: I8 holds packed int4
        # (convrot_w4a4, asym_w4a8_int8, SVDQuant) and U8 holds packed fp4 (nvfp4) or a
        # reinterpreted fp8. With one of those present and nothing declaring what is inside,
        # the width is genuinely unknown and must stay unknown.
        #
        # FP8 is not in that set. F8_E4M3 and F8_E5M2 store one weight per byte -- there is no
        # packing scheme in this ecosystem that hides narrower weights inside them -- so a file
        # whose only low-precision dtype is fp8 has a knowable width of 8.
        ambiguous = {"I8", "U8", "INT8", "UINT8"}
        widths = {"F32": 32, "FLOAT32": 32, "BF16": 16, "BFLOAT16": 16,
                  "F16": 16, "FLOAT16": 16, "F64": 64, "F8_E4M3": 8, "F8_E5M2": 8}
        if present and not (present & ambiguous):
            known = [widths[d] for d in present if d in widths]
            if not known:
                return None
            # The *narrowest* is the answer when fp8 is present: the fp32/bf16 tensors in a
            # "_fp8_scaled" checkpoint are its scales and norms, not its weights, and reporting
            # 32 because a scale is fp32 would describe the wrong tensors entirely.
            return min(known) if present & {"F8_E4M3", "F8_E5M2"} else max(known)
        return None

    known = []
    for name in quant_formats:
        if name in FORMAT_WEIGHT_BITS:
            known.append(FORMAT_WEIGHT_BITS[name])
            continue
        # Any `<method>_w<N>a<M>` name carries its own answer. Without this, every method the
        # table has not been taught yet -- awq, gptq, whatever ships next -- silently reports
        # None, which is the same "unknown looks like unquantized" failure the table was added
        # to fix, just moved one level up.
        match = re.search(r"_w(\d+)(?:a(\d+))?$", name)
        if match:
            known.append(int(match.group(1)))
    if not known:
        return None
    # A mixed file (this project writes convrot_w4a4 + asym_w4a8_int8 in one checkpoint) is
    # reported at its widest, which is the honest bound on "how compressed is this already".
    return max(known)


def classify_precision(dtype_bytes: dict[str, int], quant_formats: list[str], file_format: str) -> str:
    if file_format == "gguf":
        quantized = [(dtype, size) for dtype, size in dtype_bytes.items() if dtype not in {"F32", "F16", "BF16"}]
        if quantized:
            dominant = max(quantized, key=lambda item: item[1])[0]
            return f"GGUF {dominant}"
        return "GGUF unquantized"
    if quant_formats:
        return "+".join(sorted(set(quant_formats)))
    total = sum(dtype_bytes.values()) or 1
    ordered = sorted(dtype_bytes.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return "unknown"
    dominant, dominant_bytes = ordered[0]
    if dominant_bytes / total >= 0.9:
        return dominant
    return "mixed " + "/".join(dtype for dtype, _ in ordered[:3])


def safe_metadata(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    result = {}
    for key, value in metadata.items():
        text = str(value)
        result[str(key)] = text if len(text) <= 500 else text[:497] + "..."
    return result


def inspect_safetensors(path: Path) -> dict:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError("truncated safetensors size header")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size <= 2 or header_size > min(path.stat().st_size - 8, 1024**3):
            raise ValueError(f"invalid safetensors header size: {header_size}")
        header = json.loads(handle.read(header_size))

    raw_metadata = header.get("__metadata__", {})
    metadata = safe_metadata(raw_metadata)
    tensors = [(name, value) for name, value in header.items() if name != "__metadata__"]
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: defaultdict[str, int] = defaultdict(int)
    linear_candidates = []
    for name, info in tensors:
        dtype = str(info.get("dtype", "unknown"))
        offsets = info.get("data_offsets", [0, 0])
        stored_bytes = int(offsets[1]) - int(offsets[0])
        shape = tuple(int(dim) for dim in info.get("shape", []))
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += stored_bytes
        if len(shape) == 2 and dtype in HIGH_PRECISION_DTYPES and shape[1] % 64 == 0:
            linear_candidates.append({
                "name": name,
                "shape": list(shape),
                "stored_bytes": stored_bytes,
                "default_group_256_compatible": shape[1] % 256 == 0,
            })

    quant_formats, quant_declared = read_quant_dialects(raw_metadata)

    linear_candidates.sort(key=lambda item: item["stored_bytes"], reverse=True)
    return {
        "format": "safetensors",
        "tensor_count": len(tensors),
        "dtype_counts": dict(dtype_counts),
        "dtype_bytes": dict(dtype_bytes),
        "metadata": metadata,
        "quant_formats": sorted(set(quant_formats)),
        "quant_declared": quant_declared,
        "architecture": infer_architecture(path, [name for name, _ in tensors], metadata),
        "linear_candidate_count": len(linear_candidates),
        "linear_candidate_bytes": sum(item["stored_bytes"] for item in linear_candidates),
        "largest_linear_candidates": linear_candidates[:20],
    }


def decode_gguf_string(field) -> str | None:
    try:
        part = field.parts[field.data[0]]
        return part.tobytes().decode("utf-8", errors="replace")
    except (AttributeError, IndexError, TypeError):
        return None


def inspect_gguf(path: Path) -> dict:
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(path, "r")
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: defaultdict[str, int] = defaultdict(int)
    enum_names = {int(item.value): item.name for item in GGMLQuantizationType}
    for tensor in reader.tensors:
        dtype = enum_names.get(int(tensor.tensor_type), str(tensor.tensor_type))
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += int(tensor.n_bytes)
    architecture = None
    if "general.architecture" in reader.fields:
        architecture = decode_gguf_string(reader.fields["general.architecture"])
    return {
        "format": "gguf",
        "tensor_count": len(reader.tensors),
        "dtype_counts": dict(dtype_counts),
        "dtype_bytes": dict(dtype_bytes),
        "metadata": {"general.architecture": architecture} if architecture else {},
        "quant_formats": sorted(dtype for dtype in dtype_counts if dtype not in HIGH_PRECISION_DTYPES),
        "architecture": architecture or infer_architecture(path, [], {}),
        "linear_candidate_count": 0,
        "linear_candidate_bytes": 0,
        "largest_linear_candidates": [],
    }


def walk_tensors(value, prefix: str = ""):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix or "<root>", value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_tensors(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_tensors(child, f"{prefix}.{index}")


def inspect_torch_checkpoint(path: Path) -> dict:
    import torch

    try:
        loaded = torch.load(path, map_location="meta", weights_only=True, mmap=True)
    except RuntimeError as error:
        message = str(error)
        if "mmap can only be used" in message:
            loaded = torch.load(path, map_location="meta", weights_only=True)
        elif "TorchScript archives" in message:
            try:
                loaded = torch.jit.load(path, map_location="meta").state_dict()
            except NotImplementedError:
                loaded = torch.jit.load(path, map_location="cpu").state_dict()
        else:
            raise
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: defaultdict[str, int] = defaultdict(int)
    names = []
    linear_candidates = []
    bytes_per_dtype = {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.float8_e4m3fn: 1, torch.float8_e5m2: 1,
        torch.int8: 1, torch.uint8: 1,
    }
    for name, tensor in walk_tensors(loaded):
        dtype = str(tensor.dtype).replace("torch.", "").upper()
        stored_bytes = tensor_bytes(tuple(tensor.shape), bytes_per_dtype.get(tensor.dtype, tensor.element_size()))
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += stored_bytes
        names.append(name)
        if tensor.ndim == 2 and dtype in HIGH_PRECISION_DTYPES and tensor.shape[1] % 64 == 0:
            linear_candidates.append({
                "name": name,
                "shape": list(tensor.shape),
                "stored_bytes": stored_bytes,
                "default_group_256_compatible": tensor.shape[1] % 256 == 0,
            })
    linear_candidates.sort(key=lambda item: item["stored_bytes"], reverse=True)
    return {
        "format": "pytorch_checkpoint",
        "tensor_count": len(names),
        "dtype_counts": dict(dtype_counts),
        "dtype_bytes": dict(dtype_bytes),
        "metadata": {},
        "quant_formats": [],
        "quant_declared": {},
        "architecture": infer_architecture(path, names, {}),
        "linear_candidate_count": len(linear_candidates),
        "linear_candidate_bytes": sum(item["stored_bytes"] for item in linear_candidates),
        "largest_linear_candidates": linear_candidates[:20],
    }


def inspect_unknown_binary(path: Path) -> dict:
    return {
        "format": path.suffix.lower().lstrip("."),
        "tensor_count": None,
        "dtype_counts": {},
        "dtype_bytes": {},
        "metadata": {},
        "quant_formats": [],
        "quant_declared": {},
        "architecture": infer_architecture(path, [], {}),
        "linear_candidate_count": 0,
        "linear_candidate_bytes": 0,
        "largest_linear_candidates": [],
        "inspection_note": "container listed; tensor metadata reader not available",
    }


def assess_model(model: dict) -> None:
    role = model["role"]
    fmt = model["format"]
    dtype_bytes = model["dtype_bytes"]
    total_tensor_bytes = sum(dtype_bytes.values()) or 1
    high_ratio = sum(size for dtype, size in dtype_bytes.items() if dtype in HIGH_PRECISION_DTYPES) / total_tensor_bytes
    low_ratio = sum(size for dtype, size in dtype_bytes.items() if dtype in LOW_PRECISION_DTYPES) / total_tensor_bytes
    if fmt == "gguf":
        model["source_suitability"] = "no_gguf_quantized_source"
        model["w4a4_assessment"] = "DO NOT REQUANTIZE GGUF"
    elif role in {
        "vae", "lora", "embedding", "vision_encoder", "latent_upscaler",
        "model_patches", "tts", "chatterbox", "facerestore_models",
        "upscale_models", "audio_encoders", "geometry_estimation", "nsfw_detector",
    }:
        model["source_suitability"] = "low_priority_component"
        model["w4a4_assessment"] = "SKIP INITIAL PASS"
    elif model["quant_formats"] or low_ratio >= 0.2:
        model["source_suitability"] = "already_quantized"
        model["w4a4_assessment"] = "NO REQUANTIZE"
    elif high_ratio >= 0.8 and model["linear_candidate_count"] > 0:
        model["source_suitability"] = "high_precision_candidate"
        model["w4a4_assessment"] = "RECIPE + BACKEND REQUIRED"
    else:
        model["source_suitability"] = "review"
        model["w4a4_assessment"] = "REVIEW"
    model["current_precision"] = classify_precision(dtype_bytes, model["quant_formats"], fmt)
    model["effective_weight_bits"] = effective_weight_bits(model["quant_formats"], dtype_bytes)


def inspect_model(path: Path, models_root: Path) -> dict:
    relative = path.relative_to(models_root)
    model = {
        "path": str(path.resolve()),
        "relative_path": relative.as_posix(),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "size_human": human_size(path.stat().st_size),
        "extension": path.suffix.lower(),
        "role": infer_role(relative),
    }
    try:
        if path.suffix.lower() == ".safetensors":
            details = inspect_safetensors(path)
        elif path.suffix.lower() == ".gguf":
            details = inspect_gguf(path)
        elif path.suffix.lower() in {".ckpt", ".bin", ".pt", ".pth"}:
            details = inspect_torch_checkpoint(path)
        else:
            details = inspect_unknown_binary(path)
        model.update(details)
    except Exception as error:
        model.update(inspect_unknown_binary(path))
        model["inspection_error"] = f"{type(error).__name__}: {error}"
    assess_model(model)
    return model


def stack_snapshot(portable_root: Path) -> dict:
    import importlib.metadata
    import torch
    import comfy_kitchen as ck

    before = ck.list_backends()
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    sys.path.insert(0, str(portable_root / "ComfyUI"))
    try:
        import comfy.quant_ops  # noqa: F401
    finally:
        sys.argv = saved_argv
    after = ck.list_backends()
    selected = None
    if torch.cuda.is_available():
        weight = torch.empty((64, 64), device="cuda", dtype=torch.float16)
        implementation = ck.registry.get_implementation(
            "quantize_convrot_w4a4_weight",
            kwargs={
                "weight": weight,
                "convrot_groupsize": 64,
                "quant_group_size": 64,
                "stochastic_rounding": 0,
            },
        )
        selected = f"{implementation.__module__}.{implementation.__name__}"
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
        "backends_before_comfy_import": before,
        "backends_after_comfy_import": after,
        "convrot_quantizer_selected_after_comfy_import": selected,
        "native_convrot_ready": bool(selected and ".backends.cuda." in selected),
    }


def candidate_score(model: dict) -> float:
    if model["source_suitability"] != "high_precision_candidate":
        return -1
    role_weight = {"diffusion": 5, "checkpoint": 4, "text_encoder": 3, "controlnet": 2}.get(model["role"], 1)
    linear_ratio = model["linear_candidate_bytes"] / max(model["size_bytes"], 1)
    return model["size_bytes"] * (1 + role_weight * 0.05 + linear_ratio * 0.1)


def bits_cell(model: dict) -> str:
    """Bits per weight, or '?' -- never a number this tool did not actually derive.

    '?' is the honest rendering of "no dialect declared a format". Printing the container width
    here instead would reproduce the exact bug this column was added to fix.
    """
    bits = model.get("effective_weight_bits")
    return str(bits) if bits else "?"


def markdown_report(inventory: dict) -> str:
    stack = inventory["stack"]
    models = inventory["models"]
    candidates = sorted((model for model in models if candidate_score(model) >= 0), key=candidate_score, reverse=True)
    lines = [
        "# Quantization Inventory",
        "",
        f"Generated: `{inventory['generated_at']}`  ",
        f"Models root: `{inventory['models_root']}`  ",
        f"Files: **{len(models)}**; total size: **{human_size(inventory['total_size_bytes'])}**",
        "",
        "## Reading this file",
        "",
        "`dtype_counts` and `dtype_bytes` in the JSON describe the **container**, not the weight. "
        "A 4-bit format packs two weights per INT8 byte, so those fields report half the weight "
        "count and, if summed as if they were weights, double the real bit width. They are left "
        "that way on purpose: they are true statements about bytes on disk, and `dtype_bytes` "
        "sums to the file size. Falsifying them would break that accounting.",
        "",
        "The per-weight number is **`Bits`** below, and `effective_weight_bits` in the JSON. It "
        "is derived from the declared format, never from the dtype. Worked example from this "
        "inventory: `svdq-int4_r32-z-image-turbo.safetensors` reports `I8: 3008102400` bytes and "
        "`Bits: 4` -- those bytes hold twice 3008102400 weights, at 4 bits each.",
        "",
        "`Bits: ?` means the file carries a low-precision container (I8/U8/FP8) and **no** dialect "
        "this tool reads declared what is inside it. That is *unknown*, not *unquantized*: "
        "treating the two as the same is what reported a nunchaku INT4 checkpoint as INT8 until "
        "2026-08-19. A file with only high-precision dtypes and no markers is not ambiguous -- "
        "there is no container to hide anything -- so it reports its dtype width directly.",
        "",
        "## Installed W4A4 Stack",
        "",
        f"- Python: `{stack['python']}` (`{stack['python_executable']}`)",
        f"- Torch: `{stack['torch']}`; CUDA: `{stack['torch_cuda']}`",
        f"- GPU: `{stack['gpu']}`; compute capability: `{stack['compute_capability']}`",
        f"- comfy-kitchen: `{stack['comfy_kitchen']}`",
        f"- ConvRot implementation selected after normal ComfyUI import: `{stack['convrot_quantizer_selected_after_comfy_import']}`",
        f"- Native ConvRot ready: **{'YES' if stack['native_convrot_ready'] else 'NO'}**",
        "",
        "## Candidate Ranking",
        "",
    ]
    if not stack["native_convrot_ready"]:
        lines.append("> Conversion is blocked: normal ComfyUI startup selects the eager ConvRot implementation, not the CUDA Tensor Core backend.")
        lines.append("")
    lines.extend(["| Rank | Model | Size | Current | Bits | Role | Architecture | W4A4 |",
                  "|---:|---|---:|---|---:|---|---|---|"])
    for rank, model in enumerate(candidates, 1):
        lines.append(
            f"| {rank} | `{model['relative_path']}` | {model['size_human']} | {model['current_precision']} | "
            f"{bits_cell(model)} | {model['role']} | {model['architecture']} | {model['w4a4_assessment']} |"
        )
    lines.extend(["", "## Full Inventory (size descending)", "",
                  "| Model | Size | Format | Current | Bits | Type | Architecture | W4A4? |",
                  "|---|---:|---|---|---:|---|---|---|"])
    for model in models:
        note = model.get("inspection_error", model["w4a4_assessment"])
        lines.append(
            f"| `{model['relative_path']}` | {model['size_human']} | {model['format']} | {model['current_precision']} | "
            f"{bits_cell(model)} | {model['role']} | {model['architecture']} | {note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    models_root = args.models_root.resolve()
    if not models_root.is_dir():
        raise SystemExit(f"Models root not found: {models_root}")
    portable_root = models_root.parent.parent
    paths = sorted(
        (path for path in models_root.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    models = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.relative_to(models_root)}", flush=True)
        models.append(inspect_model(path, models_root))
    inventory = {
        # 2: added quant_declared and effective_weight_bits, and taught the reader the
        # nunchaku and modelspec metadata dialects. dtype_counts/dtype_bytes unchanged.
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "portable_root": str(portable_root),
        "models_root": str(models_root),
        "total_size_bytes": sum(model["size_bytes"] for model in models),
        "stack": stack_snapshot(portable_root),
        "models": models,
    }
    args.json.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown.write_text(markdown_report(inventory), encoding="utf-8")
    print(f"Wrote {args.json} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
