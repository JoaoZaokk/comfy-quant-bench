"""Create ComfyUI ConvRot W4A4 Safetensors from a high-precision source."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402
from _native_probe import native_backend_ready  # noqa: E402
from quant_w4a8 import is_qwen_image21  # noqa: E402


PROFILE_PATTERNS = {
    # Qwen-Image-2.1: the same regex `quant_w4a8.py` uses (derived from Comfy-Org's int8-convrot
    # selection, 192 Linears), so the W4A4 and W4A8 builds differ only in the activation path.
    "qwen_image21": re.compile(
        r"^transformer_blocks\.\d+\.(?:attn\.(?:to_[qkv]|to_out\.0)|img_mlp\.(?:gate_up|proj|gate_layer|out))\.weight$"
    ),
    "gemma": re.compile(
        r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    "qwen": re.compile(
        r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    # Qwen3-VL as a text encoder -- Krea 2's conditioner. Same decoder as `qwen`, one segment
    # deeper: `model.language_model.layers.N` because the checkpoint carries a vision tower beside
    # the language model. That extra segment is exactly why the `qwen` profile matches ZERO layers
    # here; checked against the header of qwen3vl_4b_bf16 before this entry was written, not
    # assumed from the family name.
    "qwen3vl": re.compile(
        r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    # HunyuanVideo 1.5 double-stream blocks. Names follow the checkpoint's own convention;
    # HunyuanVideo.process_unet_state_dict rewrites them to the ComfyUI module names after
    # convert_old_quants has injected the .comfy_quant keys, and its substring replacements
    # ("_attn_qkv." -> "_attn.qkv.", "mlp.fc1." -> "mlp.0.", ...) carry the injected
    # .comfy_quant and .weight_scale keys along with the weights.
    "hunyuan_video_15": re.compile(
        r"^double_blocks\.\d+\.(?:(?:img|txt)_attn_(?:qkv|proj)|(?:img|txt)_mlp\.fc[12])\.weight$"
    ),
    # LTX 2.3 / 2.5: the same regex `quant_w4a8.py` uses, so the W4A4 control arm quantizes
    # EXACTLY the 1440 layers the W4A8 build does and the two differ in activation path only.
    # Verified 2026-09-13 on both headers: 2.3 and 2.5 share 76 identical 2-D weight families and
    # this selects 1440/1772 on each. Works on the transformer-only file (2.5) and on the
    # single-file checkpoint (2.3, `model.diffusion_model.` prefix) alike.
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
}
EXCLUSIONS = {
    "qwen_image21": ["norm", "modulation", "img_in", "txt_in", "time_text_embed", "proj_out"],
    "gemma": ["embed_tokens", "norm", "lm_head", "vision"],
    "qwen": ["embed_tokens", "norm", "lm_head", "visual", "vision"],
    "qwen3vl": ["embed_tokens", "norm", "lm_head", "visual", "vision"],
    # adaLN modulation drives every block's conditioning, so *_mod.linear stays high precision
    # along with the norms, the embedders, the token refiner, and the byt5/vision/time adapters.
    "hunyuan_video_15": [
        "_mod.linear", "_norm", "norm", "img_in", "txt_in", "byt5_in", "vision_in",
        "time_in", "final_layer", "_embedding", "task_bias",
    ],
    # Second net for LTX: modulation, norms, patch/caption projections, the VAEs and the vocoder
    # of the single-file checkpoint. The allowlist already excludes all of them.
    "ltx_2_5": ["norm", "adaln_single", "scale_shift", "proj_in", "proj_out", "caption_projection",
                "patchify", "vae.", "audio_vae.", "vocoder.", "text_embedding_projection", "to_gate_logits"],
}
HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32"}
TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}
CONVROT_GROUP_SIZE = 256
QUANT_GROUP_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["auto", *PROFILE_PATTERNS], default="auto")
    # `args.auto_detect` was never read: detection is driven purely by `args.profile == "auto"`,
    # which is already the default. As a flag it was a no-op in the common case and a lie when
    # combined with an explicit `--profile gemma`, where it claimed to force detection and did
    # not. Now it actually forces it, and says so when it overrides an explicit profile.
    parser.add_argument("--auto-detect", action="store_true",
                        help="force profile auto-detection, overriding an explicit --profile")
    parser.add_argument("--dry-run", action="store_true", help="Plan the conversion without writing tensors")
    parser.add_argument(
        "--convrot-groupsize",
        type=int,
        default=CONVROT_GROUP_SIZE,
        help="Hadamard rotation group size. Smaller means a finer rotation and less quantization "
             "error, at the cost of more scale bookkeeping. The loader reads this back out of the "
             "sidecar metadata, so a file converted at 64 is executed at 64.",
    )
    return parser.parse_args()


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError


def w4a4_probe_ops(convrot_groupsize: int) -> dict:
    """The two ops this converter dispatches, at the groupsize it is about to convert at.

    This used to be a private `normal_comfy_backend()` here, one of six across `tools/`, and it
    hardcoded `convrot_groupsize: 64, quant_group_size: 64` while `--convrot-groupsize` (default
    256) went into the real call at write_streamed_checkpoint(). So the constraint validation ran
    against a configuration the conversion did not use.

    MEASURED 2026-08-22 on the 3090: that mismatch changed nothing -- cg=64 and cg=256 both
    resolve to `comfy_kitchen.backends.cuda` for both ops, and both real calls succeed. The old
    preflight was untidy, not wrong, and this is not a bug fix. It is passing the real value
    because there is no reason to pass a different one.
    """
    return {
        "quantize_convrot_w4a4_weight": {"convrot_groupsize": convrot_groupsize},
        "convrot_w4a4_linear": {"convrot_groupsize": convrot_groupsize},
    }


def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        raw_header = json.loads(handle.read(header_size))
    metadata = dict(raw_header.pop("__metadata__", {}) or {})
    header = {
        name: {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "data_offsets": list(info["data_offsets"]),
        }
        for name, info in raw_header.items()
    }
    return header, metadata


def detect_profile(path: Path, names: list[str]) -> str:
    name_set = set(names)
    # Structural first: this mirrors the HunyuanVideo branch of comfy.model_detection, so the
    # profile is derived from the checkpoint rather than from its file name.
    if {
        "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        "double_blocks.0.img_attn_qkv.weight",
    } <= name_set:
        return "hunyuan_video_15"
    if is_qwen_image21(name_set):
        return "qwen_image21"
    lowered = path.name.lower()
    if "gemma" in lowered and any(name.startswith("model.layers.") for name in names):
        return "gemma"
    # `qwen3vl` before `qwen`: the VL checkpoint has no `model.layers.` at all, so the order only
    # matters if a future file carries both, and then the deeper name is the right answer.
    if "qwen" in lowered and any(name.startswith("model.language_model.layers.") for name in names):
        return "qwen3vl"
    if "qwen" in lowered and any(name.startswith("model.layers.") for name in names):
        return "qwen"
    raise ValueError("auto-detection found no supported profile; pass a supported --profile after verifying the architecture")


def selected_layers(header: dict, profile: str, convrot_groupsize: int = CONVROT_GROUP_SIZE) -> list[str]:
    """Allowlist first, then EXCLUSIONS as a second, independent net.

    `EXCLUSIONS` used to be dead: its only reference in the file was the `excluded_patterns`
    field of the manifest, so every `.quant.json` this project has written documents a safety net
    that was never strung. Exclusion happened purely as a side effect of the allowlist not
    matching -- which is fine while the allowlist is tight, and is exactly the wrong thing to
    rely on when someone writes a looser `PROFILE_PATTERNS` for one of the four architectures
    CLAUDE.md still lists as needing a recipe (Flux, Hunyuan, SeedVR2, Z-Image).

    Applying it changes nothing for the profiles that exist today -- verified: gemma still
    selects 328/328 and ltx_2_5 still selects 1440/1440 against the shipped markers -- because
    the allowlist already excludes everything the list names. That is the point: the net should
    be there before it is needed, not after.
    """
    pattern = PROFILE_PATTERNS[profile]
    excluded = EXCLUSIONS.get(profile, [])
    selected = []
    for name, info in header.items():
        shape = info["shape"]
        if (
            pattern.fullmatch(name)
            and info["dtype"] in HIGH_PRECISION_DTYPES
            and len(shape) == 2
            and shape[1] % convrot_groupsize == 0
            and not any(token in name for token in excluded)
        ):
            selected.append(name)
    return selected


def tensor_elements(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def estimate_output(header: dict, source_size: int, selected: list[str]) -> int:
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}
    estimate = source_size
    for name in selected:
        info = header[name]
        elements = tensor_elements(info["shape"])
        source_bytes = elements * dtype_bytes[info["dtype"]]
        packed_bytes = elements // 2
        scale_bytes = info["shape"][0] * 4
        estimate += packed_bytes + scale_bytes - source_bytes
    return estimate


def output_header(header: dict, metadata: dict[str, str], selected: list[str], layers: dict) -> tuple[dict, int]:
    selected_set = set(selected)
    quant_metadata = {"format_version": "1.0", "layers": layers}
    output_metadata = dict(metadata)
    output_metadata["_quantization_metadata"] = json.dumps(quant_metadata, separators=(",", ":"))
    output_metadata["quantization"] = "ConvRot W4A4"

    result = {"__metadata__": output_metadata}
    offset = 0
    for name, info in header.items():
        if name in selected_set:
            rows, columns = info["shape"]
            packed_bytes = rows * columns // 2
            result[name] = {
                "dtype": "I8",
                "shape": [rows, columns // 2],
                "data_offsets": [offset, offset + packed_bytes],
            }
            offset += packed_bytes
            scale_name = f"{name.removesuffix('.weight')}.weight_scale"
            scale_bytes = rows * 4
            result[scale_name] = {
                "dtype": "F32",
                "shape": [rows],
                "data_offsets": [offset, offset + scale_bytes],
            }
            offset += scale_bytes
        else:
            tensor_bytes = info["data_offsets"][1] - info["data_offsets"][0]
            result[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [offset, offset + tensor_bytes],
            }
            offset += tensor_bytes
    return result, offset


def encoded_header(header: dict) -> bytes:
    payload = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return payload + b" " * (-len(payload) % 8)


def copy_range(source_handle, output_handle, start: int, size: int) -> None:
    source_handle.seek(start)
    remaining = size
    while remaining:
        chunk = source_handle.read(min(16 * 1024**2, remaining))
        if not chunk:
            raise EOFError(f"Unexpected end of source with {remaining} bytes remaining")
        output_handle.write(chunk)
        remaining -= len(chunk)


def read_tensor_range(source_handle, start: int, size: int, dtype: str, shape: list[int]) -> tuple[torch.Tensor, bytearray]:
    source_handle.seek(start)
    raw = bytearray(size)
    view = memoryview(raw)
    position = 0
    while position < size:
        count = source_handle.readinto(view[position:])
        if not count:
            raise EOFError(f"Unexpected end of source with {size - position} bytes remaining")
        position += count
    tensor = torch.frombuffer(raw, dtype=TORCH_DTYPES[dtype]).reshape(shape)
    return tensor, raw


def write_tensor(output_handle, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    output_handle.write(memoryview(array).cast("B"))


def output_metadata(metadata: dict[str, str], layers: dict) -> dict[str, str]:
    """O `__metadata__` de saida. Separado do plano porque `verificar_migracao.py` precisa dele."""
    result = dict(metadata)
    result["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": layers},
                                                  separators=(",", ":"))
    result["quantization"] = "ConvRot W4A4"
    return result


def planejar(source_handle, header: dict, selected: list[str], ck,
             convrot_groupsize: int = CONVROT_GROUP_SIZE) -> list:
    """As entradas do nucleo, na MESMA ordem que `output_header()` produzia.

    Nada quantiza aqui. `plan_lazy` carrega dtype, forma e nbytes explicitamente, entao o plano --
    e portanto o header inteiro -- sai sem tocar a GPU; os produtores so rodam dentro de
    `commit()`. E isso que permite `tools/verificar_migracao.py` conferir o layout desta ferramenta
    contra os arquivos que ela ja escreveu, sem placa nenhuma.

    Cada camada selecionada vira DUAS entradas (`qdata`, depois `weight_scale`) e o kernel roda
    UMA vez: o produtor do peso guarda as escalas num dicionario que o produtor seguinte consome.
    Isso amarra os dois a ordem em que `commit()` os chama; o segundo levanta com uma mensagem
    explicita se for chamado fora de ordem, em vez de escrever lixo silenciosamente.
    """
    source_handle.seek(0)
    source_data_start = 8 + struct.unpack("<Q", source_handle.read(8))[0]
    selected_set = set(selected)
    entradas = []
    for name, info in header.items():
        if name not in selected_set:
            entradas.append(C.plan_copy(name, info))
            continue

        rows, columns = info["shape"]
        pendente: dict = {}

        def produz_peso(name=name, info=info, pendente=pendente):
            start, end = info["data_offsets"]
            source_tensor, raw = read_tensor_range(
                source_handle, source_data_start + start, end - start,
                info["dtype"], info["shape"])
            # FP32 na entrada (26/09), como quant_int8/quant_w4a8: a rotacao em BF16 perde precisao.
            weight = source_tensor.to(device="cuda", dtype=torch.float32)
            qdata, scales = ck.quantize_convrot_w4a4_weight(
                weight, convrot_groupsize=convrot_groupsize,
                quant_group_size=QUANT_GROUP_SIZE, stochastic_rounding=0)
            qdata = qdata.cpu().contiguous()
            scales = scales.cpu().contiguous()
            expected_q_shape = (info["shape"][0], info["shape"][1] // 2)
            if qdata.dtype != torch.int8 or tuple(qdata.shape) != expected_q_shape:
                raise RuntimeError(
                    f"Unexpected packed weight for {name}: {qdata.dtype} {tuple(qdata.shape)}")
            if scales.dtype != torch.float32 or tuple(scales.shape) != (info["shape"][0],):
                raise RuntimeError(f"Unexpected scales for {name}: {scales.dtype} {tuple(scales.shape)}")
            pendente["scales"] = scales
            del source_tensor, raw, weight
            torch.cuda.empty_cache()
            return qdata

        def produz_escala(name=name, pendente=pendente):
            if "scales" not in pendente:
                raise RuntimeError(
                    f"{name}: o produtor da escala rodou antes do produtor do peso. As duas "
                    "entradas desta camada tem que ser escritas em ordem pelo commit().")
            return pendente.pop("scales")

        entradas.append(C.plan_lazy(name, "I8", [rows, columns // 2],
                                    rows * columns // 2, produz_peso))
        entradas.append(C.plan_lazy(f"{name.removesuffix('.weight')}.weight_scale", "F32",
                                    [rows], rows * 4, produz_escala))
    return entradas


def version_info(portable_root: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(portable_root / "ComfyUI"), "rev-parse", "--short", "HEAD"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.strip()
    return {
        "comfy_version": commit or "unknown",
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    portable_root = Path(__file__).resolve().parent.parent
    output = (args.output or source.with_name(f"{source.stem}_w4a4_convrot.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")

    backend = None
    if not args.dry_run:
        backend = native_backend_ready(portable_root, w4a4_probe_ops(args.convrot_groupsize))
        if not backend["native_ready"]:
            raise SystemExit(
                "Refusing conversion: normal ComfyUI resolves ConvRot to a non-CUDA backend. "
                + ", ".join(f"{op}={module}"
                            for op, module in sorted(backend["resolved"].items()))
            )

    header, metadata = read_header(source)
    # Comfy-Org and Lightricks ship per-layer `.comfy_quant` tensors with no `__metadata__` at
    # all, so the check above misses them entirely. Real example on this machine:
    # `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` has metadata
    # ['gemma_config','format'] and 328 inline markers. Today the allowlist happens to select
    # nothing and it exits with the misleading "profile selected no compatible layers"; that luck
    # runs out on a partially quantized checkpoint, which is exactly what quant_mixed.py writes.
    if args.auto_detect and args.profile != "auto":
        print(f"--auto-detect overrides --profile {args.profile}")
    profile = (detect_profile(source, list(header))
               if args.auto_detect or args.profile == "auto" else args.profile)
    selected = selected_layers(header, profile, args.convrot_groupsize)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")
    estimated_size = estimate_output(header, source.stat().st_size, selected)
    print(f"Source: {source}")
    print(f"Profile: {profile}")
    print(f"Selected Linear weights: {len(selected)}")
    print(f"Estimated output: {human_size(estimated_size)}")
    print(f"Output: {output}")
    if args.dry_run:
        for name in selected[:20]:
            print(f"  {name}: {header[name]['shape']} {header[name]['dtype']}")
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} more")
        return 0


    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    import comfy_kitchen as ck

    started = time.perf_counter()
    layers = {
        name.removesuffix(".weight"): {
            "format": "convrot_w4a4",
            "convrot_groupsize": args.convrot_groupsize,
        }
        for name in selected
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    conv = C.Conversion(source, output, sidecar)
    conv.refuse_unsafe()
    # Transmite, entao a RAM que importa e o maior tensor sozinho, nao a soma: tres copias dele
    # (bytes lidos, tensor de origem, resultado) mais a folga que a propria guarda adiciona.
    largest_selected = max(header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
                           for name in selected)
    conv.guard(estimated_size, accumulated=largest_selected * 3, label="quant_w4a4 (streaming)")

    def progresso(indice: int, total: int, chave: str) -> None:
        if chave.endswith(".weight_scale"):
            return
        print(f"[{indice}/{total}] {chave}", flush=True)

    with source.open("rb") as source_handle:
        entradas = planejar(source_handle, header, selected, ck, args.convrot_groupsize)
        conv.commit(entradas, output_metadata(metadata, layers), progress=progresso)

    elapsed = time.perf_counter() - started
    versions = version_info(portable_root)
    manifest = {
        "source": str(source),
        "source_size": source.stat().st_size,
        "output": str(output),
        "output_size": output.stat().st_size,
        "architecture": profile,
        "quantization": "ConvRot W4A4",
        "convrot_groupsize": args.convrot_groupsize,
        "layout": "TensorCoreConvRotW4A4Layout",
        "backend": backend["resolved"]["convrot_w4a4_linear"],
        "expected_kernel": "native INT4 MMA",
        "weight_storage_dtype": "INT8 packed signed INT4",
        "activation_input_dtype": "BF16/FP16; dynamically rotated and quantized to INT4 in kernel",
        "quantized_tensors": len(selected),
        "preserved_tensors": len(header) - len(selected),
        "preserved_dtype": "original BF16/FP16/FP32",
        "excluded_patterns": EXCLUSIONS[profile],
        **versions,
        "conversion_seconds": round(elapsed, 3),
    }
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({human_size(output.stat().st_size)})")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
