"""Create ComfyUI ConvRot W4A4 Safetensors from a high-precision source.

Transmite: o header sai so das formas (`_formats.ConvrotW4A4.tensors`), e cada camada e lida,
quantizada e escrita dentro do laco de escrita do nucleo (`_conversion.Conversion.commit`), com o
`.quant.json` posto no lugar junto com o modelo. Perfis e selecao em `_profiles`.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402
from _native_probe import native_backend_ready  # noqa: E402
from _profiles import (  # noqa: E402,F401  (re-exportados: verificar_migracao e scripts antigos)
    EXCLUSIONS,
    HIGH_PRECISION_DTYPES,
    PROFILE_PATTERNS,
    detect_profile,
    select_layers,
)

CONVROT_GROUP_SIZE = 256
QUANT_GROUP_SIZE = 64
QUANTIZATION = "ConvRot W4A4"

read_header = C.read_header
human_size = C.human_size


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


def selected_layers(header: dict, profile: str, convrot_groupsize: int = CONVROT_GROUP_SIZE) -> list[str]:
    """Allowlist do perfil, `EXCLUSIONS` como segunda rede, e K divisivel pelo groupsize."""
    return select_layers(header, profile, F.ConvrotW4A4(convrot_groupsize).accepts)


def output_metadata(metadata: dict[str, str], layers: dict) -> dict[str, str]:
    """O `__metadata__` de saida. Separado do plano porque `verificar_migracao.py` precisa dele."""
    return F.quant_metadata(metadata, layers, QUANTIZATION)


def planejar(source_handle, header: dict, selected: list[str], ck,
             convrot_groupsize: int = CONVROT_GROUP_SIZE) -> list:
    """As entradas do nucleo, na ordem do header. Nada quantiza aqui.

    `plan_lazy` carrega dtype, forma e nbytes explicitamente, entao o plano -- e portanto o header
    inteiro -- sai sem tocar a GPU; os produtores so rodam dentro de `commit()`. E isso que permite
    `tools/verificar_migracao.py` conferir o layout desta ferramenta contra os arquivos que ela ja
    escreveu, sem placa nenhuma (e ele chama esta funcao com `ck=None`).
    """
    source_handle.seek(0)
    start = C.data_start(Path(source_handle.name))
    fmt = F.ConvrotW4A4(convrot_groupsize, QUANT_GROUP_SIZE)

    def peso(name: str) -> torch.Tensor:
        info = header[name]
        begin, end = info["data_offsets"]
        tensor = C.read_tensor(source_handle, start + begin, end - begin, info["dtype"], info["shape"])
        # FP32 na entrada (26/09), como quant_int8/quant_w4a8: a rotacao em BF16 perde precisao.
        return tensor.to(device="cuda", dtype=torch.float32)

    return F.plan_model(header, {name: fmt for name in selected}, peso, ck)


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

    # Recusas ANTES do dry-run e do preflight (revisao 2026-09-29, achado 10): o dry-run retornava
    # antes de `refuse_unsafe`, entao ensaiava um caminho que a execucao real recusaria -- saida
    # existente, partial obsoleto, fonte ja quantizada (os `.comfy_quant` inline da Comfy-Org e da
    # Lightricks, que antes davam o enganoso "profile selected no compatible layers").
    output.parent.mkdir(parents=True, exist_ok=True)
    conv = C.Conversion(source, output, sidecar)
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata

    if args.auto_detect and args.profile != "auto":
        print(f"--auto-detect overrides --profile {args.profile}")
    profile = (detect_profile(source, list(header))
               if args.auto_detect or args.profile == "auto" else args.profile)
    selected = selected_layers(header, profile, args.convrot_groupsize)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")
    fmt = F.ConvrotW4A4(args.convrot_groupsize, QUANT_GROUP_SIZE)
    formats = {name: fmt for name in selected}
    layers = F.layer_configs(formats)
    out_meta = output_metadata(metadata, layers)
    # O plano sai so das formas e nao quantiza nada; com ele o tamanho da saida e EXATO.
    estimated_size = C.Conversion.planned_size(F.plan_model(header, formats, None, None), out_meta)
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

    backend = native_backend_ready(portable_root, w4a4_probe_ops(args.convrot_groupsize))
    if not backend["native_ready"]:
        raise SystemExit(
            "Refusing conversion: normal ComfyUI resolves ConvRot to a non-CUDA backend. "
            + ", ".join(f"{op}={module}"
                        for op, module in sorted(backend["resolved"].items()))
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    import comfy_kitchen as ck

    started = time.perf_counter()
    # Transmite, entao a RAM que importa e o maior tensor sozinho, nao a soma: tres copias dele
    # (bytes lidos, tensor de origem, resultado) mais a folga que a propria guarda adiciona.
    conv.guard(estimated_size, accumulated=F.streaming_peak(header, selected),
               label="quant_w4a4 (streaming)")

    def progresso(indice: int, total: int, chave: str) -> None:
        if chave.endswith(".weight_scale"):
            return
        print(f"[{indice}/{total}] {chave}", flush=True)

    def manifesto() -> dict:
        return {
            "source": str(source),
            "source_size": source.stat().st_size,
            "output": str(output),
            "output_size": conv.output_size,
            "architecture": profile,
            "quantization": QUANTIZATION,
            "convrot_groupsize": args.convrot_groupsize,
            "layout": "TensorCoreConvRotW4A4Layout",
            "backend": backend["resolved"]["convrot_w4a4_linear"],
            "expected_kernel": "native INT4 MMA",
            "weight_storage_dtype": "INT8 packed signed INT4",
            "activation_input_dtype": "BF16/FP16; dynamically rotated and quantized to INT4 in kernel",
            "quantized_tensors": len(selected),
            "preserved_tensors": len(header) - len(selected),
            "preserved_dtype": "original BF16/FP16/FP32",
            "excluded_patterns": list(EXCLUSIONS.get(profile, ())),
            **version_info(portable_root),
            "conversion_seconds": round(time.perf_counter() - started, 3),
        }

    with source.open("rb") as source_handle:
        entradas = planejar(source_handle, header, selected, ck, args.convrot_groupsize)
        conv.commit(entradas, out_meta, progress=progresso, sidecar=manifesto)

    print(f"Wrote {output} ({human_size(output.stat().st_size)})")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
