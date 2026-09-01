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
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

# O contrato de escrita de oito partes, escrito uma vez. Esta ferramenta foi a PRIMEIRA a adotar
# o nucleo (2026-09-01) porque e a unica das seis que nao quantiza -- roda inteira sem GPU --, e
# por isso a migracao dela pode ser verificada de ponta a ponta comparando byte a byte contra a
# saida que ja existia no disco. As outras cinco precisam de uma janela de placa.
#
# Ela tambem foi a que expos o unico buraco do nucleo: `to_native` funde `to_{q,k,v}` num `qkv`,
# entao um destino nasce de VARIAS faixas do source, e o nucleo so tinha `plan_copy` de uma faixa
# so. Dai `plan_copy_many`.
import _conversion as C  # noqa: E402

TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                "I8": torch.int8, "U8": torch.uint8}


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

    conv = C.Conversion(source, output)
    header, metadata = conv.header, conv.metadata
    # A recusa de fonte ja quantizada vale tambem em --dry-run: um plano tirado de um checkpoint
    # quantizado nao e um plano valido, mesmo que nada va ser escrito. As outras duas (saida
    # existente, partial obsoleto) so fazem sentido quando vai haver escrita -- mas entao elas vem
    # ANTES do plano, e nao depois: recusar so no fim faz o usuario esperar a construcao do plano
    # para ouvir "esse arquivo ja existe". Foi o que aconteceu na primeira versao desta migracao.
    C.refuse_already_quantized(header, metadata)
    if not args.dry_run:
        conv.refuse_unsafe()

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

    # Partes 1-8 do contrato, no nucleo. O que esta ferramenta fazia a mao ficava sem a parte 8:
    # ela NAO tinha guarda de disco nenhuma, e escrever 11 GiB num volume cheio dava um erro de
    # escrita no meio do laco em vez de uma recusa antes de comecar. `conv.guard()` fecha isso.
    entradas = []
    for name in sorted(resolved):
        entry = resolved[name]
        fontes = entry["sources"]
        if len(fontes) == 1:
            entradas.append(C.plan_copy(name, header[fontes[0]]))
        else:
            faixas = [(header[src]["data_offsets"][0],
                       header[src]["data_offsets"][1] - header[src]["data_offsets"][0])
                      for src in fontes]
            entradas.append(C.plan_copy_many(name, entry["dtype"], entry["shape"], faixas))

    output.parent.mkdir(parents=True, exist_ok=True)
    conv.guard(sum(e.nbytes for e in entradas), label="to_native")

    def progresso(indice: int, total: int, _chave: str) -> None:
        if indice % 100 == 0 or indice == total:
            print(f"[{indice}/{total}] written", flush=True)

    conv.commit(entradas, dict(metadata) if metadata else None, progress=progresso)

    print(f"wrote {output} ({output.stat().st_size / 2**30:.2f} GiB)")
    print("This is still BF16. Generate with it at a fixed seed and compare against the source "
          "before quantizing anything from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
