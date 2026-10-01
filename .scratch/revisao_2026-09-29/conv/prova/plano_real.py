"""Header PLANEJADO pelo codigo novo (so formas, sem kernel) x header GRAVADO nos arquivos reais do disco.

Para cada `.quant.json` real cuja fonte ainda existe: reconstroi a selecao, os formatos e o
metadata a partir da fonte e do sidecar, monta as entradas de `_formats.plan_model` (produtores
nunca chamados) e compara `_conversion.header_bytes` com os bytes de header do arquivo. So le
headers. Nao reconverte nada.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402
import quant_int8  # noqa: E402
import quant_mixed  # noqa: E402
import quant_w4a4  # noqa: E402
import quant_w4a8  # noqa: E402

RAIZES = [Path("F:/COMFY_PORTABLE/ComfyUI/models"), Path("D:/ComfyUI-Models")]


def header_disco(p: Path) -> bytes:
    with p.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return f.read(n)


def plano(side: dict, saida: Path):
    fonte = Path(side["source"])
    if not fonte.is_file():
        return None, "fonte ausente"
    header, meta = C.read_header(fonte)
    q = side.get("quantization", "")
    if q == "ConvRot W4A4":
        cg = side.get("convrot_groupsize", 256)
        sel = quant_w4a4.selected_layers(header, side["architecture"], cg)
        formats = {n: F.ConvrotW4A4(cg) for n in sel}
        out_meta = quant_w4a4.output_metadata(meta, F.layer_configs(formats))
    elif q == "asym_w4a8_int8":
        fmt = F.AsymW4A8(side["group_size"], side["convrot_groupsize"], side.get("codebook", True))
        sel = quant_w4a8.selected_layers(header, side["architecture"], side["group_size"], side["convrot_groupsize"])
        formats = {n: fmt for n in sel}
        out_meta = F.quant_metadata(meta, F.layer_configs(formats), q)
    elif q.startswith("int8_tensorwise") and "architecture" in side:
        convrot = bool(side.get("convrot", q.endswith("+convrot")))
        fmt = F.Int8Tensorwise(convrot, side.get("convrot_groupsize", 256))
        sel = quant_int8.selected_layers(header, side["architecture"], convrot, fmt.convrot_groupsize)
        formats = {n: fmt for n in sel}
        out_meta = F.quant_metadata(meta, F.layer_configs(formats), q)
    elif q == "mixed convrot_w4a4 / asym_w4a8_int8":
        w4 = F.ConvrotW4A4(side["convrot_groupsize"])
        w8 = F.AsymW4A8(side["group_size"], side["convrot_groupsize"], True)
        sel = quant_mixed.selected_layers(header, side["architecture"], side["convrot_groupsize"])
        dec = side["layers"]
        formats = {n: (w4 if dec[n.removesuffix(".weight")] == "convrot_w4a4" else w8)
                   for n in sel if dec.get(n.removesuffix(".weight"), "bf16") != "bf16"}
        out_meta = F.quant_metadata(meta, F.layer_configs(formats), q)
    else:
        return None, f"formato nao coberto aqui: {q!r}"
    entradas = F.plan_model(header, formats, None, None)
    blob, _ = C.header_bytes(entradas, out_meta)
    return blob, f"{q}, {len(formats)} camadas"


def main() -> int:
    linhas, ok, ruim = [], 0, 0
    for raiz in RAIZES:
        if not raiz.is_dir():
            continue
        for side_p in sorted(raiz.rglob("*.quant.json")):
            saida = side_p.with_name(side_p.name.removesuffix(".quant.json") + ".safetensors")
            if not saida.is_file():
                continue
            try:
                side = json.loads(side_p.read_text(encoding="utf-8"))
                blob, desc = plano(side, saida)
            except Exception as e:  # noqa: BLE001
                linhas.append(f"ERRO  {saida.name}: {e!r}")
                ruim += 1
                continue
            if blob is None:
                linhas.append(f"PULA  {saida.name}: {desc}")
                continue
            disco = header_disco(saida)
            if blob == disco:
                ok += 1
                linhas.append(f"IGUAL {saida.name}: {desc}, header {len(disco)} bytes")
            else:
                ruim += 1
                i = next((k for k, (a, b) in enumerate(zip(blob, disco)) if a != b), min(len(blob), len(disco)))
                linhas.append(f"DIF   {saida.name}: {desc}; primeiro byte diferente {i}: "
                              f"plano {blob[max(0, i - 60):i + 60]!r} | disco {disco[max(0, i - 60):i + 60]!r}")
    txt = "\n".join(linhas) + f"\n\n{ok} iguais, {ruim} diferentes/erro"
    print(txt)
    Path(__file__).with_name("plano_real.txt").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
