r"""Monta uma grade rotulada bracos x sementes a partir de PNGs ja decodificados.

POR QUE SEPARADO: a folha do decodificador e uma tira por semente, boa para olhar aqui e larga
demais para um card. Um card precisa de uma grade com a mesma escala em todas as celulas e o
rotulo DENTRO da imagem -- este repo ja publicou um card em que as imagens viajaram soltas com o
rotulo so no markdown e a referencia de uma semente estava quebrada sem ninguem notar.

NAO COBERTO: so recorta, reduz e cola. Nao decodifica, nao mede nada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--bracos", nargs="+", required=True,
                   help="prefixos na ordem das COLUNAS, ex: curto_bf16 curto_w4a4t curto_w4a4s")
    p.add_argument("--rotulos", nargs="+", required=True, help="um por braco, mesma ordem")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--lado", type=int, default=384)
    p.add_argument("--saida", type=Path, required=True)
    a = p.parse_args()

    if len(a.bracos) != len(a.rotulos):
        raise SystemExit(f"{len(a.bracos)} bracos e {len(a.rotulos)} rotulos: tem de bater")

    faixa, margem = 26, 30
    larg = margem + len(a.bracos) * a.lado
    alt = faixa + len(a.seeds) * a.lado
    folha = Image.new("RGB", (larg, alt), (12, 12, 12))
    d = ImageDraw.Draw(folha)

    faltando = []
    for cx, (braco, rot) in enumerate(zip(a.bracos, a.rotulos)):
        d.text((margem + cx * a.lado + 6, 8), rot, fill=(255, 255, 255))
        for cy, semente in enumerate(a.seeds):
            f = a.dir / f"{braco}_s{semente}.png"
            if not f.exists():
                faltando.append(f.name)
                continue
            im = Image.open(f).convert("RGB").resize((a.lado, a.lado), Image.LANCZOS)
            folha.paste(im, (margem + cx * a.lado, faixa + cy * a.lado))
    for cy, semente in enumerate(a.seeds):
        # semente escrita na vertical, na margem: a folha tem de dizer o que e sem o markdown
        d.text((4, faixa + cy * a.lado + a.lado // 2 - 20), f"s\n{semente}", fill=(190, 190, 190))

    if faltando:
        # Celula que falta vira buraco preto e passa por escolha estetica. Dizer alto.
        print(f"AVISO: {len(faltando)} celulas ausentes: {faltando}", file=sys.stderr)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    folha.save(a.saida, optimize=True)
    print(f"{a.saida}  ({larg}x{alt})  {a.saida.stat().st_size / 1024:.0f} KiB  "
          f"{len(a.bracos)}x{len(a.seeds)} celulas, {len(faltando)} ausentes")
    return 0 if not faltando else 3


if __name__ == "__main__":
    raise SystemExit(main())
