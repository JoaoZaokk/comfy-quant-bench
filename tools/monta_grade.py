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
    p.add_argument("--prompts", type=int, nargs="+", default=None,
                   help="indices de prompt, quando a pasta veio de uma corrida com VARIOS "
                        "prompts -- os arquivos sao <braco>__p<N>_s<semente>.png. Sem isto o "
                        "nome esperado e <braco>_s<semente>.png, que e o formato de uma "
                        "corrida de um prompt so. Ate 2026-09-12 so o segundo existia, e "
                        "apontar a ferramenta para uma pasta de 6 prompts dava uma folha 3x2 "
                        "vazia com 6 celulas ausentes em vez de erro.")
    p.add_argument("--lado", type=int, default=384)
    p.add_argument("--saida", type=Path, required=True)
    a = p.parse_args()

    if len(a.bracos) != len(a.rotulos):
        raise SystemExit(f"{len(a.bracos)} bracos e {len(a.rotulos)} rotulos: tem de bater")

    # uma linha por (prompt, semente); sem --prompts a lista tem um item nulo e o
    # comportamento antigo fica identico
    linhas = ([(pr, s) for pr in a.prompts for s in a.seeds] if a.prompts
              else [(None, s) for s in a.seeds])

    faixa, margem = 26, 30
    larg = margem + len(a.bracos) * a.lado
    alt = faixa + len(linhas) * a.lado
    folha = Image.new("RGB", (larg, alt), (12, 12, 12))
    d = ImageDraw.Draw(folha)

    faltando = []
    for cx, (braco, rot) in enumerate(zip(a.bracos, a.rotulos)):
        d.text((margem + cx * a.lado + 6, 8), rot, fill=(255, 255, 255))
        for cy, (prompt, semente) in enumerate(linhas):
            nome = (f"{braco}_s{semente}.png" if prompt is None
                    else f"{braco}__p{prompt}_s{semente}.png")
            f = a.dir / nome
            if not f.exists():
                faltando.append(f.name)
                continue
            im = Image.open(f).convert("RGB").resize((a.lado, a.lado), Image.LANCZOS)
            folha.paste(im, (margem + cx * a.lado, faixa + cy * a.lado))
    for cy, (prompt, semente) in enumerate(linhas):
        # prompt e semente escritos na margem: a folha tem de dizer o que e sem o markdown
        rot = f"s\n{semente}" if prompt is None else f"p{prompt}\ns{semente}"
        d.text((4, faixa + cy * a.lado + a.lado // 2 - 20), rot, fill=(190, 190, 190))

    if faltando:
        # Celula que falta vira buraco preto e passa por escolha estetica. Dizer alto.
        print(f"AVISO: {len(faltando)} celulas ausentes: {faltando}", file=sys.stderr)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    folha.save(a.saida, optimize=True)
    print(f"{a.saida}  ({larg}x{alt})  {a.saida.stat().st_size / 1024:.0f} KiB  "
          f"{len(a.bracos)}x{len(linhas)} celulas, {len(faltando)} ausentes")
    return 0 if not faltando else 3


if __name__ == "__main__":
    raise SystemExit(main())
