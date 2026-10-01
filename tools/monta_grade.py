r"""Monta uma grade rotulada bracos x sementes a partir de PNGs ja decodificados.

POR QUE SEPARADO: a folha do decodificador e uma tira por semente, boa para olhar aqui e larga
demais para um card. Um card precisa de uma grade com a mesma escala em todas as celulas e o
rotulo DENTRO da imagem -- este repo ja publicou um card em que as imagens viajaram soltas com o
rotulo so no markdown e a referencia de uma semente estava quebrada sem ninguem notar.

`--layout comfy` le direto da saida do ComfyUI (`<dir>/<braco>/p<N>_s<S>_NNNNN_.png`, o formato
das baterias via /prompt), resolvendo o contador com `metricas_imagem.imagem_unica` em vez de supor
`_00001_`: um grafo rodado de novo grava `_00002_`, e uma folha montada do `_00001_` fixo mostra a
imagem ANTIGA. Substitui a `folha_contato.py` que cada bateria copiava (revisao 2026-09-29).

NAO COBERTO: so recorta, reduz e cola. Nao decodifica, nao mede nada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent


def imagem_unica(pasta: Path, prefixo: str) -> Path:
    """`metricas_imagem.imagem_unica`, importado so no modo comfy (ele puxa torch no import)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metricas_imagem import imagem_unica as _unica
    return _unica(pasta, prefixo)


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
    p.add_argument("--layout", choices=("decodificado", "comfy"), default="decodificado",
                   help="decodificado: <dir>/<braco>__p<N>_s<S>.png (padrao, formato antigo); "
                        "comfy: <dir>/<braco>/p<N>_s<S>_NNNNN_.png, contador resolvido sem supor")
    p.add_argument("--corte", help="recorte relativo x0,y0,x1,y1 em fracao, ex 0.3,0.3,0.6,0.6")
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
            if a.layout == "comfy":
                chave = f"p{prompt if prompt is not None else 0}_s{semente}"
                try:
                    f = imagem_unica(a.dir / braco, chave)
                except FileNotFoundError:
                    faltando.append(f"{braco}/{chave}")
                    continue
            else:
                nome = (f"{braco}_s{semente}.png" if prompt is None
                        else f"{braco}__p{prompt}_s{semente}.png")
                f = a.dir / nome
                if not f.exists():
                    faltando.append(f.name)
                    continue
            im = Image.open(f).convert("RGB")
            if a.corte:
                x0, y0, x1, y1 = map(float, a.corte.split(","))
                w, h = im.size
                im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            im = im.resize((a.lado, a.lado), Image.LANCZOS)
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
