"""Folha de contato do teto do Krea2: 4 bracos x 5 prompts, uma folha por semente.

Um eixo varia entre as linhas -- o `convrot_groupsize` -- e nada mais: mesma fonte, mesma
calibragem, mesmas 224 camadas, mesmo prompt e mesma semente em cada coluna. A primeira linha e
o BF16, que e o controle que tem de sair bom; se ele sair ruim, nenhuma outra linha se le.

A folha nao mede nada. Quem decide se 0,1377 de erro por camada destruiu a imagem e alguem
olhando, e a comparacao util e a VERTICAL: mesma coluna, linhas diferentes.
"""
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

D = pathlib.Path(r"F:\COMFY_PORTABLE\bench\quality_ladder_krea2_teto")
LINHAS = [
    ("BF16 referencia      24,48 GiB   err -", "krea2_turbo_bf16"),
    ("W4A4 cg 256 (enviado) 7,50 GiB   err 0,1199", "krea2_turbo_w4a4"),
    ("W4A4 cg 64            7,50 GiB   err 0,1238", "krea2_turbo_w4a4_cg64"),
    ("W4A4 cg 16            7,50 GiB   err 0,1377", "krea2_turbo_w4a4_cg16"),
]
COLUNAS = ["maca (controle)", "rosto idoso", 'placa "OPEN"', "mercado noturno", "cristal de gelo"]

LADO, TOPO, ESQ = 300, 28, 330
try:
    f = ImageFont.truetype("arial.ttf", 14)
    fp = ImageFont.truetype("arial.ttf", 14)
except OSError:
    f = fp = ImageFont.load_default()

faltando = []
for semente in (1, 2):
    folha = Image.new("RGB", (ESQ + LADO * len(COLUNAS), TOPO + LADO * len(LINHAS)), "white")
    d = ImageDraw.Draw(folha)
    for j, nome in enumerate(COLUNAS):
        d.text((ESQ + j * LADO + 6, 7), nome, fill="black", font=fp)
    for i, (rotulo, prefixo) in enumerate(LINHAS):
        y = TOPO + i * LADO
        d.text((8, y + LADO // 2 - 8), rotulo, fill="black", font=f)
        for j in range(len(COLUNAS)):
            x = ESQ + j * LADO
            p = D / f"{prefixo}__p{j}_s{semente}.png"
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((LADO - 4, LADO - 4))
                folha.paste(im, (x + (LADO - im.width) // 2, y + (LADO - im.height) // 2))
            else:
                faltando.append(p.name)
                d.rectangle([x + 4, y + 4, x + LADO - 4, y + LADO - 4], outline="red", width=3)
            d.line([(x, TOPO), (x, TOPO + LADO * len(LINHAS))], fill="#cccccc")
    destino = pathlib.Path(rf"F:\COMFY_PORTABLE\bench\krea2_teto_s{semente}.png")
    folha.save(destino)
    print("gravado:", destino, folha.size)

if faltando:
    print("AUSENTES:", faltando)
    sys.exit(1)
print("NAO COBERTO: a folha nao mede. O veredito de P3 -- 'no cg 16 ainda renderiza em pelo "
      "menos 4 dos 5 prompts' -- sai de olhar a quarta linha contra a segunda.")
