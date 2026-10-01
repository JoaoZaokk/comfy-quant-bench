"""Folha de contato do controle do cachorro: 3 bracos x 8 sementes, mesma instrucao.

Uma linha por checkpoint, uma coluna por semente. Um eixo varia entre as linhas (o checkpoint);
a semente varia entre as colunas e e a MESMA em todas as linhas, que e o que torna a leitura
pareada. A primeira coluna e a imagem de origem, repetida, porque o veredito de uma edicao e
sempre relativo a ela.

A folha nao mede nada. Contar cachorro e olho humano.
"""
import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(r"F:\COMFY_PORTABLE\ComfyUI\output")
IN = pathlib.Path(r"F:\COMFY_PORTABLE\ComfyUI\input")
REG = pathlib.Path(r"F:\COMFY_PORTABLE\.scratch\edit_8sementes.json")

ROTULOS = {"int8": "int8 publico  12,57 GiB",
           "w4a4": "nosso W4A4     7,50 GiB",
           "misto": "nosso misto    7,70 GiB"}

registro = json.loads(REG.read_text(encoding="utf-8"))
sementes = sorted({r["semente"] for r in registro})
bracos = [b for b in ("int8", "w4a4", "misto") if any(r["braco"] == b for r in registro)]
por = {(r["braco"], r["semente"]): r for r in registro}

LADO, TOPO, ESQ = 250, 30, 200
try:
    f = ImageFont.truetype("arial.ttf", 15)
    fp = ImageFont.truetype("arial.ttf", 13)
except OSError:
    f = fp = ImageFont.load_default()

colunas = ["ORIGEM"] + [f"semente {s}" for s in sementes]
folha = Image.new("RGB", (ESQ + LADO * len(colunas), TOPO + LADO * len(bracos)), "white")
d = ImageDraw.Draw(folha)
for j, nome in enumerate(colunas):
    d.text((ESQ + j * LADO + 6, 8), nome, fill="black", font=fp)

faltando = []
for i, braco in enumerate(bracos):
    y = TOPO + i * LADO
    d.text((8, y + LADO // 2 - 8), ROTULOS.get(braco, braco), fill="black", font=f)
    caminhos = [IN / "example.png"]
    for s in sementes:
        r = por.get((braco, s))
        nomes = (r or {}).get("arquivos") or []
        caminhos.append(OUT / nomes[0] if nomes else None)
    for j, caminho in enumerate(caminhos):
        x = ESQ + j * LADO
        if caminho is not None and caminho.exists():
            im = Image.open(caminho).convert("RGB")
            im.thumbnail((LADO - 4, LADO - 4))
            folha.paste(im, (x + (LADO - im.width) // 2, y + (LADO - im.height) // 2))
        else:
            faltando.append((braco, colunas[j]))
            d.rectangle([x + 4, y + 4, x + LADO - 4, y + LADO - 4], outline="red", width=3)
        d.line([(x, TOPO), (x, TOPO + LADO * len(bracos))], fill="#cccccc")

destino = pathlib.Path(r"F:\COMFY_PORTABLE\bench\krea2_edit_cao_8sementes.png")
folha.save(destino)
print("gravado:", destino, folha.size)
if faltando:
    print("CELULAS VAZIAS (moldura vermelha):", faltando)
print("NAO COBERTO: a folha nao conta cachorro. O veredito e de quem olha, e a pergunta e uma "
      "so: a instrucao pedia UM.")
