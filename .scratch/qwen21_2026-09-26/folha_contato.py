"""Folha de contato: para cada prompt/seed, as imagens dos DiTs lado a lado (rotuladas), para inspecao visual.

    python_embeded\\python.exe -s folha_contato.py <saida_dir> <lado_px> <dit> [<dit> ...] [--raiz R] [--corte x0,y0,x1,y1]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

AQUI = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("saida", type=Path)
ap.add_argument("lado", type=int)
ap.add_argument("dits", nargs="+")
ap.add_argument("--raiz", type=Path, default=Path("F:/COMFY_PORTABLE/ComfyUI/output/qwen21_bateria"))
ap.add_argument("--prompts", default="0,1,2,3,4,5")
ap.add_argument("--seeds", default="42,7")
ap.add_argument("--corte", help="recorte relativo x0,y0,x1,y1 em fracao (ex 0.3,0.3,0.6,0.6) para ver detalhe")
a = ap.parse_args()
a.saida.mkdir(parents=True, exist_ok=True)
for i in map(int, a.prompts.split(",")):
    for s in map(int, a.seeds.split(",")):
        ims = []
        for d in a.dits:
            im = Image.open(a.raiz / d / f"p{i}_s{s}_00001_.png").convert("RGB")
            if a.corte:
                x0, y0, x1, y1 = map(float, a.corte.split(","))
                W, H = im.size
                im = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
            im = im.resize((a.lado, a.lado), Image.LANCZOS)
            ImageDraw.Draw(im).rectangle((0, 0, 9 * len(d) + 8, 18), fill="black")
            ImageDraw.Draw(im).text((4, 3), d, fill="white")
            ims.append(im)
        folha = Image.new("RGB", (a.lado * len(ims), a.lado))
        for k, im in enumerate(ims):
            folha.paste(im, (k * a.lado, 0))
        sufixo = "_corte" if a.corte else ""
        folha.save(a.saida / f"p{i}_s{s}{sufixo}.jpg", quality=90)
print("ok")
