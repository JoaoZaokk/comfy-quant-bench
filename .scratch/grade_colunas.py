"""Grade de render com colunas arbitrarias: grade_colunas.py saida.png "rotulo=pasta:prefixo" ...
Todas as pastas precisam ter o mesmo ladder.json de prompts (5 prompts x sementes 11 12)."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

saida, cols = Path(sys.argv[1]), []
for arg in sys.argv[2:]:
    rot, _, resto = arg.partition("=")
    pasta, _, pre = resto.rpartition(":")
    cols.append((rot, Path(pasta), pre))
ps = {json.dumps(json.loads((c[1] / "ladder.json").read_text())["prompts"]) for c in cols}
assert len(ps) == 1, "prompts diferentes entre pastas"
S, H = 256, 24
rows = [(p, s) for p in range(5) for s in (11, 12)]
g = Image.new("RGB", (S * len(cols), H + S * len(rows)), "white")
d = ImageDraw.Draw(g)
for j, (n, dd, pre) in enumerate(cols):
    d.text((j * S + 6, 6), n, fill="black")
    for i, (p, s) in enumerate(rows):
        g.paste(Image.open(dd / f"{pre}__p{p}_s{s}.png").convert("RGB").resize((S, S)), (j * S, H + i * S))
g.save(saida)
print("ok", saida, g.size)
