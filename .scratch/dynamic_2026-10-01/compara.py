"""Critério: imagem dyn x nodyn, mesmo caso/seed. Idêntica ou PSNR > 40 dB."""
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "tools"))
from metricas_imagem import carrega, identica, medir  # noqa: E402

D = pathlib.Path(__file__).resolve().parent
OUT = RAIZ / "ComfyUI/output"


def regs(b):
    linhas = (D / b / "resultados.jsonl").read_text(encoding="utf-8").splitlines()
    return {(r["caso"], r["rodada"]): r for r in map(json.loads, filter(None, linhas))}


a, b = regs("dyn"), regs("nodyn")
res = {}
for k in sorted(a):
    x, y = carrega(OUT / a[k]["files"][0]), carrega(OUT / b[k]["files"][0])
    item = {"s_dyn": a[k]["server_side_s"], "s_nodyn": b[k]["server_side_s"], "identica": identica(x, y)}
    if not item["identica"]:
        item.update({m: round(v, 3) for m, v in medir(x, y, quais=("psnr", "msssim")).items()})
    res["_".join(k)] = item
print(json.dumps(res, indent=1))
(D / "compara.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
