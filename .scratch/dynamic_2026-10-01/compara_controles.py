"""Controles: repetição dentro do braço (determinismo) e Z1 dyn x nodyn."""
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


def par(ra, rb):
    x, y = carrega(OUT / ra["files"][0]), carrega(OUT / rb["files"][0])
    item = {"identica": identica(x, y)}
    if not item["identica"]:
        item.update({m: round(v, 3) for m, v in medir(x, y, quais=("psnr", "msssim")).items()})
    return item


B = {b: regs(b) for b in ("dyn", "nodyn", "dyn_rep", "nodyn_rep")}
res = {}
for caso in ("NZ", "NQ"):
    for rod in ("frio", "quente"):
        k = (caso, rod)
        res[f"{caso}_{rod} dyn x dyn_rep"] = par(B["dyn"][k], B["dyn_rep"][k])
        res[f"{caso}_{rod} nodyn x nodyn_rep"] = par(B["nodyn"][k], B["nodyn_rep"][k])
for rod in ("frio", "quente"):
    k = ("Z1", rod)
    res[f"Z1_{rod} dyn x nodyn"] = par(B["dyn_rep"][k], B["nodyn_rep"][k])
    res[f"Z1_{rod} s dyn/nodyn"] = [B["dyn_rep"][k]["server_side_s"], B["nodyn_rep"][k]["server_side_s"]]
print(json.dumps(res, indent=1))
(D / "compara_controles.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
