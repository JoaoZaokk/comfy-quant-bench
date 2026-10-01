"""Critério 3: render depois do upgrade pip x mesmo grafo/seed/flags de hoje cedo (triton_2026-09-29/on, ambiente antigo)."""
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "tools"))
from metricas_imagem import carrega, identica, medir  # noqa: E402

OUT_DIR = RAIZ / "ComfyUI/output"


def registros(p):
    regs = [json.loads(linha) for linha in p.read_text(encoding="utf-8").splitlines() if linha.strip()]
    return {(r["caso"], r["rodada"]): r for r in regs}


def imagem(reg):
    fs = [f for f in reg.get("files") or [] if f.endswith(".png")]
    return OUT_DIR / fs[0] if len(fs) == 1 else None


antes = registros(RAIZ / ".scratch/triton_2026-09-29/on/resultados.jsonl")
depois = registros(RAIZ / ".scratch/pip_2026-09-29/depois/resultados.jsonl")
res = {}
for chave in sorted(depois):
    a, d = antes.get(chave), depois[chave]
    item = {"status_depois": d.get("status"), "s_antes": a and a.get("server_side_s"), "s_depois": d.get("server_side_s")}
    ia, idp = a and imagem(a), imagem(d)
    if ia and idp and ia.exists() and idp.exists():
        x, r = carrega(idp), carrega(ia)
        item["identica"] = identica(x, r)
        if not item["identica"]:
            item.update({k: round(v, 4) for k, v in medir(x, r, quais=("psnr", "msssim")).items()})
    else:
        item["sem_par"] = True
    res["_".join(chave)] = item
print(json.dumps(res, indent=1, ensure_ascii=False))
