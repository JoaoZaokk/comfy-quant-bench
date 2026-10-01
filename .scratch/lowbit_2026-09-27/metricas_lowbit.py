"""Metricas da bateria low-bit: cada braco contra a referencia, identidade entre packs, tempo e memoria do log.

    python_embeded\\python.exe -s metricas_lowbit.py <fase> <braco_ref> <saida.json>

Revisao 2026-09-29: o it/s vem do log do ComfyUI casado POR POSICAO com os tempos sem erro; se as
contagens diferem, recusa (antes o zip truncava calado e deslocava tempos entre bracos). A imagem de
cada prompt/seed sai de `metricas_imagem.imagem_unica` (antes: a de maior contador, calada, mesmo
que as execucoes tivessem dado imagens diferentes).
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
from metricas_imagem import carrega, identica, imagem_unica, medir  # noqa: E402

AQUI = Path(__file__).parent
FASE, REF, SAIDA = sys.argv[1], sys.argv[2], sys.argv[3]
IMG = Path("F:/COMFY_PORTABLE/ComfyUI/output/lowbit_2026-09-27") / FASE


def iguais(f, g):
    return identica(carrega(f), carrega(g))


def por_braco():
    out = {}
    for d in sorted(p for p in IMG.iterdir() if p.is_dir()):
        chaves = sorted({re.sub(r"_\d+_\.png$", "", f.name) for f in d.glob("*.png")})
        out[d.name] = {k: imagem_unica(d, k) for k in chaves}
    return out


def log_por_prompt(log):
    """Sequencia de prompts do log: it/s do tqdm (4/4) e segundos; cargas de modelo vistas no meio."""
    linhas, cargas = [], []
    its = None
    for linha in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"4/4 \[[^\]]*?([\d.]+)(it/s|s/it)\]", linha)
        if m:
            v = float(m.group(1))
            its = v if m.group(2) == "it/s" else 1 / v
        m = re.search(r"(loaded completely|loaded partially|prepared for dynamic VRAM loading)[^\n]*", linha)
        if m:
            cargas.append(m.group(0))
        m = re.search(r"Prompt executed in ([\d.]+) seconds", linha)
        if m:
            linhas.append({"it_s": its, "s": float(m.group(1)), "cargas": cargas})
            its, cargas = None, []
    return linhas


imgs = por_braco()
tempos = json.loads((AQUI / f"tempos_{FASE}.json").read_text(encoding="utf-8"))
execs = log_por_prompt(AQUI / f"comfy_{FASE}.log")
ok = [t for t in tempos if "erro" not in t]
if len(ok) != len(execs):
    raise SystemExit(f"RECUSADO: {len(ok)} execucoes sem erro em tempos_{FASE}.json e {len(execs)} "
                     f"'Prompt executed' em comfy_{FASE}.log. O casamento e por posicao; com contagens "
                     "diferentes o it/s cairia no braco errado.")
for t, e in zip(ok, execs):
    t.update(e)

res = {}
for braco, fotos in imgs.items():
    linhas = [t for t in ok if t["braco"] == braco]
    quentes = [t["it_s"] for t in linhas[1:] if t.get("it_s")] or [t["it_s"] for t in linhas if t.get("it_s")]
    r = {"n": len(fotos), "it_s_quente_mediana": st.median(quentes) if quentes else None,
         "it_s_quentes": quentes, "cargas": sorted({c for t in linhas for c in t.get("cargas", [])})}
    if REF in imgs and braco != REF:
        ms, n_iguais = [], 0
        for k, f in fotos.items():
            g = imgs[REF].get(k)
            if g is None:
                continue
            n_iguais += iguais(f, g)
            ms.append(medir(carrega(f), carrega(g), quais=("msssim",))["msssim"])
        r.update({"identicas_ref": n_iguais, "msssim_ref_min": min(ms) if ms else None, "msssim_ref_media": st.mean(ms) if ms else None})
    res[braco] = r
# identidade entre os packs do mesmo modelo (mesmos pesos -> mesmas imagens)
for modelo in ("ternario", "binario"):
    bracos = [b for b in imgs if b.startswith(modelo + "_")]
    for b in bracos[1:]:
        res[b][f"identicas_a_{bracos[0]}"] = sum(iguais(f, imgs[bracos[0]][k]) for k, f in imgs[b].items() if k in imgs[bracos[0]])
res["_erros"] = [t for t in tempos if "erro" in t]
Path(SAIDA).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
for b, r in res.items():
    if b.startswith("_"):
        continue
    print(b, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("cargas", "it_s_quentes")})
    for c in r["cargas"][:3]:
        print("    ", c[:140])
print("erros:", len(res["_erros"]))
