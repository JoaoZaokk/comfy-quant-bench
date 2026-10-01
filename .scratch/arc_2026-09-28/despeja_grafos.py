"""Despeja os grafos API gerados pelos scripts (antes/ ou atuais) para provar equivalência da refatoração.
    python despeja_grafos.py <pasta> <saida.json>"""
import json, sys, pathlib, types

def carrega(caminho, corte):
    src = pathlib.Path(caminho).read_text(encoding="utf-8")
    src = src[:src.index(corte)] if corte in src else src
    mod = types.ModuleType("m"); sys.argv = ["x", "/tmp/nada"]
    mod.__dict__["__file__"] = str(caminho)
    exec(compile(src, str(caminho), "exec"), mod.__dict__)
    return mod

def limpa(g):  # títulos de UI não afetam a execução
    return {k: {"class_type": v["class_type"], "inputs": v["inputs"]} for k, v in g.items()}

ARGS = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))  # comfy_client
p = pathlib.Path(ARGS[0]); sys.path.insert(0, str(p.resolve().parent)); sys.path.insert(0, str(p.resolve()))
out = {}
a = carrega(p / "roda_arc.py", "OUT.mkdir(")
for i in range(6):
    for s in (42, 7):
        out[f"arc/p{i}_s{s}"] = limpa(a.grafo(i, a.PROMPTS[i], s))
t = carrega(p / "roda_turbo.py", "def chama(")
for tag in ("viggle6", "viggle4", "viggle8", "v01_4", "turbo8"):
    for pk in ("p2", "p1"):
        out[f"turbo/{tag}_{pk}"] = limpa(t.grafo(tag, pk))
e = carrega(p / "roda_edit.py", "def chama(")
for tag in ("viggle6", "v01_4", "turbo8", "base25", "base25s5"):
    for pk in ("gorro", "gorro768", "frasco"):
        out[f"edit/{tag}_{pk}"] = limpa(e.grafo(tag, pk))
for nome in ("monta_wf_turbo.py", "monta_wf_edit.py"):
    m = carrega(p / nome, "for nome, (g, nota) in WF.items():")
    for k, (g, nota) in m.WF.items():
        out[f"wf/{k}"] = {"grafo": limpa(g), "nota": nota}
pathlib.Path(ARGS[1]).write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
print(len(out), "grafos")
