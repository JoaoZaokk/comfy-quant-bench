"""Testa, no ambiente atual, cada `from transformers[.x] import a, b` e `import transformers.x` dos custom nodes."""
import ast, importlib, json, pathlib, sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
RAIZ = pathlib.Path(r"F:\COMFY_PORTABLE\ComfyUI\custom_nodes")
packs = [l.split()[1] for l in open("nos_transformers.txt") if ".disabled" not in l]
usos = {}
for pk in packs:
    for f in (RAIZ / pk).rglob("*.py"):
        if any(p in f.parts for p in ("node_modules", "__pycache__", "tests", "test")):
            continue
        try:
            arv = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(arv):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == "transformers" and n.level == 0:
                for a in n.names:
                    usos.setdefault((n.module, a.name), set()).add(f"{pk}/{f.relative_to(RAIZ / pk)}:{n.lineno}")
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] == "transformers" and "." in a.name:
                        usos.setdefault((a.name, None), set()).add(f"{pk}/{f.relative_to(RAIZ / pk)}:{n.lineno}")
falhas = {}
for (mod, nome), onde in sorted(usos.items()):
    try:
        m = importlib.import_module(mod)
        if nome and nome != "*":
            getattr(m, nome)
    except Exception as e:
        falhas[f"{mod}:{nome}"] = {"erro": f"{type(e).__name__}: {str(e)[:160]}", "onde": sorted(onde)}
json.dump(falhas, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"{len(usos)} símbolos, {len(falhas)} falham")
for k, v in falhas.items():
    print(k, "|", v["erro"][:110], "|", ", ".join(v["onde"][:3]))
