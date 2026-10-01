"""Sobe o ComfyUI (mesmos argumentos) contando qual backend do comfy-kitchen atende cada op.

Não importa torch/comfy_kitchen antes do main.py (o ComfyUI configura o alocador CUDA antes do torch; importar
antes derruba a subida): uma thread espera o main.py importar o kitchen e só então embrulha o registry.
Grava em $CONTADOR_SAIDA a cada 5 s: {"op|backend": n} e, quando o escolhido não é cuda, o motivo
da recusa do cuda ("op": "param: motivo" ou "cuda nao implementa").
"""
import json
import os
import runpy
import sys
import threading
import time
from collections import Counter

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "ComfyUI"))

cont = Counter()
motivos = {}
registry = None
_orig = None


def contado(func_name, kwargs=None):
    b = _orig(func_name, kwargs)
    cont[f"{func_name}|{b}"] += 1
    if b != "cuda" and func_name not in motivos:
        if func_name in registry._capabilities.get("cuda", set()) and kwargs:
            r = registry.validate_backend_for_call("cuda", func_name, kwargs)
            motivos[func_name] = f"{r.failed_param}: {r.failure_reason}"
        else:
            motivos[func_name] = "cuda nao implementa (ou chamada sem kwargs)"
    return b


SAIDA = os.environ["CONTADOR_SAIDA"]


def grava():
    global registry, _orig
    while registry is None:
        mod = sys.modules.get("comfy_kitchen.registry")
        if mod is not None and hasattr(mod, "registry"):
            registry = mod.registry
            _orig = registry.get_capable_backend
            registry.get_capable_backend = contado
        else:
            time.sleep(0.05)
    while True:
        time.sleep(5)
        with open(SAIDA + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"contagem": dict(cont), "recusa_cuda": motivos,
                       "backends": {k: str(v) for k, v in registry.list_backends().items()}}, f, indent=1)
        os.replace(SAIDA + ".tmp", SAIDA)


threading.Thread(target=grava, daemon=True).start()
main = os.path.join(RAIZ, "ComfyUI", "main.py")
sys.argv = [main] + sys.argv[1:]
runpy.run_path(main, run_name="__main__")
