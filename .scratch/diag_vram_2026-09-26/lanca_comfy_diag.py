"""Sobe o ComfyUI (main.py, mesmos argumentos) com um registro de VRAM em cada load_models_gpu e free_memory.

Nao toca a arvore do ComfyUI: uma thread espera o servidor terminar os imports e embrulha as duas funcoes de
comfy.model_management (os chamadores usam acesso por atributo). Cada chamada grava uma linha JSON com:
argumentos de memoria, livre/alocado/reservado por placa, pico alocado desde a chamada anterior e os modelos
carregados (classe, placa, MiB na placa, MiB total).

    python_embeded\\python.exe -s .scratch/diag_vram_2026-09-26/lanca_comfy_diag.py <saida.jsonl> -- <args do main.py>
"""
import json
import os
import runpy
import sys
import threading
import time

RAIZ = r"F:\COMFY_PORTABLE\ComfyUI"
SAIDA = os.path.abspath(sys.argv[1])  # antes do chdir
ARGS = sys.argv[sys.argv.index("--") + 1:]
MIB = 2 ** 20


def estado(mm, torch):
    placas = []
    for i in range(torch.cuda.device_count()):
        livre, total = torch.cuda.mem_get_info(i)
        placas.append({"cuda": i, "livre": round(livre / MIB), "total": round(total / MIB),
                       "alocado": round(torch.cuda.memory_allocated(i) / MIB),
                       "reservado": round(torch.cuda.memory_reserved(i) / MIB),
                       "pico_alocado": round(torch.cuda.max_memory_allocated(i) / MIB)})
        torch.cuda.reset_peak_memory_stats(i)
    modelos = []
    for lm in list(mm.current_loaded_models):
        try:
            p = lm.model
            nome = type(getattr(p, "model", p)).__name__
            modelos.append({"modelo": nome, "placa": str(lm.device), "na_placa": round(lm.model_loaded_memory() / MIB),
                            "total": round(lm.model_memory() / MIB)})
        except Exception as e:  # modelo coletado no meio
            modelos.append({"erro": repr(e)[:120]})
    return {"placas": placas, "modelos": modelos}


def grava(reg):
    reg["t"] = round(time.time(), 1)
    with open(SAIDA, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")


def instala():
    while "server" not in sys.modules or "comfy.model_management" not in sys.modules:
        time.sleep(0.2)
    time.sleep(3)
    import torch
    import comfy.model_management as mm
    orig_load, orig_free = mm.load_models_gpu, mm.free_memory

    def nomes(models):
        return [type(getattr(m, "model", m)).__name__ for m in models]

    def load_models_gpu(models, *a, **k):
        mem = {x: k.get(x) for x in ("memory_required", "minimum_memory_required", "force_full_load")}
        if a:
            mem["posicionais"] = [x if isinstance(x, (int, float, bool)) or x is None else repr(x) for x in a]
        for x in ("memory_required", "minimum_memory_required"):
            if isinstance(mem.get(x), (int, float)):
                mem[x] = round(mem[x] / MIB)
        grava({"ev": "load_antes", "pede": nomes(models), **mem, **estado(mm, torch)})
        r = orig_load(models, *a, **k)
        grava({"ev": "load_depois", "pede": nomes(models), **estado(mm, torch)})
        return r

    def free_memory(memory_required, device, *a, **k):
        grava({"ev": "free", "pede_mib": round(memory_required / MIB), "placa": str(device),
               **estado(mm, torch)})
        return orig_free(memory_required, device, *a, **k)

    mm.load_models_gpu, mm.free_memory = load_models_gpu, free_memory
    grava({"ev": "instalado", "args": ARGS})


LANCADOR = os.path.abspath(__file__)
_execv = os.execv


def execv_pelo_lancador(path, argv):
    """O ComfyUI-Manager relanca o main.py com os.execv a cada boot (lazy install recorrente). Relanca por aqui,
    senao o processo novo sobe sem o registro."""
    grava({"ev": "execv_interceptado", "argv": [str(a) for a in argv]})
    q = lambda s: '"' + s + '"'  # noqa: E731 -- mesmo formato de aspas que o Manager usa no Windows
    _execv(sys.executable, [q(sys.executable), "-s", q(LANCADOR), q(SAIDA), "--", *ARGS])


os.execv = execv_pelo_lancador
os.chdir(RAIZ)
sys.path.insert(0, RAIZ)
sys.argv = [os.path.join(RAIZ, "main.py"), *ARGS]
threading.Thread(target=instala, daemon=True).start()
runpy.run_path(sys.argv[0], run_name="__main__")
