"""Levantamento de espaço em F: (só leitura). Tamanho por pasta (profundidade 3), maiores arquivos e candidatos a
duplicata (mesmo nome e tamanho em lugares diferentes, inclusive nas raízes de rede do extra_model_paths).

    python varre.py > varredura.json
"""
import json
import os
import sys
from collections import defaultdict

RAIZES = ["F:/"]
REDE = ["D:/ComfyUI-Models", "P:/ComfyBench", "U:/bancada-w4a4", "W:/ltx-2.5", "C:/ComfyBench"]
EXT_MODELO = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".onnx")
IGNORA = {"$RECYCLE.BIN", "System Volume Information"}

por_pasta = defaultdict(int)
arquivos = []
erros = 0


def varre(raiz, registra_pastas=True, prof_max=3):
    global erros
    pilha = [raiz]
    while pilha:
        d = pilha.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name in IGNORA:
                        continue
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            pilha.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            t = e.stat(follow_symlinks=False).st_size
                            p = e.path.replace("\\", "/")
                            arquivos.append((t, p))
                            if registra_pastas:
                                partes = p[len(raiz):].split("/")[:-1]
                                for k in range(0, min(len(partes), prof_max) + 1):
                                    por_pasta[raiz + "/".join(partes[:k])] += t
                    except OSError:
                        erros += 1
        except OSError:
            erros += 1


for r in RAIZES:
    varre(r)
locais = list(arquivos)
rede = []
for r in REDE:
    if os.path.isdir(r):
        antes = len(arquivos)
        varre(r, registra_pastas=False)
        rede += arquivos[antes:]
        del arquivos[antes:]

GiB = 2 ** 30
modelos = [(t, p) for t, p in locais if p.lower().endswith(EXT_MODELO) and t > 200 * 2 ** 20]
por_chave = defaultdict(list)
for t, p in modelos + [(t, p) for t, p in rede if p.lower().endswith(EXT_MODELO) and t > 200 * 2 ** 20]:
    por_chave[(os.path.basename(p).lower(), t)].append(p)
dup = [{"gib": round(t / GiB, 2), "copias": ps} for (n, t), ps in por_chave.items() if len(ps) > 1]
dup.sort(key=lambda x: -x["gib"] * (len(x["copias"]) - 1))

json.dump({
    "total_local_gib": round(sum(t for t, _ in locais) / GiB, 1),
    "erros": erros,
    "pastas": sorted(({"pasta": k, "gib": round(v / GiB, 1)} for k, v in por_pasta.items() if v > 5 * GiB),
                     key=lambda x: -x["gib"]),
    "maiores": [{"gib": round(t / GiB, 2), "arquivo": p} for t, p in sorted(locais, reverse=True)[:150]],
    "duplicatas_nome_tamanho": dup[:200],
}, sys.stdout, indent=1, ensure_ascii=False)
