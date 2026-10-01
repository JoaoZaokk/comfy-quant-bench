"""Caça ao NaN na Arc: repete o turbo 4 passos (p2/p1 alternados, para não cair no cache) salvando também o latente.

    python3 nan_teste.py <saida_dir> <n_pares> [tag]
Para cada render grava no JSONL: status, tempo, NaN/Inf no latente (SaveLatent) e média da imagem (preta = NaN no decode
ou propagado). Lê as saídas com o python do venv do ComfyUI (numpy/safetensors/PIL), chamado à parte.
"""
import json
import pathlib
import subprocess
import sys

import arc_grafos as G
import comfy_client as cc
import roda_turbo

OUT = pathlib.Path(sys.argv[1])
PARES = int(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else "v01_4"
COMFY_OUT = pathlib.Path.home() / "ComfyUI/output"
VENV = pathlib.Path.home() / "ComfyUI/venv/bin/python"
INSPECIONA = r'''
import sys, json, numpy as np
from PIL import Image
from safetensors.numpy import load_file
lat = load_file(sys.argv[1])["latent_tensor"].astype(np.float32)
img = np.asarray(Image.open(sys.argv[2])).astype(np.float32)
print(json.dumps({"lat_nan": int(np.isnan(lat).sum()), "lat_inf": int(np.isinf(lat).sum()),
                  "lat_absmax": float(np.nanmax(np.abs(lat))) if np.isfinite(lat).any() else None,
                  "img_media": float(img.mean()), "img_std": float(img.std())}))
'''


def grafo(pk, i):
    g = roda_turbo.grafo(TAG, pk)
    g["8"]["inputs"]["filename_prefix"] = f"nan_teste/{TAG}_{pk}_{i:03d}"
    g["20"] = G.n("SaveLatent", samples=["6", 0], filename_prefix=f"nan_teste/{TAG}_{pk}_{i:03d}")
    return g


def main():
    comfy = cc.Comfy("127.0.0.1:8188")
    for i in range(PARES):
        for pk in ("p2", "p1"):
            reg = cc.roda_um(comfy, grafo(pk, i), 900, rotulo=f"{TAG}_{pk}_{i:03d}", extra={"i": i, "prompt": pk})
            if reg["status"] == "success":
                png = next(f for f in reg["files"] if f.endswith(".png"))
                lat = sorted((COMFY_OUT / "nan_teste").glob(f"{TAG}_{pk}_{i:03d}_*.latent"))[-1]
                r = subprocess.run([str(VENV), "-c", INSPECIONA, str(lat), str(COMFY_OUT / png)],
                                   capture_output=True, text=True, timeout=120)
                reg.update(json.loads(r.stdout) if r.returncode == 0 else {"inspecao_erro": r.stderr[-300:]})
            print(json.dumps({k: reg.get(k) for k in ("grafo", "status", "server_side_s", "lat_nan", "lat_inf",
                                                        "lat_absmax", "img_media", "erro")}), flush=True)
            cc.grava_jsonl(OUT / "nan.jsonl", reg)


if __name__ == "__main__":
    main()
