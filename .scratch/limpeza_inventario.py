"""Inventario para a limpeza de 2026-09-14: quantizacoes locais (peso + sidecar) contra o que esta
no Hub (JoaoZaokk/*), casadas por TAMANHO EXATO do LFS. Nao apaga nada; escreve
.scratch/limpeza_inventario.json e imprime a tabela. Originais nunca entram: so arquivos que
tenham sidecar .quant.json OU cujo nome carregue marca de quantizacao nossa.
"""
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOTS = [Path("F:/COMFY_PORTABLE/ComfyUI/models"), Path("P:/ComfyBench"), Path("C:/ComfyBench"),
         Path("D:/ComfyUI-Models"), Path("W:/ltx-2.5"), Path("W:/ltx-2.3")]
MARCAS = re.compile(r"(w4a4|w4a8|int8|misto|convrot|smooth|_native|-BF16\.gguf|_bf16\.gguf|_gguf|q6_k|Q6_K)", re.I)

api = HfApi()
me = api.whoami()["name"]
hub = {}  # size -> [(repo, path, sha256)]
for m in api.list_models(author=me):
    repo = m.id
    try:
        tree = list(api.list_repo_tree(repo, repo_type="model", recursive=True, expand=True))
    except Exception as e:  # noqa: BLE001
        print("ERRO listando", repo, e); continue
    for f in tree:
        if getattr(f, "lfs", None):
            hub.setdefault(f.lfs.size, []).append((repo, f.path, f.lfs.sha256))
        elif getattr(f, "size", None):
            hub.setdefault(f.size, []).append((repo, f.path, None))
print(f"hub: {len(hub)} tamanhos distintos em {len(list(api.list_models(author=me)))} repos", flush=True)

locais = []
for root in ROOTS:
    if not root.exists():
        print("raiz ausente:", root); continue
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = Path(dp) / f
            if p.suffix.lower() not in (".safetensors", ".gguf", ".pt"):
                continue
            side = p.with_suffix(".quant.json")
            if not side.exists() and not MARCAS.search(p.name):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size < 100 * 2**20:
                continue
            casa = hub.get(size, [])
            locais.append({"path": str(p), "size": size, "gib": round(size / 2**30, 2),
                           "sidecar": side.exists(), "hub": casa})
locais.sort(key=lambda r: (-r["size"]))
for r in locais:
    h = "; ".join(f"{repo.split('/')[1]}:{path}" for repo, path, _ in r["hub"]) or "-"
    print(f"{r['gib']:8.2f}  {'S' if r['sidecar'] else '-'}  {r['path']}\n            hub: {h}")
Path(".scratch/limpeza_inventario.json").write_text(json.dumps({"hub_por_tamanho": {str(k): v for k, v in hub.items()}, "locais": locais}, indent=1), encoding="utf-8")
print(f"\n{len(locais)} arquivos locais com marca/sidecar; {sum(1 for r in locais if r['hub'])} casam com o Hub por tamanho")
print("NAO COBERTO: casamento por tamanho, nao por sha; originais (sem marca, sem sidecar) nao listados de proposito.")
