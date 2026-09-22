"""Baixa o minimo para o reverse engineering do Bonsai Image: os tres transformers
alinhaveis (original, ternario, binario) + o pack gemlite real.

Os tres transformers tem a MESMA estrutura (7.751.109.744 contra 7.751.109.712 B, 32 bytes),
entao o diff por camada e direto. O sha256 do ternario esta publicado no manifest.json deles,
o que permite conferir o download por prova positiva em vez de por tamanho.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"F:\hf-cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download

DEST = Path(r"F:\bonsai-re")
T = "transformer/diffusion_pytorch_model.safetensors"
ALVOS = [
    ("black-forest-labs/FLUX.2-klein-4B", T, 7_751_109_744, None, "original"),
    ("prism-ml/bonsai-image-ternary-4B-unpacked", T, 7_751_109_712,
     "fa5fd18239aadc41d3bec64810479fd3aed5182946f70e2d894f747cf4e35964", "ternario"),
    ("prism-ml/bonsai-image-binary-4B-unpacked", T, 7_751_109_712, None, "binario"),
    ("prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
     "transformer-gemlite-int2/state_dict.pt", 1_540_457_482, None, "pack gemlite"),
    ("prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
     "transformer-gemlite-int2/gemlite_autotune.json", 470_215, None, "autotune"),
]

for repo, arq, esperado, sha, rotulo in ALVOS:
    sub = DEST / repo.split("/")[1]
    p = hf_hub_download(repo_id=repo, filename=arq, local_dir=str(sub))
    n = Path(p).stat().st_size
    print(f"{rotulo:14s} {n:>15,}  esperado {esperado:>15,}  {'OK' if n == esperado else 'DIVERGE'}  {p}", flush=True)
    if sha:
        import hashlib
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        d = h.hexdigest()
        print(f"{'':14s} sha256 {d}  {'CONFERE com o manifest deles' if d == sha else 'NAO CONFERE'}", flush=True)
print("BONSAI_DOWNLOAD_FIM")
