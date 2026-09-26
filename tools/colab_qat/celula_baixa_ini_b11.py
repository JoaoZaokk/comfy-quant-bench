"""Celula (colab exec): baixa o checkpoint final do b11 (HF) para /content/b11_ini e confere o passo.

O b11L continua o b11 com `--inicia-de` (mestre + Adam do b11, repo novo, o b11 fica intacto no HF).
O envio do checkpoint final pode ter sido pulado ("envio anterior ainda em curso"), entao o passo gravado
no arquivo e' conferido aqui antes de lancar: esperado 2000.
"""
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

p = hf_hub_download("JoaoZaokk/klein4b-qat-b11", "ckpt/ultimo.pt", local_dir="/content/b11_ini")
est = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
print(f"b11 ckpt: {Path(p).stat().st_size / 2**30:.2f} GiB, passo {est['passo']}")
for d in ("/content/qat_b8/professor", "/content/qat_b8/professor_holdout"):
    print(d, "existe" if Path(d).is_dir() else "FALTA", len(list(Path(d).glob("*.pt"))) if Path(d).is_dir() else 0)
