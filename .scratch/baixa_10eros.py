"""Baixa o BF16 do 10Eros v1.5 para o home estabelecido dos checkpoints LTX grandes.

Token de LEITURA do ambiente (F:/hf-cache). Nao cria token. Confere o tamanho contra
o que o header remoto anunciou -- 46.139.886.366 B -- e recusa se nao bater.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"F:\hf-cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download

REPO = "TenStrip/LTX2.3-10Eros"
ARQ = "10Eros_v1.5_bf16.safetensors"
ESPERADO = 46_139_886_366
DESTINO = Path(r"P:\ComfyBench\checkpoints")

p = hf_hub_download(repo_id=REPO, filename=ARQ, local_dir=str(DESTINO))
n = Path(p).stat().st_size
print(f"baixado: {p}")
print(f"bytes:   {n:,}   esperado {ESPERADO:,}   {'OK' if n == ESPERADO else 'DIVERGE'}")
raise SystemExit(0 if n == ESPERADO else 1)
