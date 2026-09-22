"""Baixa o apoio que o render do 10Eros W4A8 precisa e que NAO estava em disco.

Medido em 2026-09-21: `10S-Comfy-nodes` nao esta instalado (instalar node e mudanca no ComfyUI,
decisao do dono), a LoRA DMD do 10Eros nao existia aqui, e a unica `dmd*` em disco era
`dmd2_sdxl_4step_lora` -- de SDXL, ou seja exatamente a armadilha de LoRA de outra arquitetura que
o CLAUDE.md registra (casa 300 chaves por NOME, falha em shape, ComfyUI loga e CONTINUA, e a camada
e requantizada com delta zero).

Tres itens:
  workflow  o JSON oficial do autor, para ler os parametros reais em vez de adivinhar
  lora      LTX2.3_DMD_hybrid_v2, que o README do 10Eros recomenda para a v1.5
  int8      o `10Eros_v1.4_DMD_int8_convrot` de terceiro que mora no proprio repo do 10Eros --
            braco de comparacao barato, ja que o BF16 de 42,97 GiB nao abre nesta maquina pelo
            leitor normal (2x em commit)
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"F:\hf-cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download

ALVOS = [
    ("TenStrip/LTX2.3-10Eros_Workflows", "10Eros_10SNodes_I2V_Basic_DMD_V5.json",
     81_724, Path(r"F:\COMFY_PORTABLE\bench\10eros"), "workflow oficial"),
    ("TenStrip/LTX2.3-10Eros_Workflows", "README.md",
     2_435, Path(r"F:\COMFY_PORTABLE\bench\10eros"), "README dos workflows"),
    ("TenStrip/LTX2.3_DMD_Lora", "LTX2.3_DMD_hybrid_v2.safetensors",
     5_095_398_920, Path(r"P:\ComfyBench\loras"), "LoRA DMD hibrida v2"),
    ("TenStrip/LTX2.3-10Eros", "INT8 diffusion_models/10Eros_v1.4_DMD_int8_convrot.safetensors",
     29_161_842_398, Path(r"P:\ComfyBench\diffusion_models"), "int8 de terceiro, braco de comparacao"),
]

for repo, arq, esperado, dest, rotulo in ALVOS:
    dest.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo_id=repo, filename=arq, local_dir=str(dest))
    n = Path(p).stat().st_size
    print(f"{rotulo:38s} {n:>15,}  esperado {esperado:>15,}  "
          f"{'OK' if n == esperado else 'DIVERGE'}", flush=True)
    print(f"{'':38s} {p}", flush=True)
print("APOIO_10EROS_FIM")
