"""Baixa o checkpoint BF16 do LTX 2.3 distilled 1.1 -- a unica fonte sem quantizacao da familia.

O maior LTX 2.3 local e `ltx-2.3-22b-dev-fp8` (ja fp8) e dois GGUF de terceiro; nenhum serve de
fonte para calibrar em ativacao real nem de braco "original". O arquivo do Hub e um checkpoint
UNICO (DiT + VAE de video + VAE de audio + vocoder + projecao de texto), 42,98 GiB, repo NAO
gated (`gated: False` lido da API em 2026-09-13). Vai para P: (2 TB livres) e entra no ComfyUI
por `extra_model_paths.yaml` (`bench_p.checkpoints`), sem copia.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "Lightricks/LTX-2.3"
REMOTO = "ltx-2.3-22b-distilled-1.1.safetensors"
DESTINO = r"P:\ComfyBench\checkpoints"

info = [p for p in HfApi().get_paths_info(REPO, [REMOTO]) if p.path == REMOTO][0]
esperado = info.size
print(f"esperado {esperado} B ({esperado / 2**30:.2f} GiB), sha {getattr(info.lfs, 'sha256', '?')}", flush=True)
alvo = os.path.join(DESTINO, REMOTO)
if os.path.exists(alvo) and os.path.getsize(alvo) == esperado:
    print("ja existe e bate", flush=True); print("LTX23_DOWNLOAD_OK", flush=True); sys.exit(0)
t0 = time.time()
p = hf_hub_download(repo_id=REPO, filename=REMOTO, local_dir=DESTINO)
if os.path.abspath(p) != os.path.abspath(alvo):
    os.replace(p, alvo)
n = os.path.getsize(alvo)
dt = time.time() - t0
print(f"{n} B em {dt:.0f}s ({n / dt / 2**20:.1f} MiB/s)  {'OK' if n == esperado else 'TAMANHO DIFERE'}", flush=True)
if n != esperado: sys.exit(1)
print("LTX23_DOWNLOAD_OK", flush=True)
