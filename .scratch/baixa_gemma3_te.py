"""Gemma 3 12B it, empacotado pela Comfy-Org em arquivo unico para o LTX 2.x -- o text encoder
de FABRICA do LTX 2.3. O unico Gemma 3 12B BF16 local e o `heretic` (abliterado): serve como
eixo fixo entre bracos, mas nao e o encoder que a Lightricks treinou contra. 22,71 GiB, repo nao
gated (lido da API em 2026-09-13). Vai para P: e entra por `bench_p.text_encoders`."""
import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import HfApi, hf_hub_download
REPO, REMOTO = "Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it.safetensors"
DEST = r"P:\ComfyBench\text_encoders"
info = [p for p in HfApi().get_paths_info(REPO, [REMOTO]) if p.path == REMOTO][0]
esperado = info.size
print(f"esperado {esperado} B ({esperado / 2**30:.2f} GiB) sha {getattr(info.lfs, 'sha256', '?')}", flush=True)
alvo = os.path.join(DEST, os.path.basename(REMOTO))
if os.path.exists(alvo) and os.path.getsize(alvo) == esperado:
    print("ja existe e bate"); print("GEMMA3_TE_DOWNLOAD_OK"); sys.exit(0)
t0 = time.time()
p = hf_hub_download(repo_id=REPO, filename=REMOTO, local_dir=DEST)
if os.path.abspath(p) != os.path.abspath(alvo):
    os.replace(p, alvo)
n = os.path.getsize(alvo); dt = time.time() - t0
print(f"{n} B em {dt:.0f}s ({n / dt / 2**20:.1f} MiB/s) {'OK' if n == esperado else 'TAMANHO DIFERE'}", flush=True)
if n != esperado: sys.exit(1)
print("GEMMA3_TE_DOWNLOAD_OK", flush=True)
