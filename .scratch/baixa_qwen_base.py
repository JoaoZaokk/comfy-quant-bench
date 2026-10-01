"""Baixa o Qwen-Image 2512 BASE (nao-edit) para P:, nao para F:.

POR QUE P: E NAO F:
-------------------
F: tem 15 GB livres depois dos 62,96 GiB de Wan; este arquivo sozinho tem 38,05 GiB. Apagar
original para abrir espaco e proibido aqui, e nao e preciso: P: tem 2,1 TB ociosos e ja esta
montado em `extra_model_paths.yaml` como `bench_p`, entao o ComfyUI le de la igual.

O ENCODER JA ESTA NO DISCO
--------------------------
`qwen_2.5_vl_7b` e o MESMO do Qwen-Image-Edit, e ja foi baixado e quantizado nesta bancada. A
cadeia e compartilhada entre base e edit; so o transformer difere. Por isso so este arquivo desce.
"""
import os, shutil, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Comfy-Org/Qwen-Image_ComfyUI"
DEST = r"P:\ComfyBench\diffusion_models"
REMOTO = "split_files/diffusion_models/qwen_image_2512_bf16.safetensors"
ESPERADO = 40861031488

os.makedirs(DEST, exist_ok=True)
alvo = os.path.join(DEST, os.path.basename(REMOTO))
if os.path.exists(alvo) and os.path.getsize(alvo) == ESPERADO:
    print("ja existe e bate", flush=True); sys.exit(0)
livre = shutil.disk_usage(DEST).free
print(f"P: tem {livre/2**30:.1f} GiB livres; o arquivo pede {ESPERADO/2**30:.1f}", flush=True)
if livre < ESPERADO + 20 * 2**30:
    print("PARANDO: folga insuficiente"); sys.exit(1)
t0 = time.time()
p = hf_hub_download(repo_id=REPO, filename=REMOTO, local_dir=DEST)
if os.path.abspath(p) != os.path.abspath(alvo):
    os.replace(p, alvo)
n = os.path.getsize(alvo)
for lixo in ("split_files", ".cache"):
    d = os.path.join(DEST, lixo)
    if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)
print(f"{n} B em {time.time()-t0:.0f}s  {'OK' if n == ESPERADO else 'TAMANHO DIFERE'}", flush=True)
print("QWEN_BASE_OK" if n == ESPERADO else "FALHOU", flush=True)
