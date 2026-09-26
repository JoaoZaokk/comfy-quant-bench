"""Qwen-Image-2.1 BF16 single-file da Comfy-Org para P:/ComfyBench/diffusion_models (pedido do dono 26/09)."""
import os, time
from huggingface_hub import HfApi, hf_hub_download
REPO, A, DEST = "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_bf16.safetensors", "P:/ComfyBench"
tam = {s.rfilename: s.size for s in HfApi().model_info(REPO, files_metadata=True).siblings}[A]
alvo = os.path.join(DEST, A)
if os.path.exists(alvo):
    print("ja existe", os.path.getsize(alvo), "esperado", tam)
else:
    t0 = time.time(); p = hf_hub_download(REPO, A, local_dir=DEST); s = os.path.getsize(p)
    print(f"ok {A} {s} B ({'confere' if s == tam else 'DIFERE ' + str(tam)}) {time.time() - t0:.0f} s", flush=True)
print("FIM", flush=True)
