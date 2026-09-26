"""Qwen-Image-2.1 da Comfy-Org para P:/ComfyBench/<pasta>/ (mesma estrutura do repo = pastas do extra_model_paths).
Autorizado pelo dono em 26/09: VAE, text encoder W4A8 e o DiT int8_convrot de referencia. Recusa sobrescrever."""
import os, time
from huggingface_hub import HfApi, hf_hub_download
REPO, DEST = "Comfy-Org/Qwen-Image-2.1", "P:/ComfyBench"
ARQ = ["vae/qwen_image_2.1_vae_bf16.safetensors", "text_encoders/qwen3vl_8b_w4a8.safetensors",
       "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"]
tam = {s.rfilename: s.size for s in HfApi().model_info(REPO, files_metadata=True).siblings}
for a in ARQ:
    alvo = os.path.join(DEST, a)
    if os.path.exists(alvo):
        print("ja existe", alvo, os.path.getsize(alvo), "esperado", tam[a], flush=True); continue
    t0 = time.time()
    p = hf_hub_download(REPO, a, local_dir=DEST)
    s = os.path.getsize(p)
    print(f"ok {a} {s} B ({'confere' if s == tam[a] else 'TAMANHO DIFERE ' + str(tam[a])}) {time.time() - t0:.0f} s", flush=True)
print("FIM", flush=True)
