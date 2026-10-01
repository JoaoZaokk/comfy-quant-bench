"""Sobe o W4A8 do Qwen-Image 2512 BASE. Repo separado do Edit, como o dono pediu."""
import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import HfApi  # noqa: E402
REPO = "JoaoZaokk/Qwen-Image-2512-W4A8-ConvRot"
CARD = r"F:\COMFY_PORTABLE\bench\hf\qwen-image-2512-quant"
P = r"P:\ComfyBench\diffusion_models"
ARQ = [
    (os.path.join(P, "qwen_image_2512_w4a8.safetensors"), "qwen_image_2512_w4a8.safetensors"),
    (os.path.join(P, "qwen_image_2512_w4a8.quant.json"), "qwen_image_2512_w4a8.quant.json"),
    (os.path.join(CARD, "images", "tres_bracos.png"), "images/tres_bracos.png"),
    (os.path.join(CARD, "README.md"), "README.md"),
]
api = HfApi()
print("autenticado como", api.whoami().get("name"), flush=True)
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
print("repo:", f"https://huggingface.co/{REPO}", flush=True)
for local, remoto in ARQ:
    if not os.path.exists(local):
        print("FALTA", local, flush=True); sys.exit(1)
    print(f"subindo {remoto} ({os.path.getsize(local):,} B) ...", flush=True)
    t0=time.time()
    api.upload_file(path_or_fileobj=local, path_in_repo=remoto, repo_id=REPO, repo_type="model")
    print(f"  ok em {time.time()-t0:.0f}s", flush=True)
print("QWEN_BASE_UPLOAD_OK", flush=True)
