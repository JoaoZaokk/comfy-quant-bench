"""Sobe o W4A8 do Wan 2.2 TI2V 5B com a prova visual dos tres bracos.

O W4A4 NAO sobe como peso: e negativo medido. A imagem que mostra o borrao sobe, porque a
afirmacao central do card -- que 4 bits de ativacao falham nesta familia tambem -- tem de ser
conferivel sem que ninguem precise baixar um arquivo que nao deve usar.
"""
import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import HfApi  # noqa: E402

REPO = "JoaoZaokk/Wan2.2-TI2V-5B-W4A8-ConvRot"
CARD = r"F:\COMFY_PORTABLE\bench\hf\wan22-ti2v-5b-w4a8"
P = r"P:\ComfyBench\diffusion_models"
ARQ = [
    (os.path.join(P, "wan2.2_ti2v_5B_w4a8.safetensors"), "wan2.2_ti2v_5B_w4a8.safetensors"),
    (os.path.join(P, "wan2.2_ti2v_5B_w4a8.quant.json"), "wan2.2_ti2v_5B_w4a8.quant.json"),
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
    t0 = time.time()
    api.upload_file(path_or_fileobj=local, path_in_repo=remoto, repo_id=REPO, repo_type="model")
    print(f"  ok em {time.time()-t0:.0f}s", flush=True)
print("WAN_UPLOAD_OK", flush=True)
