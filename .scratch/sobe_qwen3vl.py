"""Sobe ao Hub o encoder qwen3vl_4b da suite do Krea 2: W4A8 (recomendado) e W4A4 convrot (medido,
2,5x pior, publicado como comparacao), com sidecars e card. Apache 2.0 na origem. Token do ambiente."""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "JoaoZaokk/Qwen3-VL-4B-W4A8-ConvRot"
ROOT = Path(r"F:\COMFY_PORTABLE")
CARD = ROOT / "bench" / "hf" / "qwen3vl-4b-w4a8"
TE = ROOT / "ComfyUI" / "models" / "text_encoders"

provas = {
    "README.md": CARD / "README.md",
    "qwen3vl_4b_w4a8.safetensors": TE / "qwen3vl_4b_w4a8.safetensors",
    "qwen3vl_4b_w4a8.quant.json": TE / "qwen3vl_4b_w4a8.quant.json",
    "qwen3vl_4b_w4a4_convrot.safetensors": TE / "qwen3vl_4b_w4a4_convrot.safetensors",
    "qwen3vl_4b_w4a4_convrot.quant.json": TE / "qwen3vl_4b_w4a4_convrot.quant.json",
}
faltam = [k for k, v in provas.items() if not v.exists()]
if faltam:
    sys.exit(f"faltam provas, nao subo nada: {faltam}")
if "TBD" in (CARD / "README.md").read_text(encoding="utf-8"):
    sys.exit("README com TBD; nao subo")
api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
for remoto, local in provas.items():
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remoto, repo_id=REPO, repo_type="model",
                    commit_message=f"Qwen3-VL 4B text encoder for Krea 2: {remoto}")
    print(f"  subiu {remoto}  ({local.stat().st_size / 2**20:.1f} MiB)", flush=True)
no_hub = set(api.list_repo_files(REPO, repo_type="model"))
falta = [k for k in provas if k not in no_hub]
print("QWEN3VL_UPLOAD_OK" if not falta else f"FALTA NO HUB: {falta}")
print("NAO COBERTO: nenhuma imagem renderizada para este card; os numeros sao de condicionamento (bench/krea2_suite.md §8).")
