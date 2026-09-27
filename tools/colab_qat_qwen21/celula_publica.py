"""Celula (colab exec): sobe o checkpoint QAT e o relatorio para um repo PRIVADO do HF (token enviado por mim para
/root/.cache/huggingface/token, conteudo nunca lido aqui). Repo e caminhos vem do config."""
import json
from pathlib import Path
from huggingface_hub import HfApi
Q = Path("/content/qatq")
cfg = json.loads((Q / "config.json").read_text())
d = Path(cfg["dir"])
api = HfApi()
api.create_repo(cfg["repo"], private=True, exist_ok=True)
for arq in ("relatorio.json", "qat.log", "qwen_image_2.1_w4a4_qat.safetensors"):
    p = d / arq
    if p.is_file():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=arq, repo_id=cfg["repo"])
        print("enviado", arq, p.stat().st_size)
print("publicado", cfg["repo"])
