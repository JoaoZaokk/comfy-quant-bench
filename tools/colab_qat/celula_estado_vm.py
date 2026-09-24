"""Celula (colab exec): estado da VM antes de lancar a fila -- disco, GPU, shards, passo do replay."""
import json
import shutil
import subprocess
from pathlib import Path

t, u, f = shutil.disk_usage("/content")
print(f"disco /content: livre {f / 2**30:.0f} GiB de {t / 2**30:.0f}")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
for sub in ("professor", "professor_holdout"):
    print(sub, len(list(Path("/content/qat_run", sub).glob("p*_s1.pt"))), "shards semente 1")
jr = Path("/content/qat_run/journal.jsonl")
if jr.is_file():
    print("replay ultimo:", jr.read_text().strip().splitlines()[-1][:200])
cf = Path("/content/qat/config.json")
print("config atual:", cf.read_text() if cf.is_file() else "(nenhum -- VM nova)")
print("klein:", Path("/content/klein4b/transformer/diffusion_pytorch_model.safetensors").is_file())
tok = Path("/root/.cache/huggingface/token")
print("token de escrita presente:", tok.is_file())
print(json.dumps({k: v for k, v in [("hf_home", str(Path('/root/.cache/huggingface')))]}))
