"""Fecha o job da A100 #1: mata o processo, sobe o ultimo checkpoint + journal + log para o repo publico."""
import json, subprocess, time
from pathlib import Path
from huggingface_hub import HfApi
subprocess.run(["pkill", "-f", "qat_ternario_klein.py"]); time.sleep(5)
d = Path("/content/qat_run"); a = HfApi(); r = "JoaoZaokk/klein4b-qat-ckpt"
ult = json.loads(d.joinpath("journal.jsonl").read_text().strip().splitlines()[-1])
print("ultimo journal:", ult.get("passo"))
for f, alvo in ((d / "journal.jsonl", "logs/journal_a100_1.jsonl"), (d / "qat.log", "logs/qat_a100_1.log")):
    a.upload_file(path_or_fileobj=str(f), path_in_repo=alvo, repo_id=r)
t = time.time()
a.upload_file(path_or_fileobj=str(d / "ckpt" / "ultimo.pt"), path_in_repo="ckpt/ultimo.pt", repo_id=r)
import torch
print("ckpt enviado em %.0f s; passo do ckpt:" % (time.time() - t),
      torch.load(d / "ckpt" / "ultimo.pt", map_location="cpu", weights_only=False, mmap=True)["passo"])
