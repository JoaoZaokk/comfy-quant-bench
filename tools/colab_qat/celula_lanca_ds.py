"""Celula (colab exec): lanca o gerador do dataset do professor em background (setsid nohup)."""
import subprocess
import sys
from pathlib import Path

Q = Path("/content/qat")
for f in ("qat_ternario_klein.py", "ajusta_denso_diffusers.py", "gera_dataset_klein.py", "ds_config.json"):
    if not (Q / f).is_file():
        print(f"FALTA {f}")
        raise SystemExit(1)
print("token de escrita presente:", Path("/root/.cache/huggingface/token").is_file())
Path("/content/ds").mkdir(parents=True, exist_ok=True)
log = open("/content/ds/ds.log", "ab")
p = subprocess.Popen(["setsid", "nohup", sys.executable, "-u", str(Q / "gera_dataset_klein.py")],
                     stdout=log, stderr=subprocess.STDOUT, cwd=str(Q), start_new_session=True)
print(f"DATASET LANCADO pid={p.pid}")
