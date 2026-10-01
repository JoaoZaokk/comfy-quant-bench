"""Celula (colab exec): lanca o gerador do dataset do professor em background (setsid nohup)."""
import sys
import time
from pathlib import Path

Q = Path("/content/qat")
sys.path.insert(0, str(Q))
import colab_ops

colab_ops.exige([Q / f for f in colab_ops.ARQUIVOS_QAT] + [Q / "gera_dataset_klein.py", Q / "ds_config.json"])
print("token de escrita presente:", Path("/root/.cache/huggingface/token").is_file())
for velho in (Path("/content/ds/ds.log"), Path("/content/ds/status.json")):
    if velho.is_file():  # senao o probe le o FIM velho e declara DONE na hora
        velho.rename(velho.with_name(f"{velho.stem}.{time.strftime('%H%M%S')}{velho.suffix}"))
colab_ops.lanca([sys.executable, "-u", Q / "gera_dataset_klein.py"], Path("/content/ds/ds.log"))
