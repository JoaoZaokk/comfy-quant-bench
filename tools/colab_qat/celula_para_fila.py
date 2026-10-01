"""Celula (colab exec): para a fila de QAT e o braco em curso, sem apagar nada.

SIGTERM nos dois; o braco grava checkpoint ao fim do passo em curso (e o envia, se `--push-hf`) e a fila nao
lanca o proximo. So' depois de `espera_s` sem sair vem o SIGKILL (o antigo esperava 8 s e perdia ate 25 min).
"""
import subprocess
import sys

sys.path.insert(0, "/content/qat")
import colab_ops

colab_ops.para(["fila_qat_ternario_klein.py", "qat_ternario_klein.py"], espera_s=900)
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True,
                     text=True, check=False).stdout.strip())
