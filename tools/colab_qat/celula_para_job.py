"""Celula (colab exec): para o QAT (SIGTERM -> checkpoint ao fim do passo; SIGKILL so' se nao sair)."""
import glob
import sys

sys.path.insert(0, "/content/qat")
import colab_ops

colab_ops.para(["qat_ternario_klein.py"], espera_s=900)
print("shards professor:", len(glob.glob("/content/qat_run/professor/*.pt")))
