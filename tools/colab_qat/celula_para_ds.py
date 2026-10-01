"""Celula (colab exec): para o gerador do dataset (SIGTERM, espera, depois SIGKILL). O que ja' subiu fica no HF;
o que ficou local e' enviado no proximo lancamento (o gerador nao pula shard local sem envio)."""
import sys

sys.path.insert(0, "/content/qat")
import colab_ops

colab_ops.para(["gera_dataset_klein.py"], espera_s=120)
