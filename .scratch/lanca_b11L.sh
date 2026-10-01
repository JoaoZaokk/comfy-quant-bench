#!/bin/bash
# Roda DENTRO do WSL com o supervisor da qat-a100c PARADO (um dono do CLI). Continua o b11 por mais 4000 passos.
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=qat-a100c
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in tools/colab_qat/fila_b11L.json:fila.json \
           tools/colab_qat/fila_qat_ternario_klein.py:fila_qat_ternario_klein.py \
           tools/colab_qat/celula_lanca_fila.py:celula_lanca_fila.py tools/colab_qat/probe_qat.py:probe_qat.py \
           tools/colab_qat/celula_baixa_ini_b11.py:celula_baixa_ini_b11.py \
           bench/qat_klein/prompts_treino.txt:prompts_treino_124.txt \
           bench/qat_klein/prompts_holdout.txt:prompts_holdout.txt; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | grep -v '^\[colab\]' | tail -1; done
# token de escrita do HF junto com os outros arquivos (pedido do dono, 25/09): conteudo nao e' lido aqui
timeout 120 $C upload -s $S $W/.hf/token /root/.cache/huggingface/token 2>&1 | grep -v '^\[colab\]' | tail -1
echo "== ckpt b11 =="
timeout 900 $C exec -s $S --timeout 840 -f $W/tools/colab_qat/celula_baixa_ini_b11.py 2>&1 | grep -v '^\[colab\]' | tail -6
echo "== lanca =="
timeout 300 $C exec -s $S --timeout 240 -f $W/tools/colab_qat/celula_lanca_fila.py 2>&1 | grep -v '^\[colab\]' | tail -8
