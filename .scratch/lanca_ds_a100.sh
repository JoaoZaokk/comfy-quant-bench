#!/bin/bash
# Poe a qat-a100c para gerar o dataset (Parti semente 2) em paralelo com a ds-l4. Supervisor da A100 PARADO antes.
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=qat-a100c
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in tools/colab_qat/gera_dataset_klein.py:gera_dataset_klein.py tools/colab_qat/ds_config_a100_s2.json:ds_config.json \
           bench/qat_klein/prompts_treino_parti.txt:prompts_treino.txt; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | grep -v '^\[colab\]' | tail -1; done
timeout 120 $C upload -s $S $W/.hf/token /root/.cache/huggingface/token 2>&1 | grep -v '^\[colab\]' | tail -1
timeout 300 $C exec -s $S --timeout 240 -f $W/tools/colab_qat/celula_lanca_ds.py 2>&1 | grep -v '^\[colab\]' | tail -3
