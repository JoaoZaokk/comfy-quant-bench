#!/bin/bash
# Roda DENTRO do WSL (Ubuntu, usuario pipeline), com o supervisor da qat-a100b JA' PARADO (um dono do CLI).
# Sobe o tool novo + a fila e lanca; a fila espera o replay terminar sozinha. Depois disto, religar o
# supervisor (ele passa a observar /content/qat_fila, porque a celula troca o config.json).
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=qat-a100b
echo "== estado =="
timeout 180 $C exec -s $S --timeout 120 -f $W/tools/colab_qat/celula_estado_vm.py 2>&1 | tail -12
echo "== upload =="
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in tools/colab_qat/fila.json:fila.json \
           tools/colab_qat/probe_qat.py:probe_qat.py \
           tools/colab_qat/fila_qat_ternario_klein.py:fila_qat_ternario_klein.py \
           tools/colab_qat/celula_lanca_fila.py:celula_lanca_fila.py \
           bench/qat_klein/prompts_treino.txt:prompts_treino_124.txt; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | tail -1; done
echo "== lanca =="
timeout 300 $C exec -s $S --timeout 240 -f $W/tools/colab_qat/celula_lanca_fila.py 2>&1 | tail -6
