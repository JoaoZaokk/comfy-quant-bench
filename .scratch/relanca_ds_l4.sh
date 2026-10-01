#!/bin/bash
# Para o gerador na ds-l4, sobe o codigo novo (estatisticas por camada) e relanca. Supervisor da ds-l4 PARADO antes.
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=ds-l4
timeout 240 $C exec -s $S --timeout 180 -f $W/tools/colab_qat/celula_para_ds.py 2>&1 | grep -v '^\[colab\]' | tail -2
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in tools/colab_qat/gera_dataset_klein.py:gera_dataset_klein.py tools/colab_qat/probe_ds.py:probe_ds.py; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | grep -v '^\[colab\]' | tail -1; done
timeout 300 $C exec -s $S --timeout 240 -f $W/tools/colab_qat/celula_lanca_ds.py 2>&1 | grep -v '^\[colab\]' | tail -3
