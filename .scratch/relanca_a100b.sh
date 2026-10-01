#!/bin/bash
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=qat-a100b
for p in $(pgrep -f 'colab_job_supervisor.py --session qat-a100b'); do kill $p; done
sleep 3
timeout 1020 $C exec -s $S --timeout 960 -f $W/tools/colab_qat/celula_para_job.py 2>&1 | tail -2
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in tools/colab_qat/config_a100b.json:config.json tools/colab_qat/probe_qat.py:probe_qat.py tools/colab_qat/celula_lanca.py:celula_lanca.py; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | tail -1; done
timeout 1500 $C exec -s $S --timeout 1200 -f $W/tools/colab_qat/celula_lanca.py 2>&1 | grep -E "inicio|LANCADO|FALHOU|Error"
