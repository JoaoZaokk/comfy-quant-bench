#!/bin/bash
export REQUEST_TIMEOUT=600
C=/home/pipeline/.local/bin/colab
W=/mnt/f/COMFY_PORTABLE
S=qat-a100b
timeout 300 $C new -s $S --gpu A100 2>&1 | tail -2
$C sessions 2>&1 | grep -E "qat-a100" 
timeout 120 $C exec -s $S --timeout 60 -f $W/tools/colab_qat/celula_pastas.py 2>&1 | tail -1
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in bench/qat_klein/prompts_treino_parti.txt:prompts_treino.txt bench/qat_klein/prompts_holdout.txt:prompts_holdout.txt \
           .scratch/wheels_colab/diffusers-0.38.0-py3-none-any.whl:diffusers-0.38.0-py3-none-any.whl \
           tools/colab_qat/probe_qat.py:probe_qat.py tools/colab_qat/config_a100b.json:config.json tools/colab_qat/celula_lanca.py:celula_lanca.py; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | tail -1
done
timeout 1500 $C exec -s $S --timeout 1200 -f $W/tools/colab_qat/celula_lanca.py 2>&1 | grep -E "diffusers da VM|snapshot|inicio|LANCADO|FALHOU|Error"
