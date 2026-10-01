#!/bin/bash
# Sobe a ds-l4 (L4), prepara, sobe tools/prompts/token e lanca o gerador do dataset do professor.
export REQUEST_TIMEOUT=600
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=ds-l4
timeout 300 $C new -s $S --gpu L4 2>&1 | grep -v '^\[colab\]' | tail -3
timeout 180 $C exec -s $S --timeout 120 -f $W/tools/colab_qat/celula_ambiente.py 2>&1 | grep AMBIENTE
timeout 120 $C exec -s $S --timeout 60 -f $W/tools/colab_qat/celula_pastas.py 2>&1 | tail -1
timeout 1500 $C exec -s $S --timeout 1200 -f $W/tools/colab_qat/celula_prepara.py 2>&1 | grep -E "diffusers da VM|snapshot|Error|Traceback"
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
sobe_qat "$S" || exit 1
for par in tools/colab_qat/gera_dataset_klein.py:gera_dataset_klein.py tools/colab_qat/ds_config.json:ds_config.json \
           tools/colab_qat/probe_ds.py:probe_ds.py \
           bench/qat_klein/prompts_holdout.txt:prompts_holdout.txt bench/qat_klein/prompts_treino_parti.txt:prompts_treino.txt \
           bench/qat_klein/avaliacao_fixa/prompts.txt:prompts_fixa.txt bench/render_braco1_2026-09-22/prompts.txt:prompts_grade.txt; do
  timeout 300 $C upload -s $S $W/${par%%:*} /content/qat/${par##*:} 2>&1 | grep -v '^\[colab\]' | tail -1; done
# token de escrita do HF junto com os outros arquivos (pedido do dono, 25/09): conteudo nao e' lido aqui
timeout 120 $C upload -s $S $W/.hf/token /root/.cache/huggingface/token 2>&1 | grep -v '^\[colab\]' | tail -1
timeout 300 $C exec -s $S --timeout 240 -f $W/tools/colab_qat/celula_lanca_ds.py 2>&1 | grep -v '^\[colab\]' | tail -4
echo "=== FIM sobe ds $(date +%T) ==="
