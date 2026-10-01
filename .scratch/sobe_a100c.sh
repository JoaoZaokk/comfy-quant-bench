#!/bin/bash
# Sobe a qat-a100c (A100-40), mede o ambiente e baixa o snapshot do klein. Nao lanca treino.
export REQUEST_TIMEOUT=600
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=qat-a100c
timeout 300 $C new -s $S --gpu A100 2>&1 | grep -v '^\[colab\]' | tail -3
$C sessions 2>&1 | grep -v '^\[colab\]'
timeout 180 $C exec -s $S --timeout 120 -f $W/tools/colab_qat/celula_ambiente.py 2>&1 | grep AMBIENTE
timeout 120 $C exec -s $S --timeout 60 -f $W/tools/colab_qat/celula_pastas.py 2>&1 | tail -1
timeout 1500 $C exec -s $S --timeout 1200 -f $W/tools/colab_qat/celula_prepara.py 2>&1 | grep -E "diffusers da VM|snapshot|Error|Traceback"
echo "=== FIM sobe $(date +%T) ==="
