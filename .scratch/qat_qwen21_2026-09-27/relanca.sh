#!/bin/bash
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; T=$W/tools/colab_qat_qwen21; S=${1:-qwq-g4}
timeout 300 $C upload -s $S $T/qat_qwen21_blocos.py /content/qatq/qat_qwen21_blocos.py 2>&1 | grep -v '^\[colab\]' | tail -1
timeout 300 $C exec -s $S --timeout 240 -f $T/celula_lanca.py 2>&1 | grep -v '^\[colab\]' | tail -2
