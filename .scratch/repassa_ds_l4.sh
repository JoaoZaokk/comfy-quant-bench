#!/bin/bash
# Passada final na ds-l4: refaz o que falta no metadata (holdout sem `camadas` e linhas perdidas). Supervisor PARADO.
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=ds-l4
timeout 180 $C exec -s $S --timeout 120 -f $W/tools/colab_qat/celula_gira_log_ds.py 2>&1 | grep -v '^\[colab\]' | tail -2
timeout 300 $C exec -s $S --timeout 240 -f $W/tools/colab_qat/celula_lanca_ds.py 2>&1 | grep -v '^\[colab\]' | tail -3
