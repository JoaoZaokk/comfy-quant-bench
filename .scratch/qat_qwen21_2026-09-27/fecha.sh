#!/bin/bash
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; T=$W/tools/colab_qat_qwen21; S=${1:-qwq-g4}; D=$W/.scratch/qat_qwen21_2026-09-27
timeout 1500 $C exec -s $S --timeout 1400 -f $T/celula_publica.py 2>&1 | grep -v '^\[colab\]' | tail -6
timeout 300 $C download -s $S /content/qatq/run/relatorio.json $D/relatorio_vm.json 2>&1 | grep -v '^\[colab\]' | tail -1
timeout 300 $C download -s $S /content/qatq/run/qat.log $D/qat_vm.log 2>&1 | grep -v '^\[colab\]' | tail -1
timeout 300 $C stop -s $S 2>&1 | grep -v '^\[colab\]' | tail -1
$C sessions 2>&1 | grep -v '^\[colab\]' | tail -2
