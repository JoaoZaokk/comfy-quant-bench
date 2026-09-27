#!/bin/bash
# Sessao CPU (sem GPU paga): valida os imports do comfy e o smoke do QAT na VM. Um dono do CLI.
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; S=qwq-cpu; D=$W/.scratch/qat_qwen21_2026-09-27
export REQUEST_TIMEOUT=600
timeout 600 $C new -s $S 2>&1 | grep -v '^\[colab\]' | tail -2
timeout 180 $C exec -s $S --timeout 120 -f $D/celula_pastas.py 2>&1 | grep -v '^\[colab\]' | tail -1
for par in $D/comfy_v0374.zip:comfy_v0374.zip $D/config_cpu.json:config.json $W/tools/colab_qat_qwen21/qat_qwen21_blocos.py:qat_qwen21_blocos.py \
           $W/tools/colab_qat_qwen21/celula_prepara.py:celula_prepara.py; do
  timeout 600 $C upload -s $S ${par%%:*} /content/qatq/${par##*:} 2>&1 | grep -v '^\[colab\]' | tail -1; done
echo "== prepara"; timeout 900 $C exec -s $S --timeout 800 -f $W/tools/colab_qat_qwen21/celula_prepara.py 2>&1 | grep -v '^\[colab\]' | tail -30
echo "== smoke"; timeout 900 $C exec -s $S --timeout 800 -f $D/celula_smoke.py 2>&1 | grep -v '^\[colab\]' | tail -15
timeout 300 $C stop -s $S 2>&1 | grep -v '^\[colab\]' | tail -1
$C sessions 2>&1 | grep -v '^\[colab\]' | tail -3
