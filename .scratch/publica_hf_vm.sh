#!/bin/bash
for p in $(pgrep -f 'scripts/colab/colab_job_supervisor.py'); do kill $p; done
sleep 3
timeout 300 /home/pipeline/.local/bin/colab exec -s qat-a100 --timeout 120 -f /mnt/f/COMFY_PORTABLE/tools/colab_qat/celula_publico.py 2>&1 | grep -E 'publico|Error'
