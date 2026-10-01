#!/bin/bash
for p in $(pgrep -f 'colab_job_supervisor.py --session qat-a100b'); do kill $p; done
sleep 3
timeout 900 /home/pipeline/.local/bin/colab exec -s qat-a100b --timeout 800 -f /mnt/f/COMFY_PORTABLE/tools/colab_qat/celula_sobe_origem.py 2>&1 | grep -E "passo|enviada|Error"
