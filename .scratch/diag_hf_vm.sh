#!/bin/bash
for p in $(pgrep -f 'scripts/colab/colab_job_supervisor.py'); do kill $p; done
sleep 3
ps -eo pid,args | grep -E '[c]olab_job_supervisor|[c]olab exec' | cut -c1-100
timeout 300 /home/pipeline/.local/bin/colab exec -s qat-a100 --timeout 240 -f /mnt/f/COMFY_PORTABLE/tools/colab_qat/celula_diag_hf.py 2>&1 | grep -E 'conta|upload|apagado|ERRO'
