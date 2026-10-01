#!/bin/bash
for p in $(pgrep -f 'colab_job_supervisor.py --session qat-a100 '); do kill $p; done
sleep 3
timeout 900 /home/pipeline/.local/bin/colab exec -s qat-a100 --timeout 800 -f /mnt/f/COMFY_PORTABLE/tools/colab_qat/celula_fecha_a100.py 2>&1 | grep -E 'ultimo|enviado|Error'
timeout 300 /home/pipeline/.local/bin/colab download -s qat-a100 /content/qat_run/journal.jsonl /mnt/f/COMFY_PORTABLE/bench/qat_klein/journal_a100_1.jsonl 2>&1 | tail -1
timeout 120 /home/pipeline/.local/bin/colab stop -s qat-a100 2>&1 | tail -1
/home/pipeline/.local/bin/colab sessions 2>&1 | grep -E "qat|No active"
ps -eo pid,args | grep '[c]olab_job_supervisor' | cut -c1-110
