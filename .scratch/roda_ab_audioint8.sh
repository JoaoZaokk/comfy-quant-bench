#!/bin/bash
# Dois renders em sequencia no ComfyUI 8190: misto (audio INT8) e W4A8 de referencia, mesmo grafo e seed.
cd /f/COMFY_PORTABLE; SP="$1"
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8190/queue >/dev/null && break; sleep 5; done
for b in audioint8 w4a8ref; do
  ./python_embeded/python.exe -s .scratch/roda_eros_2gpu.py "$SP/prompt_${b}_api.json" .scratch/eros_${b}_amostras.csv 2>&1 | sed "s/^/$b /"
done
echo "=== FIM AB $(date +%T) ==="
