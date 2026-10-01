#!/bin/bash
# Mantem o kernel da qat-a100c ativo (celula trivial a cada 4 min) ate' existir /mnt/f/COMFY_PORTABLE/.scratch/para_keepalive_a100c
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; F=$W/.scratch/para_keepalive_a100c
rm -f $F
while [ ! -f $F ]; do
  out=$(timeout 120 $C exec -s qat-a100c --timeout 60 -f $W/tools/colab_qat/celula_vivo.py 2>&1 | grep -v '^\[colab\]' | tail -1)
  echo "$(date +%T) $out"
  for i in $(seq 24); do [ -f $F ] && break; sleep 10; done
done
echo "keepalive parado $(date +%T)"
