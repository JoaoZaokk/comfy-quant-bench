#!/bin/sh
# Orçamento Vulkan do heap de VRAM da A750 a cada ~2 s (inclui todos os processos). Uso: vigia_vram.sh <saida>
exec docker run --rm --name vigia-vram --device /dev/dri --entrypoint sh zp-wcpp:latest -c \
  'while true; do printf "%s " "$(date +%s)"; vulkaninfo 2>/dev/null | grep -A2 "memoryHeaps\[0\]" | grep budget | head -1; sleep 1; done' > "$1"
