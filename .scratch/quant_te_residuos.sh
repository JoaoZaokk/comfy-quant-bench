#!/bin/bash
# Quatro conversoes, CPU, uma de cada vez. Log: .scratch/quant_te_residuos.log
cd /f/COMFY_PORTABLE
P=/p/ComfyBench/checkpoints/ltx-2.3_text_projection_bf16.safetensors
G=ComfyUI/models/text_encoders/gemma_3_12B_it_heretic_w4a8.safetensors
for par in "$P int8" "$P fp8" "$G int8" "$G fp8"; do
  set -- $par
  echo "=== $(date +%H:%M:%S) $1 $2"
  ./python_embeded/python.exe -s tools/quant_te_residuos.py --input "$1" --format "$2" || echo "FALHOU rc=$? $1 $2"
done
echo "=== FIM $(date +%H:%M:%S)"
