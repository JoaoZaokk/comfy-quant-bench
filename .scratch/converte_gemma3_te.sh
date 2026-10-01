#!/bin/bash
# O text encoder de fabrica do LTX 2.3 em W4A8, para a "suite inteira" ter o encoder tambem.
# Espera as conversoes do DiT terminarem na mesma placa.
cd /f/COMFY_PORTABLE
until grep -q 'LTX23_CONVERSOES_DONE' .scratch/converte_ltx23.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=1
echo "=== gemma w4a8 $(date)"
./python_embeded/python.exe -s tools/quant_w4a8.py --input P:/ComfyBench/text_encoders/gemma_3_12B_it.safetensors --profile gemma --output P:/ComfyBench/text_encoders/gemma_3_12B_it_w4a8.safetensors 2>&1 | tail -12
echo "GEMMA3_W4A8_DONE $(date)"
