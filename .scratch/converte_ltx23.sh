#!/bin/bash
# LTX 2.3 distilled 1.1: W4A8 (candidato) e W4A4 (controle negativo), na 3080 Ti para nao
# contaminar o s/quadro dos renders que rodam na 3090.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=1
SRC=P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
echo "=== w4a8 $(date)"
./python_embeded/python.exe -s tools/quant_w4a8.py --input $SRC --output P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a8.safetensors 2>&1 | tail -25
echo "=== w4a4 $(date)"
./python_embeded/python.exe -s tools/quant_w4a4.py --input $SRC --profile ltx_2_5 --output P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a4.safetensors 2>&1 | tail -25
echo "LTX23_CONVERSOES_DONE $(date)"
