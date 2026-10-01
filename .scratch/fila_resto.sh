#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
D=bench/render_braco1_2026-09-22
PY=./python_embeded/python.exe
echo "=== E: epsilon $(date +%T) ==="
$PY -s .scratch/roda_eps_foradaamostra.py > $D/eps_driver.log 2>&1; echo "eps rc=$?"
echo "=== X: diffusers braco0 $(date +%T) ==="
$PY -s tools/render_klein_diffusers.py --raiz P:/ComfyBench/originais/FLUX.2-klein-4B \
  --braco braco0_ptq=P:/ComfyBench/originais/klein4b_ternario_ingenuo.safetensors \
  --prompt-file $D/prompts.txt --seeds 11 12 --steps 8 --size 1024 --out $D/diffusers \
  > $D/diffusers_b0.log 2>&1; echo "diffusers b0 rc=$?"
echo "=== FIM resto $(date +%T) ==="
bash .scratch/fila_mestre.sh
