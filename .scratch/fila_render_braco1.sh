#!/bin/bash
# Fila do criterio_render_braco1_2026-09-22.md: R (ComfyUI) -> X (diffusers) -> E (epsilon).
# Cada ferramenta toma o proprio lock; CUDA_VISIBLE_DEVICES=0 para o guard nao olhar a cuda:1 do cortex.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
D=bench/render_braco1_2026-09-22
PY=./python_embeded/python.exe
while [ ! -f ComfyUI/models/diffusion_models/klein4b_braco3_bf16_original_bfl.safetensors ]; do sleep 20; done
grep -q FIM .scratch/copia_klein_local.log || sleep 30
echo "=== R: quality_ladder $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_braco0_ternario_ingenuo_bfl.safetensors klein4b_braco1_compensado_bfl.safetensors klein4b_braco2_bonsai_ternario_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --vae flux2_klein_vae_diffusers.safetensors --out $D/comfy \
  > $D/ladder.log 2>&1; echo "ladder rc=$?"
echo "=== X: diffusers $(date +%T) ==="
$PY -s tools/render_klein_diffusers.py --raiz P:/ComfyBench/originais/FLUX.2-klein-4B \
  --braco braco3_bf16=F:/bonsai-re/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors \
  --braco braco1_ajuste=P:/ComfyBench/originais/klein4b_braco1_compensado.safetensors \
  --braco braco0_ptq=P:/ComfyBench/originais/klein4b_ternario_ingenuo.safetensors \
  --prompt-file $D/prompts.txt --seeds 11 12 --steps 8 --size 1024 --out $D/diffusers \
  > $D/diffusers.log 2>&1; echo "diffusers rc=$?"
echo "=== E: epsilon $(date +%T) ==="
$PY -s .scratch/roda_eps_foradaamostra.py > $D/eps_driver.log 2>&1; echo "eps rc=$?"
echo "=== FIM $(date +%T) ==="
