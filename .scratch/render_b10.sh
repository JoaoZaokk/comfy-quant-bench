#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe; MD=ComfyUI/models/diffusion_models; D=bench/qat_klein
[ -f $MD/klein4b_C1_corpo_q4_bfl.safetensors ] || cp /p/qat_klein_arquivo/hibridos/klein4b_C1_corpo_q4_bfl.safetensors $MD/
M=klein4b_qat_b10_melhor_bfl.safetensors
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $M \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_b10 > $D/ladder_fixa_b10.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_b10/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_b10 > $D/decode_fixa_b10.log 2>&1; echo "decode fixa rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $M \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_b10 > $D/ladder_b10.log 2>&1; echo "ladder grade rc=$?"
$PY -s tools/decode_latents.py $D/render_b10/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_b10 > $D/decode_b10.log 2>&1; echo "decode grade rc=$?"
$PY -s .scratch/roda_eps_generico.py b10 braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors \
  C1_q4=klein4b_C1_corpo_q4_bfl.safetensors b10=$M > $D/eps_b10_driver.log 2>&1; echo "eps rc=$?"
$PY -s tools/agrega_epsilon_sementes.py $D/eps_b10.log --base C1_q4 > $D/agrega_eps_b10.txt 2>&1; echo "agrega rc=$?"
echo "=== FIM render b10 $(date +%T) ==="
