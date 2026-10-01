#!/bin/bash
# Avaliacao do b7 (resto congelado). Criterio: bench/criterio_b7_b7_congela_resto_2026-09-24.md
# grade principal (5 x sementes 11 12), avaliacao fixa (10 prompts, semente 11) com controle p3024, T3_ctrl e b6 pareados, epsilon.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
REF=klein4b_braco3_bf16_original_bfl.safetensors
$PY -s tools/quality_ladder.py --reference $REF --models klein4b_qat_b7_final_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_b7 > $D/ladder_b7.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_b7/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_b7 > $D/decode_b7.log 2>&1; echo "decode rc=$?"
$PY -s tools/quality_ladder.py --reference $REF \
  --models klein4b_qat_b7_final_bfl.safetensors klein4b_qatA100_bfl.safetensors klein4b_transp_T3_ctrl_bfl.safetensors klein4b_qat_b6_final_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_b7 > $D/ladder_fixa_b7.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_b7/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_b7 > $D/decode_fixa_b7.log 2>&1; echo "decode fixa rc=$?"
$PY -s .scratch/roda_eps_generico.py b7 braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors \
  b7=klein4b_qat_b7_final_bfl.safetensors ctrl_p3024=klein4b_qatA100_bfl.safetensors T3_ctrl=klein4b_transp_T3_ctrl_bfl.safetensors b6=klein4b_qat_b6_final_bfl.safetensors \
  > $D/eps_b7_driver.log 2>&1; echo "eps rc=$?"
$PY -s tools/agrega_epsilon_sementes.py $D/eps_b7.log --base braco0_ptq > $D/agrega_eps_b7.txt 2>&1; echo "agrega rc=$?"
echo "=== FIM render b7 $(date +%T) ==="
