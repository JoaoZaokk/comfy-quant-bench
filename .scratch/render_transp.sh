#!/bin/bash
# Render dos transplantes T1/T2/T3 (PTQ <-> b4d): grade principal (5 prompts x 2 sementes), generalizacao
# (10 prompts, semente 11) e epsilon fora da amostra. Criterio: bench/criterio_transplante_flips_2026-09-24.md
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
MS="klein4b_transp_T1_b4d_bfl.safetensors klein4b_transp_T2_b4d_bfl.safetensors klein4b_transp_T3_b4d_bfl.safetensors"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_transp > $D/ladder_transp.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_transp/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_transp > $D/decode_transp.log 2>&1; echo "decode rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/generaliza/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/generaliza_transp > $D/ladder_generaliza_transp.log 2>&1; echo "ladder gen rc=$?"
$PY -s tools/decode_latents.py $D/generaliza_transp/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/generaliza_transp > $D/decode_generaliza_transp.log 2>&1; echo "decode gen rc=$?"
$PY -s .scratch/roda_eps_generico.py transp braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors \
  T1=klein4b_transp_T1_b4d_bfl.safetensors T2=klein4b_transp_T2_b4d_bfl.safetensors T3=klein4b_transp_T3_b4d_bfl.safetensors \
  b4d=klein4b_qat_b4d_final_bfl.safetensors > $D/eps_transp_driver.log 2>&1; echo "eps rc=$?"
$PY -s tools/agrega_epsilon_sementes.py $D/eps_transp.log --base braco0_ptq > $D/agrega_eps_transp.txt 2>&1; echo "agrega rc=$?"
echo "=== FIM render transp $(date +%T) ==="
