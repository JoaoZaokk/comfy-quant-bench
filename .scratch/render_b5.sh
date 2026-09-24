#!/bin/bash
# Render + epsilon do b5 a partir dos BFL ja' no disco (sem baixar). melhor = final = p4000, entao so' o final.
# So' rodar com a 3090 livre (o dono pediu para esperar). Depois: grade com .scratch/grade_colunas.py.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models klein4b_qat_b5_final_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_b5 > $D/ladder_b5.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_b5/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_b5 > $D/decode_b5.log 2>&1; echo "decode rc=$?"
$PY -s .scratch/roda_eps_generico.py b5 braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors b5_final=klein4b_qat_b5_final_bfl.safetensors \
  qatA100_3024=klein4b_qatA100_bfl.safetensors > $D/eps_b5_driver.log 2>&1; echo "eps rc=$?"
$PY -s tools/agrega_epsilon_sementes.py $D/eps_b5.log --base braco0_ptq > $D/agrega_eps_b5.txt 2>&1; echo "agrega rc=$?"
echo "=== FIM render b5 $(date +%T) ==="
