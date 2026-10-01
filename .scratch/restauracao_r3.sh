#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe; MD=ComfyUI/models/diffusion_models; D=bench/qat_klein
OR=$MD/klein4b_braco3_bf16_original_bfl.safetensors
CORPO='double_blocks\.\d+\.(img|txt)_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.\d+\.linear[12]\.weight'
H14='double_blocks\.\d+\.txt_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.1[0-9]\.linear[12]\.weight'
mk() { [ -f $MD/$1 ] || $PY -s tools/mistura_klein.py "${@:2}" --saida $MD/$1 || exit 1; }
mk klein4b_C3_corpo_q3_bfl.safetensors --base $OR --doador $OR --regex "$CORPO" --rtn4 32 --bits 3 --esperadas 80
mk klein4b_C3g16_corpo_q3_bfl.safetensors --base $OR --doador $OR --regex "$CORPO" --rtn4 16 --bits 3 --esperadas 80
mk klein4b_M43_bfl.safetensors --base $MD/klein4b_C3_corpo_q3_bfl.safetensors --doador $OR --regex "$H14" --rtn4 32 --bits 4 --esperadas 40
MS="klein4b_C3_corpo_q3_bfl.safetensors klein4b_C3g16_corpo_q3_bfl.safetensors klein4b_M43_bfl.safetensors"
echo "=== prontos $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_r3 > $D/ladder_fixa_r3.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_r3/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_r3 > $D/decode_fixa_r3.log 2>&1; echo "decode fixa rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_r3 > $D/ladder_r3.log 2>&1; echo "ladder grade rc=$?"
$PY -s tools/decode_latents.py $D/render_r3/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_r3 > $D/decode_r3.log 2>&1; echo "decode grade rc=$?"
echo "=== eps $(date +%T) ==="
$PY -s .scratch/roda_eps_generico.py rest braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors b6=klein4b_qat_b6_final_bfl.safetensors \
  H14=klein4b_b6_rest_H14_bfl.safetensors H14q4=klein4b_b6_rest_H14q4_bfl.safetensors C1_q4=klein4b_C1_corpo_q4_bfl.safetensors \
  C3_q3=klein4b_C3_corpo_q3_bfl.safetensors C3g16=klein4b_C3g16_corpo_q3_bfl.safetensors M43=klein4b_M43_bfl.safetensors > $D/eps_rest_driver.log 2>&1; echo "eps rc=$?"
$PY -s tools/agrega_epsilon_sementes.py $D/eps_rest.log --base braco0_ptq > $D/agrega_eps_rest.txt 2>&1; echo "agrega rc=$?"
echo "=== FIM r3 $(date +%T) ==="
