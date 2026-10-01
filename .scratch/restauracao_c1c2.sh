#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe; MD=ComfyUI/models/diffusion_models; D=bench/qat_klein
OR=$MD/klein4b_braco3_bf16_original_bfl.safetensors; P0=$MD/klein4b_braco0_ternario_ingenuo_bfl.safetensors
CORPO='double_blocks\.\d+\.(img|txt)_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.\d+\.linear[12]\.weight'
H14='double_blocks\.\d+\.txt_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.1[0-9]\.linear[12]\.weight'
[ -f $MD/klein4b_C1_corpo_q4_bfl.safetensors ] || $PY -s tools/mistura_klein.py --base $OR --doador $OR --regex "$CORPO" --rtn4 32 --saida $MD/klein4b_C1_corpo_q4_bfl.safetensors --esperadas 80 || exit 1
[ -f $MD/klein4b_C2_ptq_H14q4_bfl.safetensors ] || $PY -s tools/mistura_klein.py --base $P0 --doador $OR --regex "$H14" --rtn4 32 --saida $MD/klein4b_C2_ptq_H14q4_bfl.safetensors --esperadas 40 || exit 1
MS="klein4b_C1_corpo_q4_bfl.safetensors klein4b_C2_ptq_H14q4_bfl.safetensors"
echo "=== prontos $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_c1c2 > $D/ladder_fixa_c1c2.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_c1c2/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_c1c2 > $D/decode_fixa_c1c2.log 2>&1; echo "decode fixa rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_c1c2 > $D/ladder_c1c2.log 2>&1; echo "ladder grade rc=$?"
$PY -s tools/decode_latents.py $D/render_c1c2/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_c1c2 > $D/decode_c1c2.log 2>&1; echo "decode grade rc=$?"
echo "=== FIM c1c2 $(date +%T) ==="
