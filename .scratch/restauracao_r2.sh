#!/bin/bash
# Rodada 2 do bench/criterio_restauracao_grupos_2026-09-25.md: H14 (BF16) e H14q4 (RTN 4 bits g32).
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe; MD=ComfyUI/models/diffusion_models; D=bench/qat_klein
B6=$MD/klein4b_qat_b6_final_bfl.safetensors; OR=$MD/klein4b_braco3_bf16_original_bfl.safetensors
RX='double_blocks\.\d+\.txt_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.1[0-9]\.linear[12]\.weight'
[ -f $MD/klein4b_b6_rest_H14_bfl.safetensors ] || $PY -s tools/mistura_klein.py --base $B6 --doador $OR --regex "$RX" --saida $MD/klein4b_b6_rest_H14_bfl.safetensors --esperadas 40 || exit 1
[ -f $MD/klein4b_b6_rest_H14q4_bfl.safetensors ] || $PY -s tools/mistura_klein.py --base $B6 --doador $OR --regex "$RX" --rtn4 32 --saida $MD/klein4b_b6_rest_H14q4_bfl.safetensors --esperadas 40 || exit 1
MS="klein4b_b6_rest_H14_bfl.safetensors klein4b_b6_rest_H14q4_bfl.safetensors"
echo "=== hibridos prontos $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_rest2 > $D/ladder_fixa_rest2.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_rest2/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_rest2 > $D/decode_fixa_rest2.log 2>&1; echo "decode fixa rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_rest2 > $D/ladder_rest2.log 2>&1; echo "ladder grade rc=$?"
$PY -s tools/decode_latents.py $D/render_rest2/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_rest2 > $D/decode_rest2.log 2>&1; echo "decode grade rc=$?"
echo "=== FIM r2 $(date +%T) ==="
