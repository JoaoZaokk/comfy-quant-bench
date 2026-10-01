#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe; MD=ComfyUI/models/diffusion_models; D=bench/qat_klein
OR=$MD/klein4b_braco3_bf16_original_bfl.safetensors
CORPO='double_blocks\.\d+\.(img|txt)_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.\d+\.linear[12]\.weight'
H14='double_blocks\.\d+\.txt_(attn\.(qkv|proj)|mlp\.\d)\.weight|single_blocks\.1[0-9]\.linear[12]\.weight'
mk() { [ -f $MD/$1 ] || $PY -s tools/mistura_klein.py "${@:2}" --saida $MD/$1 || exit 1; }
mk klein4b_C2r_corpo_q2g16_bfl.safetensors --base $OR --doador $OR --regex "$CORPO" --rtn4 16 --bits 2 --esperadas 80
mk klein4b_M32_bfl.safetensors --base $MD/klein4b_C2r_corpo_q2g16_bfl.safetensors --doador $OR --regex "$H14" --rtn4 32 --bits 3 --esperadas 40
MS="klein4b_C2r_corpo_q2g16_bfl.safetensors klein4b_M32_bfl.safetensors"
echo "=== prontos $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_r4 > $D/ladder_fixa_r4.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_r4/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_r4 > $D/decode_fixa_r4.log 2>&1; echo "decode fixa rc=$?"
echo "=== FIM r4 $(date +%T) ==="
