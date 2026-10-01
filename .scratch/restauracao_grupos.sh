#!/bin/bash
# Criterio: bench/criterio_restauracao_grupos_2026-09-25.md. Monta H1-H4 (CPU) e renderiza na 3090.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe; MD=ComfyUI/models/diffusion_models; D=bench/qat_klein
B6=$MD/klein4b_qat_b6_final_bfl.safetensors; OR=$MD/klein4b_braco3_bf16_original_bfl.safetensors
declare -A RX=(
  [H1]='double_blocks\.\d+\.txt_(attn\.(qkv|proj)|mlp\.\d)\.weight'
  [H2]='double_blocks\.\d+\.img_(attn\.(qkv|proj)|mlp\.\d)\.weight'
  [H3]='single_blocks\.[0-9]\.linear[12]\.weight'
  [H4]='single_blocks\.1[0-9]\.linear[12]\.weight'
)
MS=""
for H in H1 H2 H3 H4; do
  S=$MD/klein4b_b6_rest_${H}_bfl.safetensors
  [ -f $S ] || $PY -s tools/mistura_klein.py --base $B6 --doador $OR --regex "${RX[$H]}" --saida $S --esperadas 20 || exit 1
  MS="$MS klein4b_b6_rest_${H}_bfl.safetensors"
done
echo "=== hibridos prontos $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models klein4b_qat_b6_final_bfl.safetensors $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/avaliacao_fixa/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/avaliacao_fixa/render_rest > $D/ladder_fixa_rest.log 2>&1; echo "ladder fixa rc=$?"
$PY -s tools/decode_latents.py $D/avaliacao_fixa/render_rest/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/avaliacao_fixa/render_rest > $D/decode_fixa_rest.log 2>&1; echo "decode fixa rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_rest > $D/ladder_rest.log 2>&1; echo "ladder grade rc=$?"
$PY -s tools/decode_latents.py $D/render_rest/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_rest > $D/decode_rest.log 2>&1; echo "decode grade rc=$?"
echo "=== FIM restauracao $(date +%T) ==="
