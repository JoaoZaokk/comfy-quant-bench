#!/bin/bash
# criterio_mestre_dtype_2026-09-22.md. Espera a fila do render acabar (ela segura a GPU).
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
D=bench/render_braco1_2026-09-22
PY=./python_embeded/python.exe
# (fila anterior ja terminou)
echo "=== ajustes $(date +%T) ==="
$PY -s .scratch/roda_ajustes_mestre.py || { echo "ajustes FALHOU"; exit 1; }
echo "=== remap $(date +%T) ==="
for n in klein4b_braco1f_mestre_fp32 klein4b_braco1s_bf16_sr; do
  $PY -s tools/aplica_mapa_diffusers_bfl.py --entrada P:/ComfyBench/originais/$n.safetensors \
    --mapa .scratch/mapa_klein_diffusers_bfl.json \
    --saida ComfyUI/models/diffusion_models/${n}_bfl.safetensors > $D/remap_$n.log 2>&1
  echo "remap $n rc=$?"
done
echo "=== eps mestre $(date +%T) ==="
$PY -s .scratch/roda_eps_mestre.py > $D/eps_mestre_driver.log 2>&1; echo "eps mestre rc=$?"
echo "=== render mestre $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_braco1_compensado_bfl.safetensors klein4b_braco1f_mestre_fp32_bfl.safetensors klein4b_braco1s_bf16_sr_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/comfy_mestre > $D/ladder_mestre.log 2>&1; echo "ladder mestre rc=$?"
$PY -s tools/decode_latents.py $D/comfy_mestre/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/comfy_mestre > $D/decode_mestre.log 2>&1; echo "decode mestre rc=$?"
echo "=== FIM mestre $(date +%T) ==="
