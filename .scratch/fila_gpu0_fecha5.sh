#!/bin/bash
# Passada 5 (fechamento 2026-09-14): a ladder do capybara W4A8 de novo, porque a passada 1 escolheu
# o VAE com `ls | grep | head -1` e o primeiro nome que bateu foi `hunyuanvideo15_vae_fp16.info`
# (o sidecar do node "Anomalous Local Engine"), nao o .safetensors. Nome fixo aqui. Espera a
# passada 4 matar o servidor m; pula se a ladder ja existir.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
until grep -q 'FILA_GPU0_FECHA4_DONE' .scratch/fila_gpu0_fecha4.log 2>/dev/null; do sleep 20; done
echo "=== passada 5: capybara $(date)"
if [ -f bench/capybara_w4a8/ladder.json ]; then echo "ladder.json ja existe; pulo"; echo "FILA_GPU0_FECHA5_DONE $(date)"; exit 0; fi
[ -f P:/ComfyBench/diffusion_models/capybara_v0.1_w4a8r.safetensors ] || { echo "SEM PESO capybara_v0.1_w4a8r; abortando"; echo "FILA_GPU0_FECHA5_DONE $(date)"; exit 1; }
[ -f ComfyUI/models/vae/hunyuanvideo15_vae_fp16.safetensors ] || { echo "SEM VAE hunyuanvideo15_vae_fp16.safetensors; abortando"; echo "FILA_GPU0_FECHA5_DONE $(date)"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $PY -s tools/quality_ladder.py --reference capybara_v0.1.safetensors --models capybara_v0.1_w4a8r.safetensors --clip qwen_2.5_vl_7b.safetensors byt5_small_glyphxl_fp16.safetensors --clip-type hunyuan_video_15 --prompt "a red apple on a wooden table, soft daylight" --seeds 12345 --steps 6 --size 480 --frames 1 --vae hunyuanvideo15_vae_fp16.safetensors --out bench/capybara_w4a8 2>&1 | tail -30
ls -la bench/capybara_w4a8/ 2>/dev/null | awk '{print $5, $9}'
echo "FILA_GPU0_FECHA5_DONE $(date)"
