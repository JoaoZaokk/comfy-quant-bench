#!/bin/bash
# Passada 3 (fechamento 2026-09-14): ladder do capybara W4A8 reconvertido em P: (a primeira
# reconversao recusou por falta de 6 GiB em F:). Espera a passada 2 matar o servidor e a
# reconversao terminar; roda a ladder do Hunyuan in-process na 3090.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
until grep -q 'FILA_GPU0_FECHA2_DONE' .scratch/fila_gpu0_fecha2.log 2>/dev/null; do sleep 20; done
until grep -q 'CAPY_CONV_P_DONE' .scratch/capy_conv.log 2>/dev/null; do sleep 20; done
echo "=== passada 3: capybara $(date)"
tail -6 .scratch/capy_conv.log
[ -f P:/ComfyBench/diffusion_models/capybara_v0.1_w4a8r.safetensors ] || { echo "SEM PESO capybara_v0.1_w4a8r; abortando"; echo "FILA_GPU0_FECHA3_DONE $(date)"; exit 1; }
sha256sum P:/ComfyBench/diffusion_models/capybara_v0.1_w4a8r.safetensors | cut -c1-64
VAE=$(ls ComfyUI/models/vae | grep -i -E 'hunyuanvideo15|hunyuan.*1\.5' | head -1); echo "vae: $VAE"
CUDA_VISIBLE_DEVICES=0 $PY -s tools/quality_ladder.py --reference capybara_v0.1.safetensors --models capybara_v0.1_w4a8r.safetensors --clip qwen_2.5_vl_7b.safetensors byt5_small_glyphxl_fp16.safetensors --clip-type hunyuan_video_15 --prompt "a red apple on a wooden table, soft daylight" --seeds 12345 --steps 6 --size 480 --frames 1 ${VAE:+--vae $VAE} --out bench/capybara_w4a8 2>&1 | tail -30
ls -la bench/capybara_w4a8/ 2>/dev/null | awk '{print $5, $9}'
echo "FILA_GPU0_FECHA3_DONE $(date)"
