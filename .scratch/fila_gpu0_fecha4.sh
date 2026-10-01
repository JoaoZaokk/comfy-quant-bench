#!/bin/bash
# Passada 4 (fechamento 2026-09-14): por-passo do int8 da Lightricks e do nosso W4A8 no 2.5. Os
# logs dos renders de 13/09 nao sobreviveram com barra completa (o BF16 sobreviveu em
# comfy_8190_b.err: 3 @ 8,48 s/it, parede 761 s -- lido so depois de a ferramenta aprender o formato
# HH:MM:SS). Mesmo protocolo do card: encoder int8 vivo, 3 passos, 249 quadros, semente 1234.
# Espera a passada 2 matar o servidor l. Re-render deve dar os mesmos pixels (ja medido 2x).
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
until grep -q 'FILA_GPU0_FECHA2_DONE' .scratch/fila_gpu0_fecha2.log 2>/dev/null; do sleep 20; done
echo "=== passada 4: por-passo do 2.5 $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 5
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_m.log 2> .scratch/comfy_8190_m.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor m up, RAM free %.1f GiB' % (d['ram_free']/2**30))"
S3="0.909375, 0.725, 0.421875, 0.0"
E25="--encoder gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors --video-vae ltx-2.5-video-vae-bf16.safetensors --audio-vae ltx-2.5-audio-vae-bf16.safetensors"
for t in "int8_t ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" "w4a8_t ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors"; do
  set -- $t
  echo "--- ltx25av_$1 $(date)"
  $PY -s tools/ltx_video.py --transformer $2 $E25 --sigmas "$S3" --frames 249 --seed 1234 --saida ltx25av_$1 --json .scratch/ltx25av_$1.json --limite 3600 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3
done
$PY -s tools/sampler_tempo_do_log.py .scratch/comfy_8190_m.err
$PY -s tools/compara_av.py --ref .scratch/ltx25av_bf16.json --arm "int8 Lightricks, re-render=.scratch/ltx25av_int8_t.json" --arm "W4A8 ours, re-render=.scratch/ltx25av_w4a8_t.json" --out .scratch/ltx25_rerender_t --titulo "LTX 2.5 re-render of two arms for per-step timing; pixels expected identical to the card's" 2>&1 | grep -E 'MAE'
echo "=== servidor m: matando $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
echo "FILA_GPU0_FECHA4_DONE $(date)"
