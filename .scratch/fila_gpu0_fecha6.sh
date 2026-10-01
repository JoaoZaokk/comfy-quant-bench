#!/bin/bash
# Passada 6 (fechamento 2026-09-14): os dois W4A4 do heretic no caminho TRAVADO (peso de 4 bits,
# matematica dequantizada -- o que o ComfyUI de fabrica faz com qualquer text encoder), sobre o
# mesmo transformer W4A8 e a mesma semente. No caminho destravado os dois mudam a cena (crepusculo
# -> dia; bench/ltx23/encoder_heretic). Isto decide se "so com as travas" e um jeito seguro de usar
# esses arquivos. Os condicionamentos travados ja existem (ltx23cond_hw4a4cL, hw4a4sL). Espera a
# passada 5 (ladder do capybara) acabar.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
until grep -q 'FILA_GPU0_FECHA5_DONE' .scratch/fila_gpu0_fecha5.log 2>/dev/null; do sleep 20; done
echo "=== passada 6: W4A4 do heretic travados $(date)"
for c in hw4a4cL hw4a4sL; do [ -f ComfyUI/models/embeddings/ltx23cond_${c}_pos.safetensors ] || echo "FALTA ltx23cond_$c"; done
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 5
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_n.log 2> .scratch/comfy_8190_n.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor n up, RAM free %.1f GiB' % (d['ram_free']/2**30))"
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors"
run() { if [ -f ".scratch/$1.json" ] && grep -q '"status": "success"' ".scratch/$1.json"; then echo "--- $1 ja renderizado, pulo"; return; fi; echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3; }
run ltx23av_w4a8_hw4a4cL --cond-from ltx23cond_hw4a4cL --frames 249 --seed 1234 --limite 3600
run ltx23av_w4a8_hw4a4sL --cond-from ltx23cond_hw4a4sL --frames 249 --seed 1234 --limite 3600
$PY -s tools/compara_av.py --ref .scratch/ltx23av_w4a8_hbf16.json --arm "W4A4 convrot, locked (stock path)=.scratch/ltx23av_w4a8_hw4a4cL.json" --arm "W4A4 smooth, locked (stock path)=.scratch/ltx23av_w4a8_hw4a4sL.json" --arm "W4A4 convrot, released=.scratch/ltx23av_w4a8_hw4a4c.json" --arm "W4A4 smooth, released=.scratch/ltx23av_w4a8_hw4a4s.json" --out bench/ltx23/encoder_heretic_locked --titulo "LTX 2.3 W4A8, 249 frames: heretic Gemma W4A4 encoders on stock ComfyUI's locked path (4-bit weight, dequantized math) against the same builds released; reference = heretic BF16 encoder" 2>&1 | grep -v -E '^\s*$' | tail -12
echo "=== servidor n: matando $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
echo "FILA_GPU0_FECHA6_DONE $(date)"
