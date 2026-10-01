#!/bin/bash
# Segunda passada na 3090 (fechamento 2026-09-14). Motivo: os encoders QUANTIZADOS (Gemma de fabrica
# W4A8, heretic W4A8/W4A4c/W4A4s) morrem de CUDA OOM na 3080 Ti (12 GB, ~8 livres com o cortex): o
# BF16 passa porque o ComfyUI carrega parcial; o quantizado no caminho dequantizado nao. Aqui eles
# rodam na 3090 com o servidor derrubado, e depois vem os renders (A) e (D) que dependem deles.
# Espera as duas filas anteriores acabarem. Nao reencoda o que ja existe.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1
until grep -q 'FILA_GPU0_FECHA_DONE' .scratch/fila_gpu0_fecha.log 2>/dev/null; do sleep 20; done
until grep -q 'FILA_ENC_FECHA_DONE' .scratch/fila_enc_fecha.log 2>/dev/null; do sleep 20; done
echo "=== passada 2: comeca $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 5
PROMPT="a lone lighthouse on a rocky cliff at dusk, waves breaking against the rocks, the beam sweeping across low clouds, seabirds circling"
run_enc() {
  if [ -f "ComfyUI/models/embeddings/$1_pos.safetensors" ]; then echo "=== $1 ja existe, pulo"; return; fi
  echo "=== $1 <- $2 na 3090 ${4:-destravado} $(date)"
  $PY -s tools/ltx_encode_lowcommit.py --encoder "$2" --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors --prompt "$PROMPT" --saida "$1" --gpu 0 --compare "$3" --json ".scratch/enc_$1.json" $4 2>&1 | grep -E 'leitor RO|construido|codificado|gravado|compare|Error|Traceback|RECUS|out of memory'
}
run_enc ltx23cond_gw4a8  gemma_3_12B_it_w4a8.safetensors ltx23condf
run_enc ltx23cond_hw4a8  gemma_3_12B_it_heretic_w4a8.safetensors ltx23cond_hbf16
run_enc ltx23cond_hw4a4c gemma_3_12B_it_heretic_w4a4_convrot.safetensors ltx23cond_hbf16
run_enc ltx23cond_hw4a4s gemma_3_12B_it_heretic_w4a4_smooth.safetensors ltx23cond_hbf16
# os mesmos quatro com as travas do ComfyUI de fabrica (matematica dequantizada): so condicionamento, sem render
run_enc ltx23cond_gw4a8L  gemma_3_12B_it_w4a8.safetensors ltx23condf --stock-locks
run_enc ltx23cond_hw4a8L  gemma_3_12B_it_heretic_w4a8.safetensors ltx23cond_hbf16 --stock-locks
run_enc ltx23cond_hw4a4cL gemma_3_12B_it_heretic_w4a4_convrot.safetensors ltx23cond_hbf16 --stock-locks
run_enc ltx23cond_hw4a4sL gemma_3_12B_it_heretic_w4a4_smooth.safetensors ltx23cond_hbf16 --stock-locks
echo "ENC_GPU0_DONE $(date)"
$PY -s tools/compara_cond.py --ref ltx23condf --arm ltx23cond_gw4a8 --arm ltx23cond_gw4a8L --arm ltx23cond_hbf16 --json bench/ltx23/encoder_cond_factory.json
$PY -s tools/compara_cond.py --ref ltx23cond_hbf16 --arm ltx23cond_hw4a8 --arm ltx23cond_hw4a8L --arm ltx23cond_hw4a4c --arm ltx23cond_hw4a4cL --arm ltx23cond_hw4a4s --arm ltx23cond_hw4a4sL --json bench/ltx23/encoder_cond_heretic.json

export CUDA_VISIBLE_DEVICES=0,1
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_l.log 2> .scratch/comfy_8190_l.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor l up, RAM free %.1f GiB' % (d['ram_free']/2**30))"
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors"
run() { if [ -f ".scratch/$1.json" ] && grep -q '"status": "success"' ".scratch/$1.json"; then echo "--- $1 ja renderizado, pulo"; return; fi; echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3; }
CMP() { $PY -s tools/compara_av.py "$@" 2>&1 | grep -v -E '^\s*$' | tail -12; }

echo "=== (A) render com o Gemma W4A8 $(date)"
run ltx23av_w4a8_gw4a8 --cond-from ltx23cond_gw4a8 --frames 249 --seed 1234 --limite 3600
CMP --ref .scratch/ltx23av_w4a8_condf.json --arm "W4A8 transformer, W4A8 Gemma encoder=.scratch/ltx23av_w4a8_gw4a8.json" --out bench/ltx23/encoder_w4a8 --titulo "LTX 2.3 W4A8, 249 frames: factory Gemma 3 12B BF16 encoder vs the same encoder in W4A8 (all options saved; encoder locked = dequantized math)"
echo "=== (D) renders dos builds do heretic $(date)"
for c in hbf16 hw4a8 hw4a4c hw4a4s; do
  run ltx23av_w4a8_$c --cond-from ltx23cond_$c --frames 249 --seed 1234 --limite 3600
done
CMP --ref .scratch/ltx23av_w4a8_hbf16.json --arm "heretic W4A8=.scratch/ltx23av_w4a8_hw4a8.json" --arm "heretic W4A4 convrot=.scratch/ltx23av_w4a8_hw4a4c.json" --arm "heretic W4A4 smooth=.scratch/ltx23av_w4a8_hw4a4s.json" --out bench/ltx23/encoder_heretic --titulo "LTX 2.3 W4A8, 249 frames: Gemma 3 12B heretic BF16 encoder vs its W4A8, W4A4-convrot and W4A4-smooth builds (same saved-conditioning path)"
CMP --ref .scratch/ltx23av_w4a8_condf.json --arm "heretic BF16 encoder=.scratch/ltx23av_w4a8_hbf16.json" --out bench/ltx23/encoder_heretic_vs_factory --titulo "LTX 2.3 W4A8, 249 frames: factory Gemma 3 12B BF16 vs heretic (abliterated) BF16 as encoder"
echo "=== servidor l: matando $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
echo "FILA_GPU0_FECHA2_DONE $(date)"
