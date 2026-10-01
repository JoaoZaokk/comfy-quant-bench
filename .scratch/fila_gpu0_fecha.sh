#!/bin/bash
# Fila da 3090 (fechamento 2026-09-14). Ordem: servidor novo -> (G) empate do audio a 3 passos
# (BF16 GGUF, Q6_K, W4A8, condicionamento ltx23condf) -> espera o encode do Gemma W4A8 -> (A) render
# W4A8 sobre ele + compara com o render BF16-encoder -> espera os encodes do heretic -> (D) quatro
# renders W4A8 (hbf16, hw4a8, hw4a4c, hw4a4s) + comparas -> (C) os dois int8 nossos do 2.5 com o
# encoder int8 da Lightricks VIVO (como os bracos do card) + compara com o BF16 do 2.5 -> mata o
# servidor -> espera a reconversao do capybara -> (E) ladder do capybara (in-process, GPU 0).
# Criterio antes: bench/criterio_fechamento_2026-09-14.md.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
ENC=.scratch/fila_enc_fecha.log
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 5
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_k.log 2> .scratch/comfy_8190_k.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor up, RAM free %.1f GiB' % (d['ram_free']/2**30))"
curl -s -m 10 http://127.0.0.1:8190/object_info/VoidLoadConditioningFull | $PY -s -c "import json,sys; d=json.load(sys.stdin); print('no VoidLoadConditioningFull:', 'VoidLoadConditioningFull' in d)"
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors"
S3="0.909375, 0.725, 0.421875, 0.0"
run() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3; }
CMP() { $PY -s tools/compara_av.py "$@" 2>&1 | grep -v -E '^\s*$' | tail -12; }

echo "=== (G) empate do audio a 3 passos $(date)"
run ltx23s3_bf16 --cond-from ltx23condf --frames 249 --seed 1234 --limite 3600 --sigmas "$S3" --gguf ltx-2.3-22b-distilled-1.1-BF16.gguf --video-vae LTX23_video_vae_bf16.safetensors
run ltx23s3_q6k  --cond-from ltx23condf --frames 249 --seed 1234 --limite 3600 --sigmas "$S3" --gguf ltx-2.3-22b-distilled-1.1-Q6_K.gguf --video-vae LTX23_video_vae_bf16.safetensors
run ltx23s3_w4a8 --cond-from ltx23condf --frames 249 --seed 1234 --limite 3600 --sigmas "$S3"
CMP --ref .scratch/ltx23s3_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23s3_q6k.json" --arm "W4A8 ours=.scratch/ltx23s3_w4a8.json" --out bench/ltx23/av_3steps --titulo "LTX 2.3, 249 frames + audio at 3 steps (sigmas 0.909/0.725/0.422/0), same saved conditioning -- does the sound still tie W4A8 with Q6_K?"

echo "=== (A) encoder Gemma de fabrica W4A8 $(date)"
until grep -q 'ENC_GW4A8_DONE' $ENC 2>/dev/null; do sleep 15; done
[ -f ComfyUI/models/embeddings/ltx23cond_gw4a8_pos.safetensors ] || echo "FALTA ltx23cond_gw4a8"
run ltx23av_w4a8_gw4a8 --cond-from ltx23cond_gw4a8 --frames 249 --seed 1234 --limite 3600
CMP --ref .scratch/ltx23av_w4a8_condf.json --arm "W4A8 transformer, W4A8 Gemma encoder=.scratch/ltx23av_w4a8_gw4a8.json" --out bench/ltx23/encoder_w4a8 --titulo "LTX 2.3 W4A8, 249 frames: factory Gemma 3 12B BF16 encoder vs the same encoder in W4A8 (both encoded on the 3080 Ti, all options saved)"

echo "=== (D) heretic: BF16 e tres builds $(date)"
until grep -q 'ENC_HERETIC_DONE' $ENC 2>/dev/null; do sleep 15; done
for c in hbf16 hw4a8 hw4a4c hw4a4s; do
  [ -f ComfyUI/models/embeddings/ltx23cond_${c}_pos.safetensors ] || echo "FALTA ltx23cond_$c"
  run ltx23av_w4a8_$c --cond-from ltx23cond_$c --frames 249 --seed 1234 --limite 3600
done
CMP --ref .scratch/ltx23av_w4a8_hbf16.json --arm "heretic W4A8=.scratch/ltx23av_w4a8_hw4a8.json" --arm "heretic W4A4 convrot=.scratch/ltx23av_w4a8_hw4a4c.json" --arm "heretic W4A4 smooth=.scratch/ltx23av_w4a8_hw4a4s.json" --out bench/ltx23/encoder_heretic --titulo "LTX 2.3 W4A8, 249 frames: Gemma 3 12B heretic BF16 encoder vs its W4A8, W4A4-convrot and W4A4-smooth builds (same saved-conditioning path)"
CMP --ref .scratch/ltx23av_w4a8_condf.json --arm "heretic BF16 encoder=.scratch/ltx23av_w4a8_hbf16.json" --out bench/ltx23/encoder_heretic_vs_factory --titulo "LTX 2.3 W4A8, 249 frames: factory Gemma 3 12B BF16 vs heretic (abliterated) BF16 as encoder"

echo "=== (C) int8 nossos do 2.5, encoder int8 da Lightricks vivo $(date)"
curl -s -m 30 -X POST http://127.0.0.1:8190/free -H 'Content-Type: application/json' -d '{"unload_models": true, "free_memory": true}' -o /dev/null; sleep 5
E25="--encoder gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors --video-vae ltx-2.5-video-vae-bf16.safetensors --audio-vae ltx-2.5-audio-vae-bf16.safetensors"
for a in int8 int8_convrot; do
  echo "--- ltx25av_ours_$a $(date)"
  $PY -s tools/ltx_video.py --transformer ltx-2.5-22b-distilled-transformer-bf16_$a.safetensors $E25 --sigmas "$S3" --frames 249 --seed 1234 --saida ltx25av_ours_$a --json .scratch/ltx25av_ours_$a.json --limite 3600 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3
done
CMP --ref .scratch/ltx25av_bf16.json --arm "comfy-int8-convrot (Lightricks)=.scratch/ltx25av_int8.json" --arm "int8_tensorwise ours=.scratch/ltx25av_ours_int8.json" --arm "int8_tensorwise+convrot ours=.scratch/ltx25av_ours_int8_convrot.json" --arm "W4A8 ours=.scratch/ltx25av_w4a8.json" --out bench/ltx25/int8_ours --titulo "LTX 2.5 22B distilled, 249 frames + audio, 3 steps -- reference BF16; Lightricks' int8 against our two int8 builds and our W4A8 (live int8 encoder in every arm)"

echo "=== servidor k: matando $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 8
echo "=== (E) capybara W4A8 reconvertido, ladder do Hunyuan $(date)"
until grep -q 'CAPY_CONV_DONE' $ENC 2>/dev/null; do sleep 15; done
VAE=$(ls ComfyUI/models/vae | grep -i -E 'hunyuan.*(1\.5|15)|hunyuanvideo15|hunyuanvideo1\.5' | head -1); echo "vae: $VAE"
CUDA_VISIBLE_DEVICES=0 $PY -s tools/quality_ladder.py --reference capybara_v0.1.safetensors --models capybara_v0.1_w4a8r.safetensors --clip qwen_2.5_vl_7b.safetensors byt5_small_glyphxl_fp16.safetensors --clip-type hunyuan_video_15 --prompt "a red apple on a wooden table, soft daylight" --seeds 12345 --steps 6 --size 480 --frames 1 ${VAE:+--vae $VAE} --out bench/capybara_w4a8 2>&1 | tail -25
echo "FILA_GPU0_FECHA_DONE $(date)"
