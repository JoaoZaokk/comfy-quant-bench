#!/bin/bash
# Substitui fila_ltx23.sh a partir do segundo braco, lendo tudo do disco local W: em vez do SMB:
# cada braco pagava ~42 GB de leitura de P: so para carregar, e sao seis bracos. Espera o braco
# W4A8 (que ja esta rodando de P:) terminar e as copias chegarem; entao mata a fila antiga,
# reinicia o servidor (o yaml ganhou `text_encoders: ltx-2.3`, lido so no arranque) e segue.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until [ -f .scratch/ltx23av_w4a8.json ] && grep -q '"status": "success"' .scratch/ltx23av_w4a8.json; do sleep 20; done
echo "=== w4a8 terminou; esperando copias $(date)"
until grep -q 'COPIAS_W_OK' .scratch/copia_ltx23_aux_W.log 2>/dev/null; do sleep 20; done
echo "=== matando a fila antiga e reiniciando o servidor $(date)"
kill 5180 2>/dev/null; sleep 2
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 8
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_ltx23b.log 2> .scratch/comfy_8190_ltx23b.err &
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/object_info/LTXAVTextEncoderLoader | grep -o -E '"gemma_3_12B_it_W[A-Za-z0-9_.-]*' | sort -u
CK=ltx-2.3-22b-distilled-1.1_W.safetensors
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
ENC=gemma_3_12B_it_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors"
run() { if [ -f ".scratch/$1.json" ] && grep -q '"status": "success"' ".scratch/$1.json"; then echo "--- $1 ja feito pela fila antiga, pulando"; return; fi; echo "--- $1 $(date)"; $PY -s tools/ltx_video.py $AUX --encoder $ENC --frames 249 --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|distorch|modelo' | head -6; }
echo "=== (e) LTX 2.3 bracos restantes, de W: $(date)"
run ltx23av_w4a4 --checkpoint ltx-2.3-22b-distilled-1.1_w4a4.safetensors
run ltx23av_q6k  --checkpoint $W4A8 --gguf ltx-2.3-22b-distilled-1.1-Q6_K.gguf
run ltx23av_bf16 --checkpoint $CK --distorch --alocacao "cpu,40gb" --limite 10800
$PY -s tools/compara_av.py --ref .scratch/ltx23av_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23av_q6k.json" --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 512px, 8 steps, seed 1234 -- reference BF16" 2>&1 | grep -v -E '^\s*$' | tail -12
echo "=== (f) LoRA LTX 2.3 $(date)"
$PY -s tools/probe_lora_requant.py --ckpt $W4A8 --checkpoint-mode --lora LTX23_Product_Commercial_LoRA.safetensors --source W:/ltx-2.3/ltx-2.3-22b-distilled-1.1_W.safetensors --n 24 --device 0 --json .scratch/lora_requant_ltx23_product.json 2>&1 | grep -v -E 'lora key not loaded|^\s*$' | tail -16
runl() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --encoder $ENC --frames 49 --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback' | head -3; }
runl ltx23lora_base   --seed 1234
runl ltx23lora_merge  --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors
runl ltx23lora_bypass --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors --lora-bypass
runl ltx23lora_seed2  --seed 4321
$PY -s tools/compara_av.py --ref .scratch/ltx23lora_base.json --arm "LoRA merged=.scratch/ltx23lora_merge.json" --arm "LoRA bypass=.scratch/ltx23lora_bypass.json" --arm "no LoRA, other seed=.scratch/ltx23lora_seed2.json" --out bench/ltx23/lora --titulo "LTX 2.3 W4A8 + Product Commercial LoRA, 49 frames -- reference: same seed, no LoRA" 2>&1 | grep -v -E '^\s*$' | tail -10
echo "FILA_LTX23_DONE $(date)"
