#!/bin/bash
# Sexta fila do 2.3: braco BF16 num servidor COM dynamic VRAM (Linear preguicoso = sem commit de
# 39 GiB no torch.empty; leitor aimdo = zero-copy), transformer de C:, condicionamento salvo, SEM
# DisTorch. Depois volta ao servidor normal (--disable-dynamic-vram) para a rodada de LoRA.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor dyn-vram up, argv', d.get('argv')[-8:], 'RAM free %.1f GiB' % (d['ram_free']/2**30))"
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
ENC=gemma_3_12B_it_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors"
echo "=== (e) BF16 do 2.3: dynamic VRAM, transformer de C:, condicionamento salvo, sem DisTorch $(date)"
$PY -s tools/ltx_video.py --checkpoint $W4A8 --transformer ltx-2.3-22b-distilled-1.1_transformer_bf16_C.safetensors --video-vae LTX23_video_vae_bf16.safetensors --audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --cond-from ltx23cond --frames 249 --saida ltx23av_bf16 --json .scratch/ltx23av_bf16.json --limite 7200 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|condicion' | head -6
if [ -f .scratch/ltx23av_bf16.json ] && grep -q '"status": "success"' .scratch/ltx23av_bf16.json; then
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23av_q6k.json" --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 512px, 8 steps, seed 1234 -- reference BF16" 2>&1 | grep -v -E '^\s*$' | tail -12
else
  echo "BF16_23_FALHOU $(date)"
fi
echo "=== reinicio do servidor com --disable-dynamic-vram para a rodada de LoRA $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 8
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_g.log 2> .scratch/comfy_8190_g.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
echo "=== (f) LoRA LTX 2.3 $(date)"
$PY -s tools/probe_lora_requant.py --ckpt $W4A8 --checkpoint-mode --lora LTX23_Product_Commercial_LoRA.safetensors --source P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors --n 24 --device 0 --json .scratch/lora_requant_ltx23_product.json 2>&1 | grep -v -E 'lora key not loaded|^\s*$' | tail -16
runl() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --encoder $ENC --frames 49 --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback' | head -3; }
runl ltx23lora_base   --seed 1234
runl ltx23lora_merge  --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors
runl ltx23lora_bypass --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors --lora-bypass
runl ltx23lora_seed2  --seed 4321
$PY -s tools/compara_av.py --ref .scratch/ltx23lora_base.json --arm "LoRA merged=.scratch/ltx23lora_merge.json" --arm "LoRA bypass=.scratch/ltx23lora_bypass.json" --arm "no LoRA, other seed=.scratch/ltx23lora_seed2.json" --out bench/ltx23/lora --titulo "LTX 2.3 W4A8 + Product Commercial LoRA, 49 frames -- reference: same seed, no LoRA" 2>&1 | grep -v -E '^\s*$' | tail -10
echo "FILA_LTX23F_DONE $(date)"
