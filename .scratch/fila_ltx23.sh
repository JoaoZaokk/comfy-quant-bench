#!/bin/bash
# LTX 2.3 distilled 1.1: os bracos de 10 s com audio, depois LoRA. Espera (1) a fila da 3090
# terminar, (2) as duas conversoes, (3) o gemma 3 de fabrica. Reinicia o servidor 8190 antes,
# porque `bench_p.checkpoints` e `bench_p.loras` entraram no yaml DEPOIS do servidor subir e
# `extra_model_paths.yaml` so e lido no arranque -- sem isto o CheckpointLoaderSimple recusa o
# arquivo como "value not in list".
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until grep -q 'FILA_GPU0_DONE' .scratch/fila_gpu0.log 2>/dev/null; do sleep 30; done
until grep -q "LTX23_W4A4_DONE" .scratch/retoma_bf16.log 2>/dev/null; do sleep 30; done
until grep -q 'GEMMA3_TE_DOWNLOAD_OK' .scratch/baixa_gemma3_te.log 2>/dev/null; do sleep 30; done
# a copia do BF16 para W: (disco local) tem de estar inteira: 46149345334 B
until [ "$(stat -c %s W:/ltx-2.3/ltx-2.3-22b-distilled-1.1_W.safetensors 2>/dev/null)" = "46149345334" ]; do sleep 30; done
echo "=== reinicio do servidor 8190 $(date)"
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 8
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_ltx23.log 2> .scratch/comfy_8190_ltx23.err &
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/object_info/CheckpointLoaderSimple | grep -o -E 'ltx-2.3[A-Za-z0-9_.-]*|LTX23_audio[A-Za-z0-9_.-]*' | sort -u
CK=ltx-2.3-22b-distilled-1.1_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16.safetensors"
ENC=gemma_3_12B_it.safetensors
run() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py $AUX --encoder $ENC --frames 249 --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|distorch|modelo' | head -6; }
echo "=== (e) LTX 2.3 bracos de 10 s $(date)"
run ltx23av_w4a8 --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors
run ltx23av_w4a4 --checkpoint ltx-2.3-22b-distilled-1.1_w4a4.safetensors
run ltx23av_q6k  --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors --gguf ltx-2.3-22b-distilled-1.1-Q6_K.gguf
run ltx23av_bf16 --checkpoint $CK --distorch --alocacao "cpu,40gb" --limite 10800
$PY -s tools/compara_av.py --ref .scratch/ltx23av_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23av_q6k.json" --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 512px, 8 steps, seed 1234 -- reference BF16" 2>&1 | grep -v -E '^\s*$' | tail -12
echo "=== (f) LoRA LTX 2.3 $(date)"
$PY -s tools/probe_lora_requant.py --ckpt ltx-2.3-22b-distilled-1.1_w4a8.safetensors --checkpoint-mode --lora LTX23_Product_Commercial_LoRA.safetensors --source P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors --n 24 --device 0 --json .scratch/lora_requant_ltx23_product.json 2>&1 | grep -v -E 'lora key not loaded|^\s*$' | tail -16
runl() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors $AUX --encoder $ENC --frames 49 --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback' | head -3; }
runl ltx23lora_base   --seed 1234
runl ltx23lora_merge  --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors
runl ltx23lora_bypass --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors --lora-bypass
runl ltx23lora_seed2  --seed 4321
$PY -s tools/compara_av.py --ref .scratch/ltx23lora_base.json --arm "LoRA merged=.scratch/ltx23lora_merge.json" --arm "LoRA bypass=.scratch/ltx23lora_bypass.json" --arm "no LoRA, other seed=.scratch/ltx23lora_seed2.json" --out bench/ltx23/lora --titulo "LTX 2.3 W4A8 + Product Commercial LoRA, 49 frames -- reference: same seed, no LoRA" 2>&1 | grep -v -E '^\s*$' | tail -10
echo "FILA_LTX23_DONE $(date)"
