#!/bin/bash
# Nona fila do 2.3: o braco BF16 por GGUF (memmap somente-leitura, 0 commit; GGMLOps sem
# torch.empty), 249 quadros com audio, condicionamento salvo. Espera a rodada de LoRA e o controle
# de identidade (fila h) e a conversao (tools/safetensors_to_gguf_bf16.py). Se morrer, compara os
# nossos contra o Q6_K de terceiro como referencia, rotulado.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until grep -q 'FILA_LTX23H_DONE' .scratch/fila_ltx23h.log 2>/dev/null; do sleep 20; done
until grep -q '^escrito' .scratch/gguf_bf16_ltx23.log 2>/dev/null; do sleep 20; done
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 30 -X POST http://127.0.0.1:8190/free -H 'Content-Type: application/json' -d '{"unload_models": true, "free_memory": true}' -o /dev/null
sleep 5
echo "=== gguf visivel ao servidor? ==="
curl -s -m 10 http://127.0.0.1:8190/object_info/UnetLoaderGGUF | grep -o -E 'ltx-2.3-22b-distilled-1.1-[A-Za-z0-9_.-]*gguf' | sort -u
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
echo "=== (e8) BF16 do 2.3 por GGUF BF16 (UnetLoaderGGUF), sem DisTorch, sem dynamic VRAM, condicionamento salvo $(date)"
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('argv', d.get('argv')[-8:], 'RAM free %.1f GiB' % (d['ram_free']/2**30))"
$PY -s tools/ltx_video.py --checkpoint $W4A8 --gguf ltx-2.3-22b-distilled-1.1-BF16.gguf --video-vae LTX23_video_vae_bf16.safetensors --audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --cond-from ltx23cond --frames 249 --seed 1234 --saida ltx23av_bf16 --json .scratch/ltx23av_bf16.json --limite 7200 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de|condicion' | head -6
if [ -f .scratch/ltx23av_bf16.json ] && grep -q '"status": "success"' .scratch/ltx23av_bf16.json; then
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23av_q6k.json" --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 512px, 8 steps, seed 1234 -- reference BF16 (GGUF container, lossless)" 2>&1 | grep -v -E '^\s*$' | tail -12
else
  echo "BF16_23_FALHOU_8 $(date)"
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_q6k.json --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av_vs_q6k --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 8 steps, seed 1234 -- reference: third-party GGUF Q6_K (BF16 unreachable)" 2>&1 | grep -v -E '^\s*$' | tail -10
fi
echo "FILA_LTX23I_DONE $(date)"
