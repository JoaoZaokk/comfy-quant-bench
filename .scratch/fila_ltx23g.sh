#!/bin/bash
# Setima e ULTIMA tentativa do braco BF16 do 2.3: loader simples do ComfyUI (UNETLoader), sem
# DisTorch, sem dynamic VRAM, sem encoder residente (condicionamento salvo). Commit esperado ~39 GiB
# (torch.empty do modelo) contra ~52 GiB livres. Espera a rodada de LoRA terminar (servidor normal
# ja de pe). Se morrer, o BF16 do 2.3 fica registrado como inalcancavel nesta maquina, com as sete
# tentativas e o mecanismo, e a comparacao usa o Q6_K como referencia, rotulado.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until grep -q 'FILA_LTX23F_DONE' .scratch/fila_ltx23f.log 2>/dev/null; do sleep 20; done
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 30 -X POST http://127.0.0.1:8190/free -H 'Content-Type: application/json' -d '{"unload_models": true, "free_memory": true}' -o /dev/null
sleep 5
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
echo "=== (e7) BF16 do 2.3: UNETLoader simples, sem DisTorch, sem dynamic VRAM, condicionamento salvo $(date)"
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('argv', d.get('argv')[-9:], 'RAM free %.1f GiB' % (d['ram_free']/2**30))"
$PY -s tools/ltx_video.py --checkpoint $W4A8 --transformer ltx-2.3-22b-distilled-1.1_transformer_bf16_C.safetensors --video-vae LTX23_video_vae_bf16.safetensors --audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --cond-from ltx23cond --frames 249 --saida ltx23av_bf16 --json .scratch/ltx23av_bf16.json --limite 7200 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|condicion' | head -6
if [ -f .scratch/ltx23av_bf16.json ] && grep -q '"status": "success"' .scratch/ltx23av_bf16.json; then
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23av_q6k.json" --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 512px, 8 steps, seed 1234 -- reference BF16" 2>&1 | grep -v -E '^\s*$' | tail -12
else
  echo "BF16_23_FALHOU_7 $(date)"
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_q6k.json --arm "W4A8 ours=.scratch/ltx23av_w4a8.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4.json" --out bench/ltx23/av_vs_q6k --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 8 steps, seed 1234 -- reference: third-party GGUF Q6_K (BF16 unreachable)" 2>&1 | grep -v -E '^\s*$' | tail -10
fi
echo "FILA_LTX23G_DONE $(date)"
