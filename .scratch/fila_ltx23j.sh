#!/bin/bash
# Decima fila do 2.3, com o condicionamento CERTO. O que a fila h/i mediu foi renderizado sobre o
# condicionamento do LTXVSaveConditioning, que perde a chave `unprocessed_ltxav_embeds` e da ruido
# (MAE 75,9 contra o encoder vivo: bench/ltx23/cond_identity_ltxv_saver). Agora o condicionamento
# vem de tools/ltx_encode_lowcommit.py (formato completo, float32) e entra por
# VoidLoadConditioningFull (custom_nodes/comfy-void-stage-tools), novo no -> servidor reiniciado.
# Encoder NUNCA residente no servidor (commit). Ordem por modelo, porque --cache-none recarrega:
#   W4A8: identidade 249 (vs encoder vivo) | LoRA farol x4 (49) | LoRA gatilho x4 (49)
#   Q6_K 249 | W4A4 249 | BF16 por GGUF 249 -> comparacao dos 4 bracos, todos no MESMO condicionamento.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until grep -q 'ENCODE2_ALL_DONE' .scratch/encode_lowcommit2.log 2>/dev/null; do sleep 15; done
for f in ltx23condf_pos ltx23condf_neg ltx23cond_srx_pos ltx23cond_srx_neg; do
  [ -f "ComfyUI/models/embeddings/$f.safetensors" ] || { echo "FALTA ComfyUI/models/embeddings/$f.safetensors -- abortando $(date)"; exit 1; }
done
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 6
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_j.log 2> .scratch/comfy_8190_j.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor up, argv', d.get('argv')[-8:], 'RAM free %.1f GiB' % (d['ram_free']/2**30))"
echo "=== no VoidLoadConditioningFull registrado? ==="
curl -s -m 10 http://127.0.0.1:8190/object_info/VoidLoadConditioningFull | $PY -s -c "import json,sys; d=json.load(sys.stdin); print('sim:', list(d)[:1], [x for x in d['VoidLoadConditioningFull']['input']['required']['file_name'][0] if 'ltx23cond' in x])"
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors"
run() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3; }
echo "=== (a) identidade: W4A8 249 quadros, condicionamento completo, vs encoder vivo $(date)"
run ltx23av_w4a8_condf --cond-from ltx23condf --frames 249 --seed 1234 --limite 3600
$PY -s tools/compara_av.py --ref .scratch/ltx23av_w4a8.json --arm "W4A8, saved conditioning (all options)=.scratch/ltx23av_w4a8_condf.json" --out bench/ltx23/cond_identity --titulo "LTX 2.3 W4A8, 249 frames: live Gemma 3 encoder vs conditioning saved with all options (float32, tools/ltx_encode_lowcommit.py)" 2>&1 | grep -E 'saved conditioning|controle' | head -3
echo "=== (b) LoRA, prompt do farol (sem gatilho), 49 quadros $(date)"
run ltx23lora_base   --cond-from ltx23condf --frames 49 --seed 1234 --limite 1800
run ltx23lora_merge  --cond-from ltx23condf --frames 49 --seed 1234 --limite 1800 --lora LTX23_Product_Commercial_LoRA.safetensors
run ltx23lora_bypass --cond-from ltx23condf --frames 49 --seed 1234 --limite 1800 --lora LTX23_Product_Commercial_LoRA.safetensors --lora-bypass
run ltx23lora_seed2  --cond-from ltx23condf --frames 49 --seed 4321 --limite 1800
$PY -s tools/compara_av.py --ref .scratch/ltx23lora_base.json --arm "LoRA merged=.scratch/ltx23lora_merge.json" --arm "LoRA bypass=.scratch/ltx23lora_bypass.json" --arm "no LoRA, other seed=.scratch/ltx23lora_seed2.json" --out bench/ltx23/lora --titulo "LTX 2.3 W4A8 + Product Commercial LoRA, 49 frames, lighthouse prompt (no trigger) -- reference: same seed, no LoRA" 2>&1 | grep -E 'LoRA|no LoRA|controle|referencia' | head -6
$PY -s tools/compara_av.py --ref .scratch/ltx23lora_merge.json --arm "LoRA bypass vs merged=.scratch/ltx23lora_bypass.json" --out bench/ltx23/lora_par --titulo "LTX 2.3 W4A8 + Product Commercial LoRA: merged vs bypass, 49 frames" 2>&1 | grep -E 'bypass vs merged' | head -1
echo "=== (c) LoRA, prompt comercial COM gatilho srx_commercial, 49 quadros $(date)"
run ltx23trig_base   --cond-from ltx23cond_srx --frames 49 --seed 1234 --limite 1800
run ltx23trig_merge  --cond-from ltx23cond_srx --frames 49 --seed 1234 --limite 1800 --lora LTX23_Product_Commercial_LoRA.safetensors
run ltx23trig_bypass --cond-from ltx23cond_srx --frames 49 --seed 1234 --limite 1800 --lora LTX23_Product_Commercial_LoRA.safetensors --lora-bypass
run ltx23trig_seed2  --cond-from ltx23cond_srx --frames 49 --seed 4321 --limite 1800
$PY -s tools/compara_av.py --ref .scratch/ltx23trig_base.json --arm "LoRA merged=.scratch/ltx23trig_merge.json" --arm "LoRA bypass=.scratch/ltx23trig_bypass.json" --arm "no LoRA, other seed=.scratch/ltx23trig_seed2.json" --out bench/ltx23/lora_trigger --titulo "LTX 2.3 W4A8 + Product Commercial LoRA, 49 frames, trigger word srx_commercial in the prompt -- reference: same seed, no LoRA" 2>&1 | grep -E 'LoRA|no LoRA|controle|referencia' | head -6
$PY -s tools/compara_av.py --ref .scratch/ltx23trig_merge.json --arm "LoRA bypass vs merged=.scratch/ltx23trig_bypass.json" --out bench/ltx23/lora_trigger_par --titulo "LTX 2.3 W4A8 + Product Commercial LoRA with trigger: merged vs bypass, 49 frames" 2>&1 | grep -E 'bypass vs merged' | head -1
echo "=== (d) Q6_K e W4A4, 249 quadros, condicionamento completo $(date)"
run ltx23av_q6k_condf  --cond-from ltx23condf --frames 249 --seed 1234 --limite 3600 --gguf ltx-2.3-22b-distilled-1.1-Q6_K.gguf --video-vae LTX23_video_vae_bf16.safetensors
echo "--- ltx23av_w4a4_condf $(date)"; $PY -s tools/ltx_video.py --checkpoint ltx-2.3-22b-distilled-1.1_w4a4.safetensors $AUX --saida ltx23av_w4a4_condf --json .scratch/ltx23av_w4a4_condf.json --cond-from ltx23condf --frames 249 --seed 1234 --limite 3600 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3
echo "=== (e) BF16 por GGUF, 249 quadros, condicionamento completo $(date)"
curl -s -m 30 -X POST http://127.0.0.1:8190/free -H 'Content-Type: application/json' -d '{"unload_models": true, "free_memory": true}' -o /dev/null; sleep 5
run ltx23av_bf16 --cond-from ltx23condf --frames 249 --seed 1234 --limite 7200 --gguf ltx-2.3-22b-distilled-1.1-BF16.gguf --video-vae LTX23_video_vae_bf16.safetensors
if [ -f .scratch/ltx23av_bf16.json ] && grep -q '"status": "success"' .scratch/ltx23av_bf16.json; then
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_bf16.json --arm "GGUF Q6_K (third party)=.scratch/ltx23av_q6k_condf.json" --arm "W4A8 ours=.scratch/ltx23av_w4a8_condf.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4_condf.json" --out bench/ltx23/av --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio, 512px, 8 steps, seed 1234 -- reference BF16 (lossless GGUF container); same saved conditioning in every arm" 2>&1 | grep -v -E '^\s*$' | tail -12
else
  echo "BF16_23_FALHOU_9 $(date)"
  $PY -s tools/compara_av.py --ref .scratch/ltx23av_q6k_condf.json --arm "W4A8 ours=.scratch/ltx23av_w4a8_condf.json" --arm "W4A4 ours (control)=.scratch/ltx23av_w4a4_condf.json" --out bench/ltx23/av_vs_q6k --titulo "LTX 2.3 22B distilled 1.1, 249 frames + audio -- reference: third-party GGUF Q6_K (BF16 unreachable)" 2>&1 | grep -v -E '^\s*$' | tail -10
fi
echo "FILA_LTX23J_DONE $(date)"
