#!/bin/bash
# Oitava fila do 2.3: rodada de LoRA + controle de identidade do condicionamento salvo.
# Mecanismo medido ANTES desta fila (scratchpad/probe_commit_mmap.py, probe_safeopen_trace.py,
# probe_double_map.py, 2026-09-14 00:18-00:30): `safetensors.safe_open` = duas views copy-on-write
# = 2x o arquivo em commit; o `torch.empty` do modelo = mais 1x; quando a cobranca precisa expandir
# o pagefile, a view (ou o heap) volta sem lastro e o primeiro toque da access violation -- em C: e
# em W:, em processo nu. Entao: (1) encoder NUNCA residente (condicionamento salvo = 0 commit);
# (2) o W4A8 de 15,5 GiB custa 31 GiB transientes via safe_open, cabe nos ~53 livres sem expandir;
# (3) o BF16 de 39 GiB entra por GGUF na fila seguinte (memmap somente-leitura, 0 commit).
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
PID=$(netstat -ano | grep LISTEN | grep ':8190 ' | awk '{print $5}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F >/dev/null 2>&1
sleep 5
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COMFYUI_MGPU_DISABLED=1 COMFY_BYPASS_WMI_UNAME=1 CUDA_VISIBLE_DEVICES=0,1
nohup $PY -s ./ComfyUI/main.py --windows-standalone-build --use-sage-attention --disable-dynamic-vram --cache-none --preview-method none --listen 127.0.0.1 --port 8190 > .scratch/comfy_8190_h.log 2> .scratch/comfy_8190_h.err &
for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8190/system_stats >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('servidor up, argv', d.get('argv')[-8:], 'RAM free %.1f GiB' % (d['ram_free']/2**30))"
W4A8=ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
AUX="--audio-checkpoint LTX23_audio_vae_bf16_W.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors --cond-from ltx23cond"
echo "=== (f) LoRA LTX 2.3, W4A8, 49 quadros, condicionamento salvo (encoder nao carregado) $(date)"
runl() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --frames 49 --saida "$1" --json ".scratch/$1.json" --limite 1800 "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3; }
runl ltx23lora_base   --seed 1234
runl ltx23lora_merge  --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors
runl ltx23lora_bypass --seed 1234 --lora LTX23_Product_Commercial_LoRA.safetensors --lora-bypass
runl ltx23lora_seed2  --seed 4321
$PY -s tools/compara_av.py --ref .scratch/ltx23lora_base.json --arm "LoRA merged=.scratch/ltx23lora_merge.json" --arm "LoRA bypass=.scratch/ltx23lora_bypass.json" --arm "no LoRA, other seed=.scratch/ltx23lora_seed2.json" --out bench/ltx23/lora --titulo "LTX 2.3 W4A8 + Product Commercial LoRA, 49 frames -- reference: same seed, no LoRA" 2>&1 | grep -v -E '^\s*$' | tail -10
echo "=== (g) controle de identidade: W4A8 249 quadros, condicionamento salvo, contra o render com encoder vivo $(date)"
$PY -s tools/ltx_video.py --checkpoint $W4A8 $AUX --frames 249 --seed 1234 --saida ltx23av_w4a8_cond --json .scratch/ltx23av_w4a8_cond.json --limite 3600 2>&1 | grep -E 'terminou|STATUS|RECUSOU|Error|Traceback|passou de' | head -3
$PY -s tools/compara_av.py --ref .scratch/ltx23av_w4a8.json --arm "W4A8, saved conditioning=.scratch/ltx23av_w4a8_cond.json" --out bench/ltx23/cond_identity --titulo "LTX 2.3 W4A8, 249 frames: live Gemma 3 encoder vs saved conditioning (LTXVSaveConditioning, bf16)" 2>&1 | grep -v -E '^\s*$' | tail -6
echo "FILA_LTX23H_DONE $(date)"
