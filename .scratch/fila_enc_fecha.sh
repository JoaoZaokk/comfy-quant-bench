#!/bin/bash
# Fila da 3080 Ti (fechamento 2026-09-14): cinco encodes do farol em formato completo -- Gemma de
# fabrica W4A8 (comparado com o BF16 de fabrica ltx23condf), heretic BF16, e os tres builds do
# heretic (comparados com o heretic BF16) -- e depois a reconversao do capybara em W4A8.
# Criterio antes: bench/criterio_fechamento_2026-09-14.md (A, D, E).
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PROMPT="a lone lighthouse on a rocky cliff at dusk, waves breaking against the rocks, the beam sweeping across low clouds, seabirds circling"
run_enc() {
  echo "=== $1 <- $2 $(date)"
  if [ -n "$3" ]; then CMP="--compare $3"; else CMP=""; fi
  $PY -s tools/ltx_encode_lowcommit.py --encoder "$2" --proj-checkpoint ltx-2.3_text_projection_bf16_W.safetensors --prompt "$PROMPT" --saida "$1" --gpu 1 $CMP --json ".scratch/enc_$1.json" 2>&1 | grep -E 'leitor RO|construido|codificado|gravado|compare|Error|Traceback|RECUS|NAO COBERTO'
}
run_enc ltx23cond_gw4a8  gemma_3_12B_it_w4a8.safetensors ltx23condf
echo "ENC_GW4A8_DONE $(date)"
run_enc ltx23cond_hbf16  gemma_3_12B_it_heretic.safetensors ltx23condf
run_enc ltx23cond_hw4a8  gemma_3_12B_it_heretic_w4a8.safetensors ltx23cond_hbf16
run_enc ltx23cond_hw4a4c gemma_3_12B_it_heretic_w4a4_convrot.safetensors ltx23cond_hbf16
run_enc ltx23cond_hw4a4s gemma_3_12B_it_heretic_w4a4_smooth.safetensors ltx23cond_hbf16
echo "ENC_HERETIC_DONE $(date)"
echo "=== metricas de condicionamento (rel-L2, cosseno) $(date)"
$PY -s tools/compara_cond.py --ref ltx23condf --arm ltx23cond_gw4a8 --arm ltx23cond_hbf16 --json bench/ltx23/encoder_cond_factory.json
$PY -s tools/compara_cond.py --ref ltx23cond_hbf16 --arm ltx23cond_hw4a8 --arm ltx23cond_hw4a4c --arm ltx23cond_hw4a4s --json bench/ltx23/encoder_cond_heretic.json
echo "=== capybara W4A8, reconversao na 3080 Ti $(date)"
CUDA_VISIBLE_DEVICES=1 $PY -s tools/quant_w4a8.py --input ComfyUI/models/diffusion_models/capybara_v0.1.safetensors --output ComfyUI/models/diffusion_models/capybara_v0.1_w4a8r.safetensors --profile hunyuan_video_15 2>&1 | tail -20
ls -la ComfyUI/models/diffusion_models/capybara_v0.1_w4a8r.* 2>&1
echo "CAPY_CONV_DONE $(date)"
echo "FILA_ENC_FECHA_DONE $(date)"
