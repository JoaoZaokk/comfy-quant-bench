#!/bin/bash
# QAT 4-bit na 3090 (criterio_qat_klein_2026-09-22, desvio das 22:12) + avaliacao automatica.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
R=/f/qat_klein/run_4bit
D=bench/qat_klein
until grep -q "=== FIM smoke4" .scratch/qat_smoke/roda4.log; do sleep 20; done
echo "=== QAT $(date +%T) ==="
$PY -s .scratch/qat_smoke/guarda.py $PY -s tools/qat_ternario_klein.py \
  --raiz P:/ComfyBench/originais/FLUX.2-klein-4B \
  --professor F:/bonsai-re/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors \
  --prompts $D/prompts_treino.txt --prompts-holdout $D/prompts_holdout.txt \
  --sementes 1 2 --dir $R --otim adamw4bit-sr --max-passos 4000 --lote 1 \
  --log-cada 50 --holdout-cada 500 --ckpt-min 25 >> $R.log 2>&1
rc=$?; echo "qat rc=$rc"; [ $rc -ne 0 ] && exit $rc
echo "=== remap $(date +%T) ==="
$PY -s tools/aplica_mapa_diffusers_bfl.py --entrada $R/aluno_ternario_diffusers.safetensors \
  --mapa .scratch/mapa_klein_diffusers_bfl.json \
  --saida ComfyUI/models/diffusion_models/klein4b_qat4bit_bfl.safetensors > $D/remap_qat4bit.log 2>&1
echo "remap rc=$?"
echo "=== eps $(date +%T) ==="
$PY -s .scratch/roda_eps_qat.py > $D/eps_qat_driver.log 2>&1; echo "eps rc=$?"
echo "=== render $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_braco1s_bf16_sr_bfl.safetensors klein4b_qat4bit_bfl.safetensors klein4b_braco2_bonsai_ternario_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render > $D/ladder.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render > $D/decode.log 2>&1; echo "decode rc=$?"
echo "=== FIM qat run $(date +%T) ==="
