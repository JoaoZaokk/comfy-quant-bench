#!/bin/bash
# Fila da 3090 depois dos tres bracos do LTX 2.5 com audio. Sequencial: uma placa, um eixo por vez.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until grep -q 'LTX25_AV_3BRACOS_DONE' .scratch/ltx25_av_3bracos.log 2>/dev/null; do sleep 30; done
echo "=== (a) compara_av LTX 2.5 $(date)"
$PY -s tools/compara_av.py --ref .scratch/ltx25av_bf16.json --arm "int8 Lightricks 20.03 GiB=.scratch/ltx25av_int8.json" --arm "W4A8 ours 11.66 GiB=.scratch/ltx25av_w4a8.json" --out bench/ltx25/av --titulo "LTX 2.5 22B distilled, 249 frames + audio, 512px, seed 1234 -- reference BF16" 2>&1 | grep -v -E '^\s*$' | tail -12
echo "=== (b) probes LoRA LTX 2.5 W4A8 $(date)"
$PY -s tools/probe_lora_requant.py --ckpt ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors --lora ltx2-squish.safetensors --source W:/ltx-2.5/ltx-2.5-22b-distilled-transformer-bf16.safetensors --n 24 --device 0 --json .scratch/lora_requant_ltx25_squish.json 2>&1 | grep -v -E 'lora key not loaded|^\s*$' | tail -16
$PY -s tools/probe_lora_requant.py --ckpt ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors --lora LTX23_Product_Commercial_LoRA.safetensors --source W:/ltx-2.5/ltx-2.5-22b-distilled-transformer-bf16.safetensors --n 24 --device 0 --json .scratch/lora_requant_ltx25_product.json 2>&1 | grep -v -E 'lora key not loaded|^\s*$' | tail -16
echo "=== (c) Qwen Edit Lightning 4 passos $(date)"
LORA=Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
run_q() { # rotulo, transformer, par, imagem, instrucao, extra...
  $PY -s tools/qwen_edit_test.py --transformer "$2" --imagem "$4" --instrucao "$5" --seed 1 --steps 4 --cfg 1.0 --saida "qedit_$1_$3_s1" --json ".scratch/qedit_$1_$3_s1.json" "${@:6}" 2>&1 | grep -E 'status|lora|RECUSOU|Error' | head -3; }
for par in "pera|edit_maca.png|change the apple to a green pear, keep the table, the window light and the composition exactly the same" \
           "cachecol|edit_pescador.png|add a red knitted scarf around his neck, change nothing else" \
           "closed|edit_placa.png|change the word on the sign to CLOSED, keep the same enamel sign, the same brick wall and the same lighting"; do
  IFS='|' read -r id img ins <<< "$par"
  run_q int8semlora qwen_image_edit_2511_int8_convrot.safetensors "$id" "$img" "$ins"
  run_q int8lora    qwen_image_edit_2511_int8_convrot.safetensors "$id" "$img" "$ins" --lora $LORA
  run_q w4a8semlora qwen_image_edit_2511_w4a8.safetensors "$id" "$img" "$ins"
  run_q w4a8lora    qwen_image_edit_2511_w4a8.safetensors "$id" "$img" "$ins" --lora $LORA
  run_q w4a8bypass  qwen_image_edit_2511_w4a8.safetensors "$id" "$img" "$ins" --lora $LORA --lora-bypass
done
echo "=== (d) LTX 2.5 W4A8 + squish, 49 quadros $(date)"
ENC=gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
T=ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors
run_l() { $PY -s tools/ltx_video.py --transformer $T --encoder $ENC --frames 49 --saida "$1" --json ".scratch/$1.json" "${@:2}" 2>&1 | grep -E 'terminou|STATUS|RECUSOU|lora' | head -3; }
run_l ltx25lora_base   --seed 1234
run_l ltx25lora_merge  --seed 1234 --lora ltx2-squish.safetensors
run_l ltx25lora_bypass --seed 1234 --lora ltx2-squish.safetensors --lora-bypass
run_l ltx25lora_seed2  --seed 4321
$PY -s tools/compara_av.py --ref .scratch/ltx25lora_base.json --arm "squish merged=.scratch/ltx25lora_merge.json" --arm "squish bypass=.scratch/ltx25lora_bypass.json" --arm "no LoRA, other seed=.scratch/ltx25lora_seed2.json" --out bench/ltx25/lora --titulo "LTX 2.5 W4A8 + ltx2-squish, 49 frames -- reference: same seed, no LoRA" 2>&1 | grep -v -E '^\s*$' | tail -10
echo "FILA_GPU0_DONE $(date)"
