#!/bin/bash
# Os tres bracos do LTX 2.5 em 10 s, AGORA com audio. Sequencial: uma placa.
cd /f/COMFY_PORTABLE
ENC=gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
run() { echo "=== $1 $(date)"; ./python_embeded/python.exe -s tools/ltx_video.py --encoder $ENC --frames 249 "${@:2}" 2>&1 | tail -12; }
run w4a8  --transformer ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors --saida ltx25av_w4a8 --json .scratch/ltx25av_w4a8.json
run int8  --transformer ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors --saida ltx25av_int8 --json .scratch/ltx25av_int8.json
run bf16  --transformer ltx-2.5-22b-distilled-transformer-bf16.safetensors --distorch --alocacao "cpu,40gb" --saida ltx25av_bf16 --json .scratch/ltx25av_bf16.json --limite 10800
echo "LTX25_AV_3BRACOS_DONE $(date)"
