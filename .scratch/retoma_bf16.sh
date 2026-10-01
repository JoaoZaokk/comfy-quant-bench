#!/bin/bash
# Retoma o braco BF16 do LTX 2.5 depois de ele derrubar o servidor: `access violation` em
# torch/storage.py __getitem__ dentro de load_torch_file -- o mmap do arquivo de 39 GiB, lido do
# compartilhamento SMB (D:), morreu com ~24 GiB de RAM livre. O mesmo arquivo existe em W: (disco
# local); um hardlink com outro nome (`..._W.safetensors`, mesmo inode) faz o resolvedor ler de W:
# em vez de D:. Espera a conversao do gemma acabar para nao competir por RAM, e so DEPOIS solta
# a conversao W4A4 do 2.3 (que le 43 GiB de P:).
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until grep -q 'GEMMA3_W4A8_DONE' .scratch/converte_gemma3_te.log 2>/dev/null; do sleep 20; done
echo "=== bf16 retomado $(date)"
curl -s -m 5 http://127.0.0.1:8190/system_stats | $PY -s -c "import json,sys; d=json.load(sys.stdin)['system']; print('RAM free %.1f GiB' % (d['ram_free']/2**30))"
$PY -s tools/ltx_video.py --encoder gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors --frames 249 --transformer ltx-2.5-22b-distilled-transformer-bf16_W.safetensors --distorch --alocacao "cpu,40gb" --saida ltx25av_bf16 --json .scratch/ltx25av_bf16.json --limite 7200 2>&1 | tail -12
if [ -f .scratch/ltx25av_bf16.json ] && grep -q '"status": "success"' .scratch/ltx25av_bf16.json; then
  echo "LTX25_AV_3BRACOS_DONE $(date)" >> .scratch/ltx25_av_3bracos.log
  echo "bf16 OK, marcador escrito $(date)"
else
  echo "BF16_FALHOU $(date)"
fi
echo "=== w4a4 ltx 2.3 $(date)"
export CUDA_VISIBLE_DEVICES=1
$PY -s tools/quant_w4a4.py --input P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors --profile ltx_2_5 --output P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a4.safetensors 2>&1 | tail -6
echo "LTX23_W4A4_DONE $(date)"
