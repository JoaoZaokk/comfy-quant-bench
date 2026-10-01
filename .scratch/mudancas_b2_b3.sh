#!/bin/bash
# P3 do plano da noite: o L1 (b3) rebaixa mais do que promove? b2 e b3 diferem SO' no L1. So' CPU.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
until $PY -s -c "
from huggingface_hub import HfApi; import sys
sys.exit(0 if HfApi().file_exists('JoaoZaokk/klein4b-qat-b3','final/final.json', token=False) else 1)" 2>/dev/null; do sleep 120; done
for b in b2 b3; do
  $PY -s -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('JoaoZaokk/klein4b-qat-$b','final/aluno_ternario_diffusers.safetensors',local_dir='F:/qat_klein/mud_$b',token=False))"
done
$PY -s tools/analisa_mudancas_bonsai.py \
  --bf16 F:/bonsai-re/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors \
  --bonsai F:/bonsai-re/bonsai-image-ternary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors \
  --braco b2_p3000=F:/qat_klein/mud_b2/final/aluno_ternario_diffusers.safetensors \
  --braco b3_l1_p3000=F:/qat_klein/mud_b3/final/aluno_ternario_diffusers.safetensors \
  --saida bench/qat_klein/mudancas_b2_b3.json > bench/qat_klein/mudancas_b2_b3.log 2>&1; echo "analise rc=$?"
rm -rf F:/qat_klein/mud_b2 F:/qat_klein/mud_b3
grep -v "^  [0-9]*/100" bench/qat_klein/mudancas_b2_b3.log | head -40
