#!/bin/bash
# Pesos dos bracos da noite contra o Bonsai: codigos (concorda/acerto/recall) e tipos de mudanca
# (promove/rebaixa, distancia do limiar, profundidade). Referencia: QAT local p4000. So' CPU.
cd /f/COMFY_PORTABLE
PY=./python_embeded/python.exe
for b in b1 b2 b3 b4d; do
  $PY -s -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('JoaoZaokk/klein4b-qat-$b','final/aluno_ternario_diffusers.safetensors',local_dir='F:/qat_klein/pesos_$b',token=False))" || exit 1
done
BR="--braco qat_local_p4000=F:/qat_klein/run_4bit/aluno_ternario_diffusers.safetensors"
for b in b1 b2 b3 b4d; do BR="$BR --braco ${b}=F:/qat_klein/pesos_$b/final/aluno_ternario_diffusers.safetensors"; done
$PY -s tools/compara_codigos_bonsai.py \
  --bf16 F:/bonsai-re/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors \
  --bonsai F:/bonsai-re/bonsai-image-ternary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors \
  --ingenuo P:/ComfyBench/originais/klein4b_ternario_ingenuo.safetensors \
  $BR --saida bench/qat_klein/codigos_noite_vs_bonsai.json > bench/qat_klein/codigos_noite_vs_bonsai.log 2>&1; echo "codigos rc=$?"
$PY -s tools/analisa_mudancas_bonsai.py \
  --bf16 F:/bonsai-re/FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors \
  --bonsai F:/bonsai-re/bonsai-image-ternary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors \
  $BR --saida bench/qat_klein/mudancas_noite_vs_bonsai.json > bench/qat_klein/mudancas_noite_vs_bonsai.log 2>&1; echo "mudancas rc=$?"
rm -rf F:/qat_klein/pesos_b1 F:/qat_klein/pesos_b2 F:/qat_klein/pesos_b3 F:/qat_klein/pesos_b4d
echo "=== FIM pesos $(date +%T) ==="
