#!/bin/bash
# Baixa o final de um braco do HF, remapeia para BFL e roda o K0 (resto identico ao BF16). So' CPU/disco.
cd /f/COMFY_PORTABLE; B=$1; PY=./python_embeded/python.exe; W=${PREPARA_W:-F:/qat_klein/noite_$B}; MD=ComfyUI/models/diffusion_models
until $PY -s -c "
from huggingface_hub import HfApi; import sys
sys.exit(0 if HfApi().file_exists('JoaoZaokk/klein4b-qat-$B','final/final.json', token=False) else 1)" 2>/dev/null; do sleep 120; done
echo "=== $B final no HF $(date +%T) ==="
$PY -s -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('JoaoZaokk/klein4b-qat-$B', 'final/aluno_ternario_diffusers.safetensors', local_dir='$W', token=False); print(p)
print(open(hf_hub_download('JoaoZaokk/klein4b-qat-$B', 'final/final.json', local_dir='$W', token=False)).read())"
$PY -s tools/aplica_mapa_diffusers_bfl.py --entrada $W/final/aluno_ternario_diffusers.safetensors --mapa .scratch/mapa_klein_diffusers_bfl.json \
  --saida $MD/klein4b_qat_${B}_final_bfl.safetensors > bench/qat_klein/remap_${B}_final.log 2>&1; RC=$?; echo "remap rc=$RC"; [ $RC -eq 0 ] || exit 1
rm -f $W/final/aluno_ternario_diffusers.safetensors
$PY -s .scratch/k0_resto_identico.py $MD/klein4b_qat_${B}_final_bfl.safetensors $MD/klein4b_braco3_bf16_original_bfl.safetensors
echo "=== FIM prepara $B $(date +%T) ==="
