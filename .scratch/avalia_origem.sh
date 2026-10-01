#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
$PY -s -c "
from huggingface_hub import hf_hub_download
import time; t=time.time()
p=hf_hub_download('JoaoZaokk/klein4b-qat-replay','origem/ckpt_origem.pt',local_dir='F:/qat_klein/origem',token=False); print('baixado', p, round(time.time()-t),'s')"
echo "download rc=$?"
$PY -s .scratch/exporta_ckpt_qat.py F:/qat_klein/origem/origem/ckpt_origem.pt F:/qat_klein/qatOrigem_diffusers.safetensors; echo "export rc=$?"
$PY -s tools/aplica_mapa_diffusers_bfl.py --entrada F:/qat_klein/qatOrigem_diffusers.safetensors \
  --mapa .scratch/mapa_klein_diffusers_bfl.json --saida ComfyUI/models/diffusion_models/klein4b_qatOrigem8159_bfl.safetensors > $D/remap_origem.log 2>&1; echo "remap rc=$?"
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_qatOrigem8159_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_replay1 > $D/ladder_origem.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_replay1/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_replay1 > $D/decode_origem.log 2>&1; echo "decode rc=$?"
echo "=== FIM origem $(date +%T) ==="
