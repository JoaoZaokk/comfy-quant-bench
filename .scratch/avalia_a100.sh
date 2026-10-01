#!/bin/bash
# Puxa o checkpoint da A100 do HF, exporta e avalia no MESMO protocolo do QAT local.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
echo "=== download $(date +%T) ==="
$PY -s - <<'PYEOF'
import time, torch
from huggingface_hub import hf_hub_download
t = time.time()
p = hf_hub_download("JoaoZaokk/klein4b-qat-ckpt", "ckpt/ultimo.pt", local_dir="F:/qat_klein/a100_hf")
import os
s = os.path.getsize(p)
print(f"baixado {s/2**30:.2f} GiB em {time.time()-t:.0f} s ({s/2**20/(time.time()-t):.0f} MB/s)")
e = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
print("passo do checkpoint A100:", e["passo"], e["extra"])
PYEOF
echo "download rc=$?"
$PY -s .scratch/exporta_ckpt_qat.py F:/qat_klein/a100_hf/ckpt/ultimo.pt F:/qat_klein/qatA100_diffusers.safetensors; echo "export rc=$?"
$PY -s tools/aplica_mapa_diffusers_bfl.py --entrada F:/qat_klein/qatA100_diffusers.safetensors \
  --mapa .scratch/mapa_klein_diffusers_bfl.json --saida ComfyUI/models/diffusion_models/klein4b_qatA100_bfl.safetensors > $D/remap_qatA100.log 2>&1; echo "remap rc=$?"
until grep -q "=== FIM qat run" .scratch/qat_4bit_run.log; do sleep 30; done
echo "=== eps A100 $(date +%T) ==="
$PY -s .scratch/roda_eps_qatA100.py > $D/eps_qatA100_driver.log 2>&1; echo "eps A100 rc=$?"
echo "=== render A100 $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_qat4bit_bfl.safetensors klein4b_qatA100_bfl.safetensors klein4b_braco2_bonsai_ternario_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_dois > $D/ladder_dois.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_dois/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_dois > $D/decode_dois.log 2>&1; echo "decode rc=$?"
echo "=== FIM avalia A100 $(date +%T) ==="
