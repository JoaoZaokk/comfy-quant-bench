#!/bin/bash
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
echo "=== download replay $(date +%T) ==="
$PY -s - <<'PYEOF'
import time, os, torch
from huggingface_hub import hf_hub_download
t = time.time()
p = hf_hub_download("JoaoZaokk/klein4b-qat-replay", "ckpt/ultimo.pt", local_dir="F:/qat_klein/replay_hf1", token=False)
print(f"baixado {os.path.getsize(p)/2**30:.2f} GiB em {time.time()-t:.0f} s")
print("passo do checkpoint replay:", torch.load(p, map_location="cpu", weights_only=False, mmap=True)["passo"])
PYEOF
$PY -s .scratch/exporta_ckpt_qat.py F:/qat_klein/replay_hf1/ckpt/ultimo.pt F:/qat_klein/qatReplay1_diffusers.safetensors; echo "export rc=$?"
$PY -s tools/aplica_mapa_diffusers_bfl.py --entrada F:/qat_klein/qatReplay1_diffusers.safetensors \
  --mapa .scratch/mapa_klein_diffusers_bfl.json --saida ComfyUI/models/diffusion_models/klein4b_qatReplay1_bfl.safetensors > $D/remap_replay1.log 2>&1; echo "remap rc=$?"
echo "=== render replay $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_qat4bit_bfl.safetensors klein4b_qatA100b_bfl.safetensors klein4b_qatReplay1_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_replay1 > $D/ladder_replay1.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_replay1/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_replay1 > $D/decode_replay1.log 2>&1; echo "decode rc=$?"
echo "=== FIM replay1 $(date +%T) ==="
