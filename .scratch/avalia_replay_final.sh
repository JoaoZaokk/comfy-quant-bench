#!/bin/bash
# Fecha o criterio de repeticao: espera o checkpoint FINAL (passo 6000) do replay no HF, exporta e renderiza
# no protocolo do render_replay1. O ckpt do replay e' sobrescrito a cada envio, entao so' aceita passo 6000.
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein
until $PY -s - <<'PYEOF'
import sys, torch
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
c = api.list_repo_commits("JoaoZaokk/klein4b-qat-replay", token=False)[0]
print("ultimo commit", c.created_at, flush=True)
if c.created_at.strftime("%Y-%m-%d %H:%M") < "2026-09-24 00:17":
    sys.exit(1)
p = hf_hub_download("JoaoZaokk/klein4b-qat-replay", "ckpt/ultimo.pt", local_dir="F:/qat_klein/replay_final", token=False)
passo = torch.load(p, map_location="cpu", weights_only=False, mmap=True)["passo"]
print("passo", passo, flush=True)
sys.exit(0 if passo >= 6000 else 1)
PYEOF
do sleep 300; done
$PY -s .scratch/exporta_ckpt_qat.py F:/qat_klein/replay_final/ckpt/ultimo.pt F:/qat_klein/qatReplay6000_diffusers.safetensors; echo "export rc=$?"
$PY -s tools/aplica_mapa_diffusers_bfl.py --entrada F:/qat_klein/qatReplay6000_diffusers.safetensors \
  --mapa .scratch/mapa_klein_diffusers_bfl.json --saida ComfyUI/models/diffusion_models/klein4b_qatReplay6000_bfl.safetensors > $D/remap_replay6000.log 2>&1; echo "remap rc=$?"
rm -f F:/qat_klein/qatReplay6000_diffusers.safetensors
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_qatReplay6000_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_replay1 > $D/ladder_replay6000.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_replay1/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_replay1 > $D/decode_replay6000.log 2>&1; echo "decode rc=$?"
echo "=== FIM replay final $(date +%T) ==="
