#!/bin/bash
# Avalia um braco da fila da noite (bench/plano_qat_noite_2026-09-23.md): espera o `final/final.json` no
# repo publico JoaoZaokk/klein4b-qat-<braco>, baixa o exportado `melhor` (se houver) e o `final`,
# remapeia para nomes BFL, renderiza (5 prompts x 2 sementes, mesmo protocolo do render_replay1) e mede
# epsilon fora da amostra. Apaga o download diffusers; o BFL fica para a grade entre bracos.
#   uso: avalia_braco.sh b1
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
B=$1
PY=./python_embeded/python.exe
D=bench/qat_klein
REPO=JoaoZaokk/klein4b-qat-$B
W=F:/qat_klein/noite_$B
MD=ComfyUI/models/diffusion_models
echo "=== $B: esperando final.json em $REPO $(date +%T) ==="
until $PY -s -c "
from huggingface_hub import HfApi; import sys
sys.exit(0 if HfApi().file_exists('$REPO','final/final.json', token=False) else 1)" 2>/dev/null; do sleep 240; done
echo "=== $B: final no HF $(date +%T) ==="
$PY -s - <<PYEOF
import json, time
from huggingface_hub import hf_hub_download, HfApi
api = HfApi()
for sub in ("final", "melhor"):
    if not api.file_exists("$REPO", f"{sub}/aluno_ternario_diffusers.safetensors", token=False):
        print(sub, "ausente"); continue
    t = time.time()
    p = hf_hub_download("$REPO", f"{sub}/aluno_ternario_diffusers.safetensors", local_dir="$W", token=False)
    print(sub, "baixado em", round(time.time() - t), "s")
    j = "final/final.json" if sub == "final" else "melhor/melhor.json"
    print(sub, open(hf_hub_download("$REPO", j, local_dir="$W", token=False)).read())
PYEOF
MODELS=""
ARMS=""
for sub in melhor final; do
  f=$W/$sub/aluno_ternario_diffusers.safetensors
  [ -f "$f" ] || continue
  $PY -s tools/aplica_mapa_diffusers_bfl.py --entrada "$f" --mapa .scratch/mapa_klein_diffusers_bfl.json \
    --saida $MD/klein4b_qat_${B}_${sub}_bfl.safetensors > $D/remap_${B}_${sub}.log 2>&1; echo "remap $sub rc=$?"
  rm -f "$f"
  MODELS="$MODELS klein4b_qat_${B}_${sub}_bfl.safetensors"
  ARMS="$ARMS ${B}_${sub}=klein4b_qat_${B}_${sub}_bfl.safetensors"
done
# a 3090 pode estar com o dono (outros testes): so' renderiza depois de .scratch/gpu_livre existir
echo "=== $B: esperando .scratch/gpu_livre $(date +%T) ==="
until [ -f .scratch/gpu_livre ]; do sleep 60; done
echo "=== $B: render $(date +%T) ==="
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors --models $MODELS \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file bench/render_braco1_2026-09-22/prompts.txt --seeds 11 12 \
  --steps 8 --cfg 1.0 --size 1024 --out $D/render_$B > $D/ladder_$B.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/render_$B/latents --vae flux2_klein_vae_diffusers.safetensors --out $D/render_$B > $D/decode_$B.log 2>&1; echo "decode rc=$?"
echo "=== $B: eps $(date +%T) ==="
$PY -s .scratch/roda_eps_generico.py $B braco0_ptq=klein4b_braco0_ternario_ingenuo_bfl.safetensors $ARMS \
  qatA100_3024=klein4b_qatA100_bfl.safetensors > $D/eps_${B}_driver.log 2>&1; echo "eps rc=$?"
$PY -s tools/agrega_epsilon_sementes.py $D/eps_$B.log --base braco0_ptq > $D/agrega_eps_$B.txt 2>&1; echo "agrega rc=$?"
# os BFL ficam para a grade entre bracos (4 x 2 x 7,2 GB cabem nos ~98 GB livres)
echo "=== FIM avalia $B $(date +%T) ==="
