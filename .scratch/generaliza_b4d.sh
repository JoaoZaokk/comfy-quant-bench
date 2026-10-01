#!/bin/bash
# Generalizacao do b4d: 10 prompts fora do treino (6 do holdout curado, 4 PartiPrompts), semente 11.
# Espera o epsilon do b4d terminar (mesma placa).
cd /f/COMFY_PORTABLE
export CUDA_VISIBLE_DEVICES=0
PY=./python_embeded/python.exe
D=bench/qat_klein/generaliza
until grep -q "FIM avalia b4d" .scratch/avalia_b4d.log; do sleep 30; done
$PY -s tools/quality_ladder.py --reference klein4b_braco3_bf16_original_bfl.safetensors \
  --models klein4b_qat_b4d_final_bfl.safetensors klein4b_qat4bit_bfl.safetensors klein4b_qatA100_bfl.safetensors \
  --clip qwen_3_4b.safetensors --clip-type flux2 --prompt-file $D/prompts.txt --seeds 11 \
  --steps 8 --cfg 1.0 --size 1024 --out $D > $D/ladder.log 2>&1; echo "ladder rc=$?"
$PY -s tools/decode_latents.py $D/latents --vae flux2_klein_vae_diffusers.safetensors --out $D > $D/decode.log 2>&1; echo "decode rc=$?"
$PY -s - <<'PYEOF'
from pathlib import Path
from PIL import Image, ImageDraw
D = Path("bench/qat_klein/generaliza")
cols = [("BF16", "klein4b_braco3_bf16_original_bfl"), ("b4d p4000", "klein4b_qat_b4d_final_bfl"),
        ("QAT local 4bit", "klein4b_qat4bit_bfl"), ("controle A100 p3024", "klein4b_qatA100_bfl")]
S, H = 256, 24
g = Image.new("RGB", (S * len(cols), H + S * 10), "white")
d = ImageDraw.Draw(g)
for j, (n, pre) in enumerate(cols):
    d.text((j * S + 6, 6), n, fill="black")
    for p in range(10):
        g.paste(Image.open(D / f"{pre}__p{p}_s11.png").convert("RGB").resize((S, S)), (j * S, H + p * S))
g.save(D / "grade_generaliza.png"); print("grade ok")
PYEOF
echo "=== FIM generaliza $(date +%T) ==="
