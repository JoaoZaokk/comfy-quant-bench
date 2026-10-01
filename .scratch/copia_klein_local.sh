#!/bin/bash
# Copia os 4 bracos BFL do NAS (P:) para ComfyUI/models/diffusion_models (F:, local).
# MESMO nome: o resolvedor do ComfyUI percorre models/ antes das raizes do yaml, entao a copia
# local vence. Copias MINHAS, descartaveis; o original em P: nao e tocado.
set -e
for f in klein4b_braco0_ternario_ingenuo_bfl klein4b_braco1_compensado_bfl klein4b_braco2_bonsai_ternario_bfl klein4b_braco3_bf16_original_bfl; do
  src="/p/ComfyBench/diffusion_models/$f.safetensors"
  dst="/f/COMFY_PORTABLE/ComfyUI/models/diffusion_models/$f.safetensors"
  t0=$(date +%s)
  cp "$src" "$dst.partial" && mv "$dst.partial" "$dst"
  a=$(stat -c %s "$src"); b=$(stat -c %s "$dst")
  echo "$f  src $a  dst $b  $([ "$a" = "$b" ] && echo TAMANHO_OK || echo TAMANHO_DIFERE)  $(( $(date +%s)-t0 ))s"
done
echo FIM
