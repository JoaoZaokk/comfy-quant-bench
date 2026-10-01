#!/bin/bash
# Copias para o disco local W:, com outro nome: cada braco do 2.3 estava lendo ~42 GB do SMB so
# para carregar (W4A8 16,6 + gemma 22,7 + projecao 2,3 + audio VAE 0,4), seis vezes ao todo.
cd /f/COMFY_PORTABLE
cp P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a8.safetensors W:/ltx-2.3/ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors
cp P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a8.quant.json W:/ltx-2.3/ltx-2.3-22b-distilled-1.1_w4a8_W.quant.json
cp P:/ComfyBench/checkpoints/ltx-2.3_text_projection_bf16.safetensors W:/ltx-2.3/ltx-2.3_text_projection_bf16_W.safetensors
cp P:/ComfyBench/checkpoints/LTX23_audio_vae_bf16.safetensors W:/ltx-2.3/LTX23_audio_vae_bf16_W.safetensors
cp P:/ComfyBench/text_encoders/gemma_3_12B_it.safetensors W:/ltx-2.3/gemma_3_12B_it_W.safetensors
echo "COPIAS_W_OK $(date)"
