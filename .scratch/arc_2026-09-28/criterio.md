# Critério: Qwen-Image 2.1 "leve" na Arc A750 8 GB (2026-09-28)

Escrito antes de qualquer render. Pedido do dono: "faz o ssh na arc ... baixa o modelo levinho e faz o patch", depois
"para o container, testa". O container `spark-x25-heretic` (llama.cpp) foi parado para liberar a VRAM; religar ao final.

## Montagem

- Máquina `ssh arc`: VM Ubuntu, A750 8 GB, Xe KMD, torch 2.14+xpu, ComfyUI 0.37.0 como serviço (8188).
- DiT `qwen_image_2.1_bf16_Q4_1.gguf` (ComfyUI-GGUF @6ea2651), VAE `qwen_image_2.1_vae_bf16`, TE zen
  (Qwen3.5-0.8B + adapter_v12, bf16, zen @7a29e58 + patch `patches/zen_image_edit_offload_encoder.patch`).
- Mesmo grafo e mesmas entradas das baterias da 3090: 6 prompts x seeds 42/7, 1024², 25 passos, euler/simple, cfg 1.
- Referências, todas geradas na 3090 (não é o mesmo dispositivo, então MS-SSIM mede distância, não julga qualidade):
  - `zen_v12`: DiT BF16 + zen. Isola o efeito do DiT Q4_1 e da troca de dispositivo (XPU contra CUDA).
  - `w4a16_q4_1`: DiT Q4_1 GGUF + TE nativo W4A8. Isola o efeito do TE.
  - `bf16`: DiT BF16 + TE nativo W4A8. Mostra a distância do conjunto inteiro.

## Previsões (antes)

A1. Roda de ponta a ponta sem OOM; as 12 imagens saem.
A2. VRAM (drm fdinfo do processo do ComfyUI, pico total na região vram): <= 7,0 GiB. Com o patch, o encoder não fica
    na placa durante a amostragem (log do zen e pico por fase).
A3. Velocidade: 0,25-0,4 it/s depois do aquecimento, ou seja ~60-100 s por imagem de 25 passos. Primeira imagem mais lenta
    (carga + compilação de kernels).
A4. MS-SSIM contra `zen_v12` ~0,9 (Q4_1 contra BF16 na 3090 deu 0,912); contra `bf16` abaixo de 0,841.
A5. Visual: mesmo aceite prático do zen, >= 10/12 aderentes e sem artefato grosseiro. Imagens sem NSFW, posso olhar.

Primeiro 1 imagem (p2 s42) para validar; se passar, as 12.
