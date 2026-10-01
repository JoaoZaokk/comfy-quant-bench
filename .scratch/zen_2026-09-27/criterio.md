# Critério: zen-image-edit (Qwen3.5-0.8B + adaptador) como text encoder do Qwen-Image 2.1

Escrito antes de qualquer render. 2026-09-27. Pedido do dono: "baixa e testa".

## Montagem

- Mesmo grafo da bateria Qwen 2.1 (`qwen21_2026-09-26/monta_bateria.py`): DiT `qwen_image_2.1_bf16`, VAE
  `qwen_image_2.1_vae_bf16`, 6 prompts x seeds 42/7, 1024², 25 passos, euler/simple, cfg 1, sem shift extra.
- Só troca o nó de texto: `TextEncodeQwenImage21` (Qwen3-VL-8B **W4A8**, o que a bateria usou) ->
  `ZenImage21TextEncode` (Qwen3.5-0.8B + adapter_v12, bf16).
- Referência = imagens `bf16` já existentes da bateria. **Ressalva: a referência usa o TE nativo em W4A8, não BF16**
  (não temos o BF16 de 17,5 GB no disco).
- Isolamento: transformers >= 5.17 em pasta própria (`P:\ComfyBench\zen_test\pydeps`), ComfyUI de teste com
  `--disable-all-custom-nodes --whitelist-custom-nodes zen-image-edit-comfyui`; o ambiente principal não muda.

## Previsões (antes)

Z1. O nó carrega e as 12 imagens saem sem erro, com o DiT original.
Z2. Troca de encoder muda composição: MS-SSIM contra a referência vai ser BAIXO (espero 0,4-0,7) mesmo se a
    qualidade for boa. MS-SSIM aqui não julga qualidade, só distância; o julgamento é visual.
Z3. Visual (eu olho, sem NSFW nesta bateria): aderência ao prompt nas 12; texto "SLOW MORNINGS" legível; o letreiro
    "QWEN IMAGE 2.1" provavelmente com erro no número (limitação declarada). Aceite prático: >= 10/12 aderentes
    e sem artefato grosseiro.
Z4. Memória: TE zen ~2,4 GB contra 6,3 GB (W4A8) / 17,5 GB (BF16 nativo). Tempo de encode medido pelo log.
