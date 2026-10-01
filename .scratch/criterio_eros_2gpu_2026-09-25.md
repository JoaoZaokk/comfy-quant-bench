# Eros I2V em duas placas sem transbordo para a RAM — critério (25/09, escrito antes de medir)

Workflow: `ComfyUI/user/default/workflows/10Eros_v1.5_W4A8_I2V_DMD_2gpu.json` (cópia do `_split_tiled_loraadv`).
Mudanças: checkpoint via `CheckpointLoaderSimpleDisTorch2MultiGPU` (computa cuda:0, até 4 GB de pesos em cuda:1
em vez da CPU); `VRAM_Debug` com `unload_all_models` entre o `LTXVConditioning` e os samplers (tira o text encoder
FP8 da 3080 Ti antes da amostragem); ID-LoRA TalkVid adicionada desligada na pilha do Power Lora Loader.

Referência (24/09, `_split_tiled`, text encoder BF16): 2ª passada com 1,79 GB do LTXAV em offload para a RAM;
text encoder 1,54 GB em offload; tempo 16 min 45 s frio (1º após restart) e 13 min 25 s com modelos em memória.

Passa se, na mesma entrada (grafo `prompt_tiled_api.json`, seed fixa), primeira execução após restart:
1. LTXAV sem nenhuma parte na CPU nas duas passadas (log DisTorch/core: só cuda:0 e cuda:1).
2. Text encoder inteiro em cuda:1.
3. VAE de vídeo com folga na 3090 no decode em tiles (não 0 MB usable).
4. Sem OOM em nenhuma placa.
5. Tempo total ≤ 16 min 45 s × 1,10 (frio contra frio). Pior que isso = o transbordo em GPU custa mais que a RAM.
Conteúdo do vídeo/áudio não é aberto (regra NSFW); avaliação de qualidade é do dono.

## Correção 1 (25/09 ~19:30, antes da rodada v2)
DisTorch2 + `VRAM_Debug` abandonados: o DisTorch estimou o W4A8 em 39,11 GB (real 11,9) e pôs tudo em cuda:0; o
`unload_all_models` jogou ~20 GB para a RAM e a recarga levou 2,5 min; a 2ª tentativa caiu com access violation ao ler
a LoRA do P: com a RAM em 63,4/63,6 GB. Medição de base (`_loraadv` + TE FP8, frio): 1090 s; TE inteiro na 3080 Ti;
entre as passadas o LTXAV sai inteiro da 3090 (11,9 GB para a RAM) para o decode normal da 1ª passada; RAM pico 63,6 GB;
2ª passada com 1,79 GB em offload. v2 = base + decode da 1ª passada em tiles.
Passa se: nenhum "Unloaded ... 11921 MB" entre as passadas; pico de RAM na transição menor que 62,9 GB; tempo frio ≤ 1090 s.

## Resultado v2 (25/09 ~19:55, frio, uma rodada de cada — sem dispersão medida)
| | base (`_loraadv` + TE FP8) | v2 (+ decode 1ª passada em tiles) |
|---|---|---|
| tempo | 1090 s | 1034 s (−5%, dentro do que uma rodada não separa) |
| LTXAV entre as passadas | sai inteiro (11,9 GB para a RAM) | fica; só 1,81 GB saem na 2ª passada |
| RAM na transição | 62,9 GB | 49,4 GB |
| TE | inteiro na 3080 Ti (7,86 GB) | idem |
| RAM no fim (decode 2ª passada + VHS) | 60,8 GB | 63,6 GB (cheia), commit 201,6 GB |
Passa nos três critérios da correção 1. Não resolve o pico final: decodificar 361 quadros ~1024×1344 cria o vídeo em
float32 na CPU (~6 GB por buffer, o tiled usa mais de um) e a RAM já parte de ~29–42 GB ocupados por outros programas.
Áudio: DMD com audio/v2a em 0.5 (base e v2) ficou muito pior segundo o dono → sliders voltaram a 1.0.
