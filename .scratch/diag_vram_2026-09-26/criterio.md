# Offload da 2ª passada do 10Eros na 3090 — critério (escrito antes de medir, 26/09/2026)

Pedido do dono (26/09): medir o que ocupa a 3090 no início da 2ª passada e testar a 3090 só com o transformer.

## Contexto (log de 25/09, `.scratch/comfy_8190_audioint8.log`)
- misto audioint8 (14.273 MB): 2ª passada deixou 10.138 MB na placa, 3.897 MB foram para a RAM, 93 s/passo.
- W4A8 logo depois, no mesmo servidor: 2ª passada com 0 MB na placa, 120 s/passo. A/B não limpo.
- O decode tiled não é o culpado: o VAE entra com 1,4 GB e tirou 0,86 GB do transformer.
- Suspeita lida no código (não medida): `LTXVLatentUpsamplerTiled` (10S-Comfy-nodes) chama
  `free_memory(module + tile_volume*3000 + saída*4, cuda:0)`. Com tile 11 e 46 quadros latentes, isso dá
  ~8,5 GB e expulsa o transformer antes da 2ª passada.

## Montagem
- Grafo: o da rodada audioint8 de 25/09 (mesma imagem, prompt, seed 635141064074927, 1024×1376, 361 quadros,
  DMD r256 a 1.0, 9 + 3 passos). Só muda o prefixo de saída.
- `base`: VAE de vídeo (do checkpoint) e VAE de áudio na cuda:0.
- `vae1`: VAE de vídeo avulso `LTX23_video_vae_bf16` (170/170 tensores byte a byte iguais ao `vae.*` do checkpoint,
  conferido por `compara_vae.py`) via `VAELoaderMultiGPU` na cuda:1; VAE de áudio via `LTXV2AudioVAELoaderMultiGPU`
  na cuda:1. O upscaler latente segue na cuda:0 (o nó usa `get_torch_device()` e devolve o modelo para a CPU).
- Servidor reiniciado antes de cada variante (frio), lançado por `lanca_comfy_diag.py`, que grava cada
  `load_models_gpu`/`free_memory` com a VRAM por placa e os modelos carregados. Mesmos argumentos de 25/09:
  `--windows-standalone-build --use-sage-attention --disable-dynamic-vram`, `COMFYUI_MGPU_DISABLED=1`.
- Não abro vídeo nem áudio. Só números do log.

## Métricas
1. No `load_antes` do LTXAV da 2ª passada: livre/alocado na cuda:0 e quem está carregado.
2. Linha "loaded completely/partially" e MB em offload do LTXAV na 2ª passada.
3. s/passo da 2ª passada e tempo total (secundário: 1 amostra, primeira rodada lendo do compartilhamento P:).
4. Pico de RAM/commit (amostrado a cada 5 s) — a variante tira o text encoder da 3080 Ti para a RAM.

## Aceitação
- Diagnóstico: identificar, com número, o que ocupa a cuda:0 quando o LTXAV da 2ª passada pede carga.
- `vae1` só é melhor se a 2ª passada imprimir "loaded completely" (ou offload claramente menor) sem subir o pico
  de RAM a ponto de encostar no limite (commit livre < 4 GB = falha).
- Se o culpado for o `free_memory` do upscaler, `vae1` não resolve sozinho; registrar e propor a correção ao
  dono (é nó de terceiro, não edito sem autorização).

## Adendo antes de medir `chunk4` (26/09, depois da base)
Base medida: no pedido do LTXAV da 2ª passada o ComfyUI estimou `memory_required` 24.522 MiB e mínimo 12.261 MiB
(fator 0,077 do LTXAV). O modelo recebe livre − (mínimo + 700 de reserva) = 10.138 MiB; 4.146 MiB ficam na RAM.
O pico real de ativações na 2ª passada foi ~8,7 GB (18.858 de pico − 10.154 residentes). O upscaler não foi o
culpado (tirou 248 MiB do LTXAV). A suspeita do adendo anterior está refutada.
- `chunk4` = base + `ModelMemoryUsageFactorOverride` 0,059 (mínimo estimado ~9,4 GB) + `LTXVChunkFeedForward`
  4 fatias (só `transformer_blocks.*.ff`, vídeo; a ativação do W4A8 é quantizada por linha, então fatiar tokens
  é exato).
- Espera: offload ~1,3 GB (limitado pela estimativa 0,059), e pico real menor que 8,7 GB. Com o pico medido,
  calculo o fator da rodada `final`, que deve carregar os 14.284 MiB inteiros, com margem ≥ 0,5 GB no pico.
- Falha: OOM ou pico que não caia.

## Adendo antes de medir `final` (26/09)
- `vae1` (medido): 2ª passada igual à base (10.156 MiB na placa; mesma estimativa). Decode final na 3080 Ti com
  ~1,7 GB livres (text encoder 4,7 GB + área de trabalho) passou de 39 min sem terminar (base: 165 s); abortado.
  Rejeitada.
- `chunk4` (medido): estimativa mínima 9.395 MiB; 13.041 MiB na placa; pico de ativações ~6,5 GB
  (19.522 − 13.057); 89 s/passo. O decode final não terminou: meu amostrador caiu com WinError 1455 (commit do
  Windows no limite: RAM 63,6 GiB) e o driver derrubou o ComfyUI. Amostrador corrigido para não cair.
- `final` = fator 0,046 (mínimo ≈ 7,3 GB) + chunk 4. Disponível na 2ª passada ≈ 22,6 GB; 14.284 + 6,5 GB = 20,8 GB.
  Aceita se imprimir "loaded completely" na 2ª passada sem OOM; s/passo comparado com base e chunk4.

## Resultado `final` (medido)
Estimativa mínima 7.325 MiB; 2ª passada "loaded completely; 14273 MB"; pico 20.765 MiB (598 MiB livres);
88,6 s/passo; prompt 17 min 23 s. 1ª passada base vs final/chunk4: PSNR infinito (idênticas). 2ª passada base vs
final: PSNR médio 44,2 dB, mín 38,5. RAM: pico 63,6 GiB, commit 193 GiB no decode final (problema à parte).
**Aceito** pelo critério. Workflow: `ComfyUI/user/default/workflows/10Eros_v1.5_W4A8audioINT8_I2V_DMD_2gpu_semoffload.json`.
