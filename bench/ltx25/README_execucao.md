# LTX 2.5, o que já rodou nesta máquina — registro de execução

Atualizado 2026-09-12. Tudo aqui é **executado**, não traçado.

## O braço original funciona, e precisa das duas placas

`ltx-2.5-22b-distilled-transformer-bf16.safetensors`, 42.018.190.584 B (39,13 GiB),
com o encoder `gemma4-12b-with-proj-ltx-2.5-bf16` (26.263.858.182 B) e o
`ltx-2.5-video-vae-bf16`. Ambos byte a byte iguais aos do `Lightricks/LTX-2.5` no Hub.

**Primeira tentativa, sem DisTorch2: `CUDA error: out of memory`**, levantado de dentro de
`torch.cuda.mem_get_info` chamado por `model_management.free_memory` → `unload_all_models`.
Não foi uma alocação infeliz: a placa esgotou. 39,13 GiB contra 24 GiB não cabe nem com o
offload normal do ComfyUI.

**Com DisTorch2 espalhando blocos, roda.** Alocação aceita e confirmada no log do próprio
pacote:

```
[MultiGPU DisTorch V2] Full allocation string: cuda:1,6.0gb;cpu,*#cuda:0;4.0;cpu
Model LTXAV prepared for dynamic VRAM loading. 40048MB Staged.
```

São **6 GiB** doados pela 3080 Ti, não 12: o embedding do cortex mora nessa placa com
~2,3 GiB medidos e não pode ser expulso.

## Medição

| corrida | quadros | resolução | passos | tempo | s/quadro |
|---|---|---|---|---|---|
| fumaça BF16 | 49 (1,96 s) | 512×512 | 3 | **717,8 s** | 14,65 |

Tempo lido do `/history` do worker (`execution_success` − `execution_start`), não cronometrado
por fora. Status `success`, 49 PNGs.

`3` passos porque é o destilado: sigmas manuais `0.909375, 0.725, 0.421875, 0.0`, cfg 1.0 —
o regime do workflow de aceitação que já rodava aqui, não um regime inventado.

![tira do braço BF16](tira_smoke_bf16_49q.png)

Quadros 1, 17, 33 e 49. Farol, ondas quebrando, gaivotas — coerente e com movimento real.

## A armadilha que custou uma corrida

Com `COMFYUI_MGPU_DISABLED` diferente de 1, o ComfyUI-MultiGPU sobe **um worker por placa** em
portas escolhidas em tempo de execução e encaminha o trabalho. O resultado aparece no
`/history` **do worker**; a fila do servidor principal fica vazia enquanto a GPU está a 85%.
Isso é indistinguível de "já terminou" e de "nunca começou". `tools/ltx25_video.py` e
`tools/qwen_edit_test.py` agora leem as portas dos logs do pacote e olham nos dois lugares.

## O que NÃO está coberto

- Uma resolução (512), um prompt, uma semente, um sampler, um `frame_rate`.
- O **áudio não foi decodificado**. O latente de áudio entra porque o grafo concatena, mas só o
  ramo de vídeo é decodificado.
- Nada aqui compara qualidade entre formatos ainda — só estabelece que o original roda e
  quanto custa.
- O `s/quadro` inclui carga e staging de 40 GB. Não é custo marginal por quadro, e **não
  escala linearmente**: uma corrida de 249 quadros amortiza a carga sobre 5,08x mais trabalho.
