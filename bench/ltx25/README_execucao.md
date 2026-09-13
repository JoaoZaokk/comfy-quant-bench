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
Isso é indistinguível de "já terminou" e de "nunca começou". `tools/ltx_video.py` e
`tools/qwen_edit_test.py` agora leem as portas dos logs do pacote e olham nos dois lugares.

## O que NÃO está coberto

- Uma resolução (512), um prompt, uma semente, um sampler, um `frame_rate`.
- O **áudio não foi decodificado**. O latente de áudio entra porque o grafo concatena, mas só o
  ramo de vídeo é decodificado.
- Nada aqui compara qualidade entre formatos ainda — só estabelece que o original roda e
  quanto custa.
- O `s/quadro` inclui carga e staging de 40 GB. Não é custo marginal por quadro, e **não
  escala linearmente**: uma corrida de 249 quadros amortiza a carga sobre 5,08x mais trabalho.

---

## Duas armadilhas que custaram 36 minutos, registradas para não custarem de novo

### 1. O worker do MultiGPU ignora o `compute_device` do nó

Com `COMFYUI_MGPU_DISABLED` diferente de 1, o pacote sobe um worker por placa e distribui os
trabalhos **por rodízio**. A primeira corrida foi para o worker da GPU 0; a segunda foi para o
worker da **GPU 1** — a 3080 Ti de 12 GB — mesmo com o nó pedindo `compute_device: cuda:0`.

O resultado, lido do log do worker:

```
mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
torch.AcceleratorError: CUDA error: out of memory
```

39,13 GiB numa placa de 12 GB. E o mais caro: a fila continuou dizendo `rodando: 1` **depois**
do OOM, com a GPU 0 ociosa em 659 MiB — 36 minutos parecendo progresso.

**A alocação da DisTorch2 não protege contra isso**, porque ela decide onde os *blocos* moram
dentro do processo que já foi escolhido. Quem escolhe o processo é o rodízio, antes.

Correção: `COMFYUI_MGPU_DISABLED=1`. Os nós `*DisTorch2MultiGPU` continuam registrados e
funcionando — a variável controla o *spawn de workers*, não o registro dos nós.

### 2. Matar o servidor pelo arquivo de pid pode matar o processo errado

O `.pid` foi sobrescrito por um lançamento posterior, então `taskkill` matou um número que já
não era o servidor. O antigo continuou dono da porta 8190, e o teste de saúde respondeu
**`ONLINE após 5s`** — rápido demais para o ComfyUI, que leva ~60 s para subir. Essa velocidade
foi o sinal, e quase passou batido.

O que decide é quem **possui a porta**, não quem responde nela:

```powershell
Get-NetTCPConnection -LocalPort 8190 -State Listen | Select-Object OwningProcess
```

Deu `17964` quando o esperado era `34100`. Confirmar isso antes de acreditar que o servidor foi
reiniciado.
