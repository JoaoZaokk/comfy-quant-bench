# Plano do goal de 2026-09-12 -- estado vivo, atualizado a cada etapa

Cinco etapas encadeadas. Este arquivo existe para o trabalho sobreviver a uma troca de contexto.
Cada linha diz o que FOI MEDIDO, nao o que se espera medir.

## Etapa 1 -- Krea2: teto, edicao, documentacao, cards em ingles

- [x] Criterio escrito ANTES de medir: `bench/criterio_teto_krea2.md`, 6 previsoes
- [x] `quant_group_size` investigado e **FECHADO como eixo**: `comfy/ops.py:1201` fixa 64 no
      codigo enquanto le `convrot_groupsize` e `linear_dtype` do JSON da camada. Escrever outro
      valor produz arquivo que o loader le errado.
- [x] cg 64 e cg 16 convertidos, `--somente-w4a4 --uncalibrated fail`, 224/224 calibradas
- [x] Erro por camada na intersecao (224 nas tres): 256=0,1199  64=0,1238  16=0,1377
      P1 (direcao) confirmada, 215/224 camadas monotonicas. P4 confirmada.
      **P2 refutada**: previ 1,155x/1,468x (razoes do Z-Image); medi 1,033x/1,149x.
      **P5 refutada na letra, confirmada no mecanismo**: payload identico (8.057.415.984 B nos
      dois), delta de 224 B inteiro no JSON -- um caractere por camada.
- [ ] Ladder de 4 bracos rodando -> testa P3 e P6
- [ ] Edicao: 8 sementes x 3 bracos (`.scratch/edit_8sementes.py`), fecha o desvio do cachorro
      E traz o braco `misto` para a edicao pela primeira vez
- [ ] `avaliar_despacho` nos dois builds novos
- [ ] `bench/krea2_suite.md` (ingles) + cards em ingles + upload

## Etapa 2 -- pipeline COMPLETO, nao um modelo solto

**Esclarecido pelo dono em 2026-09-12:** "switch" era exemplo. O alvo e o **no completo** -- para
cada projeto, tudo que a cadeia usa (difusao, checkpoint, text encoder, VAE, o que houver),
quantizar tudo e achar o ponto onde o custo de qualidade ainda paga o ganho de tamanho e
velocidade.

Isso muda o eixo do trabalho: deixa de ser "converter o modelo X" e passa a ser "fechar a cadeia
do projeto X". O elo que a bancada nunca tocou e o **VAE** (nenhum dos 11 no disco esta
quantizado), e o elo que virou o MAIOR arquivo depois da quantizacao e o **text encoder**.

    projeto          difusao (nosso)              text encoder            VAE
    Krea 2 Turbo     7,50 GiB w4a4 / 7,70 misto   8,27 GiB BF16 CRU       242 MiB cru
    Z-Image v2       3,06 GiB w4a4 / 3,18 misto   7,49 GiB BF16 cru       242 MiB cru
                                                  (w4a4 publicado, apagado do disco)
    LTX 2.5          11,66 GiB w4a8 (em D:)       9,88 GiB w4a8 (Comfy)   1,35 GiB cru
    Qwen Image Edit  NADA AINDA                   6,33 GiB w4a4 (nosso)   242 MiB cru

**No Krea2 e no Z-Image o encoder ja e maior que o modelo de difusao quantizado.** Essa e a
resposta direta a pergunta do "ponto perfeito": depois de 3,26x no DiT, o proximo ganho de tamanho
esta no encoder, nao em apertar mais o DiT -- que, medido, nem quebra no menor groupsize legal.


## Etapa 3 -- Z-Image: esta tudo convertido?

**Resposta ja medida: NAO.**

    checkpoint                        BF16 no disco  analise  w4a4  misto  publicado
    beyond-reality-zimage-v2          sim            0,1241   sim   sim    sim
    z_image_turbo_bf16 (oficial)      sim            0,1257   NAO   NAO    NAO
    z_image_de_turbo_v1_bf16          sim            0,1213   NAO   NAO    NAO

Os dois oficiais **ja estao na nomenclatura do ComfyUI** (453 tensores, `attention.qkv`), entao
NAO precisam de `to_native`, e as analises ja existem (`calib/xfer_*.analysis.json`), entao nao
precisam de recalibragem. 4 conversoes de ~90 s.

Tambem sem arquivo, so sidecar: `zimage-v2-teto-cg16/64/256`, `zimage-v2-mixed-t0.05/0.10/0.20`,
`zimage-v2-sigma-*`. Esses foram experimentos, nao entregaveis.

## Etapa 4 -- Qwen Image / Qwen Image Edit

**Versao mais atual confirmada pela API do HF (fonte primaria, nao blog):**

    Qwen/Qwen-Image-Edit-2511      17 dez 2025   <- mais recente da familia Edit
    Qwen/Qwen-Image-2512           30 dez 2025   <- mais recente da familia base
    Qwen/Qwen-Image-Layered        17 dez 2025

Nada mais novo na org `Qwen`. Formato ComfyUI de arquivo unico em
`Comfy-Org/Qwen-Image-Edit_ComfyUI`:

    qwen_image_edit_2511_bf16.safetensors           38,05 GiB   <- fonte
    qwen_image_edit_2511_int8_convrot.safetensors   19,09 GiB   <- braco de referencia publico
    qwen_image_edit_2511_fp8mixed.safetensors       19,12 GiB

Ja no disco (terceiros, SVDQuant/nunchaku): `nunchaku_qwen_image_edit_2511_best_quality_int4`,
`svdq-int4_r128-qwen-image-edit-2511`, `svdq-int4_r32-qwen-image-edit-2511-lightning`,
`svdq-int4-qwen-image-2512-balance`, `qwen-image-Q8_0.gguf`.
Encoder `qwen_2.5_vl_7b` ja convertido e publicado (`JoaoZaokk/Qwen2.5-VL-7B-W4A4-ConvRot`).

Disco: F: tem 161 GB livres. 38+19 = 57 GB de download, mais ~22 GB de saidas.

## Etapa 5 -- LTX / video

Ja no disco: `ltx-2.5-22b-distilled-transformer-bf16` (39,1 GiB, em D:), variantes int8/w4a8/
nvfp4/int8-convrot, `LTX25-distilled-DiT-comfy-w4a4` (riftcast, 1440 camadas, 4 bits de verdade).
Pede `--disable-dynamic-vram` (medido: 35 GB staged num cartao de 24).

## Regras que valem em todas as etapas

- So `.\python_embeded\python.exe -s`. Nunca Python global.
- Nunca apagar/sobrescrever original.
- NAO tomar `Assert-GpuLock` antes de ferramenta que entra no `BenchGuard` sozinha
  (`quality_ladder`, tudo que passa por `_timing.compare()`). Tomar por conversor e sonda.
- `CUDA_VISIBLE_DEVICES=0` nos benchmarks: o cortex ocupa ~2,3 GiB da 3080 Ti e o teto do guarda
  e 2,00 GiB.
- Matar ComfyUI pela ARVORE (`.scratch/derruba_comfy.ps1`), nunca pelo pid.
- Cards e README em ingles.
