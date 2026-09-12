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
- [x] Ladder: P3 e P6 confirmadas, 40/40 boas, teto NAO alcancado (0,1377 tolerado)
- [x] Edicao: 9 sementes x 3 bracos x 2 instrucoes. Desvio do cachorro FECHADO -- reproduz na
      semente 1002 SO no W4A4 (1 de 9 contra 0 de 9 nos outros dois) e some com numeral explicito
- [x] Despacho nos dois builds do teto: 224 modulos, 8/8, 0 dequantize, int4
- [x] `bench/krea2_suite.md` + `bench/hf/krea2-turbo-quant/README.md`, ambos em ingles
- [x] Encoder do Krea2 quantizado nos dois formatos e medido: W4A8 e a escolha
- [ ] Upload do Krea2 NAO feito: modelo gated com licenca propria, e o pedido dele era card

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


## Etapa 3 -- Z-Image: FECHADA em 2026-09-12

Os dois oficiais convertidos, medidos (72 renderizacoes, nenhuma quebrada) e PUBLICADOS, com os
8 arquivos conferidos byte a byte contra o Hub. Achado: a receita mista vence 11/12 no Turbo
oficial e 7/12 no De-Turbo -- nao transfere entre um modelo e o fine-tune dele.

**Estado anterior, que motivou tudo:**

    checkpoint                        BF16 no disco  analise  w4a4  misto  publicado
    beyond-reality-zimage-v2          sim            0,1241   sim   sim    sim
    z_image_turbo_bf16 (oficial)      sim            0,1257   NAO   NAO    NAO
    z_image_de_turbo_v1_bf16          sim            0,1213   NAO   NAO    NAO

Os dois oficiais **ja estao na nomenclatura do ComfyUI** (453 tensores, `attention.qkv`), entao
NAO precisam de `to_native`, e as analises ja existem (`calib/xfer_*.analysis.json`), entao nao
precisam de recalibragem. 4 conversoes de ~90 s.

Tambem sem arquivo, so sidecar: `zimage-v2-teto-cg16/64/256`, `zimage-v2-mixed-t0.05/0.10/0.20`,
`zimage-v2-sigma-*`. Esses foram experimentos, nao entregaveis.

## Etapa 4 -- Qwen Image Edit 2511  [EM ANDAMENTO]

- [x] BF16 (40.861.031.560 B) e int8_convrot (20.499.083.824 B) baixados e conferidos contra o Hub
- [x] Perfil `qwen_image` derivado do checkpoint do Comfy-Org: 840/840, zero falso positivo,
      zero falso negativo, conferido tambem contra `named_modules()` do modelo carregado
- [ ] Guarda de referencia no BF16 RODANDO -- risco real: modelo de EDICAO sem imagem de entrada
      pode gerar lixo, como o Wan VACE fez. Sem esse controle a calibragem seria sobre ativacoes
      nao representativas.
- [ ] Calibrar, converter w4a4 + misto, ladder, comparar com int8 publico e com os SVDQuant
- [ ] Encoder `qwen_2.5_vl_7b` ja publicado, mas ANTES do cadeado sair -- remedir

### Referencia

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

## Etapa 5 -- LTX 2.5: inventario conferido no disco 2026-09-12

Nada precisa ser baixado. Tudo em `D:` (rede) menos os nossos, que estao em `F:`.

    ltx-2.5-22b-distilled-transformer-bf16        42.018.190.584 B   FONTE destilada
    ltx-2.5-22b-dev-transformer-bf16              42.018.190.584 B   FONTE dev (nao destilada)
    ...-distilled-...-comfy-int8-convrot          21.504.034.224 B   int8 do Comfy-Org
    ...-distilled-...bf16_w4a8                    12.520.267.816 B   NOSSO
    ...-distilled-...nvfp4                        18.721.548.408 B   terceiros
    LTX25-distilled-DiT-comfy-w4a4                11.236.345.048 B   riftcast, 1440 camadas, 4 bits reais
    LTX25-distilled-DiT-comfy-mix4x8              13.810.250.240 B   terceiros

    encoder gemma4-12b-with-proj-ltx-2.5-bf16     26.263.860.594 B   FONTE
    encoder ...-comfy-int8-convrot                15.372.971.786 B   Comfy-Org
    encoder gemma4-12b-ltx25-comfy-w4a8           10.604.342.914 B   (em F:)

    vae ltx-2.5-video-vae-bf16                     1.472.223.346 B
    vae ltx-2.5-audio-vae-bf16                       364.866.540 B

Workflow de referencia pronto: `LTX25-int8-acceptance-v2.json` -- cadeia A/V completa
(`EmptyLTXVLatentVideo` + `LTXVEmptyLatentAudio` + `LTXVConcatAVLatent` + `SamplerCustomAdvanced`
+ `LTXVSeparateAVLatent`). Hoje em 512x512 x 49 quadros.

**O video de 10 s pedido:** LTX quer contagem de quadros 8n+1. A 25 fps, 10 s = 249 quadros
(8x31+1). Isso e 5,08x o latente do workflow atual -- e o ponto de risco da etapa, nao a
conversao.

**ARMADILHA JA REGISTRADA:** LTX 2.5 exige `--disable-dynamic-vram`. Sem a flag, morre com
`aimdo: hostbuf_read_file_slice: device copy failed` depois de tentar montar 35 GB num cartao
de 24, e a mensagem NAO nomeia o subsistema culpado.
