# O que resta de modelo de IMAGEM no disco — triagem de 2026-09-13

O trilho do dono diz: **todos os modelos de imagem no disco primeiro, depois os de vídeo**. Este
arquivo é a prestação de contas dessa metade — o que foi feito, o que falta, e **o que está fora
de escopo por arquitetura em vez de por falta de trabalho**, que é uma distinção que sem isto
escrito vira "não fez".

Contados os `.safetensors` acima de 3 GiB em `diffusion_models`, `checkpoints` e `unet`,
descartando o que já passou por esta bancada: **11 candidatos**. Nem todos são de imagem e nem
todos são conversíveis.

## Fora de escopo porque não são imagem

| arquivo | GiB | o que é |
|---|---|---|
| `ace_step_1.5_turbo_aio` | 9,34 | geração de **áudio/música** |
| `stable_audio_3_medium` | 8,59 | geração de **áudio** |
| `svd_xt` | 8,90 | Stable Video Diffusion — **vídeo**, entra na segunda metade do trilho |

## Fora de escopo por ARQUITETURA, e isto é medido

| arquivo | GiB | 2-D (÷256) | conv 4-D |
|---|---|---|---|
| `sd_xl_base_1.0` | 6,46 | 949 (867) | **123** |
| `Juggernaut-XL_v9` | 6,62 | — | UNet, mesma família |

O ConvRot quantiza `Linear`. O SDXL é **UNet**: seu caminho de difusão é dominado por `Conv2d`, e
os 123 tensores de 4 dimensões são exatamente isso. As 949 matrizes 2-D que o header mostra são em
boa parte do **text encoder embutido** (`conditioner.embedders.*`), não do difusor.

Quantizar SDXL neste formato converteria a parte errada do modelo: encolheria o encoder e deixaria
a convolução — onde o tempo é gasto — intocada. **Não é "não deu tempo", é que a ferramenta não
serve para esta arquitetura**, e forçar produziria um arquivo menor que não roda mais rápido e
perde qualidade à toa.

## Já quantizado por terceiro, não é fonte

| arquivo | GiB | nota |
|---|---|---|
| `hidream_o1_image_dev_fp8_scaled` | 7,51 | já fp8; a fonte BF16 está na lista abaixo |
| `flux-2-klein-base-4b-fp8` | 3,81 | já fp8, **e já medido aqui**: 0 forwards quantizados, 8 `dequantize`, `dequantize_per_tensor_fp8`. Economiza memória, não tempo |

## O que de fato falta converter — três

| arquivo | GiB | 2-D (÷256) | conv | forma dos nomes |
|---|---|---|---|---|
| `hidream_o1_image_bf16` | 15,24 | 375 (266) | **0** | `model.final_layer2.linear`, `model.language_model.embed_tokens` |
| `Flux_Realistic_NSFW_02` | 11,08 | 314 (313) | **0** | `double_blocks.N.img_attn.qkv`, `.img_attn.proj` |
| `void_pass1` / `void_pass2` | 10,38 cada | 342 (341) | **0** | `blocks.N.attn_out`, `blocks.N.ff_out` |

Todos DiT puros, zero convolução, e com quase todas as matrizes 2-D divisíveis por 256 — que é o
filtro que o ConvRot exige.

**O Flux é o mais valioso dos três**: é arquitetura pública grande, e a forma dos nomes
(`double_blocks.N.img_attn.qkv`) é próxima do perfil `hunyuan_video_15` que já existe aqui, então
o perfil sai de uma derivação conferível em vez de um chute. O `hidream` carrega um
`language_model` embutido, o que exige decidir antes se ele entra ou fica de fora — decisão que
esta bancada já tomou em outros perfis (o encoder fica fora do perfil do difusor).

## O que este arquivo NÃO cobre

- Só olhou `.safetensors` acima de 3 GiB. GGUF e arquivos menores não entraram.
- A contagem de 2-D vem do **header**, não de um modelo carregado. O perfil do
  `hunyuan_video_15` já ensinou nesta bancada que nomes de chave de arquivo podem não casar com
  `named_modules()` — a derivação de qualquer perfil novo tem de ser conferida **dos dois lados**.
- Nenhum dos três foi carregado, calibrado ou renderizado. Isto é triagem, não medição.
