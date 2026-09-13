# O que resta de modelo de IMAGEM no disco — triagem de 2026-09-13

O trilho do dono diz: **todos os modelos de imagem no disco primeiro, depois os de vídeo**. Este
arquivo é a prestação de contas dessa metade — o que foi feito, o que falta, e **o que está fora
de escopo por arquitetura em vez de por falta de trabalho**, que é uma distinção que sem isto
escrito vira "não fez".

Contados os `.safetensors` acima de 3 GiB em `diffusion_models`, `checkpoints` e `unet`,
descartando o que já passou por esta bancada: **11 candidatos**. Nem todos são de imagem e nem
todos são conversíveis.

## O "switch" do goal NÃO é um modelo, e isso foi conferido

O goal original manda "converter todo o **switch**". Não existe nenhum checkpoint com esse nome
nesta máquina — procurado em `ComfyUI/models` e no compartilhamento `D:\ComfyUI-Models` com
`find -iname "*switch*" -size +100M`, resultado **vazio**.

E o dono esclareceu na terceira mensagem da conversa que era exemplo:

> "switch = node completo, eu dei como exemplo ainda, todo o projeto, se ele usa diffuser,
> checkpoint, text encoder e etc, quantizar tudo"

Então o alvo é a **cadeia completa por projeto** — difusor, checkpoint, text encoder — e não um
modelo chamado Switch. Registrado aqui porque a palavra reaparece em toda leitura do goal e sem
isto escrito ela conta como etapa não feita, quando na verdade é uma etapa que não existe.

O VAE fica de fora por instrução explícita do dono: *"VAE nao quantiza, nao precisa, só pra deixar
claro!"*

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

## O que de fato falta converter — UM, e a primeira versão desta seção estava errada

> **CORREÇÃO, 2026-09-13, poucas horas depois de escrever a seção original.** Ela listava três
> candidatos com base em **contagem de matrizes 2-D e divisibilidade por 256**, e **não olhou o
> dtype**. Dois dos três não são fonte. O erro custou uma calibração inteira de GPU no Flux antes
> do conversor recusar — corretamente — com `Profile 'flux_1' selected no compatible layers`,
> porque `HIGH_PRECISION_DTYPES` exclui fp8. **A guarda funcionou; a triagem é que estava errada.**

| arquivo | GiB | dtype real | veredito |
|---|---|---|---|
| `Flux_Realistic_NSFW_02` | 11,08 | **780 tensores, TODOS `F8_E4M3`** | **NÃO é fonte** — já é fp8 |
| `hidream_o1_image_bf16` | 15,24 | 758 BF16 | **não é DiT** — ver abaixo |
| `void_pass1` / `void_pass2` | 10,38 cada | 1024 BF16 | **fonte válida, DiT limpo** |

**O Flux é fp8.** 11,08 GiB para um modelo de 12 B já dizia isso, e eu não somei. Quantizar fp8
em 4 bits mediria degradação sobre degradação, nunca contra o original — e o erro por camada
sairia contra uma referência que já está errada.

**O HiDream o1 não é um transformer de difusão.** Dos 375 módulos 2-D:

```
253  model.language_model.layers.N.*   <- um LLM inteiro (q/k/v/o_proj, mlp gate/up/down)
108  model.visual.blocks.N.*           <- torre de visao
 14  o resto (final_layer2, t_embedder1, patch embeds)
```

É um modelo de **visão-linguagem com cabeça de difusão**. A parte de difusão são ~14 módulos.
Aplicar um perfil de difusor aqui converteria o LLM embutido — que é exatamente o erro que esta
bancada documenta para o SDXL, só que ao contrário. Se ele for quantizado um dia, é um trabalho em
estilo **text encoder**, com a advertência das duas travas do ComfyUI junto.

**Sobra o `void`**, e ele é DiT limpo: 42 blocos × 8 famílias = **336 de 342** módulos 2-D, todas
as colunas divisíveis por 256, e as 6 de fora são as pontas de sempre (`patch_embed`, `proj_out`,
`time_embedding_*`, `norm_out`).

```
42  blocks.N.{q, k, v, attn_out}        [3072, 3072]
42  blocks.N.ff_proj                    [12288, 3072]
42  blocks.N.ff_out                     [3072, 12288]
42  blocks.N.{norm1, norm2}.linear      [18432, 512]   <- modulacao
```

O que o segura não é o perfil: é que `void_pass1` e `void_pass2` são **duas passadas** de um
pipeline próprio do dono, então medir qualidade exige o workflow das duas etapas, e comparar só a
primeira passada mediria outra coisa. **Não convertido ainda, e a razão é essa.**

## O que este arquivo NÃO cobre

- Só olhou `.safetensors` acima de 3 GiB. GGUF e arquivos menores não entraram.
- A contagem de 2-D vem do **header**, não de um modelo carregado. O perfil do
  `hunyuan_video_15` já ensinou nesta bancada que nomes de chave de arquivo podem não casar com
  `named_modules()` — a derivação de qualquer perfil novo tem de ser conferida **dos dois lados**.
- Nenhum dos três foi carregado, calibrado ou renderizado. Isto é triagem, não medição.
