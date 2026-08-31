# Resumo — modelos testados, instrumentos criados, e o que mudou do original

2026-08-31. Companheiro de [HANDOFF_2026-08-31_tarde.md](HANDOFF_2026-08-31_tarde.md).

**Uma correção de vocabulário antes de tudo:** não criei *workflows* do ComfyUI. Criei
**instrumentos de linha de comando** em `tools/`, que carregam o ComfyUI como biblioteca. Os
workflows `.json` que existem na bancada são antigos e não foram tocados hoje. Está tudo listado
abaixo pelo que é.

---

## 1. Os modelos

Todos conferidos lendo o header do arquivo, camada a camada, não o campo de resumo.

### Nossos, feitos nesta bancada

| arquivo | tamanho | formato | contra o original |
|---|---|---|---|
| `zimage-v2-w4a4` | 3,06 GiB | `convrot_w4a4` x170 | **3,75x mais leve** que os 11,46 GiB do BF16 |
| `zimage-v2-sigma-none` | 3,17 GiB | 117 w4a4 + 53 w4a8 | 3,62x mais leve |
| `zimage-v2-sigma-none56` | 3,18 GiB | 114 w4a4 + 56 w4a8 | 3,60x mais leve |
| `zimage-v2-sigma-sigma2` | 3,18 GiB | 114 w4a4 + 56 w4a8 | 3,60x mais leve |
| `zimage-v2-sigma-high` | 3,18 GiB | 114 w4a4 + 56 w4a8 | 3,60x mais leve |
| `qwen_3_4b_w4a4_convrot` | 2,42 GiB | `convrot_w4a4` x252 | **3,10x mais leve** que os 7,49 GiB do BF16 |
| `gemma_3_12B_it_heretic_w4a8` | 7,53 GiB | `asym_w4a8_int8` x336 | **2,91x mais leve** que os 21,93 GiB da fonte |

Os quatro `sigma-*` existem só para responder uma pergunta (o critério ponderado por sigma vale a
pena?). A resposta é não — ver o handoff. **Eles ficam**, e passaram pelo portão de aceitação
(seção 3).

### Públicos, baixados e auditados

| arquivo | tamanho | formato real | veredito |
|---|---|---|---|
| `LTX25-distilled-DiT-comfy-w4a4` (riftcast) | 10,46 GiB | 1440 `convrot_w4a4`, `linear_dtype` ausente | **roda 4 bits de verdade** |
| `qwen3vl_32b_minimax_h3-int4_convrot` (Winnougan) | 13,20 GiB | 350 `convrot_w4a4` + 1 int8, `linear_dtype` ausente | kernel roda 4 bits; **mas é text encoder, então o ComfyUI não deixa** |
| `minimax_h3_fl2va_pruned-w4a8_convrot_pruned` (Winnougan) | 11,68 GiB | 200 `asym_w4a8_int8` | roda quantizado |
| `MiniMax_H3_FL2VA_..._int4_int8_convrot` (Abiray, 791k dl) | 14,81 GiB | 117 `convrot_w4a4` com `linear_dtype: "int8"` + 83 int8 | roda quantizado, **ramo INT8, nunca int4** |
| `MiniMax_H3_Ref2VA_...` (Abiray, 1,09M dl) | 14,06 GiB | mesma estrutura | idem |

**Duas armadilhas nos públicos, as duas confirmadas por execução:**

1. Os dois da Abiray têm **bytes depois do último tensor** (83 e 64). `safetensors.safe_open`
   recusa com `incomplete metadata, file not fully covered`. Não é download quebrado — o
   `Content-Length` do servidor bate byte a byte e um `Range: bytes=-83` devolve a mesma cauda.
   Só o leitor do dynamic-VRAM aceita. **Com `--disable-dynamic-vram` eles não carregam.**
2. O resumo do arquivo da Abiray se contradiz: `"linear_dtype": "int4"` no topo,
   `"w4a4_int4mm_layers": 0` três chaves abaixo, e as 117 camadas dizendo `"int8"`.
   **Ler as camadas, nunca o resumo.**

---

## 2. Os instrumentos criados hoje

Cada um imprime, em toda execução, o que **não** cobriu.

| arquivo | o que responde |
|---|---|
| `tools/probe_winnougan_int4.py` | os bytes deste checkpoint emitem MMA de 4 bits? (A/B da flag, pesos reais lidos por faixa de bytes) |
| `tools/probe_winnougan_load.py` | o ComfyUI carrega e codifica pelo caminho de estoque? |
| `tools/probe_te_fullprecision_mm.py` | **conta** as chamadas ao kernel durante uma codificação real, e instrumenta cada termo de `_use_quantized` |
| `tools/probe_quant_dispatch.py` | qual implementação o registry escolheu, em que device, com qual `linear_dtype` — modo `--forward-only` independe da arquitetura |
| `tools/probe_te_lock_cost.py` | quanto custa e quanto rende destravar um text encoder: 3 braços (BF16 / travado / destravado), tempo por prompt |
| `tools/probe_epsilon_ckpt_ab.py` | N checkpoints contra o MESMO BF16, passo a passo, **entradas casadas** (a trajetória do BF16 é imposta) |
| `tools/test_quant_mixed_sigma.py` | 15 checagens da aritmética da ponderação, sem GPU |

E dois modificados:

- `tools/calibrate_activations.py` — grava agora `sample_sigma`, um sigma por linha amostrada, nos
  mesmos slots do reservoir. Sem isso a ponderação por sigma não pode nem ser perguntada, porque o
  reservoir mistura os passos por construção.
- `tools/quant_mixed.py` — `--sigma-weight none|sigma|sigma2|high`, default `none` para não
  reinterpretar análise já gravada. Recusa reusar análise medida em outro modo.
- `tools/quality_ladder.py` — consertado: sob o ComfyUI 0.33 ele **caía sem traceback nenhum**,
  deixando o lock preso, porque o `CoreModelPatcher` aloca um HostBuffer do comfy-aimdo que um
  script sem `comfy_aimdo.control.init()` não tem. Agora passa `disable_dynamic=True`.

---

## 3. Portão de aceitação — imagens de verdade

Rodado com `tools/quality_ladder.py`: mesmo prompt, mesmas sementes, referência BF16, imagens
decodificadas por `ae.safetensors`.

Prompt único, 2 sementes, 8 passos, 1024 px, sampler `euler`/`simple`, cfg 1.0, na RTX 3090.

```
checkpoint                          divergência  espalh.  s/passo    GiB   runs
beyond-reality-zimage-v2_native (BF16)        -        -    1,128  11,46      2
zimage-v2-w4a4                           0,7173   0,0664    0,594   3,06      2
zimage-v2-sigma-none56                   0,6584   0,0575    0,584   3,18      2
zimage-v2-sigma-sigma2                   0,7156   0,0627    0,616   3,18      2
zimage-v2-sigma-high                     0,7406   0,1671    0,586   3,18      2
```

**Os dois números que interessam:**

- **s/passo: 1,128 → 0,584-0,616.** Os quantizados são **1,83x a 1,93x mais rápidos** que o BF16
  por passo de amostragem. Esse é o segundo-por-iteração que faltava.
- **Disco: 11,46 GiB → 3,06-3,18 GiB.** 3,60x a 3,75x mais leve.

E o que **não** dá para concluir da tabela: qual dos quatro quantizados é melhor. O espalhamento
entre duas sementes (até 0,167) é da ordem das diferenças entre eles (0,658 a 0,741), e a própria
ferramenta avisa: *"Split decisions, i.e. not separated at 2 run(s)"*. Bate com as oito sementes do
epsilon casado, que deram os três critérios indistinguíveis.

**Divergência de latente é distância, não qualidade.** Um checkpoint pode ficar mais perto do BF16
e produzir imagem que alguém prefira menos. As imagens estão em `bench/aceitacao_sigma/` para olho
humano, e comparadas lado a lado no artifact.

### O que este portão cobriu, e o que não

Cobriu: carrega pelo loader real, amostra de ponta a ponta, decodifica imagem, mede s/passo e
divergência pareada por semente. **Não cobriu:** métrica perceptual, mais de um prompt, mais de
duas sementes, e nenhum teste de vídeo.

### Um bug real da ferramenta, consertado no caminho

`quality_ladder.py` foi escrita quando a bancada rodava ComfyUI 0.29 e **caía sem traceback
nenhum** sob a 0.33, deixando o lock da GPU preso. Três causas encadeadas:

1. O `CoreModelPatcher` (novo na 0.33) aloca um `HostBuffer` do comfy-aimdo para todo CLIP e VAE.
   Um script que importa `comfy.sd` direto não faz o `comfy_aimdo.control.init()` que o
   `main.py:63` faz.
2. Inicializar o aimdo **no início** do programa, com o CLIP ainda no caminho dinâmico, derruba o
   processo sem mensagem. A saída é `disable_dynamic=True` no `load_clip` e no
   `load_diffusion_model` — que para um benchmark é o certo de qualquer jeito, porque dynamic VRAM
   só adiciona variância.
3. O VAE não aceita esse parâmetro, então precisa do aimdo de verdade — **e inicializá-lo tarde não
   basta**: `comfy_aimdo/host_buffer.py:6` faz `lib = control.lib` **no import**, e o
   `comfy/memory_management.py:7` importa esse módulo no import do ComfyUI. A `lib` fica `None`
   congelada. É preciso religar `host_buffer.lib` depois do `init()`.

O relatório agora é gravado **antes** das imagens: uma execução anterior perdeu vinte minutos de
amostragem já medida porque o decode morreu depois.

---

## 4. O que muda do original para o quantizado, em números

Três medidas diferentes, e elas **não** dizem a mesma coisa. Essa é a lição do dia anterior.

| medida | o que compara | serve para |
|---|---|---|
| **erro por camada** | uma Linear, ativação real, contra float32 | escolher formato por camada |
| **epsilon por passo** | previsão do modelo, **entrada casada** com o BF16 | comparar dois checkpoints |
| **imagem final** | processo livre, 8 passos | **nada** — a trajetória diverge e o destino continua bom |

A terceira foi usada como juiz em 30/08 e não carregava sinal. Por isso o `probe_epsilon_ckpt_ab`
impõe a trajetória do BF16 aos dois braços: divergência de trajetória deixa de existir por
construção.

### Text encoder: quanto custa quantizar, e quanto custa destravar

Só o `qwen_3_4b` tem gêmeo BF16 no disco, então só nele dá para separar as duas metades:

```
peso de 4 bits, matemática BF16 (como o ComfyUI carrega hoje)   1,44e-1   cosseno 0,9896
peso de 4 bits, matemática de 4 bits (destravado)               6,09e-1   cosseno 0,9492
```

E o tempo depende do comprimento do prompt, não do tamanho do arquivo:

```
tokens   travado  destravado
    22     80,7      120,1 ms   1,49x MAIS LENTO
   199    151,9       95,2 ms   1,60x mais rápido    <- cruza entre 75 e 199
   850    456,1      124,3 ms   3,67x mais rápido
```

---

## 5. Não coberto

Nenhuma métrica perceptual em lugar nenhum — as imagens estão aí para olho humano, e é só isso que
elas são. Uma placa por medição (3090 e 3080 Ti, as duas sm86). Sem SASS: "ramo nativo" sempre
significa "produz número diferente do fallback", nunca "a instrução foi observada emitindo".
Nenhum modelo de **vídeo** foi gerado hoje — o MiniMax H3 quer uma lista de latentes (vídeo e
áudio) que os instrumentos genéricos daqui não montam, e o LTX 2.5 não foi rodado. E as travas do
text encoder só se soltam por monkeypatch: **não existe caminho no ComfyUI que um usuário possa
usar.**
