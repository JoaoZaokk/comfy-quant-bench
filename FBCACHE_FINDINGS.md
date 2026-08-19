# FBCache (Comfy-WaveSpeed-Fixed) — o que estava quebrado e o que foi medido

Trabalho em `ComfyUI/custom_nodes/Comfy-WaveSpeed-Fixed`, fork de
`chengzeyi/Comfy-WaveSpeed` mantido em `yannickcruz/Comfy-WaveSpeed-Fixed`.
Base: `8e75e34`. Nada foi enviado para nenhum remoto — não há PR aberto.

## Veredito

O nó estava **decorativo** para quase todo modelo que dizia suportar. Não travava,
não avisava, não acelerava: devolvia saída bit-idêntica ao modelo sem patch, gastando
100% do compute. Sete bugs, dos quais o central torna o cache impossível de disparar.

O commit-título do fork —
`8e75e34 "ApplyFirstBlockCache is now compatible with Z-Image Turbo | Thereshold tested: 0.200"` —
descreve um caminho que, pela aritmética, nunca podia acertar uma vez. Isso está
demonstrado, não deduzido, em `tools/fbcache_clone_test.py`.

## Os sete bugs

### 1. Detecção de FLUX excessivamente ampla

`fbcache_nodes.py` em HEAD:

```python
is_flux = has_double_blocks and has_single_blocks
```

Qualquer modelo com as duas listas tinha o `forward_orig` inteiro substituído pelo
rewrite do FLUX, argumentos e tudo. Vários modelos do ComfyUI têm as duas listas e
assinatura diferente:

| modelo | por que o rewrite não serve |
|---|---|
| HunyuanVideo, HunyuanVideo 1.5 | `vector_in` é `None`, `txt_in` é um `TokenRefiner(x, c, mask)` |
| Chroma, Chroma Radiance | não têm `vector_in` |
| Hunyuan3D | não tem `txt_in` (é `cond_in`) |

Sintoma no HunyuanVideo 1.5: `TypeError: 'NoneType' object is not callable`.

**Correção.** A detecção passou a testar os atributos que o rewrite realmente
desreferencia, e a testar `txt_in` pela **aridade**, não pelo tipo — `comfy.ops`
escolhe uma classe `Linear` por formato de peso, e a usada em fp8 não herda de
`torch.nn.Linear` (MRO `Linear → Module → CastWeightBiasOp`).

### 2. A cópia que faltava — o bug central

```python
clone_original_hidden_states=model_class_name == "LTXVModel",   # fbcache_nodes.py:301
```

O FBCache decide tudo a partir de `first_residual = saída(bloco 0) − entrada(bloco 0)`.
Os blocos do ComfyUI atualizam a entrada **in place** (`img += ...`,
`comfy/ldm/flux/layers.py:246`) e devolvem o mesmo objeto. Sem cópia, a subtração é
`x − x`.

Zero é o pior valor possível porque falha em silêncio nas duas direções ao mesmo tempo:
o MAE relativo vira `0/clamp(0) = 0`, que passa em qualquer threshold, enquanto o cosseno
de um vetor nulo é `0`, que reprova no portão de direção. Miss eterno, saída bit-idêntica,
nenhum erro.

Medido em `tools/fbcache_clone_test.py`, mesmo modelo, única diferença sendo a cópia:

| | primeiro residual | (hits, misses) | desvio vs. sem patch |
|---|---|---|---|
| com a cópia | 0.639719 | (5, 1) | 0.058021 |
| sem a cópia (HEAD) | **0.000000** | (0, 6) | **0.000000** |

**Correção.** A cópia é incondicional. Não existe configuração correta para desligá-la.

### 3. Wan recusado sem motivo

A lista de atributos da rota genérica em HEAD era `layers`, `transformer_blocks`,
`double_blocks`, `joint_blocks`. Wan guarda a pilha em `blocks`, que não estava na lista —
única razão pela qual Wan inteiro era recusado.

**Correção.** `blocks` entrou na lista, por último, para que um modelo com nome específico
*e* genérico fique com o específico.

### 4. Igualdade de string perde subclasses

```python
return_hidden_states_only = model_class_name in ("LTXVModel", "NextDiT") or ...
```

LTX-2.5 é `LTXAVModel(LTXVModel)`. Igualdade de string não vê herança, então ele não
recebia nem a cópia (bug 2) nem `return_hidden_states_only`.

**Correção.** A decisão passou a olhar a MRO.

### 5. Fix do upstream perdido num rewrite

`7723fc6` (upstream #120) removia `modulation_dims_img` / `modulation_dims_txt` antes dos
single blocks — o `SingleStreamBlock` aceita um `modulation_dims` só. O fix existe em
`7723fc6` e **não existe** em HEAD.

**Correção.** Restaurado nos dois pontos de chamada, com teste de regressão.

### 6. Um cache só para duas ramificações de condicionamento

`comfy/samplers.py` só junta a passada positiva e a negativa numa chamada quando as formas
batem. Quando não batem — comprimentos de texto diferentes — o modelo é chamado duas vezes
no mesmo timestep com o mesmo shape de latente, e um cache indexado por `(shape, timestep)`
não distingue as duas. Observado no HunyuanVideo 1.5 com prompt de 25 tokens contra negativo
de 6:

```
RuntimeError: The size of tensor a (25) must match the size of tensor b (6)
```

Pior: onde o encoder preenche até largura fixa, como o T5 faz no FLUX, as formas batem e a
passada negativa **continua silenciosamente do cache da positiva**.

**Correção.** Um cache por ramo, indexado por `cond_or_uncond`. O contador de
`max_consecutive_cache_hits` também passou a ser por ramo — compartilhado, um ramo gastava a
cota do outro.

### 7. Portões invisíveis

Dois. O `cos_sim > 0.95` era adição do fork (entrou em `ec3d421`; o upstream `8253745`
decide por L1 relativo sozinho), fixo no código e sem qualquer indicação. Um modelo cujos
residuais giram mais rápido que isso recusa todo hit enquanto aumentar
`residual_diff_threshold` não faz absolutamente nada. E `cat_hidden_states_first` estava
preso a `model_class_name == "HunyuanVideo"` numa ramificação que HunyuanVideo nunca
alcançava (ele caía em `is_flux`) — flag nasceu inalcançável; quando ligada de verdade,
custa relL2 0.1466, porque o ComfyUI concatena `(txt, img)`.

**Correção.** `direction_threshold` virou entrada opcional do nó (`-1` desliga, igual
upstream), e todo run termina com uma linha dizendo quantos hits, quantos misses e **qual
critério recusou**:

```
[WaveSpeed] FBCache: 16 hits, 14 misses -- 2 over tolerance,
            6 vetoed by window or consecutive-hit limit, 6 with nothing cached yet
```

A conta fecha por construção: `2 + 6 + 6 = 14`.

## O que eu errei

Três, todos pegos por rodar a condição de controle, nenhum por reler o código.

1. **Portão por `isinstance`.** Usei `isinstance(txt_in, torch.nn.Linear)` para separar FLUX.
   O `Linear` que um checkpoint fp8 recebe não herda de `nn.Linear`, então isso mandaria todo
   FLUX real para a rota genérica. Trocado por aridade.
2. **Medição contaminada.** Publiquei 1.62x/1.90x/2.89x para o Hunyuan medindo enquanto outro
   job ocupava a 3090. Retratei. A checagem que pegou foi medir o baseline **duas vezes**,
   primeiro e último: 1153.14s contra 103.56s, 91% de discordância.
3. **Contagem de hits.** `cache_hits` era incrementado dentro do teste de similaridade, antes
   do veto de `validate_can_use_cache_function`. O relatório inflava acertos que não
   aconteceram — ironia, dado que o relatório existe para impedir exatamente isso. Separado em
   `_record_similarity` (o porquê) e `note_cache_decision` (o que foi feito).

## Recusas deliberadas

Recusar é uma saída válida e ruidosa: o grafo roda em velocidade cheia, sem aceleração, e o
motivo é impresso. O que não pode existir é o no-op silencioso do bug 2.

| modelo | motivo |
|---|---|
| Wan VACE | dirige `vace_blocks` pelo índice do laço principal |
| Wan S2V | dirige `audio_injector` pelo índice do laço principal |
| desconhecido | volta ao runtime padrão, com aviso |

A rota genérica troca a lista de blocos por **um** wrapper, então o laço do modelo roda uma
vez só. Se o laço faz trabalho dependente do índice fora da chamada do bloco, esse trabalho
some. Medido no `wan2.1_vace_1.3B`: **1.42x mais rápido com zero cache hits** e relL2 0.59 —
trabalho pulado, não cacheado. Um `1.42x` que parece um cache funcionando é a razão de isso
ser recusado em vez de aproximado.

O guarda cobre os dois atributos: achei o `audio_injector` do `WanModel_S2V` investigando o
VACE, e um guarda só de VACE teria deixado passar.

## Medições

Todas com baseline medido duas vezes, primeiro e último. Só o HunyuanVideo foi medido com
pesos fixados residentes (`--highvram`) — Wan e FLUX.2 Klein foram medidos antes dessa opção
existir, e os dois cabem folgados na 3090, então provavelmente não houve thrash. "Provavelmente"
não é medido: veja *pendências*.

### HunyuanVideo 1.5 (54 blocos, 20 steps)

| threshold | s | speedup | hits | relL2 |
|---|---|---|---|---|
| baseline | 105.08 | 1.00x | — | — |
| baseline2 | 108.96 | 0.96x | — | — |
| 0.08 | 66.04 | **1.59x** | 8/20 | 0.1987 |
| 0.12 | 55.69 | **1.89x** | 10/20 | 0.2157 |
| 0.20 | 34.35 | **3.06x** | 14/20 | 0.2639 |

Faixa residual do modelo: **0.0209 a 0.3872**. Remedido depois em corrida independente e
resolução menor: baselines 40.79s / 42.26s (3%), 25.48s **1.60x**, 13.31s **3.07x** — reproduz
dentro de 1%, o que é a evidência de que a razão não depende da resolução escolhida.

Baselines concordam em 4%.

### FLUX.1-dev fp8 (rota `flux`, 20 steps, pesos residentes)

| threshold | s | speedup | hits | relL2 |
|---|---|---|---|---|
| baseline | 13.22 | 1.00x | — | — |
| baseline2 | 13.87 | 0.95x | — | — |
| 0.12 | 8.22 | **1.61x** | 16/40 | 0.3375 |
| 0.20 | 6.30 | **2.10x** | 22/40 | 0.3790 |

Baselines concordam em 5%. Faixa residual do modelo: **0.0434 a 0.3653**. Os 40 são 20 steps
× 2 ramos de condicionamento (cond e uncond contam separado, que é o bug 6).
Estes números substituem os 1.30x/1.64x citados antes, medidos sem pesos fixados.

Remedido depois em corrida independente: 12.92s / 13.23s de baseline (2%), 7.98s **1.62x**,
6.12s **2.11x**. Reproduz dentro de 1%.

**Correção de um erro meu.** Eu tinha escrito aqui que "a contagem de hits marca `0/0` porque a
instrumentação só cobre a rota genérica". Falso. O nó conta **todas** as rotas —
`first_block_cache.py:944` e `:957` incrementam `ctx.cache_hits`/`cache_misses` dentro do
rewrite da rota flux, e o relatório sai normalmente:

```
[WaveSpeed] FBCache: 16 hits, 24 misses -- 11 over tolerance, 1 with nothing cached yet
[WaveSpeed] This model's step-to-step residual distance ranged 0.0434 to 0.3653.
```

Quem não instrumenta a rota flux é o contador **do probe**, que embrulha
`CachedTransformerBlocks`. Eu li o `0/0` da coluna do probe e concluí coisa sobre o nó. O texto
que o probe imprimia reforçava a confusão; foi reescrito para apontar o leitor à linha
`[WaveSpeed] FBCache` em vez de sugerir que a contagem não existe.

### Wan 2.2 t2v (12 steps, pesos residentes)

| threshold | s | speedup | hits | relL2 |
|---|---|---|---|---|
| baseline | 18.18 | 1.00x | — | — |
| baseline2 | 19.39 | 0.94x | — | — |
| 0.08 | 18.80 | 0.97x | 0/12 | 0.0000 |
| 0.15 | 11.55 | **1.57x** | 5/12 | 0.1229 |

Baselines em 6%. Em 0.08 o diagnóstico dá `11 over tolerance` e a distância residual do modelo
vai de 0.0832 a 0.2531 — o threshold fica no piso e nunca passa. Substitui o 1.30x/1.90x
anterior, medido sem pesos fixados.

### FLUX.2 Klein — funciona e ainda assim não vale ligar

5 double blocks contra 20 single, e só a pilha double pode ser cacheada (o modelo reconstrói
`vec` entre os dois laços). Remedido com pesos residentes, 12 steps: 0.35 → 7.28s **1.07x**
(5/12 hits, relL2 0.0801); 0.5 → 7.31s **1.07x** (6/12, relL2 0.0656), contra baselines de
7.79s e 8.08s. Mais hits, mesmo tempo. **Disparar não é o mesmo que ajudar.**

### LTX-2.5 — dispara, mas só bem acima do que FLUX pede

`ltx-2.5-22b-distilled-transformer-bf16_w4a8`, 8 steps, pesos residentes:

| threshold | s | speedup | hits | relL2 | misses |
|---|---|---|---|---|---|
| baseline | 27.42 | 1.00x | — | — | — |
| baseline2 | 28.56 | 0.96x | — | — | — |
| 0.15 | 28.40 | 0.97x | 0/8 | 0.0000 | 7 por tolerância |
| 0.35 | 18.06 | **1.52x** | 3/8 | 0.8627 | 4 por tolerância |
| 0.60 | 14.63 | **1.87x** | 4/8 | 0.8203 | 2 tolerância, 1 direção (pior cos 0.8870) |

A rota sai certa (`LTXAVModel` → `transformer_blocks`, `return_only=True`, via MRO) e o
`relL2 0.0000` em 0.15 prova que o wrapper reproduz o modelo exatamente através da tupla
(vídeo, áudio) — que era o ponto do bug 4.

Os zero hits em 0.15 **não eram bug**. O diagnóstico do bug 7 responde direto: a distância
residual step-a-step deste modelo vai de 0.2015 a 0.9283, então um threshold de 0.15 fica
abaixo do piso e nenhum step jamais passa. A família LTX (2.3 e 2.5, ambas `LTXAVModel`)
precisa de um threshold 2–4x maior que o de FLUX. O `relL2` alto em 0.35/0.6 é o custo:
esses valores movem bastante a trajetória.

Corolário de método: `0 hits` sozinho é pergunta, não achado. O probe não imprimia o relatório
porque a linha sai de `patch_get_output_data`, que embrulha `execution.get_output_data` —
módulo que o probe nunca carrega. Corrigido chamando `clear_all_cache_contexts(report=True)`
depois de cada run, e foi isso que fechou a questão.

### LTX-2.3 22B fp8 — só encanamento

Mesma classe, mesmo diagnóstico: `0 hits, 6 misses -- 5 over tolerance`, faixa residual
0.2752–0.4657 em 0.15. Rota correta, `relL2 0.0000`, 29 GB não cabem residentes, então os
tempos (35.35s / 37.50s de baseline) valem só como confirmação de que nada quebrou.

### Wan VACE 1.3B — recusa verificada no modelo real

```
[WaveSpeed] NOT APPLIED to VaceWanModel: drives `self.vace_blocks` from the index of its
            main block loop ...
[WaveSpeed] The model runs unchanged at full speed. This is a fallback, not a silent skip.
```

`relL2 0.0000`, baselines em 2%. O grafo roda, sem aceleração, com o motivo impresso.

### Método

Um tempo medido num modelo que não cabe na VRAM não é um tempo. Sem os pesos fixados, o
ComfyUI reenvia peso por step, e um threshold mais alto pula os blocos **e** pula o
recarregamento deles — dois efeitos que não se separam depois. O mesmo baseline do Hunyuan
deu 105s residente e 806s em streaming, e nesse run os dois baselines discordaram entre si
em 87%.

## Testes

Todos em CPU, todos passando.

| arquivo | cobre |
|---|---|
| `tools/fbcache_routing_test.py` | 18 arquiteturas, com as classes `Linear` reais do `comfy.ops` |
| `tools/fbcache_catorder_test.py` | ordem da concatenação nos dois ramos |
| `tools/fbcache_tuple_test.py` | estado oculto em tupla (vídeo+áudio do LTX-2.5) |
| `tools/fbcache_branch_test.py` | cache por ramo de condicionamento e limite de hits |
| `tools/fbcache_kwargs_test.py` | strip do `modulation_dims_*`, bypass de STG, `direction_threshold` |
| `tools/fbcache_clone_test.py` | prova do bug 2 |

Ferramentas: `tools/fbcache_probe.py` (A/B com baseline duplo), `tools/fbcache_audit.py`
(instrumenta a cadeia de decisão por step).

## Pendências

- **Nenhuma medição pendente.** Hunyuan 1.5, FLUX.1, FLUX.2 Klein, Wan 2.2, LTX 2.3 e LTX 2.5
  saíram todos com `--highvram` e baseline duplo. Os números antigos de FLUX.1 (1.30x/1.64x) e
  Wan (1.30x/1.90x) foram substituídos acima.

## Não feito

- **Nenhum PR.** Não autorizado.
- **Z-Image**: não testado, por dois motivos independentes. `z_image_bf16.safetensors` está
  truncado no disco (2.197.236.293 bytes contra 12.309.866.400 esperados pelo header — falha
  de download pré-existente, não do nó). E no arquivo íntegro `z_image_turbo_bf16` o **baseline
  sem patch** morre antes de amostrar:
  `RuntimeError: Given normalized_shape=[2560], expected input with shape [*2560], but got input of size[1, 64, 4096]`
  em `comfy/ldm/lumina/model.py:656`. É o contexto sintético do probe, largo demais para o
  `cap_embedder` do Lumina — bug do probe, não do nó, e nada pode ser concluído enquanto o
  caminho sem patch não roda.
- **Hunyuan3D**: não testado. Falha na condicionação sintética do probe, e o baseline falha
  junto — não é bug do nó, mas também não é evidência a favor dele.
- **Bug do ComfyUI, não corrigido, só reportado**: `comfy/utils.py:1108` decora
  `tiled_scale_multidim` com `@torch.inference_mode()`, e `comfy/sd.py:1141` chama
  `process_output` (`image.add_(1.0)`) sobre a saída dela →
  `RuntimeError: Inplace update to inference tensor outside InferenceMode` em decode tiled 3D.
- `create_patch_zimage_forward` continua no arquivo e continua sem ser chamada. Já era morta
  em HEAD (só alcançável por `get_model_patch_function`, que ninguém chamava). Ligar um rewrite
  não testado é exatamente o erro que quebrou o FLUX para quatro arquiteturas.
