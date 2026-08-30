# NOVO — 2026-08-30

## O achado que fecha o dia: a 3090 está na única faixa que pega o MMA de 4 bits

Um agente da rodada 5 voltou afirmando *"ninguém executa INT4 GEMM do ConvRot; o
comfy-kitchen dequantiza INT4 → INT8 e usa CUTLASS INT8"*, e concluiu que o rótulo W4A4
virou comercial. O comfy-kitchen 0.2.31 **está instalado aqui**, então isso se lê em vez de
se acreditar. `backends/cuda/__init__.py:1252` bifurca:

```python
if linear_dtype == "int8" or not (
    _cuda_device_supports_native_int4_mma(x2d) or _should_use_turing_int4(x2d)
):
    ...  # peso INT4 x ativação INT8, _int4_weight_int8_act_gemm_dequant_chunked
...      # senão: quantize_int4_rowwise_convrot64 -> MMA INT4
```

E quem decide, em `:293`:

```python
def _cuda_device_supports_native_int4_mma(tensor) -> bool:
    if not tensor.is_cuda or _FORCE_INT4_INT8_FALLBACK:
        return False
    major, _minor = _cuda_device_capability(tensor.get_device())
    # The current ConvRot W4A4 kernel emits m16n8k64 s4 MMA, which is the
    # sm80+ integer MMA shape. Hopper is routed through the INT8 fallback for
    # better behavior with this implementation.
    return major == 8
```

**`major == 8`, exato.** A 3090 é 8.6 e a 3080 Ti é 8.6 — as duas caem no caminho nativo.
E a comparação é igualdade, não `>=`, então:

| arquitetura | major | caminho |
|---|---|---|
| Turing 7.5 | 7 | rota própria (`_should_use_turing_int4`, `cutlass_turing_int4_dequant`) |
| **Ampere / Ada 8.x** | **8** | **MMA `m16n8k64 s4` nativo** |
| Hopper 9.x | 9 | fallback INT8, **de propósito** — o comentário diz "for better behavior" |
| Blackwell 10/12 | 10/12 | fallback INT8 |

**Ampere é a única faixa que pega o MMA de 4 bits neste kernel.** Não é acidente de
hardware velho: Hopper é desviado deliberadamente.

Isso reorganiza tudo o que apareceu hoje:

1. **O agente errou pelo motivo de sempre** — descreveu o caminho de fallback como se fosse
   o único. Ele existe e é o das linhas 1255-1296; só não é o desta placa.
2. **Explica `w4a4_int4mm_layers: 0` sem hipótese nenhuma.** A ferramenta do Abiray rotula a
   camada por qual caminho ela tomaria. Zero em `int4mm` e 117 em `int8mm` é exatamente o
   que este código produz em hardware que não seja major 8. O metadado deles é honesto e
   preciso; o que eu li como deficiência é o mesmo dispatch visto do outro lado.
3. **Explica o ConvRot oficial exigir Blackwell/NVFP4.** É outro formato numérico, para o
   hardware que *não* tem o s4 MMA nesta implementação.
4. **E dá a posição desta bancada, agora com mecanismo:** os dois maiores distribuidores
   públicos entregam ConvRot que multiplica em INT8, e esta placa é de uma geração que
   multiplica em INT4. Não é sorte — é a faixa que o kernel atende.

- **CONFERIDO:** o despacho, lido do pacote instalado; e a medição de 2026-08-22 registrada
  no `CLAUDE.md` (nativo contra peso-dequantizado a **1,43e-1**, não 1e-6) é consistente
  com ele.
- **NÃO conferido, e a distinção importa:** isto é código lido, não instrução observada
  executando. `major == 8` diz qual ramo o Python escolhe; que o `m16n8k64 s4` de fato
  emita, e com que ganho, é `tools/probe_backend_resolution.py` mais um profile — e precisa
  da placa, que segue com a sessão irmã. Note também `_FORCE_INT4_INT8_FALLBACK`: existe um
  interruptor que desliga tudo isso, e ninguém aqui checou o que o liga.

**Consequência para a medição de 22/08:** aquele 1,43e-1 comparou nativo contra *peso*
dequantizado, o que prova que a ativação é quantizada — mas sozinho **não separa A4 de A8**,
porque INT8 na ativação também produziria diferença contra BF16. O que separa é este
dispatch. O par (código + medição) sustenta a afirmação; nenhum dos dois sozinho sustentava,
e este arquivo dizia que sim.

---

# Rodadas 1 e 2 — o que abriu caminho

## O achado do dia: ConvRot INT4/INT8 misto, em produção, com 67 mil downloads

`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` no ModelScope. **67114 downloads** — duas ordens
de grandeza acima de qualquer outra coisa achada hoje. Listagem de arquivos conferida pelo
endpoint `/repo/files`:

```
MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors              20,97 GB
MiniMax_H3_FL2VA_pruned_mixed_int4_int8_convrot.safetensors   15,90 GB
MiniMax_H3_FL2VA_pruned_nvfp4.safetensors                     12,53 GB
MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors             20,97 GB
MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors  15,09 GB
MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors                    12,53 GB
MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors                     24,44 GB
```

`mixed_int4_int8_convrot` **é o que este repo constrói**: rotação ConvRot com INT4 e INT8
misturados por camada. Existe publicado, em MiniMax-H3, em escala.

Do README, citado:

- INT8 convrot: *"Recommended for 24GB GPUs"*, e a **RTX 3090 é citada por nome**. Roda em
  Ampere.
- misto int4/int8: *"Requires 15+ VRAM (~15.5 GB)"*
- NVFP4: *"Blackwell Architecture Required"* — confirmação independente do que o
  `nunchaku/utils.py` já dizia, por caminho separado.

**E o que ele NÃO diz é a abertura.** Perguntado diretamente, o README responde NOT STATED
para: o que "convrot" significa ali, como a atribuição INT4/INT8 por camada foi decidida,
que ferramenta gerou os arquivos, e qualquer número de qualidade ou velocidade. Ele se
descreve como *"community-compiled collection"* — é redistribuição, não quem produziu.

Ou seja: estão distribuindo ConvRot misto sem documentar o método, para rodar na mesma
placa que está aqui, e **este repo tem o conversor, a calibração e o preflight que
decidem exatamente isso**.

### O header conta o que o README esconde — 102 KB para um arquivo de 15,9 GB

O header do safetensors fica no começo do arquivo, e o ModelScope aceita `Range`. HTTP 206,
102632 bytes lidos, **nenhum peso baixado**. O `__metadata__` traz a receita inteira:

```json
"convrot_w4a4_mixed": {
  "int8_ratio": 0.2,          "int8_mm_ratio": 0.3,
  "selection": "calibrated+", "calibrated_plus_preset": "BalancedQ",
  "linear_dtype": "int4",     "convrot_groupsize": 256,
  "w4a4_int4mm_layers": 0,    "w4a4_int8mm_layers": 117,  "int8_layers": 83,
  "prune_donor": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
  "paper": "arXiv:2512.03673"
}
```

Mais `int8_selected_layers` e `int8mm_selected_layers` — as listas nominais, camada por
camada. 932 tensores: 220 BF16, 210 F32, 200 I8, 200 U8, 102 F16.

**1. `convrot_groupsize: 256` é exatamente o `CONVROT_GROUP_SIZE = 256` deste repo.** O
formato é o mesmo família, e a checagem de compatibilidade deixa de ser especulação.

**2. `paper: arXiv:2512.03673` — conferido, existe, e é o paper deste projeto:** *"ConvRot:
Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers"* (Huang, Han,
Zhou, Chen, Zhu, Wang; 2025-12-03). Hadamard em grupo, introduz `ConvLinear4bit`, reporta
2,26× de velocidade e 4,05× de memória em FLUX.1-dev, e se apresenta como a primeira
quantização por rotação plug-and-play W4A4 para transformer de difusão. É outro paper que
o OrbitQuant (2607.02461) — duas linhas de rotação distintas, e esta é a nossa.

**3. E aqui está o que importa mais: `w4a4_int4mm_layers: 0`.**

Um checkpoint chamado `w4a4`, com `convrot` no nome, 67 mil downloads, declara no próprio
metadado que **nenhuma camada usa o matmul INT4**. Cento e dezessete usam matmul INT8, e
oitenta e três são INT8 puro. Os pesos são INT4 (`linear_dtype: int4`) — confirma-se pelas
formas: `attn.out_proj.weight` é `I8 [5376, 3584]` e `3584 = 7168/2`, com
`7168 = 56 cabeças × 128`, ou seja INT4 empacotado dois por byte. Mas o `qkv_proj` é
`I8 [21504, 5376]`, largura cheia — INT8, e está na lista `int8mm_selected_layers`.

Peso em 4 bits, multiplicação em 8. **É a coisa exata contra a qual a regra dura deste
repo foi escrita** — "W4A4 significa execução ConvRot nativa, não INT4 só no peso seguido
de GEMM em outra precisão" — e está rodando em escala, com o nome W4A4 no arquivo.

- **CONFERIDO:** o campo existe e vale `0`, lido do header do arquivo real por Range.
- **NÃO conferido, e a distinção é a regra desta casa:** a *semântica* desses campos é da
  ferramenta que gerou, não minha. `w4a4_int4mm_layers: 0` pode significar "nenhum GEMM de
  4 bits executa" ou algo mais estreito no vocabulário deles. Nenhum arquivo foi carregado,
  nenhum kernel foi resolvido, nenhuma GPU foi tocada. Para virar afirmação sobre execução,
  precisa do `tools/probe_backend_resolution.py` contra o arquivo — e da placa livre.

**Correção de escala:** o ModelScope é a cópia. O original está no HuggingFace, com
**791069 downloads e 215 likes** (`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`, mexido
2026-08-15). Quase oitocentas mil, não sessenta e sete mil.

### 4. E o ConvRot oficial não roda nesta placa — mas o comfy-kitchen roda

`github.com/feice-huang/ConvRot`, 28 estrelas, Python, empurrado 2026-07-03. Conferido:
existe, e a descrição é *"Official ConvRot implementation... enabling efficient W4A4
quantization for diffusion models, achieving 4× memory savings and 2× faster inference"*.

Do README dele, citado:

> "Runs in a environment with torchao on an **NVFP4-capable GPU (Blackwell / sm_120)**."

E `rot_size: 256` — o mesmo 256 do `CONVROT_GROUP_SIZE` daqui. Traz `quantize.py` como
produtor de checkpoint, e uma variante `nvfp4-convrot-selfcalib` com escala de ativação
estática auto-calibrada. **Suporte a ComfyUI: NOT STATED.** A distinção
`int4mm` versus `int8mm`: **NOT STATED** — não vem do ConvRot oficial.

**A implicação é a posição desta bancada.** A implementação oficial do paper exige
Blackwell, torchao e NVFP4. Este repo executa ConvRot W4A4 em **sm86**, por
`comfy_kitchen.backends.cuda`, e isso não é alegação de leitura: em 2026-08-22 mediu-se
aqui que o caminho nativo difere do peso-dequantizado por **1,43e-1** e não 1e-6, ou seja,
a metade A4 é real (`CLAUDE.md`, e `tools/probe_backend_resolution.py` reproduz).

Se isso se sustentar, **o comfy-kitchen tem um ConvRot em Ampere que os próprios autores do
paper não distribuem.**

**Hipótese, explicitamente não medida:** talvez seja essa a razão do `w4a4_int4mm_layers: 0`
— um checkpoint ConvRot adaptado para hardware sem NVFP4, onde o matmul de 4 bits não está
disponível e a saída cai para INT8. Isso *explicaria* o campo, e explicar não é medir. Fica
como pergunta, não como achado.

### 5. A ferramenta que escreveu `calibrated+` / `BalancedQ` não foi encontrada

Buscadas as strings literais (`calibrated+`, `BalancedQ`, `int8_mm_ratio`,
`w4a4_int4mm_layers`, `convrot_w4a4_mixed`) em GitHub, HuggingFace e Gitee. Nada. O
instrumento não estava cego — a mesma varredura achou o `feice-huang/ConvRot` e o controle
`comfyanonymous/ComfyUI` responde com 130748 estrelas.

Presets com nome e razões configuráveis não se escrevem à mão uma vez. A ferramenta existe
e é privada, ou não foi publicada. **Consequência para cá: o `tools/quant_mixed.py` não está
reimplementando algo público.**

### 6. São DUAS ferramentas privadas, e nenhuma das duas executa GEMM de 4 bits

A rodada 4 achou `joeygambino/joyai-echo-ltx23-echoVid-ltxAud-surgical-int8` — a **mesma
pessoa** do LTX-2.5 `mix4x8`, a única que documentou o método — e o arquivo dela chama-se
`ltx23_echoVid-ltxAud_surgical_int8_convrot.safetensors`. ConvRot de novo.

Header lido por Range (1112720 bytes, nenhum peso baixado). E o esquema é **outro**:

```
artifact_contract   = int8_tensorwise_inference_checkpoint.v1
convrot             = true
convrot_groupsize   = 256
target_dtype        = int8_tensorwise
artifact_target     = comfyui_diffusion_model
scale_axis          = out_features        scale_granularity = per_channel
quant_storage_dtype = int8                quantized_tensor_count = 1496
model_family = ltx2   model_id = Lightricks/LTX-2.3   project = ltx2-int8-tensorwise-v0
rift_built_from = SURGICAL MERGE: video/conditioning+av_ca-modulation from JoyAI-Echo...
```

7440 tensores: 2903 BF16, 1545 F32, 1496 I8, 1496 U8 — o mesmo pareamento I8/U8 do Abiray.

**O que a comparação entrega:**

1. **Não é a mesma ferramenta.** Abiray fala `convrot_w4a4_mixed`, `selection`, `preset`,
   listas nominais de camada. joeygambino fala `artifact_contract` versionado,
   `scale_granularity`, `project`, `rift_built_from`. Vocabulários disjuntos. São **duas**
   cadeias privadas, não uma.
2. **`convrot_groupsize = 256` nas duas**, e `rot_size: 256` no repo oficial, e
   `CONVROT_GROUP_SIZE = 256` aqui. Quatro procedências independentes no mesmo 256.
3. **E nenhuma das duas executa multiplicação em 4 bits.** A do joeygambino é
   `int8_tensorwise` explícito — rotação ConvRot com armazenamento e execução em INT8. A do
   Abiray declara `w4a4_int4mm_layers: 0`. Dois produtores independentes, os dois aplicando
   a rotação, **nenhum fechando a metade A4.**

Esta bancada mediu o contrário em 2026-08-22: caminho nativo contra peso-dequantizado deu
**1,43e-1**, não 1e-6. Se isso se mantiver sob nova medição, a diferença entre aqui e os
dois maiores distribuidores públicos de ConvRot não é o formato — é qual kernel de fato roda.

- **CONFERIDO:** os dois headers, lidos dos arquivos reais por Range, com os campos citados.
- **NÃO conferido:** nada foi carregado nem executado. Que `int8_tensorwise` e
  `int4mm_layers: 0` signifiquem "não há GEMM de 4 bits" é leitura do vocabulário deles, não
  medição minha. O `probe_backend_resolution.py` contra esses arquivos é o que decide, e
  precisa da placa.

### Achado de vídeo com abordagem diferente

`Harahan/QVGen-CogVideoX-2B-W4A4` — 960 downloads, `arxiv:2505.11497`, tags `qat`, `iclr`,
`text-to-video`, base `CogVideoX-2b`, licença Apache-2.0. É **QAT** (quantização durante o
treino), não PTQ como tudo o mais acima. Caminho distinto do desta bancada, e o único W4A4
de vídeo achado que não vem de ConvRot. **NÃO conferido:** header não lido, paper não lido.

---

# Rodada 1 — o que abriu caminho

Três agentes em paralelo: ModelScope, Gitee, HuggingFace. Cada linha abaixo com achado foi
**reconferida por quem orquestrou**, chamando o endpoint de detalhe direto — os agentes
marcaram coisas como CONFERIDO admitindo, no mesmo parágrafo, não ter lido metadado.

## O que muda uma decisão desta bancada

### 1. LTX-2.5 já existe quantizado, e em formato misto 4/8

`joeygambino/LTX-2.5-Quantized` — HuggingFace, 4625 downloads, mexido 2026-08-21.

```
LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors
LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors
LTX25-distilled-DiT-comfy-nvfp4.safetensors
LTX25-distilled-DiT-comfy-int8.safetensors
LTX25-distilled-DiT-Q2_K.gguf .. Q8_0.gguf
```

**`mix4x8` é o mesmo conceito do `tools/quant_mixed.py`**: 4 bits e 8 bits misturados por
camada, num único arquivo, com o prefixo `comfy-`. Duas variantes de tamanho sugerem dois
pontos de corte diferentes na mesma decisão que este repo toma por medição de erro.

**O critério da mistura está no README, e é medido — não heurística.** Citado literal:

> "All 1440 quantised layers were reconstructed at both precisions against the bf16
> original, then promoted by **error-removed-per-byte** until the budget ran out — the
> greedy solution to minimising total squared reconstruction error under a size cap."
>
> "Ranking by relative error does not work... **Weighting each layer by ‖W‖² is what
> separates them.**"
>
> "**363 of the first 386 promotions land in the audio tower**, only 23 in the video tower."

Três coisas que isso entrega de graça:

1. **É o mesmo método deste repo, um passo à frente.** O `quant_mixed.py` mede
   `err_bf16` / `err_w4a4` / `err_w4a8` por camada contra referência float32. Ele mede a
   mesma coisa e depois **aloca sob orçamento de tamanho**, ordenando por erro removido por
   byte. As duas variantes (13,8 GB e 17 GB) são dois tetos do mesmo greedy, como se
   suspeitava.
2. **"Ranking por erro relativo não funciona; pesar por ‖W‖² é o que separa."** Isto é
   testável aqui, direto, com os dados que o `calibrate_activations.py` já produz. Este
   repo já mediu que crest factor não prediz nada (Spearman +0,10) e que o erro W4A8 prediz
   (+0,978); ‖W‖² é um terceiro eixo, e é de **alocação de orçamento**, não de predição de
   formato. São complementares, não concorrentes.
3. **A torre de áudio é que come o 8-bit.** 363 das 386 primeiras promoções. Se valer para
   o LTX daqui, muda onde procurar regressão de qualidade.

E a confissão dele, que é o que abre espaço:

> "neither w4a4 nor w4a8 **has been run here on an Ada or Ampere 16 GB card**."

**Ele não testou em Ampere. Esta bancada é Ampere, com 24 GB.** Temos o hardware que falta
ao autor do checkpoint.

- **CONFERIDO:** existência, 21 arquivos, tags, `base_model: Lightricks/LTX-2.5`, e o
  critério de mistura, citado do README dele.
- **NÃO conferido:** nada foi baixado, então não sei se carrega no ComfyUI daqui, nem se o
  `comfy_quant` por camada é byte-compatível com o que o `quant_mixed.py` escreve. O README
  diz `NVFP4 is Blackwell-only by construction`, o que bate com o `nunchaku/utils.py`.

### 2. Quatro W4A4 de vídeo/difusão no ModelScope

Todos verificados por `GET /api/v1/models/<owner>/<nome>` (controle: Qwen, 7,1 M downloads).

| modelo | downloads | por que interessa |
|---|---|---|
| `XXXXinXXXXX/Wan22-i2v-w4a4` | 1439 | maior adoção do grupo; W4A4 em image-to-video |
| `ApacheOne/Wan2.2-Animate-2-14B-OrbitQuant-W4A4` | 38 | **OrbitQuant** é método que esta bancada não conhece |
| `ultranationalism/Z-Image-Turbo-SVDQuant-NVFP4` | 35 | **Z-Image** é justamente o modelo em que o `quant_mixed.py` foi calibrado aqui |
| `ModelsLab/MiniMax-H3-svdquant-nvfp4_r32` | 2 | MiniMax-H3 é o que a 0.34 trouxe |

- **CONFERIDO:** os quatro existem, com downloads e data de criação reais.
### 2b. NVFP4 em Ampere: respondido pela fonte primária no disco

A rodada 2 mandou um agente responder isso. Ele voltou com **"(a) não roda, 0% de
compatibilidade, o hardware simplesmente não existe"**, citando *NVIDIA oficial + X/Twitter
(Grok)*. Grok não é fonte primária para compatibilidade de hardware, e a conclusão está
**errada na parte que importa**.

O nunchaku 1.2.1 está instalado aqui. `python_embeded/Lib/site-packages/nunchaku/utils.py`,
em `check_hardware_compatibility`:

```python
if sm in ["120", "121"]:                       # Blackwell: só fp4
    if ...["dtype"] != "fp4_e2m1_all": raise ValueError('Please use "fp4" ...')
elif sm in ["75", "80", "86", "89"]:           # Turing, Ampere, Ada: só int4
    if ...["dtype"] != "int4":         raise ValueError('Please use "int4" ...')
else:
    raise ValueError(f"Unsupported GPU architecture {sm} due to the lack of 4-bit tensorcores...")
```

E em `get_precision`: `precision = "fp4" if sm in ["120","121"] else "int4"`.

O correto, então:

- **sm86 está na lista de suportados.** Turing, Ampere e Ada têm tensor core de 4 bits — a
  mensagem de "lack of 4-bit tensorcores" é para as arquiteturas de *fora* dessa lista.
  Dizer que o hardware não existe inverte o que o código diz.
- **O que é exclusivo do Blackwell é o formato NVFP4** (`fp4_e2m1_all`, com escala
  `fp8_e4m3_nan`, grupo 16), não a capacidade de 4 bits.
- **Um checkpoint NVFP4 aqui é RECUSADO em voz alta**, com `ValueError`. Não dequantiza em
  silêncio. Para este projeto isso é a boa notícia: o modo de falha que mataria a premissa
  — cair para BF16 sem avisar — não acontece por esse caminho.

**Consequência prática, e ela muda o alvo:** os três NVFP4 saem, mas *por causa do formato*.
`XXXXinXXXXX/Wan22-i2v-w4a4` — o de 1439 downloads, o mais adotado dos quatro — **não tem
`nvfp4` no nome.** Se for int4, é o único candidato que pode carregar nesta placa, e é o
que vale conferir primeiro.

- **NÃO conferido:** o `dtype` real dentro do `Wan22-i2v-w4a4`; nenhum arquivo foi baixado.
  Se o ComfyUI (fora do nunchaku) tem outro caminho para nvfp4. Se o `OrbitQuant` roda sem
  o runtime próprio dele.

### 2c. OrbitQuant é real, e mira o mesmo modelo que calibramos aqui

`arxiv.org/abs/2607.02461` — **"OrbitQuant: Data-Agnostic Quantization for Image and Video
Diffusion Transformers"**. Confirmado buscando a página, porque o ID veio de um agente e
tinha cara de inventado.

Rotação Hadamard em blocos, permutada e randomizada, com codebook único entre camadas e
modalidades; alega empurrar PTQ de transformer de difusão até **W2A4** com qualidade usável.
Avaliado em FLUX.1, **Z-Image-Turbo**, Wan 2.1 e CogVideoX.

É a mesma família do ConvRot — rotacionar antes de quantizar — por um caminho diferente, e
ataca justamente o Z-Image, que é onde o `quant_mixed.py` foi calibrado aqui. **NÃO
conferido:** o paper não foi lido além do resumo, e o `OrbitQuant` exige runtime próprio
segundo o agente, o que não foi verificado.

### 3. Alguém mais faz W4A4 com calibração em ativação real

`AlperKTS/Krea-2-SVDQuant-ComfyUI` — 4830 downloads, tags `w4a4, int4, svdquant, comfyui`,
125 arquivos, incluindo `calibration/krea2_act_stats_base.safetensors` e `_turbo`.

Estatística de ativação empacotada junto do modelo é exatamente o que o
`tools/calibrate_activations.py` produz aqui. É o primeiro caso externo comparável.

Menor, mesma família: `Patil/krea-turbo-svdquant` (39 downloads, `svdquant_config.json`).

## O que NÃO existe

Com consulta de controle validada:

- **`gitee.com/comfyui-cn` e `gitee.com/ComfyUI-Extensions` não existem** — 404 em API
  (`/api/v5/orgs/`, `/api/v5/users/`) e em HTML. Controle `mindspore` → 200 nos dois.
  Os dois links da lista original são becos.
- Gitee: 9 achados descartados por serem espelho declarado de GitHub/HF. Um único original
  chinês, `deeprd/Wan2GP` (0 estrelas, FP8 Scaled, não W4A4).

## O erro da rodada 1, registrado

Um agente escreveu *"LTX-Video quantizado → 0, controle validado"* e no mesmo relatório
entregou `LTX-2.5-Quantized`. A busca não estava cega e o controle estava certo — **o
modelo mudou de nome.** Ele segurou fixo o eixo que era a variável.

Regra nova na seção 1.2 do `AGENTS.md`: toda ausência lista os apelidos tentados.

Segundo erro: os três agentes escreveram `NOVO.md` por cima uns dos outros, e sobrou o do
último parecendo o total. Agora só quem orquestra escreve este arquivo.

## Não coberto nesta rodada

Conteúdo de qualquer arquivo — nenhum peso foi baixado, por regra. Nenhum teste de carga no
ComfyUI. Nenhuma GPU tocada: o lock estava com a sessão irmã (`glm-w8a8-test`), medido em
atividade real (5 MiB → 12 GiB, 29% util ao longo de 50 s).
