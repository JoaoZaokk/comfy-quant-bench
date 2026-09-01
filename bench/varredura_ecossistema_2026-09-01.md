# O que existe la fora que serve aqui -- varredura de 2026-09-01

O dono apontou a conta `brahianrosswill` (497 repos) e pediu para achar o que valia. Cinco agentes
leram; tudo abaixo e **LIDO**, exceto o que esta marcado MEDIDO AQUI.

## A conta em si nao e o achado

    497 repos    486 forks    11 "proprios"    9,3 GB forkados    so 14 com estrela

E uma conta de **bookmark por fork**. Os 11 nao-forks nao sao dele: **nenhum commit de codigo e
assinado por ele em nenhum dos onze** -- sao copias por push (por isso o GitHub nao mostra "forked
from"). Um deles, `Face`, e um bypass de filtro NSFW do FaceFusion; fica registrado como risco, nao
como tecnica.

O valor e o **mapa do ecossistema** que ele juntou sem querer. Sempre ir ao upstream: os forks dele
estao ate 8 meses atrasados e com zero commits proprios.

## 1. A lista de camadas do Z-Image: encontrada -- e esta secao foi CORRIGIDA pela segunda rodada

> **Leia a segunda rodada, no fim deste arquivo, antes de agir por esta secao.** O que esta abaixo
> conclui que a lista de terceiro esta "errada para nos". Lendo o **dtype por camada** do build
> OFICIAL do Comfy-Org, a conclusao correta e outra: eles quantizam `adaLN_modulation` a **INT8**, e
> nos a deixamos em BF16 porque o resto vai a **INT4**. Ninguem poe modulacao em 4 bits. Sao dois
> pontos do mesmo trade, nao um certo e um errado.

Este era o bloqueio da janela de GPU de hoje -- `quant_w4a4.py` nao tem perfil `zimage`. Duas fontes
independentes deram a lista, e **elas se contradizem entre si**:

    ussoewwin/Hybrid-Sensitivity-Weighted-Quantization   INCLUI cap_embedder, x_embedder,
                                                          noise_refiner, context_refiner,
                                                          t_embedder, final_layer
    tritant/ComfyUI_Kitchen_nvfp4_Converter (Turbo)      EXCLUI exatamente esses seis

Contado contra o nosso `beyond-reality-zimage-v2_native` (453 tensores, 208 com `.weight` e ndim>=2)
-- **MEDIDO AQUI**:

    HSWQ allowlist          208 camadas
    tritant Z-Image-Turbo   180
    tritant Z-Image-Base     90
    NOSSO zimage-v2-w4a4    170   <- o que de fato foi quantizado e funciona

Nenhuma das tres bate. A diferenca contra o Turbo e nos DOIS sentidos:

    so no Turbo   30x  layers.N.adaLN_modulation.0   [15360, 256]
    so no nosso   20x  context_refiner.* e noise_refiner.*  (attention.qkv/out, feed_forward.w1/w2/w3)

**`adaLN_modulation` produz escala e deslocamento de condicionamento**, entao erro ali multiplica no
bloco inteiro -- e o nosso `quant_w4a4.py` ja exclui a camada equivalente do Hunyuan (`_mod.linear`)
pelo mesmo motivo. Copiar a lista deles teria quantizado 30 camadas que evitamos de proposito, e o
arquivo resultante carregaria e despacharia normalmente.

**Consequencia:** o perfil `zimage` do `quant_w4a4.py` deve sair do NOSSO conjunto de 170, que ja
esta validado por render, e nao das listas de terceiros. Elas serviram para levantar a hipotese e
para mostrar que ha desacordo real no ecossistema sobre modular quantizavel ou nao.

Brinde, do mesmo arquivo do HSWQ: `convrot_group_size_for_features()` escolhe o **maior potencia-de-4
que divide n**, com fallback 256 -> 64 -> 16 -> 4. E a terceira confirmacao independente da regra de
potencia de 4 que medimos hoje, e da um degrade por camada que nao temos.

## 2. O unico par de verdade: `NidAll/comfyui-mixed-quantizer`

Conversor standalone de ~13 mil linhas que **reimplementa a matematica** em vez de delegar a
comfy-kitchen (o nosso `quant_w4a8.py` tem 374 linhas e delega). Mesmos tres formatos que os nossos;
nenhum formato novo no ecossistema inteiro.

O que ele tem e nos nao:

- **codebook adaptativo por curtose** -- mede curtose excedente numa amostra e escolhe entre tabela
  gaussiana fixa e Lloyd-Max de 25 iteracoes. O nosso e liga/desliga.
- **refinamento ALS das escalas de grupo** (2 a 3 passes).
- **gate global ponderado por parametro** + promocao gulosa por erro-removido-por-byte, contra o
  nosso limiar fixo por camada e budget por contagem de camadas.
- **`testdata/runtime_equivalence.py`**: compara o simulador deles contra o kernel eager REAL da
  comfy-kitchen exigindo 1e-4 relativo, em K dificeis (320/640/1152/1408/1920/2520/3360), e valida
  com **decodificadores de referencia independentes** escritos da spec. O nosso `verify_w4a4.py`
  compara byte-identidade dos preservados e faz smoke contra `F.linear` -- **o mesmo codigo dos dois
  lados**. Isto e teste diferencial de verdade e e o que mais falta aqui.

Duas alegacoes do README deles que batem com medicao nossa: W4A8 e 256-only (idem nosso
`probe_convrot_groupsize`), e W4A4 ~2x mais ruidoso que W4A8 (mesma direcao dos nossos 1,40-1,49x).
Uma discordancia a checar: eles dizem que W4A4 exige `K % 64 == 0`; nos medimos aceitacao em
16/64/256/1024.

## 3. Uma armadilha, confirmada na nossa arvore

`ussoewwin/ComfyUI-DistorchMemoryManager` faz, no `__init__.py`:

    mm.EXTRA_RESERVED_VRAM = non_torch

**MEDIDO AQUI** (`ComfyUI/comfy/model_management.py:853-854`): essa e exatamente a global que
`--reserve-vram` escreve, em codigo de modulo que roda no import -- **antes** dos custom_nodes.
Entao o node sobrescreve a flag em silencio. Nenhum node instalado nesta bancada faz isso hoje
(grep em `custom_nodes/`, zero ocorrencias), mas vale saber antes de instalar.

O nome do pacote tambem mente: nao tem relacao com DisTorch2 e **nao toca no aimdo** -- o no
principal e o `Purge VRAM V2` do LayerStyle renomeado.

Uma ideia dali vale copiar em cinco linhas em vez de instalar 2694: `non_torch = NVML.used -
(total - torch.mem_get_info().free)`, isto e, reservar o que o vizinho ocupa. Na nossa placa com
inquilino, onde ja medimos 15 GiB de discordancia entre torch e nvidia-smi, isso daria uma reserva
grande o bastante para impedir o ComfyUI de despejar o `glm-w4`. Mas deles e **estatico** (calculado
uma vez no startup, indice 0 fixo); teria de virar dinamico e por dispositivo, somando ao
`--reserve-vram` em vez de sobrescrever.

## 4. Duas saidas para o dynamic-vram, que hoje so sabemos desligar por inteiro

- **`model.clone(disable_dynamic=True)`** -- por modelo, sem flag global. **MEDIDO AQUI**: a
  assinatura existe em `comfy/model_patcher.py:430` e o proprio ComfyUI a usa em `:2173`. Candidato
  a substituir o `--disable-dynamic-vram` global; nao testado com Nunchaku.
- **`comfy_aimdo.control.init(simple_vram_headroom=<bytes>)`** -- um knob do alocador que estourou
  com "35 GB staged" no LTX 2.5. Visto em `Windecay/ComfyUI-ReservedVRAM`. Nunca soubemos que
  existia.

O `ReservedVRAM` em si **nao** resolve o bypass de `--gpu-only`/`--highvram`: ele escreve na mesma
variavel que ja medimos nao ser consultada.

## 5. `molbal/ComfyUI-GGUF`: INT8 ConvRot dentro de um GGUF

Tem um quant type `Q8_CR` que chama `comfy_kitchen.tensor.int8_utils._build_hadamard/_rotate_weight`
-- a nossa propria rota de INT8 ConvRot, feita por terceiro, com fallback de groupsize 256/64/16/4.
Vale comparar contra o nosso `quant_int8.py --convrot`. Detecta Z-Image por
`cap_embedder.1.weight` + `context_refiner.0.attention.qkv.weight`.

O INT4 dele esta **aposentado**, comentado como *"pending a performant Ampere W4A16 backend"* --
sm86 e exatamente a nossa placa, e nos temos o backend. Vale saber que o ecossistema considera esse
o buraco.

## O que NAO serve, e por que

- **NVFP4 / MXFP8** -- emulados fora de Blackwell; a propria tabela do autor diz que na sm86 so
  INT8/INT4 ConvRot sao acelerados.
- **`silveroxides/ComfyUI-QuantOps`** -- marcado `[DEPRECATED]` pelo autor: a funcao foi absorvida
  pelo ComfyUI que ja rodamos.
- **`ComfyUI-fni8`** -- alvo e sm_70 (Volta), com gate de arquitetura que recusa outras GPUs, e o
  pacote de kernels da 404.
- **`ComfyUI-INT8-Fast`** -- encerrado pelo autor ("INT8 virou nativo").
- **`comfyui-sdnq`** -- nao escreve checkpoint do ComfyUI; e pipeline diffusers.
- **`wavespeed-comfyui`** -- e o cliente da API cloud da wavespeed.ai e faz `pip install` no import,
  o que viola a regra de nao mexer no stack pinado. O first-block cache que ja temos e o
  `chengzeyi/Comfy-WaveSpeed`, projeto sem relacao apesar do nome quase igual.
- **`facok/comfyui-meancache-z`** -- cache de passo para Z-Image, ortogonal a quantizacao, e
  **conflita com o nosso instrumento**: ocupa o mesmo `model_function_wrapper` que o
  `probe_epsilon_per_step.py` usa para impor a trajetoria BF16. Os dois nao coexistem.
- **`Starnodes2024/comfyui-starnodes-modelconverter`** -- mais arquiteturas, nenhum metodo novo: o
  "mixed precision" dele e reaplicar um mapa camada->formato extraido de arquivo alheio.
- Os **11 repos proprios** -- nenhum e dele; das tecnicas alheias so duas mereciam leitura, e as
  duas se leem melhor no upstream vivo.

## A observacao que vale mais que qualquer item

**O ecossistema inteiro faz quantizacao so-de-peso, sem calibragem.** Dos conversores lidos, nenhum
captura ativacao real nem mede erro por camada; escolhem por lista de nomes ou heuristica posicional.
O `quant_mixed.py` desta bancada -- que roda amostragem real com forward-pre-hooks e decide por erro
medido do kernel -- nao tem par entre os lidos. O `NidAll` chega mais perto pelo gate global, mas
tambem sem ativacao real.

## Nao coberto

Nada disto foi executado: nenhum repo foi clonado, instalado ou medido. Onde um README alega numero,
esta escrito "alega". As unicas linhas MEDIDO AQUI sao as contagens de camada contra o nosso
checkpoint, o `EXTRA_RESERVED_VRAM` em `model_management.py:853`, a assinatura de `clone()` em
`model_patcher.py:430` e o grep de `custom_nodes/`. Cinco agentes leram; um sexto olhando os mesmos
repos poderia discordar.

---

# Segunda rodada: o conversor que fez um arquivo nosso, e a resposta oficial do Z-Image

## `jlucasmcrell/ltx25-quant-lab` construiu dois arquivos do nosso disco -- PROVADO

O dono apontou `jlucasmcrell`. E o mesmo `joeygambino` que o `CLAUDE.md` ja cita: todos os commits
dos 7 repos dele sao `joeygambino <jlucasmcrell@gmail.com>`. O `MANIFEST.json` do `ltx25-quant-lab`
publica tamanho e sha256; conferido contra o disco -- **MEDIDO AQUI**:

    LTX25-distilled-DiT-comfy-w4a4   11 236 345 048 B   aed01441...ac44d   IDENTICO
    LTX25-distilled-DiT-comfy-w4a8   12 520 362 840 B   7f7c0fc3...a7776   IDENTICO

Byte a byte. O `LTX25-distilled-DiT-comfy-w4a4` -- que o `CLAUDE.md` cita como "um segundo W4A4
publico que realmente executa 4 bits", 1440 camadas -- tem procedencia fechada: ferramenta, metodo
e autor.

**Metodo deles (LIDO):** sem calibragem, sem ativacoes, sem hook. Chamam
`QuantizedTensor.from_float(w, layout)` do proprio ComfyUI e gravam `<layer>.comfy_quant` como
tensor uint8 -- o segundo dialeto.

**A ideia que vale mais que o arquivo: `canon_layers()`.** Em vez de manter lista por arquitetura,
abrem o release oficial ja quantizado, leem o cabecalho e extraem as chaves terminadas em
`.comfy_quant`. Espelham a escolha do autor do modelo em vez de adivinhar.

**A lacuna deles que nos ja fechamos:** o README diz que `w4a4` e `nvfp4` foram construidos e
testados em Blackwell e que "nenhum dos dois foi rodado aqui em Ada ou Ampere". Nos rodamos o w4a4
deles na sm86 e ele **despacha nativo** -- camada 2, 1440 camadas, 8/8 forwards quantizados, 0
dequantize. Material para issue ou PR: informacao que so nos temos.

## A resposta OFICIAL do Z-Image, obtida com 91 KB em vez de 6,2 GB

`canon_layers()` aplicado ao `Comfy-Org/z_image_turbo` (7,1 M downloads). O build
`z_image_turbo_int8_convrot.safetensors` tem 6,2 GB, mas o cabecalho fica no comeco: um
`Range: bytes=0-7` da o tamanho, um segundo Range traz os **91 000 bytes** do header. Nenhum peso
baixado. **MEDIDO AQUI -- o oficial quantiza 202 camadas:**

    30x layers.N.{adaLN_modulation.0, attention.out, attention.qkv, feed_forward.w1/w2/w3}
     +  context_refiner.0-1.* e noise_refiner.* (inclusive o adaLN do noise_refiner)
    fora: final_layer, x_embedder, cap_embedder, t_embedder

Contra as contagens da primeira rodada:

    HSWQ allowlist          208
    tritant Z-Image-Turbo   180
    OFICIAL Comfy-Org       202
    NOSSO zimage-v2-w4a4    170

## E aqui a leitura muda, contra o que a primeira rodada escreveu

A primeira rodada concluiu que copiar a lista de terceiro "teria quantizado 30 camadas que evitamos
de proposito", como se eles estivessem errados. **Comparando o dtype POR CAMADA, a conclusao e
outra:**

    OFICIAL   layers.0.adaLN_modulation.0   I8    [15360, 256]   <- INT8, largura cheia
    NOSSO     layers.0.adaLN_modulation.0   BF16  [15360, 256]   <- nao quantizado
    NOSSO     layers.0.feed_forward.w1      I8    [10240, 1920]  <- INT4 empacotado (3840/2)

O oficial **nao** poe modulacao em 4 bits: poe em **8**. Nos deixamos em BF16 e pomos o resto em 4.
Dois pontos do mesmo trade, nao um certo e um errado -- e **ninguem quantiza `adaLN_modulation` a 4
bits**, o que endossa a nossa exclusao em vez de contradize-la.

**A opcao nova que isso abre:** o `quant_mixed.py` poderia promover `adaLN_modulation` a **W4A8** em
vez de deixar em BF16 -- casando o espirito do build oficial (8 bits na modulacao) sem perder os 4
bits no resto. Sao 31 camadas hoje intocadas, testaveis com o maquinario existente e sem perfil novo.

## Correcao de rota sobre o `DistorchMemoryManager`

A primeira rodada descreveu a armadilha de um jeito que da a entender que ela esta ativa aqui. **Nao
esta**: `grep -rnE "EXTRA_RESERVED|set_extra_reserved_vram"` em `custom_nodes/` da **zero**,
inclusive dentro dos `.disabled` (761 arquivos `.py` no escopo, verificado). Nao ha o que remover.

## `Scottcjn/ComfyUI-TurboQuant`: o nome promete o que o codigo nao faz

**Nao quantiza peso nenhum** -- so K/V. E nao ha cache: comprime K e V e **descomprime na linha
seguinte**, dentro do mesmo `attn_patch`. Round-trip puro; o `llms.txt` deles admite que "nao e um
KV cache persistente ainda". Em difusao isso e coerente: o DiT recomputa a atencao a cada passo. O
README **alega** 4,6x de VRAM; 4,57x e a razao do formato (128 floats fp16 = 256 B -> 56 B), nao uma
medicao.

## Nao coberto nesta rodada

Os sha256 e os cabecalhos foram medidos; **o codigo do `ltx25-quant-lab` foi so LIDO** e nenhuma
alegacao de qualidade deles foi reproduzida. O header oficial diz quais camadas e em que dtype --
**nao** diz por que, e nao substitui medir. A ideia de promover `adaLN_modulation` a W4A8 tem zero
medicoes.

---

# Terceira rodada: esparsidade 2:4, e duas conclusoes minhas derrubadas em seguida

Pergunta do dono: *"a gente nunca fez o teste bf16/fp16 > sparsity > quantizacao, correto?"*

**Correto, nunca fizemos.** A unica "sparsity" na arvore e o **SpargeAttn** -- esparsidade de
ATENCAO em runtime, pulando blocos, medida no `nunchaku_compare.py`. Peso: zero. Sem poda, sem 2:4,
sem SparseGPT nem Wanda.

## O bloqueio de hoje e de BIBLIOTECA, nao de placa -- MEDIDO AQUI

    cusparse64_12.dll                          existe, 143 MB   <- e cuSPARSE, outra lib
    torch.backends.cusparselt.is_available()   False, version None
    to_sparse_semi_structured(...)             RuntimeError: cuSPARSELt not supported on your machine
    compute capability                         8.6  -- 2:4 e suportado por HARDWARE desde 8.0

O dono confirmou que cuSPARSELt roda em sm86. Instalar e mudanca de pacote (decisao dele), e da
para fazer com o mesmo `-c constraints.txt` que usamos no pytest/ruff.

## Meu primeiro teste era TAUTOLOGICO, e o dono apontou

Medi que aplicar a rotacao Hadamard num peso 2:4 destroi o padrao (zeros 0,500 -> 0,000; grupos
exatos 1,000 -> 0,000) e escrevi que "esparsidade antes da ConvRot nao sobrevive". A pergunta dele
desmonta: *"hadamard faz isso, verdade, mas o que ele faria se nao tivesse zeros?"*

**Faltou o controle.** Rotacao densa preenche zero de QUALQUER matriz -- isso nao e um fato sobre
esparsidade, e o que rotacao faz. Apresentei uma trivialidade como achado.

## O teste que presta, e ele derruba a MINHA hipotese seguinte

Hipotese: a rotacao espalha magnitude, entao podar 2 de 4 depois doeria mais. **MEDIDO AQUI**, peso
real `layers.0.feed_forward.w1` do Z-Image, [10240, 3840]:

    rotacao e inversivel                    erro round-trip 1,77e-07
    A  podar 2:4 direto                     erro relativo 0,3617
    B  rodar -> podar 2:4 -> desrodar       erro relativo 0,3641

    razao (2 menores)/(2 maiores) por grupo de 4:
      denso original      0,3485
      depois da rotacao   0,3496

**Praticamente identicos.** A rotacao NAO uniformiza a magnitude do peso, e a hipotese morre. Faz
sentido em retrospecto: Hadamard de um peso quase-gaussiano continua quase-gaussiano (invariancia
rotacional da gaussiana). O ConvRot existe contra outlier de ATIVACAO, nao contra distribuicao de
peso -- e este numero e a primeira medicao disso nesta bancada.

**Consequencia pratica:** a ordem nao importa para o erro, e existe receita viavel que eu tinha
descartado cedo demais -- **rodar -> podar 2:4 no dominio rodado -> quantizar**. Ali o peso
armazenado e 2:4 E rodado, entao o tensor core esparso funciona, e o custo de precisao e o mesmo de
podar direto.

## O numero que realmente decide

    podar 2:4 nesta camada      erro relativo 0,3617
    nosso W4A4 inteiro          mediana       0,1241

**A esparsidade custa ~3x o que a quantizacao a 4 bits custa**, numa camada real. Isso reordena a
prioridade: antes de perseguir 2:4, o ganho por bit esta muito melhor onde ja estamos.

**Ressalva que anda junto:** e poda por magnitude PURA, sem reconstrucao. SparseGPT e Wanda
atualizam os pesos restantes para compensar e derrubam bastante esse erro -- 0,3617 e o teto
ingenuo, nao o custo de um metodo serio. Uma camada, uma metrica, sem render.
