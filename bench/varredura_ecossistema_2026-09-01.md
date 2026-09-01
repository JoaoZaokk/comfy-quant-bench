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

## 1. A lista de camadas do Z-Image: encontrada, e ERRADA para nos

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
