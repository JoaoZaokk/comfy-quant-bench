# 10Eros v1.5 em W4A8: o que foi medido, e a folha que falta o dono rodar (2026-09-21)

Pedido de `Dalvogalbo2` no card de `JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot`. Criterio
escrito **antes** de qualquer numero: [criterio_10eros_w4a8.md](criterio_10eros_w4a8.md).

**Nada foi publicado.** A licenca e a renderizacao sao decisao do dono; ver a secao final.

---

## As quatro previsoes: tres passaram, a quarta nao foi rodada

    P1  1440 camadas selecionadas / 4507 preservadas       PASSOU    --dry-run deu exatamente 1440
    P2  saida a menos de 0,5% dos 16.651.422.654 B do gemeo PASSOU   0,0568% (9.459.352 B)
    P3  zero quantizado fora de `model.diffusion_model.`    PASSOU   0 de 2880, e as 4 familias intactas
    P4  zero dequantize numa contagem de despacho       NAO RODADO   bloqueado, motivo medido abaixo

## O arquivo

    fonte   P:\ComfyBench\checkpoints\10Eros_v1.5_bf16.safetensors        46.139.886.366 B   5947 tensores
    saida   P:\ComfyBench\checkpoints\10Eros_v1.5_bf16_w4a8.safetensors   16.641.963.302 B  10267 tensores
    sidecar 10Eros_v1.5_bf16_w4a8.quant.json                                        719 B

    dtypes da saida:  BF16 4507   I8 1440   U8 1440   F32 2880
    _quantization_metadata: 1440 camadas, todas `asym_w4a8_int8`

    familia                       tensores        GiB    origem
    model (o DiT)                     8764     11,652   era 39,12 GiB -> 3,36x menor
    text_embedding_projection            4      2,153   INTACTA, byte a byte
    vae                                170      1,352   INTACTA
    vocoder                           1227      0,240   INTACTA
    audio_vae                          102      0,099   INTACTA

Sidecar: `architecture ltx_2_5`, `quantization asym_w4a8_int8`, `layout AsymW4A8Int8Layout`,
`backend comfy_kitchen.backends.cuda`, `group_size 16`, `convrot_groupsize 256`, `symmetric true`,
`codebook true`, `quantized_tensors 1440`, `preserved_tensors 4507`, `comfy_kitchen 0.2.31`,
`torch 2.13.0+cu130`, `gpu NVIDIA GeForce RTX 3080 Ti`, `conversion_seconds 946.132`,
`source_size 46139886366`.

## Como foi feito, e o que a corrida ensinou

`quant_w4a8.py --profile ltx_2_5`, **na 3080 Ti** (`CUDA_VISIBLE_DEVICES=1`), com
`Assert-GpuLock -Owner 'bench:10eros_w4a8_conversao'` tomado a mao -- o converter nao passa por
`_timing.compare()`, entao o lock e dever de quem chama. A 3090 estava a 92-94% com trabalho de
terceiro e **nao foi tocada**. Durante a conversao a 3080 Ti marcou 2981 MiB / 8%: o cortex, que mora
nela, nao foi incomodado. `Release-GpuLock` depois de a ultima medicao terminar.

**946,1 s**, contra 1296,5 s do gemeo destilado. Nenhuma calibracao: o perfil `ltx_2_5`
(`tools/quant_w4a8.py:54-66`) e allowlist por regex.

**O limite que morde e commit, nao disco.** `quant_w4a8.py:286` e de dois passos -- *"pass one fills
`quantized` with every layer, pass two writes"* -- entao o payload inteiro fica em RAM antes de um
byte sair. Observado: **15,38 GiB privados com 3,45 GiB de commit livre**, e nenhum `.partial` no
disco durante ~35 dos 47 minutos, o que faz a conversao parecer travada. O crescimento e previsivel
(1056 -> 1248 camadas custaram 0,57 GiB), entao da para extrapolar de um terco do caminho. Para um
modelo maior que este, a guarda existente nao pega: ela checa `3 x maior tensor`, nao a soma.

## Verificacao (A4): estrutura e fonte, PASS

    Structural verification: PASS (1440 asym_w4a8_int8)
    Source comparison:       PASS          <- byte a byte em todos os 4507 preservados
    Normal ComfyUI backend:  w4a8_int8_linear -> comfy_kitchen.backends.cuda
    kernel smoke, 1 de 1440 camadas, M=2, entrada aleatoria:
      relative_rmse 0,074666   max_abs_error 0,3125   output_dtype bfloat16

`verify_w4a4.py` (que aceita `asym_w4a8_int8`) le por `seek` + byte range, sem mmap e sem
`safe_open`, entao roda sem risco de commit -- conferido no codigo antes de rodar.

**P3 conferido tambem pela letra da previsao**, direto no header: 2880 tensores I8/U8, **0 fora de
`model.diffusion_model.`**, e `vae` 170 / `audio_vae` 102 / `vocoder` 1227 /
`text_embedding_projection` 4 com **zero** quantizados cada.

E os ops ainda resolvem para o backend nativo HOJE (`probe_backend_resolution.py`, na 3080 Ti):
`w4a8_int8_linear -> comfy_kitchen.backends.cuda`, e o nativo difere do GEMM com peso dequantizado
por rel-L2 **1,43e-1** -- a metade A4 e real, nao e so peso de 4 bits com matematica de 16.

## P4 nao fechou, e o caminho ate a parede foi MEDIDO -- inclusive contra o que eu havia previsto

**A primeira versao desta secao dizia que a tentativa "morreria com `access violation`". Isso era
inferencia declarada como limite, que e o modo de falha que a REGRA ZERO proibe.** Tentei. Nao morreu.
O que aconteceu de verdade, em ordem:

**Primeira tentativa: `FileNotFoundError`, e a causa que eu havia escrito estava INCOMPLETA.** Criei um
**hardlink** em `P:/ComfyBench/diffusion_models/` -- funciona sobre SMB, 16.641.963.302 bytes, zero
disco extra -- e o probe ainda assim nao achou. Motivo real: **`probe_quant_dispatch.py` nunca
carregava o `extra_model_paths.yaml`**, que o `ComfyUI/main.py:130-132` carrega. Sem isso ele so ve
`ComfyUI/models/*`, e **todos os roots montados ficam invisiveis** -- D:, W:, P:, U:, C:\ComfyBench,
**119 dos 359 arquivos de modelo** desta instalacao, incluindo justamente `P:\ComfyBench`, onde as
conversoes daqui sao escritas. O `avaliar_despacho.py:75` documentava a limitacao por PASTA e nao
conhecia esta, por ROOT, e o `NAO_PROBAVEL` do `CLAUDE.md` tinha a causa incompleta. **Consertado**
(commit `cb29b85`): carregado o yaml como o `main.py` faz, com aviso em stderr se falhar.

**Segunda tentativa, com o yaml carregado: NAO deu access violation. Deu thrashing.** O processo
subiu a **31,70 GiB privados** -- os 2x do `safe_open` sobre 16,64 GiB, como a aritmetica previa -- e
**sobreviveu**, porque o pagefile cresceu e o limite de commit foi de 98,72 (documentado) para
**137,91 GiB**. Mas ali platoou: 15 minutos com o CPU avancando ~9 s por minuto de relogio (15% de um
nucleo), commit livre em 16,78 GiB, e **nada nunca chegou na placa** (a 3080 Ti ficou em 2557 MiB do
comeco ao fim). Estava paginando, nao progredindo.

**Parei por decisao, e o motivo nao e tecnico.** O vizinho estava com **23,7 GiB de VRAM** na 3090 em
trabalho ativo, e o meu processo segurando 31,70 GiB de commit num limite de 137,91 com 16,78 livres
punha a maquina perto da borda -- se ele precisasse de commit, a minha medicao poderia derrubar o
trabalho **dele**. A regra permite eu matar os meus proprios processos de bancada; e a medicao e
desejavel, mas desestabilizar a maquina na ausencia do dono nao e. Lock liberado, hardlink removido,
original conferido intacto (16.641.963.302 B), commit de volta a 27,77 GiB.

**Entao a barreira real de P4 nao e "morre", e "nao cabe sem paginar"**, e sobra um terceiro motivo
que continua valido: este e um **checkpoint empacotado** (DiT + VAE + audio VAE + vocoder + projecao),
que o proprio yaml anota carregar por `CheckpointLoaderSimple`, e o probe so tem os modos `te` e
`diffusion`. Mesmo com commit sobrando, faltaria um modo `checkpoint` usando
`load_checkpoint_guess_config`.

O que EXISTE no lugar, e e mais fraco de proposito: o kernel smoke acima roda um kernel real sobre
**uma** camada real deste arquivo e prova que o backend nativo resolve e executa. Ele **nao** conta
quantos dos 1440 forwards passam pelo caminho quantizado nem quantos `dequantize` acontecem, que e a
pergunta de P4. **Nao tratar "kernel smoke PASS" como "despacha".**

Para fechar P4 depois: (a) rodar com a 3090 livre, ou com o vizinho fora, para nao competir por
commit; (b) um modo `checkpoint` no probe; e (c) considerar o leitor do dynamic-VRAM, que nao cobra 2x,
**com a ressalva medida** de que este arquivo tem 2-D em BF16 e em F32 (os `weight_s_rel`), que e
exatamente a armadilha de dtype do lazy-load registrada no `CLAUDE.md`.

---

## FOLHA DE TESTES -- o que falta, e e do dono

O que sobrou depende de julgamento visual e da placa grande. O modelo e adulto; as imagens sao dele.

### 1. Renderizar o W4A8

O arquivo esta em `P:/ComfyBench/checkpoints/`, que a estrofe `bench_p` monta como `checkpoints`,
entao **`CheckpointLoaderSimple` ja o ve** como `10Eros_v1.5_bf16_w4a8.safetensors` -- confirmado
lendo o workflow oficial, que usa exatamente esse loader.

**Os parametros reais, lidos do JSON oficial** (`bench/10eros/10Eros_10SNodes_I2V_Basic_DMD_V5.json`,
85 nodes, baixado do repo do autor) -- nao adivinhados:

    loader            CheckpointLoaderSimple            (no original: 10Eros_v1.3_fp8mixed_learned)
    LoRA              LoraLoaderModelOnly, strength 1   LTX2.3_DMD_reshaped_r256.safetensors
    DUAS passadas de SamplerCustom, com sigmas manuais:
      passada 1   euler_ancestral        9 passos   1.000 0.955 0.893 0.812 0.715 0.603 0.482 0.241 0.121 0.0
      passada 2   euler_ancestral_cfg_pp 3 passos   0.92 0.725 0.421875 0.0
    entre elas        LTXVLatentUpsamplerTiled com ltx-2.3-spatial-upscaler-x2-1.1.safetensors
    cfg 1, seed 42 fixa
    latente de audio  LTXVEmptyLatentAudio [1, 24, 1]
    resolucao         vem da imagem de entrada (I2V, via ImageResizeKJv2 / GetImageSize),
                      nao do EmptyLTXVLatentVideo

**Atencao a LoRA: o workflow V5 usa `LTX2.3_DMD_reshaped_r256`, e o README do 10Eros recomenda
`LTX2.3_DMD_hybrid_v2` para a v1.5.** Sao arquivos diferentes (5.095.405.082 contra 5.095.398.920 B).
**As duas ja estao em `P:/ComfyBench/loras/`**, baixadas e conferidas por tamanho, para nao esperar
download na hora de comparar.

**O upscaler espacial JA esta em disco:**
`ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`.

**Nodes: `10S-Comfy-nodes` NAO esta instalado** (nada casa `*10S*` / `*TenStrip*` em
`ComfyUI/custom_nodes/`). Instalar pacote de node e mudanca no ComfyUI e **e decisao sua** -- nao
instalei. Um grep dos 41 tipos usados pelo workflow contra o codigo nao localizou 9 deles:

    GetNode  SetNode  mxSlider  TwoWaySwitch  MarkdownNote  Power Lora Loader (rgthree)
    LTXReferenceConditioning  LTXReferenceEnable  LTXVLatentUpsamplerTiled

**Esse grep e fraco e provavelmente tem falso negativo**: `ImageResizeKJv2` do MESMO pacote
(KJNodes) FOI encontrado, e `GetNode`/`SetNode` moram nele -- um node registrado por dicionario
computado nao casa em busca de texto. `MarkdownNote` e nativo do frontend. **A checagem que vale e
abrir o JSON na UI**, que lista os faltantes com precisao. O que da para afirmar sem ressalva e so o
`10S-Comfy-nodes`, procurado por nome de pasta.

O servidor precisa de `--disable-dynamic-vram` (duas razoes independentes ja medidas nesta bancada) e
de um launcher que passe a flag: `run_nvidia_gpu_8190_ultra_video.bat` passa.

### 2. O que olhar para dizer bom ou ruim

Nao e "a imagem ficou boa". Nesta familia os modos de falha medidos sao especificos:

- **Estatica / chuvisco colorido** e como o W4A4 do Qwen Edit falhou.
- **Borrao** e como o W4A4 do Wan 2.2 falhou -- e esse e o traicoeiro, porque **borrar APROXIMA o
  latente da referencia**: a divergencia elogia um build que suaviza. Nao aceitar por divergencia.
- **O AUDIO.** O LTX gera imagem e som juntos, e o card do 2.5 desta bancada ja errou uma vez
  medindo so metade do modelo. Ouvir: nivel (RMS), se o som esta em fase com o movimento, e se nao
  virou ruido branco. No 2.3 a metrica de audio **saturou** e empatou W4A8 com Q6_K a 8 passos,
  entao ouvido vale mais que numero aqui.
- **Movimento.** O 2.3 W4A8 obedeceu "rotating slowly" com movimento 4,6 contra 1,25 da referencia
  sem LoRA. Se o movimento morrer, e sinal.

### 3. O controle que TEM de falhar

Sem esse controle a folha se auto-aprova. Renderizar tambem **sem a LoRA DMD**, a 4 passos: o Qwen
Edit a 4 passos sem a Lightning falhou nas tres instrucoes, e era o controle que tinha de falhar.
Se o braco sem LoRA sair igualmente bom, o teste nao esta medindo o que diz.

### 4. Se passar: o que falta para publicar

- **Decisao de licenca.** A cadeia e a **LTX-2 Community License Agreement** (2026-01-05), lida do
  `__metadata__` do `10Eros_v1.4_DMD_int8_convrot` que mora no proprio repo do 10Eros. O README do
  10Eros **nao declara licenca nenhuma** e linka aprovando tres quants de terceiros
  (`vantagewithai` GGUF, `CornLogic` INT8, `LokkenJP` fp8) -- o que e evidencia de que o autor quer
  quants, nao de que a licenca permite. Conferir os termos de redistribuicao e a politica de uso
  aceitavel antes de subir.
- **O card sobe ANTES do peso**, pela regra desta bancada: negativo medido nunca existe sem etiqueta.
- **O braco de referencia BF16** custa caro aqui: 42,97 GiB x 2 de commit contra ~98-137 GiB de
  limite com ~70 ja comprometidos. Caminho: GGUF sem perda (`tools/safetensors_to_gguf_bf16.py`),
  que foi escrito para arquivo **so-transformer** e aqui exigiria extrair as 4444 chaves `model.*`
  primeiro.

### O braco de comparacao JA ESTA EM DISCO, e foi validado

`P:/ComfyBench/diffusion_models/10Eros_v1.5_INT8_TFO.safetensors` (CornLogic), **25.032.524.432 B**,
tamanho conferido. `int8_tensorwise` pelo dialeto `comfy_quant` por tensor, **1232 camadas
quantizadas**, **so-transformer** (6908 tensores, todos sob `model.`, 23,312 GiB -- sem vae, vocoder,
audio_vae ou projecao). Os 3212 BF16 + 1232 quantizadas = **4444**, exatamente o numero de tensores
`model.*` da nossa fonte: estrutura identica.

**Validado como MESMA BASE que o nosso W4A8, por dois caminhos.** Primeiro o decisivo:
`int8_tensorwise` preserva sinal, e a concordancia de sinal com o `10Eros_v1.5_bf16` deu **1,0000 em
6 de 6** camadas amostradas (blocos 0, 10, 24, 47 e um `embeddings_connector`).

Segundo, e ele corrige uma leitura minha: o `__metadata__` deles traz `blend_base: 10Eros_v1.4_bf16`,
`blend_source: 10Eros_v1.2_bf16`, `blend_mode slerp`, `blend_alpha 0.7->0.0:linear`,
`blend_scope attn1,attn2` -- e eu li isso como "eles remesclaram, entao nao e a v1.5 publicada". **A
nossa propria fonte `10Eros_v1.5_bf16` carrega esses mesmos campos, identicos um por um** (29 chaves
contra 30; a unica a mais deles e `REFCLPP`). Aquilo e a procedencia que o **TenStrip** gravou de como
a v1.5 foi construida, repassada verbatim pelo quantizador de terceiro.

**Armadilha geral que vale guardar: metadata que descreve uma mesclagem pode descrever a historia da
FONTE, nao a producao daquele arquivo.** Um campo `blend_base` nao diz que quem escreveu o arquivo
mesclou nada.

Entao o A/B limpo e **W4A8 nosso (4 bits, ativacao 8) contra INT8 de terceiro (8 bits)**, mesmos pesos
de partida, e esta bancada ja mediu quatro vezes em quatro familias que o int8 ganha em fidelidade --
direcao esperada, e um resultado invertido seria motivo para conferir a medicao, nao para celebrar.
Ressalva: o deles e so-transformer, entao o VAE, o vocoder e a projecao tem de vir de outro lugar no
workflow; o nosso ja os traz dentro.

**E NAO usar o `fp8mixed_experimental_learned` do LokkenJP como braco**, apesar de ser da mesma fonte
v1.5: ele carrega parametros de quantizacao ajustados por camada contra um objetivo ponderado por
canal ([10eros_fp8_terceiro_promocao_medida.md](10eros_fp8_terceiro_promocao_medida.md)), entao a
comparacao mediria **formato + criterio de ajuste**, nao formato.

---

## NAO COBERTO

Nenhuma imagem, nenhum audio, nenhum latente. Nenhuma medicao de tempo por passo (e quando vier, sai
da barra de progresso do sampler, nunca do JSON do `ltx_video.py`). P4 nao rodou. O `10S-Comfy-nodes`
nao foi conferido. A LoRA DMD nao foi baixada nem aplicada. O braco BF16 nao existe. E **nada foi
publicado**.
