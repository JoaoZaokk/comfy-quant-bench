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

## Por que P4 nao rodou -- tres motivos, todos medidos

`probe_quant_dispatch.py --mode diffusion` e a ferramenta de camada 2 desta bancada, e ela nao
alcanca este arquivo por tres razoes independentes:

1. **Resolucao de nome.** Ela chama `folder_paths.get_full_path_or_raise("diffusion_models", CKPT)`.
   O arquivo mora em `P:/ComfyBench/checkpoints/`, e a estrofe `bench_p` do
   `extra_model_paths.yaml` declara `checkpoints: checkpoints` -- outra pasta. E o caso
   `NAO_PROBAVEL` que o `CLAUDE.md` ja registra.
2. **Loader errado.** Este e um **checkpoint empacotado** (DiT + VAE + audio VAE + vocoder +
   projecao). O proprio yaml anota que o LTX 2.3 se carrega por `CheckpointLoaderSimple`, nao por
   `load_diffusion_model`, e o probe so tem os modos `te` e `diffusion`.
3. **Commit.** Qualquer um desses caminhos passa por `load_torch_file` -> `safe_open`, que cobra
   **2x o arquivo**: 16,64 GiB x 2 = **33,3 GiB**, contra **23,00 GiB** de commit livre medidos as
   23:25 (a 3090 do vizinho segura muito). Morreria com `access violation` em
   `torch/storage.py`, que e a assinatura que esta bancada ja pagou quatro vezes.

O que EXISTE no lugar, e e mais fraco de proposito: o kernel smoke acima roda um kernel real sobre
**uma** camada real deste arquivo e prova que o backend nativo resolve e executa. Ele **nao** conta
quantos dos 1440 forwards passam pelo caminho quantizado nem quantos `dequantize` acontecem, que e a
pergunta de P4. **Nao tratar "kernel smoke PASS" como "despacha".**

Para fechar P4 depois seria preciso: (a) um hardlink ou copia do arquivo em
`P:/ComfyBench/diffusion_models/`, (b) um modo `checkpoint` no probe usando
`load_checkpoint_guess_config`, e (c) commit livre acima de ~35 GiB, ou o leitor do dynamic-VRAM --
com a ressalva de que este arquivo tem **2-D em BF16 e em F32** (os `weight_s_rel`), o que e
exatamente a armadilha de dtype do lazy-load que o `CLAUDE.md` registra.

---

## FOLHA DE TESTES -- o que falta, e e do dono

O que sobrou depende de julgamento visual e da placa grande. O modelo e adulto; as imagens sao dele.

### 1. Renderizar o W4A8

O arquivo esta em `P:/ComfyBench/checkpoints/`, que a estrofe `bench_p` monta como `checkpoints`,
entao `CheckpointLoaderSimple` ja o ve como `10Eros_v1.5_bf16_w4a8.safetensors`.

O workflow oficial do autor e
`TenStrip/LTX2.3-10Eros_Workflows -> 10Eros_10SNodes_I2V_Basic_DMD_V5.json`, e ele exige os nodes de
`github.com/TenStrip/10S-Comfy-nodes` -- **nao conferi se estao instalados aqui**. A v1.5 e hibrida e
o README do autor recomenda uma LoRA DMD (`TenStrip/LTX2.3_DMD_Lora -> LTX2.3_DMD_hybrid_v2.safetensors`).

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
  primeiro. Alternativa mais barata: comparar contra o
  **`10Eros_v1.4_DMD_int8_convrot` de terceiro que ja esta no repo do autor** (27,16 GiB) -- nao e o
  BF16, mas e um braco int8 do mesmo modelo, e esta bancada ja mediu quatro vezes que o int8 ganha
  do 4-bit em fidelidade, entao a comparacao tem direcao esperada.

---

## NAO COBERTO

Nenhuma imagem, nenhum audio, nenhum latente. Nenhuma medicao de tempo por passo (e quando vier, sai
da barra de progresso do sampler, nunca do JSON do `ltx_video.py`). P4 nao rodou. O `10S-Comfy-nodes`
nao foi conferido. A LoRA DMD nao foi baixada nem aplicada. O braco BF16 nao existe. E **nada foi
publicado**.
