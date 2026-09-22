# Criterio: 10Eros v1.5 em W4A8 (pedido de terceiro no Hub)

Escrito **2026-09-21, antes de qualquer numero**. Pedido de `Dalvogalbo2` no card do LTX 2.3:
subir um W4A8 de `TenStrip/LTX2.3-10Eros`.

## O que ja se sabe por MEDICAO, antes de converter

Header remoto lido por `Range` request, sem baixar o peso (`.scratch/le_header_remoto.py`,
`anatomia_10eros.py`):

    10Eros_v1.5_bf16.safetensors   46.139.886.366 B   5947 tensores   23.069.505.387 params
      model.*                        39,12 GiB   21.005.004.544   <- o DiT, AVTransformer3DModel, 48 blocos
      text_embedding_projection       2,15 GiB    1.156.061.184
      vae                             1,35 GiB      726.106.225   <- NAO se quantiza
      vocoder                         0,24 GiB      129.085.032   <- NAO se quantiza
      audio_vae                       0,10 GiB       53.248.402   <- NAO se quantiza
      dtypes: BF16 em 5947/5947 (2067 tensores 2-D)

**E o irmao estrutural ja foi convertido nesta bancada.** Sidecar de
`ltx-2.3-22b-distilled-1.1_w4a8`: fonte **46.149.345.334 B**, `quantized_tensors` **1440**,
`preserved_tensors` **4507** — soma 5947, o mesmo numero de tensores do 10Eros v1.5. Saida
16.651.422.654 B, `asym_w4a8_int8`, `group_size 16`, `convrot_groupsize 256`, codebook,
`backend comfy_kitchen.backends.cuda`, convertido na **3080 Ti** em 1296 s.

**Nao precisa da 3090.** `quant_w4a8.py` nao calibra: o perfil `ltx_2_5` e uma allowlist por
regex (`tools/quant_w4a8.py:54-66`) e a conversao e streaming, camada por camada.

**Licenca: LTX-2 Community License Agreement**, de 2026-01-05 — lida do `__metadata__` do
`10Eros_v1.4_DMD_int8_convrot.safetensors`, que mora no proprio repo do 10Eros. O README do
10Eros nao declara licenca e **linka aprovando tres quants de terceiros** (GGUF de
`vantagewithai`, INT8 de `CornLogic`, fp8 de `LokkenJP`). Base da cadeia: `ltx-2.3-22b-dev`
(campo `reshape_ref`) + `SulphurAI/Sulphur-2-base`. **Conferir os termos de redistribuicao da
LTX-2 antes de publicar** — isso e decisao do dono, nao medicao.

## Previsoes, escritas antes de rodar

- **P1** `--profile ltx_2_5 --dry-run` seleciona **exatamente 1440** camadas e preserva 4507.
  Contagem derivada a mao do regex contra os nomes do header: 48 blocos x 28 + 96 dos dois
  `embeddings_connector` = 1440. **Refuta:** qualquer outro numero — o merge mudou a estrutura
  e o perfil tem de ser rederivado antes de gastar GPU.
- **P2** Saida a menos de 0,5% de **16.651.422.654 B**. **Refuta:** desvio acima de 2%.
- **P3** Zero tensor quantizado sob `vae.`, `audio_vae.`, `vocoder.`, `text_embedding_projection.`
  — e todos eles **byte a byte identicos** a fonte no `verify_w4a8`. O int8 de terceiro no mesmo
  repo tambem tocou **0** desses. **Refuta:** um unico I8 fora de `model.diffusion_model.`.
- **P4** `probe_quant_dispatch --forward-only`: 1440 modulos `asym_w4a8_int8`, **0 dequantize**,
  `backends.cuda`. **Refuta:** qualquer dequantize — seria memoria economizada sem tempo.
- **P5** [JULGAMENTO] a imagem presta. No irmao destilado o W4A8 ficou a MAE 10,39 / SSIM 0,829 /
  log-mel 0,163 do BF16 e era usavel. **Mas 10Eros nasce do `dev`, nao do destilado**, e o proprio
  README diz que v1.5 e hibrido e pede uma LoRA DMD — ou seja, outro regime de passos.
  **Refuta:** borrao ou estatica, como nos bracos W4A4 desta bancada.
- **P6** [JULGAMENTO] o `10Eros_v1.4_DMD_int8_convrot` de terceiro (1232 camadas) deve ser **mais
  fiel** que o nosso W4A8. Quatro medicoes independentes nesta bancada dizem que o int8 ganha em
  fidelidade. **Refuta:** nosso W4A8 mais fiel que o int8 — ai a medicao e que tem de ser
  conferida, nao celebrada.

## O braco de referencia e CARO, e isso e parte do criterio

O BF16 tem 42,97 GiB. `safe_open` cobra **2x o arquivo em commit** e o limite desta maquina e
**98,72 GiB** com ~70 ja comprometidos: abrir o BF16 pelo leitor normal do ComfyUI e a assinatura
de `access violation` em `torch/storage.py` que ja matou o servidor quatro vezes. Entao o A/B
contra BF16 exige o caminho GGUF sem perda (`tools/safetensors_to_gguf_bf16.py`), que foi escrito
para um arquivo **so-transformer** e aqui teria de lidar com um checkpoint empacotado.

Por isso o teste vem em duas camadas, e a primeira **ja decide**:

    camada 1  estrutura + despacho + uma renderizacao do W4A8 sozinho   -> decide se publica
    camada 2  A/B contra BF16 por GGUF, frames e audio                  -> prova para o card

**Nao coberto de proposito nesta rodada:** nenhuma comparacao com o fp8mixed do S1LV3RC01N nem
com o GGUF do `vantagewithai`; nenhuma LoRA DMD aplicada; nenhum numero de s/passo (vem da barra
de progresso do sampler, nunca do JSON do `ltx_video.py`).
