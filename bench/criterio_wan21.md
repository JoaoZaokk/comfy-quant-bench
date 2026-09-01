# Criterio previo — Wan 2.1 VACE 1.3B, terceira familia

Escrito **antes** de rodar `quant_mixed.py` contra este modelo. Nada abaixo foi ajustado depois de
olhar o resultado; o que mudar depois disso entra como paragrafo novo, datado, nao como edicao.

Data: 2026-09-01. Bancada: 3090, torch 2.13.0+cu130, comfy-kitchen 0.2.31.

## O que esta sendo testado

A regra publicada em `W4A4_PROGRESS.md` parte 39, derivada de **duas** familias:

    mediana err_w4a4 > 0,21   ->  W4A4 puro quebra
    mediana err_w4a4 < 0,15   ->  W4A4 puro funciona
    entre os dois             ->  correta, granulada

Pontos que a sustentam hoje:

| modelo | familia | mediana err_w4a4 | desfecho |
|---|---|---|---|
| Z-Image v2 (Beyond Reality) | Z-Image / Lumina | 0,1241 | boa |
| capybara | Z-Image / Lumina | 0,2163 | quebrou (previsto antes) |
| HunyuanVideo 1.5 720p T2V | Hunyuan | 0,2136 | destruida |

Sao **duas** familias, e a critica externa que motivou esta rodada esta certa: com N=2 a regra pode
ser uma coincidencia entre dois modelos, nao uma propriedade do formato. Wan 2.1 e a **terceira**
familia, e o unico ponto barato que a arrisca em vez de confirmar dentro de casa — perfil
`wan_2_1` ja existe, o checkpoint ja esta no disco (4,31 GiB, fp16), e a calibragem de 2026-08-19
levou 8,26 s de amostragem.

## Previsao

**Mediana `err_w4a4` acima de 0,21. W4A4 puro quebra.**

Duas razoes, e as duas sao argumento, nao medicao:

1. **Tamanho.** 1,3 B de parametros contra ~6 B do Z-Image e ~13 B do Hunyuan. Menos redundancia
   por peso, entao cada peso carrega mais informacao e 4 bits deveriam doer mais.
2. **Video.** O unico modelo de video ja medido (Hunyuan) ficou em 0,2136.

Confianca: moderada. A razao 1 e um argumento de capacidade que esta bancada **nunca mediu**, e a
razao 2 e um n=1 disfarcado de tendencia.

## O que cada desfecho decide

| mediana medida | leitura |
|---|---|
| **> 0,21** | previsao acerta. Regra sobrevive em 3 familias. Ainda nao prova que "video quebra" — Wan tambem e o menor modelo, e os dois eixos estao confundidos. |
| **0,15 a 0,21** | **o caso mais informativo**, porque e a faixa que nenhum modelo ocupou ainda. A regra preve "correta, granulada"; se a imagem sair correta e granulada, a regra passa no seu unico teste de meio. Se sair destruida, o piso de 0,21 cai. |
| **< 0,15** | previsao erra. "Video quebra" morre na hora (Wan e video e funciona), e a leitura por tamanho tambem. A regra sobrevive so como numero por modelo, sem mecanismo. |

Em **qualquer** dos tres a regra so vira afirmacao depois de uma renderizacao real. A mediana
prediz; a imagem decide. Foi assim que o capybara fechou.

## Como vai ser medido

A calibragem existente (`calib/xfer_wan21_vace.calib.pt`, 2026-08-19) **nao serve**: e anterior a
2026-08-22 e nao carrega as chaves de proveniencia, e `quant_mixed.py` recusa por isso, de
proposito. Recalibrar, e a recalibragem entra no registro como parte do custo.

    python_embeded\python.exe -s tools/calibrate_activations.py \
        --model wan2.1_vace_1.3B_fp16.safetensors --profile wan_2_1 ...

    python_embeded\python.exe -s tools/quant_mixed.py \
        --input ...wan2.1_vace_1.3B_fp16.safetensors --calibration calib/wan21_2026-09-01.calib.pt \
        --dry-run

## Nao coberto por este criterio

Uma semente por enquanto, um tamanho, um numero de quadros. O ramo `vace_blocks` fica **fora** do
perfil por construcao (so roda com entrada de controle, entao uma calibragem sem controle nao
gravaria nada nele) — entao isto mede o tronco do Wan 2.1, nao o caminho VACE. E o checkpoint e
**fp16**, nao bf16: o mesmo eixo que quebrou o capybara com `mat1 and mat2 must have the same
dtype` ate passar `--bf16-unet`.

---

# Resultado, 2026-09-01

Escrito depois de medir. Nada acima foi editado.

## A previsao errou, e errou para o lado mais informativo

**Mediana `err_w4a4` = 0,1602.** Previ acima de 0,21. Caiu na faixa do meio, que o criterio
chamou de "o caso mais informativo, porque e a faixa que nenhum modelo ocupou ainda". A regra
prevê ali **correta, granulada**.

A imagem diz outra coisa. Tres sementes, 25 passos, 33 quadros, 480x480, `vace_strength 0`:

    referencia FP16          oficina nitida, pessoa na bancada, janela, engrenagens
    W4A4 puro (0,1602)       borrao marrom, sem sujeito, sem bancada -- DESTRUIDA

Nao e granulacao. E perda de estrutura. **A faixa do meio nao se sustenta neste modelo.**

## E o piso e muito mais baixo aqui do que nos outros dois

| build | erro efetivo mediano | 4 bits / 8 bits | GiB | resultado |
|---|---|---|---|---|
| quase W4A8 (`--promote-error 0,05`) | 0,0546 | 2 / 298 | 2,15 | **correta** |
| misto (`--promote-error 0,15`) | 0,0793 | 134 / 166 | 2,11 | estrutura volta, tudo borrado, inutilizavel |
| W4A4 puro | 0,1602 | 300 / 0 | 2,07 | destruida |

A linha do Wan 2.1 fica **entre 0,0546 e 0,0793**. O misto em 0,0793 ja e pior que o Z-Image em
0,1241, que sai bom.

## O que isso faz com a regra

| modelo | parametros | erro efetivo tolerado | fonte |
|---|---|---|---|
| Wan 2.1 VACE | 1,3 B | 0,0546 correta, 0,0793 nao | aqui |
| Z-Image v2 | ~6 B | 0,1241 correta, 0,2163 nao (capybara) | partes 38-40 |
| HunyuanVideo 1.5 | ~13 B | 0,1837 correta granulada, 0,2147 nao | parte 39 |

**Nao ha um limiar do formato.** Ha um limiar por modelo, e nestes tres pontos ele cresce
monotonicamente com o tamanho do modelo -- 2,4x a 3,4x entre o menor e o maior. A leitura por
capacidade, que entrou aqui como argumento nao medido para prever ">0,21", sobrevive numa forma
diferente da que eu previ: nao "modelo pequeno tem erro por camada maior" (o Wan tem o MENOR dos
tres), e sim **"modelo pequeno aguenta menos erro por camada"**.

Isso e exatamente a critica externa que motivou a rodada, agora medida em vez de argumentada:
0,15-0,21 nao e limite universal. Com N=3 a monotonia em tamanho tambem nao e -- e uma hipotese
com tres pontos, e o proximo modelo pode derruba-la do mesmo jeito que este derrubou a anterior.

## Um achado que nao e de quantizacao, e custou quatro renderizacoes

A referencia FP16 saiu destruida nas tres primeiras configuracoes (6 passos/1 quadro; 25/33;
cfg 6 e cfg 1; com e sem `ModelSamplingSD3 shift 8`, que aplica mas nao muda sigma nenhum no
scheduler `simple`). Quando o braco **nao quantizado** tambem quebra, nao ha nada de quantizacao
no resultado -- e o `divergence 1,2365` da primeira rodada nao media qualidade nenhuma.

Causa, com um eixo variado e o resto fixo (`scratchpad/probe_vace_strength.py`):

    vace_strength 1.0 (default do ComfyUI)   |latente| 607,6    trama tecida
    vace_strength 0.0                        |latente| 1543,2   oficina, pessoa, cena

`WAN21_Vace.extra_conds` (`comfy/model_base.py:1710-1737`) preenche `vace_frames` com zeros quando
nao ha no VACE, passa cada bloco por `process_latent_in` -- que subtrai a media do formato latente,
entao **zero vira um valor nao nulo** --, concatena uma mascara toda de UNS e aplica com forca
**1,0**. Um checkpoint VACE num workflow T2V comum sai destruido, sem erro, sem aviso. Agora e
`--vace-strength` no `quality_ladder.py`.

## Confirmado de passagem

- **Dispatch real**, nao so metadado: `probe_quant_dispatch.py --forward-only` conta 300 modulos
  quantizados, 12/12 forwards com matematica quantizada, 0 dequantize, `linear_dtype int4`,
  `comfy_kitchen.backends.cuda`. O `WARNING: unet unexpected [...comfy_quant]` que aparece no load
  e cosmetico -- os tensores sao consumidos antes e reclamados depois.
- **Razao a4/a8 = 2,932**, contra 2,996 / 3,021 / 3,202 das outras familias. Terceira familia
  independente, espalhamento 9% enquanto o erro absoluto varia 78%: o custo de descer a ativacao
  de 8 para 4 bits e quase constante, e o que muda entre modelos e o peso.
- **Velocidade muda de sinal com o tamanho do lote**, de novo. 1 quadro: 0,204 contra 0,363 s/passo,
  W4A4 **1,78x mais lento**. 33 quadros: 0,870 contra 0,653, W4A4 **1,33x mais rapido**. A mesma
  forma da curva `m_crossover`, agora num modelo de video.

## Nao coberto

Um prompt, tres sementes, 480x480, 33 quadros, um scheduler, uma placa. O checkpoint e **fp16** e
e uma variante **VACE** rodada como T2V comum, que nao e o uso para o qual foi treinada -- entao a
tolerancia medida aqui pode ser do modo, nao do modelo. Os `vace_blocks` ficam fora do perfil por
construcao e permaneceram fp16 (com forca 0 nao contribuem, entao nao houve carona). Nenhuma
metrica perceptual: "correta" e "borrada" sao julgamento de quem olhou. E a monotonia em tamanho
tem tres pontos.

---

# Correcao, 2026-09-01 (mesmo dia)

A tabela de tres familias acima diz `Z-Image v2 | 0,1241 correta, 0,2163 nao (capybara)`, e a
tabela de pontos iniciais lista `capybara | Z-Image / Lumina`. **As duas estao erradas.**

`capybara_v0.1` nao e um checkpoint Z-Image. Lido do arquivo:

    capybara_v0.1              1364 tensores   54 double_blocks   byt5_in, final_layer
    hunyuanvideo1.5_720p       1361 tensores   54 double_blocks   byt5_in, final_layer
    beyond-reality-zimage-v2    453 tensores    0 double_blocks   layers, cap_embedder

E por tamanho: 16 653 435 264 bytes contra 16 653 368 128 do Hunyuan, 67 KiB de diferenca. Os
432 modulos que o perfil `hunyuan_video_15` casou nele ja diziam isso e ninguem leu.

Consequencias, e a segunda e a que importa:

1. O 0,2163 passa para a linha de ~13 B, onde vira a **segunda** quebra independente medida nessa
   arquitetura e concorda com o 0,2147 do proprio Hunyuan. A evidencia nao se perde.
2. **O teto do Z-Image nunca foi medido.** Sabe-se que 0,1241 funciona. Nada acima disso foi
   tentado nele. A celula fica vazia em vez de preenchida com o numero de outro modelo.

A monotonia na coluna do *tolerado* -- 0,0546 < 0,1241 < 0,1837 -- nao depende da celula errada e
sobrevive. O que some e a largura da faixa do Z-Image.

Corrigido no mesmo dia nos tres cards do HuggingFace, no README publico, no `CLAUDE.md` e aqui.
