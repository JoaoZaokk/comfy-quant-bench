# O fp8 de terceiro do 10Eros v1.5: promocao MEDIDA por camada, e o resgate que e ORCAMENTO (2026-09-21)

Achado de raspao enquanto eu procurava um braco de comparacao para o nosso W4A8. `LokkenJP` publica,
ao lado do peso, um **manifest de quantizacao de 6,2 MB com uma entrada por camada** --
`10Eros_v1.5_fp8mixed_experimental_learned.safetensors.quantization.json`. E um terceiro fazendo o
mesmo trabalho, no **mesmo modelo** que acabamos de converter, com o processo de decisao inteiro
aberto. Baixado e lido; **nada foi executado**.

Isto e a outra metade da resposta a pergunta do dono sobre promocao/democao de precisao por camada. O
[Bonsai](bonsai_image_engenharia_reversa.md) nao mede camada nenhuma e **treina**. Este mede tudo e
**nao treina**. Sao duas escolas, e a bancada esta na segunda.

---

## A receita deles, dos campos do proprio manifest

    converter_version                        3.2.0 / convert_to_quant 1.3.1
    fonte                                    10Eros_v1.5_bf16.safetensors  (a MESMA base que a nossa)
    saida                                    30.235.567.135 B
    placa                                    RTX 4080
    candidate_strategy                       three_learned_ranks_plus_simple_scaled
    candidate_tensor_count                   1232
    quantized_tensor_count                   1151
    learned_tensor_count                     1060
      learned_low / mid / high                 100 / 744 / 216
    simple_fallback_count                      91
    bf16_rescue_count                          81   (todos automaticos)
    learned_candidate_evaluation_count       3453
    total_learned_candidate_convert_seconds  28748,04   = 7 h 59 min
    maximum_chosen_nrmse                     0,031519
    maximum_chosen_projected_nrmse           0,012208
    optimizer_oom_policy                     fail_closed_no_rank_reduction

**Quatro candidatos por camada**, avaliados de verdade (3453 avaliacoes para 1232 camadas), e o
vencedor escolhido por camada:

    744  learned_mid       \
    216  learned_high       >  1060 camadas com correcao aprendida de baixo rank
    100  learned_low       /
     91  simple_scaled         fp8 puro, sem correcao

    por que:  1057  learned_projected_win
                90  simple_raw_tiebreak
                 3  learned_raw_tiebreak
                 1  simple_projected_win

Uma entrada tipica carrega, entre outros campos: `selected_top_p 0.4`,
`selected_effective_k 1638`, `audit_svd_rank 64`, `learned_nrmse`, `simple_nrmse`, `chosen_nrmse`,
`learned_projected_nrmse`, `simple_projected_nrmse`, `projected_improvement`, `weight_scale`,
`zero_fraction`, `learned_seed`, `chosen_candidate_sha256` e a lista `candidate_metrics` com os
quatro candidatos. Ou seja: **cada decisao e reproduzivel a partir do arquivo publicado.**

---

## Achado 1: o "resgate para bf16" e um ORCAMENTO DE 1 GiB, nao um criterio de qualidade

O campo se chama `automatic_bf16_rescue_count: 81`, o que se le como "81 camadas fragis demais foram
promovidas de volta". Medido, e nao e isso:

    automatic_bf16_rescue_bytes     1.073.739.313
    1 GiB                           1.073.741.824
    diferenca                               2.511 bytes   = 0,000234% abaixo de 1 GiB
    soma dos 81 additional_bf16_bytes, um por um: bate exatamente com o total

Um numero que para **2.511 bytes** abaixo de 1 GiB nao e acidente. E os outros dois controles fecham:

- **A ordem e por erro, decrescente.** Os resgatados vao de `simple_nrmse` 0,026626 (o pior) a
  0,026539, e existem **exatamente 2** camadas quantizadas com erro acima desse corte (0,026541 e
  0,026540) -- que e o comportamento de um guloso por orcamento: as duas nao couberam no que sobrou.
- **Os tamanhos sao de quatro valores so** (4.194.273 / 8.388.577 / 16.777.185 / 67.108.833 bytes --
  cada um e 2^n + 33), e **38 dos 81 sao o menor**. Guloso por erro, limitado por bytes.

Entao a regra deles e: **ordene por erro, resgate ate gastar 1 GiB, pare.** O limiar nao e uma
propriedade do modelo; e onde o orcamento acabou.

## Achado 2: em fp8 o erro por camada quase nao varia, e isso esvazia o ranking

    simple_nrmse, 1232 camadas candidatas:   0,02625  a  0,02663      <- espalhamento total 1,4%
      das quantizadas                        0,02625  a  0,02654
      das resgatadas                         0,02654  a  0,02663
    chosen_nrmse (depois da correcao)        0,02625  a  0,03152      <- espalhamento 20%

O corte que separa resgatado de quantizado cai em **0,02654**, e a diferenca entre o pior quantizado e
o melhor resgatado e de **0,00001** -- uma parte em 2600. Ordenar 1232 camadas por uma metrica cujo
espalhamento total e 1,4% e depois cortar no meio dela nao localiza camada fragil: **em fp8 o erro
relativo e quase constante**, porque fp8 tem precisao relativa fixa e nao depende da distribuicao do
peso. E o contrario de int4/ternario, onde o erro depende fortemente da distribuicao -- e e por isso
que esta bancada consegue ordenar camadas em W4A4 e eles nao conseguem em fp8.

**Consequencia:** o valor da pipeline deles nao esta no resgate. Esta no Achado 3.

## Achado 3: o que `learned` significa de fato -- e a correcao de uma coisa que eu escrevi errado aqui

**Eu escrevi, na primeira versao desta secao, que `learned_low/mid/high` era uma "correcao aprendida de
baixo rank" adicionada ao peso, e que isso demonstrava o estagio de compensacao que eu havia
hipotetizado no relatorio do Bonsai horas antes. Lendo os quatro candidatos na integra, nao e isso**, e
a diferenca importa porque a leitura errada transformava um heuristico de quantizacao no achado que eu
queria encontrar. Os quatro candidatos de uma camada `[4096, 4096]`:

    nome            kind     top_p    effective_k    nrmse (peso inteiro)
    simple_scaled   simple    null       null        0,02650228973694793
    learned_low     learned   0,125       512        0,02661555195450032
    learned_mid     learned   0,4        1638        0,028136915116186684   <- ESCOLHIDO
    learned_high    learned   1,0        4096        0,02650228973694793   <- IDENTICO ao simple

Tres coisas saem disso:

1. **`top_p` e uma fracao da dimensao de ENTRADA** (`effective_k = top_p x K`: 512, 1638, 4096 de
   4096), nao um rank de um termo somado. E em `top_p 1,0` o candidato "learned" da `nrmse` **identico
   ao `simple_scaled` nos 17 digitos** -- ele degenera no simples. Um termo de correcao somado nao
   degeneraria; um objetivo ponderado por canal, com peso 1 em todos os canais, degenera exatamente
   assim. Entao `learned` e **o ajuste dos parametros de quantizacao contra um objetivo ponderado por
   canal**, nao capacidade nova no modelo.
2. **O escolhido tem erro de peso PIOR que o simples** -- 0,028137 contra 0,026502 -- e ganhou porque o
   `projected_nrmse` dele e menor (0,005870 contra 0,006044). Ou seja: aceitam mais erro total no peso
   em troca de menos erro nos 40% de canais que a projecao considera importantes.
3. **A projecao NAO usa ativacao real.** `activation_key` vem `None` e `activation_rows` vem `0` em
   **todas as 1151** entradas, e `bias_adjustment_norm` e `0,0` em todas as 1151 (coerente com
   `activation_corrected_bias_count: 0`). A importancia dos canais sai do proprio peso.

E o tamanho do arquivo confirma que nao ha termo somado: 30.235.567.135 B, que e o que se espera de
peso fp8 + escalas + 1 GiB de resgate + os 3,84 GiB de `vae`/`vocoder`/`audio_vae`/`projection`
intocados. Se guardassem uma correcao de baixo rank por camada, o arquivo seria maior.

**Entao o [JULGAMENTO] que eu fiz no relatorio do Bonsai -- que o proximo ganho vem de um estagio de
compensacao -- NAO esta demonstrado por este artefato.** Ele segue sendo julgamento. O que este arquivo
demonstra e outra coisa, menor mas real: **da para escolher parametros de quantizacao por camada
medindo, com quatro candidatos por camada, em 8 horas de uma placa que esta bancada tem.**

O custo esta publicado: **28.748 s = 7 h 59 min numa RTX 4080**, `maximum_candidate_peak_allocated_gib
3,40` -- **caberia na 3080 Ti desta maquina**, que e a placa que sobra. Nao e escala de pre-treino.

[JULGAMENTO] O que ainda vale testar daqui e a **ponderacao por canal do objetivo**, que e barata e que
esta bancada nunca tentou: todos os nossos criterios minimizam erro uniforme no peso ou medem erro em
ativacao real, e nenhum aceita mais erro nos canais pouco usados de proposito. O que me faria mudar de
ideia: o ganho deles esta numa metrica projetada que **eles proprios nunca compararam com uma imagem**
em nada que eu tenha lido -- e esta bancada ja mediu tres vezes que erro por camada ordena formatos e
nao localiza penhascos. Aceitar essa ideia sem uma renderizacao seria repetir o erro que a REGRA ZERO
descreve.

---

## O que isso muda no braco de comparacao do nosso W4A8

O arquivo deles e da **mesma fonte v1.5** (`source: E:\ltx-quant\10Eros_v1.5_bf16.safetensors`), o que
o torna comparavel com o nosso -- ao contrario do `10Eros_v1.4_DMD_int8_convrot` que eu quase usei, que
e de outra versao. Mas atencao a um confounder que sobra: **o deles tem correcao aprendida de baixo
rank e o nosso nao.** Uma comparacao W4A8-nosso contra fp8-deles mede **formato + correcao**, nao
formato. O braco limpo continua sendo o INT8 puro de mesma base
(`CornLogic/10EROS-INT8 -> 10Eros_v1.5_INT8_TFO.safetensors`), que esta baixando.

## NAO COBERTO

Nada executado: nenhum peso deles foi baixado (so o manifest de 6,2 MB, o sha256 declarado, o README e
o workflow), nenhuma imagem, nenhuma ativacao, nenhum tempo. Nao conferi o sha256 declarado contra o
arquivo, porque o arquivo nao esta aqui.

Do `candidate_metrics` eu li os quatro candidatos de UMA camada na integra e os campos escalares das
1151; nao li as 1151 listas inteiras, entao "learned_high degenera no simple" esta medido em **uma**
camada, o que e indicio forte e nao censo. O que exatamente a metrica "projetada" projeta segue nao
resolvido: sei que **nao** usa ativacao (1151/1151 com `activation_key None`, `activation_rows 0` e
`bias_adjustment_norm 0,0`) e que `effective_k = top_p x K`, mas nao sei como os canais sao ordenados,
porque o conversor deles nao e publico. `audit_svd_rank 64` aparece em toda entrada e eu **nao** sei o
que ele audita -- tratei como metrica de auditoria e nao como rank de um termo somado, apoiado no
tamanho do arquivo, que e argumento indireto. E `zero_fraction 4,6e-05` numa camada fp8 e um numero que
eu nao sei interpretar sem o codigo deles.

E o registro do erro: **a primeira versao da secao "Achado 3" afirmava que este arquivo demonstrava o
estagio de compensacao que eu havia hipotetizado horas antes no relatorio do Bonsai.** Nao demonstra. Eu
li `learned` + `audit_svd_rank 64` como "correcao de baixo rank somada ao peso" porque era o achado que
eu estava procurando. Os quatro candidatos, lidos na integra, mostram `top_p` como fracao de canais de
entrada e `learned_high` degenerando no `simple_scaled`. Corrigido acima, com o motivo a vista.
