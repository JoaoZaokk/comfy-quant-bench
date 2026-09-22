# Bonsai Image 4B: engenharia reversa, MEDIDA (2026-09-21)

Pergunta do dono: *"projeto original, projeto bonsai, o que foi feito, e como podemos entender o que
faz uma promocao/democao de values no tamanho faz o modelo se comportar bem desse jeito."*

**Resposta curta: nao e promocao nem democao. Eles TREINARAM o modelo com a grade ternaria dentro do
laco, e as camadas deixadas em FP16 nao sao as "sensiveis" -- sao a capacidade de ADAPTACAO que
absorve o dano.** Abaixo, como isso foi medido.

Correcao de premissa feita no comeco: **nao e Flux Schnell, e FLUX.2-klein-4B.** Schnell aparece so
como linha de comparacao na tabela deles (23,8 GB, GenEval 0,716).

---

## 1. O que existe, e o que e alinhavel

Seis repos `prism-ml/bonsai-image-*`, 18 a 21/05/2026 (primeiro commit do git: **2026-05-26**),
`diffusers:Flux2KleinPipeline`, apache-2.0. `binary` e `ternary`, cada um em `unpacked` (safetensors
densos), `gemlite-Nbit` (CUDA) e `mlx-Nbit` (Apple).

    arquivo                                                        bytes            
    black-forest-labs/FLUX.2-klein-4B  transformer/...safetensors   7.751.109.744   original
    bonsai-image-ternary-4B-unpacked   idem                         7.751.109.712   ternario
    bonsai-image-binary-4B-unpacked    idem                         7.751.109.712   binario
    bonsai-image-ternary-4B-gemlite-2bit  .../state_dict.pt         1.540.457.482   pack de deploy

**32 bytes de diferenca entre os transformers**, 169 tensores em cada, **zero chave a mais ou a
menos**, mesmos shapes, todos BF16. Por isso o diff por camada e direto. O sha256 do ternario
(`fa5fd182...cf4e35964`) **confere com o `manifest.json` publicado por eles**, entao o download foi
conferido por prova positiva, nao por tamanho. VAE e text encoder tem o MESMO tamanho do original nos
dois repos unpacked: eles so tocaram o transformer.

**O confounder que mataria a conclusao esta morto.** Se a BFL tivesse revisado o original depois de
maio, o "drift" que eu medi poderia ser revisao deles, nao treino do Bonsai. Historico de commits do
`FLUX.2-klein-4B`: pesos em **2026-01-15** (`add diffusers weights`), e o unico commit posterior e
**2026-02-24 `Update README.md`**. Quatro meses antes do Bonsai. Confirmado por um segundo caminho:
`proj_out` sai **byte a byte identico** entre original e Bonsai nos dois bracos -- se a base tivesse
mudado, ele teria mudado tambem.

---

## 2. O mapa de promocao/democao: publicado, e sem criterio nenhum

`bonsai-image-ternary-4B-gemlite-2bit/transformer-gemlite-int2/quantization_config.json`, 6042 B:

    format gemlite-int2-ternary-g128   bits 2   group_size 128   solver ternary
    quantized_count 100    skipped_count 9    pack_seconds 150,48

    QUANTIZADAS (100), todas no MESMO tratamento:
      transformer_blocks.0..4  (5 double-stream) x 12
        attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_add_out,to_out.0}
        ff.{linear_in,linear_out}   ff_context.{linear_in,linear_out}
      single_transformer_blocks.0..19 (20 single) x 2
        attn.to_qkv_mlp_proj   attn.to_out

    PULADAS (9), por `skip_patterns` -- lista escrita a mao:
      x_embedder  context_embedder  proj_out  norm_out.linear
      time_guidance_embed.timestep_embedder.linear_{1,2}
      double_stream_modulation_{img,txt}.linear   single_stream_modulation.linear

**Achado: nao existe erro medido por camada na receita deles.** `skip_patterns` casa por NOME, e as
100 escolhidas recebem formato, group size e escala identicos. E a mesma forma do nosso
`PROFILE_PATTERNS` (`tools/quant_w4a8.py:54`). Esta bancada construiu `quant_mixed.py` para escolher
formato por camada medindo em ativacao real; a referencia publica de 1,58 bit nao faz isso, e nao
precisa -- pelo motivo da secao 4.

**Achado: eles pulam a modulacao, que esta bancada mediu como a camada MAIS FACIL.** Os tres
`*_modulation.linear` estao entre os pulados. O `CLAUDE.md` deste repo registra, medido:
`adaLN_modulation` e a camada de **menor** erro em W4A4 de todo o bloco -- **0,1263 contra 0,1569**
das demais -- e registra que "ninguem poe modulacao em baixa precisao, logo nao se deve" era consenso
usado como evidencia. A secao 4 explica por que os dois fatos convivem sem se contradizer.

---

## 3. O teste decisivo: PTQ ou treino?

`tools/probe_bonsai_ptq_ou_treino.py`. Le por **mmap somente-leitura**, nao `safe_open`, porque nesta
maquina `safe_open` cobra 2x o arquivo em commit e dois arquivos de 7,22 GiB estourariam o limite de
98,72 GiB.

O teste e **receita-agnostico**, e ai esta o valor dele. Um PTQ de magnitude -- absmean, absmax,
Lloyd-Max, HQQ, qualquer um -- mapeia `w -> s * round(clip(w/d, -1, 1))`. Disso seguem duas
propriedades que valem por CONSTRUCAO, sem saber o limiar deles:

1. **Nao pode inverter um sinal.** O sinal da saida e o sinal de `w`, ou zero.
2. **Nao pode desordenar magnitudes dentro do grupo.** Todo peso com codigo 0 tem `|w|` menor que
   todo peso com codigo +-1. Separabilidade perfeita, AUC = 1,000000.

### Braco ternario, 100 camadas

    mediana fracao de zeros                     0,3318
    mediana magnitudes nao-nulas por grupo K      1,00   <- ternario g128 VERDADEIRO
    idem no eixo N                               57      <- confirma que o grupo e no eixo K
    mediana concordancia de sinal               0,9999   (PTQ exige 1,000000)
    mediana cosseno com o original              0,8760
    mediana AUC |orig| preve codigo != 0        0,9826   (PTQ exige 1,000000)
    camadas com ZERO inversao de sinal            0/100  (PTQ puro exige 100/100)
    inversoes de sinal, total             2.727.589 de 2.355.364.875 posicoes nao-nulas (0,1158%)

**Nao e 1. Logo nao e PTQ deste original.** E a prova nao e a mediana, e a contagem: **2.727.589
inversoes de sinal**, e **nenhuma das 100 camadas** esta limpa. Um PTQ de magnitude produz zero
inversoes, em todas as camadas, sempre. A distancia de 1 tambem nao e ruido de arredondamento: ela
cresce monotonicamente com a profundidade, o que ruido nao faz.

    single_transformer_blocks.N.attn.to_qkv_mlp_proj
    bloco     0      3      5      8     11     14     17     19
    zeros  0,347  0,340  0,340  0,357  0,374  0,411  0,440  0,409
    sinal  0,9998 0,9996 0,9979 0,9990 0,9982 0,9967 0,9951 0,9957
    AUC    0,9816 0,9785 0,9683 0,9706 0,9634 0,9476 0,9370 0,9503

### Braco binario, 100 camadas -- a prova mais limpa

    mediana fracao de zeros                     0,0000   <- binario de verdade, sem estado zero
    mediana magnitudes nao-nulas por grupo K      1,00
    mediana concordancia de sinal               0,9394
    mediana cosseno com o original              0,7744
    AUC                                            nan   <- indefinida, e corretamente: sem zeros nao
                                                            existe contraste "codigo 0 vs +-1"

**Um PTQ binario E `sign(w) * escala`. A concordancia de sinal teria de ser 1,000000 por definicao
do formato.** Medido **0,9394**: cerca de **6,06% dos 3,68 bilhoes de pesos tiveram o sinal
trocado** em relacao ao original. Nenhuma quantizacao pos-treino pode fazer isso. **Os pesos foram
treinados.**

### Veredito

**Treinado, partindo deste original, com a grade dentro do laco.** Nao e PTQ (ha inversoes de sinal,
impossiveis sob quantizacao) e nao e treino do zero (sinal 0,94 a 0,9999 contra 0,50 de um treino sem
essa inicializacao; cosseno 0,77 a 0,88). E **QAT com peso latente**, que e exatamente o que o
`manifest.json` deles diz em uma linha: `"model_version": "ternary g128 (bf16 master)"`. O
`transformer/config.json` deles tambem tem tres chaves que o original nao tem -- `enable_time_sign_embed`,
**`musubi_block_swap_device`** e **`musubi_blocks_to_swap`** -- e `musubi-tuner` e ferramenta de
fine-tuning com block-swap para VRAM curta. Configuracao de treino vazada no artefato de release.

**Correcao de ferramenta feita aqui:** a primeira versao deste probe classificou 0,9999/0,9826 como
"PTQ", porque eu escrevi o limiar como `> 0,98`. O limiar certo nao e "perto de 1", e **igual a 1** --
ler "quase 1" como "1" e exatamente o erro que a prova exclui. Consertado na ferramenta, com a
contagem absoluta de inversoes impressa junto, nao so a fracao.

---

## 4. O mecanismo: as camadas em FP16 sao CAPACIDADE DE ADAPTACAO, nao camadas sensiveis

Aqui esta a resposta a pergunta dele, e ela sai de um controle que **falhou** -- do jeito informativo.

O controle era: se for "PTQ com allowlist", as 9 camadas declaradas puladas tem de sair byte a byte
identicas ao original. Medido:

    camada                                          ternario    binario   veredito
    proj_out                                         0,00e+00   0,00e+00  IDENTICO nos dois
    x_embedder                                       6,851e-02  7,986e-02 MUDOU
    context_embedder                                 2,347e-01  2,550e-01 MUDOU
    time_guidance_embed...linear_1                   1,625e-01  1,492e-01 MUDOU
    time_guidance_embed...linear_2                   1,043e-01  1,074e-01 MUDOU
    double_stream_modulation_img.linear              4,879e-02  4,976e-02 MUDOU
    double_stream_modulation_txt.linear              4,361e-02  4,531e-02 MUDOU
    single_stream_modulation.linear                  1,481e-02  1,421e-02 MUDOU
    norm_out.linear                                  1,996e-02  2,402e-02 MUDOU
    (rel-L2 contra o original)

    e os 60 tensores fora das DUAS listas (norm_q / norm_k de cada bloco):  0/60 identicos

**8 das 9 "puladas" mudaram, e todas as 60 normas tambem.** "Pulada" no config deles significa **nao
quantizada**, nao **nao modificada**. Essas camadas ficaram em FP16 e foram TREINADAS.

E o dano das camadas ternarias e maior do que o desvio do proprio codigo sugere: `context_embedder`
se move **rel-L2 0,235**, cinco vezes mais do que a maior discordancia de codigo medida no braco
ternario.

**Isso reinterpreta o mapa inteiro.** As 9 camadas em FP16 nao foram escolhidas por serem sensiveis a
quantizacao. Elas ficaram fora da grade para poderem **se mover livremente** e compensar as 100 que
foram esmagadas. Sao os graus de liberdade do treino. Por isso a receita deles nao precisa de criterio
por camada: a escolha nao e "qual camada aguenta 2 bits", e **"qual camada eu deixo solta para
consertar o resto"** -- e para isso serve qualquer conjunto pequeno que toque todo o fluxo
(entrada, saida, tempo, modulacao, normas).

**E a colisao com a modulacao se resolve sem nenhum dos dois lados estar errado.** Esta bancada mediu
que a modulacao *aguenta* 4 bits melhor que o resto (0,1263 contra 0,1569) -- isso segue verdadeiro.
O Bonsai nao a pula por ser fragil: a pula por ser **barata e bem posicionada** para absorver
correcao (`single_stream_modulation.linear` sozinha alimenta 20 blocos). Aguentar quantizacao e ser
util como compensador sao propriedades diferentes, e o mapa deles otimiza a segunda.

**Ha dose-resposta, mas ela e tendencia e nao lei: 6 das 8 camadas mudadas se movem MAIS no braco
binario que no ternario** -- quantizacao mais dura, compensacao maior. As duas excecoes sao
`time_guidance_embed.linear_1` e `single_stream_modulation.linear`, que se movem menos. Com n=8 isso e
um indicio, nao um resultado.

### O que isso significa para esta bancada

Todo o esforco daqui -- `quant_mixed.py`, erro por camada em ativacao real, `--promote-error`, os
criterios escritos antes de medir -- responde **"como escolher o formato de cada camada sem
treinar"**. Bonsai nao responde essa pergunta: ele **treina**. As tres hipoteses de transferencia
que morreram aqui (monotonia por tamanho, razao entre groupsizes, erro mediano x numero de camadas) e
a conclusao de que o criterio por camada "ordena formatos e nao localiza penhascos" seguem validas
para PTQ -- e nao sao comparaveis ao que o Bonsai faz. **O teto do PTQ nao e o teto deles**, e por
isso 1,58 bit funciona la e W4A4 quebra aqui em varias familias.

[JULGAMENTO] Isso sugere que o proximo ganho real nesta bancada nao vem de um criterio de promocao
melhor, e sim de um **estagio de compensacao** -- deixar um conjunto pequeno de camadas em alta
precisao e ajusta-las contra a saida do modelo denso, sem retreinar o corpo. O que me faria mudar de
ideia: se o Bonsai tiver treinado com dataset e escala de pre-treino (nao um ajuste curto), o
mecanismo e "retreinar", nao "compensar", e nada disso cabe num orcamento de bancada. O sinal a favor
do ajuste CURTO e que os pesos ternarios ficaram tao perto do original (sinal 0,9999) -- um treino
longo os teria movido muito mais.

---

## 5. O pack de deploy: onde os numeros publicados nao fecham

`state_dict.pt`, 1.540.457.482 B, lido com `torch.load(mmap=True, weights_only=True)` --
`weights_only` porque e pickle de terceiro e nao se executa codigo de terceiro para ler peso.

    sufixo          n        GiB   % do arquivo   dtype      shape de exemplo
    W_q           100     0,8569        59,73%   uint8      (768, 27648)
    weight         69     0,3633        25,32%   bfloat16   (3072, 3072)
    scales        100     0,1071         7,47%   float32    (24, 27648)
    zeros         100     0,1071         7,47%   float32    (24, 27648)
    metadata      100     ~0             0,00%   int32      (12,)
    orig_shape    100     ~0             0,00%   int32      (2,)
    overhead do container    0,0002       0,01%

**O grupo de 128 no eixo K, confirmado por dois caminhos independentes:** o peso desempacotado da
**1,00 magnitude nao-nula por grupo de 128 em 100,00% dos grupos** no eixo K (e 57 no eixo N, que
portanto nao e o eixo); e no pack `scales` tem shape `[24, 3072]` contra `orig_shape` 3072, e
3072/24 = 128.

**Ele e ASSIMETRICO no container, simetrico na matematica.** Ha `zeros` alem de `scales`, e
`zeros = -scales` exatamente (min/max espelhados: scales 0,004639 a 0,05078; zeros -0,05078 a
-0,004639). Ou seja o afim mapeia codigo 0 -> -s, 1 -> 0, 2 -> +s: e ternario, expresso pela
interface afim do gemlite. A matematica do README esta certa; o CUSTO nao.

**Bits por peso: o arquivo diz 2,5000, o README diz 1,71.** Sobre os 3.680.501.760 parametros das
100 camadas:

    W_q                920.125.440 B   2,0000 bits/peso   <- log2(3)=1,585 guardado num slot de 2 bits
    scales fp32        115.015.680 B   0,2500                 (eles mesmos dizem que o 4o codigo sobra)
    zeros  fp32        115.015.680 B   0,2500
    ------------------------------------------------
    real             1.150.156.800 B   2,5000 bits/peso
    alegado no README                  1,7100  (log2(3) + 16/128)
    razao                               1,462x

A diferenca e inteiramente explicavel e nao e erro de medicao: eles contam 1,585 bits de *informacao*
onde o arquivo gasta 2 bits de *armazenamento*, e contam **uma** escala **fp16** onde o arquivo tem
**duas** tabelas **fp32**. Trocar as duas por fp16 economizaria 0,25 bit/peso, ou 115 MB.

**E "menos de 5% dos parametros em FP16" vira 25,32% dos BYTES.** Os 69 tensores densos bf16 (as 9
puladas + as 60 normas) somam 390.085.632 B. A afirmacao pode estar certa em contagem de parametro e
e enganosa em tamanho de arquivo, que e o que o usuario baixa.

O numero a citar e o do arquivo: **1,435 GiB / 1,54 GB decimais**. O "1,21 GB" do README e a conta
idealizada de 1,71 bit/peso, nao um arquivo que exista.

`gemlite_autotune.json`: cinco chaves de topo, e as populadas sao **GEMV**, `GEMV_REVSPLITK` e
`GEMV_SPLITK`, com configs para `(1, 4096, 4096, ...)` -- **M = 1**. Eles afinaram o caminho
matriz-vetor.

[JULGAMENTO, por leitura de hardware e nao por execucao] **a matematica nao pode ser de 2 bits.** Nao
existe instrucao MMA de 2 bits em nenhuma GPU NVIDIA: a mais estreita da Ampere e `m16n8k64 s4`
(INT4), e ela saiu na Hopper. gemlite e Triton, que nao expoe MMA sub-INT4. Entao o peso e 2 bits **em
memoria** e o produto acontece em outra precisao -- o ganho e banda, nao tensor core. Isso espelha o
que esta bancada ja mediu no proprio ConvRot, onde o ramo INT8 e mais fiel e o "nativo INT4" so ganha
em vazao com M grande. **O que faria isso virar medicao:** instalar gemlite e contar o despacho, como
`probe_quant_dispatch.py` faz aqui.

---

## 6. A qualidade que eles publicam

Numeros DELES, em H100, nao replicados aqui:

    modelo                     transformer GB   GenEval   HPSv3   DPG-Bench
    FLUX.2 Klein 4B (base)           7,75        0,819    12,84     0,853
    Bonsai Ternary 4B                1,21        0,723    12,22     0,851
    Bonsai Binary 4B                 0,93        0,671    11,15     0,822
    FLUX.1-schnell                  23,8         0,716    12,67     0,848

GenEval cai **11,7%** do base para o ternario e **18,1%** para o binario; DPG-Bench quase nao se move
(0,853 -> 0,851 -> 0,822). **As tres metricas discordam sobre o tamanho do dano**, que e o mesmo
padrao que esta bancada ja mediu entre divergencia de latente e imagem final. O README chama 0,723
contra 0,819 de "muito perto"; e 11,7% menos numa metrica de seguimento de instrucao.

---

## 7. Nao coberto

- **Nada executado no modelo.** Nenhuma imagem gerada, nenhuma ativacao medida, nenhuma velocidade.
  Todo numero acima e de PESO (executado) ou de metadado publicado (lido).
- **O caminho de execucao do gemlite nao foi medido**, so inferido do hardware. gemlite nao esta
  instalado aqui.
- **As metricas deles nao foram replicadas** e nao procurei avaliacao independente de terceiro.
- O pack **binario** (INT1) e os packs **MLX** nao foram abertos.
- O **text encoder** (Qwen3-4B) e o **VAE** nao foram comparados tensor a tensor; so os tamanhos, que
  batem com o original nos repos unpacked.
- `group_size 128` veio do config DELES e depois foi confirmado por duas medicoes minhas; nao foi
  procurado por varredura cega.
- A AUC agrega grupos mistos; grupos inteiramente zero ou inteiramente nao-zero nao informam e ficaram
  fora da media. No braco binario ela e indefinida por construcao.
- **Quanto de treino** eles gastaram e desconhecido: nao ha numero de passos, dataset, ou GPU-hora em
  nenhum artefato publicado. O whitepaper nao descreve o metodo (medido antes: outline sem secao de
  metodo, `QAT` 0, `PTQ` 0, `we train` 0 em 25 paginas por pypdf 6.19.0).
