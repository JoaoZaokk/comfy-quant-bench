# Os tres packs do Bonsai Image, abertos. 6 previsoes, 6 confirmadas, o controle falhou como devia

Criterio em `bench/criterio_bonsai_packs_2026-09-22.md`, escrito **antes de qualquer numero** (e
corrigido duas vezes, tambem antes de qualquer numero, porque a premissa da P2/P3 estava errada: eu
nunca desempacotei os bits do gemlite, so contei bytes dele).

Baixados so os transformers, 3,26 GiB em vez de 10,62 -- text encoder e VAE de cada repo ficaram de
fora de proposito, ja conferidos por tamanho contra os unpacked.

---

## 1. P1 CONFIRMADA no digito: o pack binario custa 1,5000 bit/peso

Contabilidade sobre as 100 camadas quantizadas apenas, denominador vindo do `orig_shape` do proprio
pack, nao estimado (`tools/probe_bonsai_bits_por_peso.py`):

    gemlite int2 (ternario)   3.680.501.760 parametros
      W_q            920.125.440 B    2,0000 bit/peso     slot de 2 bits, 4 pesos por byte
      scales         115.015.680 B    0,2500              fp32, grupo 128
      zeros          115.015.680 B    0,2500              fp32, grupo 128
      TOTAL        1.150.162.400 B    2,5000

    gemlite int1 (binario)    3.680.501.760 parametros
      W_q            460.062.720 B    1,0000 bit/peso     slot de 1 bit, 8 pesos por byte
      scales         115.015.680 B    0,2500
      zeros          115.015.680 B    0,2500
      TOTAL          690.099.680 B    1,5000

`1,0 + 32/128 + 32/128 = 1,5000`, exato. **A alternativa que eu tinha nomeado como interessante --
1,2500, o formato largando a tabela de zero no caso binario -- esta refutada: a tabela de zero esta
la, em fp32, mesmo num codigo de dois niveis que nao precisa dela.** O gemlite tem layout unico por
largura de slot, e a minha correcao dos 2,5000 generaliza.

## 2. O que eu NAO tinha previsto: o mesmo modelo custa 2,25 no MLX, e nenhum dos dois e 1,71

O pack MLX carrega `scales` e `biases` em **BF16**, nao fp32:

    pack                       slot   tabelas   TOTAL bit/peso
    gemlite int2              2,0000   0,5000       2,5000
    MLX 2bit                  2,0000   0,2500       2,2500
    gemlite int1              1,0000   0,5000       1,5000
    MLX 1bit                  1,0000   0,2500       1,2500
    README deles                                    1,7100  (log2(3) + 16/128)

**O overhead e escolha da empacotadora, nao do formato** -- 0,25 contra 0,50 bit/peso pela mesma
informacao, e o arquivo ternario mais barato que existe (MLX, 2,2500) ainda esta **32% acima** do que
o README anuncia. O unico numero publicado que chega perto de 1,71 e o **1,2500 do MLX binario**, que
e outro modelo, com GenEval 0,671 em vez de 0,723.

E o motivo do slot de 2 bits para tres niveis fica a vista: **o codigo usa 3 dos 4 valores**, medido.
Um quarto do espaco de slot nao e usado, e e por isso que `log2(3) = 1,585` nao aparece em disco.

## 3. P2 e P3 CONFIRMADAS BIT A BIT: as tres distribuicoes sao o mesmo modelo

`tools/probe_bonsai_mlx_vs_unpacked.py`. Layout do MLX **lido** da convencao publicada do
`mx.quantize` e confirmado pelos shapes (`3072/192 = 16` valores por palavra no pack de 2 bits,
`3072/96 = 32` no de 1 bit, `3072/24 = 128` no grupo), nunca ajustado ao resultado:

    w = scales[n, k//128] * codigo + biases[n, k//128]

    braco                              rel-L2      elementos identicos   niveis   camadas
    MLX 2bit  vs  unpacked ternario   0,000e+00        1,000000            3        8/8
    MLX 1bit  vs  unpacked binario    0,000e+00        1,000000            2        8/8

**Zero. Nao "perto de zero".** Acertou na primeira tentativa, sem tocar no layout. Ninguem
requantizou por pack: unpacked, gemlite e MLX sao views do mesmo peso treinado.

### O controle falhou como tinha de falhar

MLX **ternario** contra unpacked **BINARIO** -- modelos diferentes:

    rel-L2  5,8909e-01     elementos identicos  0,010287

1,03% de coincidencia contra 100,0000% no braco certo. A comparacao nao concorda com tudo.

## 4. P5 CONFIRMADA: "binario" nao mente

    transformer_blocks.0.ff.linear_in, primeiro grupo de 128:
      2 valores distintos:  -0,017822265625  e  +0,017822265625
      soma dos dois = 0,000e+00   -> simetrico +-s/2
      zeros na camada inteira: 0 de 56.623.104

Dois niveis, simetricos, **zero nunca ocorre**. Nao e ternario esparso disfarcado.

## 5. P4 CONFIRMADA, e a minha ferramenta errou TRES vezes antes de chegar la

Grupo **128 no eixo K, em 100 de 100 camadas**, nos dois packs gemlite, e 128 tambem no MLX.

Levou tres versoes da mesma funcao de 6 linhas, e vale registrar porque o modo de falha foi o mesmo
das outras duas vezes hoje -- **remendar a heuristica em vez de olhar o dado cru**:

1. assumi que o eixo empacotado era o 0. Saiu `{1: 30, 36: 20, 4: 40, 24: 10}` pesos por byte e grupos
   `{32, 1152, 128, 768, 42}`. **36 pesos num byte e impossivel** e devia ter me parado na hora.
2. procurei o eixo que "casava". Devolveu `-1` em **60 das 100** camadas, e de forma assimetrica
   entre os dois packs -- que e sinal de que o problema nao era o caso, era o modelo.
3. imprimi os shapes crus. **O pack TRANSPOE.** `orig_shape` segue a convencao do Linear,
   `(N_saida, K_entrada)`; o pack grava `(K/r, N)`. Um exemplo basta para ver:
   `orig=(27648, 3072) -> W_q=(768, 27648)`, que e `(3072/4, 27648)`. Cinco padroes distintos, todos
   obedecendo a mesma regra.

O conserto nao foi um caso especial: foi trocar adivinhacao por leitura, e a funcao devolve `-1`
quando N nao casa, em vez de produzir um numero plausivel.

## 6. P6 CONFIRMADA, com um achado de brinde que fecha o confound de ontem

    densos BF16 no pack MLX ternario                          69
      identicos ao unpacked ternario                        69/69   (esperado)
      DIFERENTES do MLX binario                             68/69   (esperado: modelos diferentes)

A excecao e uma so, e e a que importa:

    proj_out.weight  (128, 3072)   identico nos TRES: ternario, binario, e o FLUX.2-klein-4B ORIGINAL

Ontem eu medi isso em dois arquivos; agora sao **quatro**, incluindo duas empacotadoras que nunca se
falaram. **Nenhum passo global toca todos os pesos.** Se houvesse rotacao, rescale global ou
reescrita de tudo, `proj_out` nao sobreviveria byte a byte em tres modelos diferentes -- e sobrevive.
Isso ancora os dois lados: o arquivo e derivado daquele original exato, e o que mudou nos outros
pesos mudou **por escolha**, nao por transformacao de pipeline.

## 7. A parte que fica pior com os packs abertos: o denso domina

    arquivo                 quantizado      denso BF16     denso como % do arquivo
    gemlite int2         1.150.162.400   390.085.632            25,33%
    gemlite int1           690.099.680   390.085.632            36,11%

Os 69 tensores densos nao mudam de tamanho quando o slot encolhe. **No pack binario, mais de um terco
do arquivo sao as camadas que eles chamam de "menos de 5% dos parametros".** Empurrar o slot de 2
para 1 bit corta 40% do arquivo, nao 50%, e o proximo bit corta menos ainda -- o piso nao e o
formato, e a parte que ficou em alta precisao de proposito.

---

## Nao coberto

- **Nenhuma imagem, nenhuma ativacao, nenhum kernel, GPU nunca tocada.** Tudo aqui e byte e codigo.
- **O caminho de execucao do gemlite segue nao medido.** Este trabalho diz quanto o arquivo GASTA,
  nao se o kernel faz GEMM na largura do slot ou desempacota antes. gemlite nao esta instalado.
- P2/P3 em **8 de 100 camadas** por pack, amostradas com passo espalhado, nao no arquivo todo.
- P5 conferida numa camada, nao nas 100.
- Os text encoders (`hqq-4bit` no gemlite, `mlx-4bit` no MLX) e os VAEs **nao foram abertos**: sao
  2,1 GB por repo e nao respondem nenhuma pergunta deste criterio.
- As metricas publicadas deles (GenEval, HPSv3, DPG) seguem **nao replicadas**, e nada aqui tentou.
