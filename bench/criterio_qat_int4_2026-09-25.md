# Critério — professor-aluno em 4 bits (b8, b9) — escrito 25/09, antes de subir a VM

Pedido do dono: "joga pra 4 bit, verifica se professor e aluno melhora em algo". Ponto de partida medido
hoje (`bench/criterio_restauracao_grupos_2026-09-25.md`): o corpo inteiro em 4 bits RTN g32, sem treino
(C1), já sai quase idêntico ao BF16 (epsilon 0,221; render 20/20 igual ao BF16 salvo detalhe).

`tools/qat_ternario_klein.py --formato int4 --grupo 32`: código absmax/7 (-7..7) por grupo de 32 no eixo K,
escala ótima L2 dado o código, STE exato; mesmo laço de destilação dos braços ternários. Resto congelado
(`--congela-resto`, K0 obrigatório), lr 1e-5, 8-bit+SR, lote 1, holdout a cada 500, A100 do Colab
(treino na 3090 vetado).

| braço | dado | passos | por quê |
|---|---|---|---|
| b8 | 124 curados x sementes 1 2 | 2000 | mesmo dado do b6 (o que gerou atrator no ternário) |
| b9 | 1.756 (124 + Parti) x semente 1 | 2000 | pouca repetição, o do b7 |

Avaliação: K0; render da avaliação fixa (10) e da grade (5 x 11 12) contra BF16 e contra o C1 (o mesmo
4 bits sem treino); epsilon fora da amostra (registrado, não decide: mentiu em todas as rodadas de 24-25/09).

## Previsões

- **Z0:** o passo 0 (int4 RTN, escala L2) já tem holdout_rel muito abaixo do ternário ingênuo (~1,09).
- **Z1:** o holdout_rel cai com o treino (o laço sabe ajustar o aluno ao professor nos dados dele).
- **Z2 [JULGAMENTO, decide]:** no render, b8 e b9 ficam **indistinguíveis do C1** -- não há o que
  consertar num 4 bits que já é quase o BF16. Refutada para "melhora" se b8 ou b9 ficarem visivelmente
  mais perto do BF16 que o C1 em ≥ 3 das 10 células da fixa; refutada para "piora" se aparecer atrator,
  quimera ou deriva de composição maior que a do C1 em ≥ 3 células. Aposta secundária: se alguém piora,
  é o b8 (dado repetido).

Não coberto: uma semente de embaralhamento por braço; julgamento meu; o formato continua simulado
(BF16 com valores de 4 bits), sem arquivo 4 bits nativo.

## Desvio 25/09 ~14:45 local — b8 parado no passo 1000; fila trocada (escrito antes dos novos braços)

b8 (int4, pesos inteiros, lr 1e-5, 124 x 2): passo 0 holdout_rel 0,2315 / sens 0,990; passo 500 0,3897 /
0,641; passo 1000 0,3795 / 0,703; perda de treino 0,15-0,22 contra 0,067 de holdout no passo 0. **Z1
refutada**: o treino afasta o aluno do professor, inclusive nos exemplos de treino (não é overfit, é o
otimizador). Hipóteses: escala por absmax do grupo move os 32 valores quando o maior peso se mexe; Adam com
lr 1e-5 e lote 1 anda ~lr por peso por passo mesmo com gradiente ruidoso; o ponto de partida já é quase
ótimo. O "melhor" da ferramenta ignora o passo 0 (regra do ternário) -- aqui o melhor é o passo 0.

Fila nova, professor reaproveitado de /content/qat_b8 (mesmo dado 124 x 2 + holdout), 2000 passos, holdout
a cada 250, resto congelado:

| braço | o que treina | lr | pergunta |
|---|---|---|---|
| b10 | int4 g32, SÓ as escalas (`--so-escalas`, códigos do RTN congelados) | 1e-6 | refino sem mexer na grade melhora o int4? |
| b11 | ternário g128, SÓ as escalas (códigos do PTQ ingênuo congelados) | 1e-5 | só escala conserta o ternário? |
| b9 | int4 g32, pesos inteiros | 1e-6 | lr 10x menor ainda estraga? |

Previsões: **Z3:** b10 fica com holdout_rel ≤ 0,2315 em todo holdout (não piora) e cai um pouco. **Z4:** b11
não sai do ruído no render (só 29 M escalas contra código errado), mesmo que o holdout_rel caia.
**Z5:** b9 piora menos que o b8, mas ainda sobe acima de 0,2315.

## Resultado b10 (int4, só escalas, lr 1e-6) — 25/09 ~16:08 local

K0 passou (69/69). Holdout: passo 0 0,2315 -> passo 250 **0,2098** (melhor) -> 500 0,2103 -> 750 0,2119 -> 1000
0,2140, parada antecipada no 1000 (3 avaliações sem melhorar); sens 0,99 -> 0,93-0,96. Exportado o do passo 250.
Epsilon fora da amostra, 8 sementes, pareado contra o C1 (o mesmo int4 sem treino): **b10 0,1989 contra 0,2207,
10% menor, vence 8/8, 7,2x o erro-padrão.**

Render (`bench/qat_klein/avaliacao_fixa/render_b10/folha_fixa.png`, `bench/qat_klein/render_b10/folha_grade.png`,
colunas BF16 | C1 | b10): **indistinguíveis** a olho. b10 um pouco mais perto do BF16 no bonde (tamanho) e no
filhote s12 (pose); mais longe no "OPEN" s12 (letras brancas em vez de coloridas). Bem abaixo das 3/10 células
exigidas para "melhora".

- **Z3 confirmada:** só escalas não piora e melhora um pouco o número (holdout e epsilon).
- **Z2 confirmada:** no render, o professor-aluno em 4 bits não muda nada que se veja; o C1 já estava no teto.
  Resposta ao dono: com só as escalas, melhora o número (~10%) e não a imagem; com os pesos inteiros (b8), piora.

## Resultado b11 (ternário PTQ, SÓ escalas, lr 1e-5) — 25/09 ~16:49 local

K0 passou (69/69). Holdout: passo 0 1,0951 / sens 0,157 -> 250 0,751 -> 500 0,650 -> 1000 0,590 -> 1500 0,569 ->
2000 **0,557 / sens 0,681** (ainda caindo no fim). Epsilon fora da amostra, 8 sementes: **b11 0,549, b6 0,552**,
braco0_ptq 1,059.

Render (`bench/qat_klein/avaliacao_fixa/render_b11/folha_fixa.png`, `bench/qat_klein/render_b11/folha_grade.png`,
colunas BF16 | b11 | b6): **mesmo patamar do b6**, com o mesmo tipo de defeito (zebra, pavão e moinho
quiméricos), e melhor em algumas células: navio a lápis monocromático (o b6 colore), maçã única (o b6 faz
quimera), sopa mais perto do BF16, menos do respingo branco do atrator.

- **Z4 refutada.** Eu previ que só escala (29 M parâmetros, 1 por grupo de 128) não tiraria o PTQ do ruído.
  Tirou, e chegou onde o b6 chegou treinando os 3,68 B pesos -- sem trocar um código.
- **O que isso muda.** O Bonsai troca 2,7 M sinais, e a leitura de 21/09 foi "treinaram os códigos". Aqui, com
  os códigos do PTQ intactos, a escala sozinha já faz o trabalho que os braços com código treinado faziam. O
  transplante T1/T2 (códigos do PTQ + escalas do b4d = bege) não contradiz: aquelas escalas eram de outro código.
- **Pendente:** o b11 ainda melhorava no passo 2000; escala + H14q4 (os 2 grupos em 4 bits) é o misto óbvio.

## Resultado b9 (int4, pesos inteiros, lr 1e-6) — 25/09 ~20:13 VM

Holdout: passo 0 0,2315 -> 250 **0,2188** (melhor) -> 500 0,2214 -> 750 0,2244 -> 1000 0,2241, parada antecipada;
sens 0,99 -> 0,91-0,94; códigos trocados 0,56% no 250, 1,36% no 1000. **Z5 refutada:** com lr 10x menor os
pesos inteiros não estragam -- melhoram um pouco no começo e escorregam depois, como o b10, mas o melhor do b9
(0,2188) fica atrás do melhor do b10 (0,2098). O que estragou o b8 foi o lr, não o professor-aluno. Render não
feito: no b10, com holdout melhor, o render já saiu indistinguível do C1.

**Fechamento do int4 (b8, b9, b10):** o professor-aluno ajuda o int4 só no número, e só com passo pequeno;
só escalas é o melhor dos três e o mais barato (13,9 GiB, 1,03 s/passo contra 22,8 GiB e 1,64 s/passo).

## b11L — continuação do b11 (25/09 ~21:50–23:03 VM UTC, A100 qat-a100c)
Mesmos argumentos do b11, `--inicia-de` o `ckpt/ultimo.pt` do b11 (passo 2000 conferido no arquivo), repo novo
`JoaoZaokk/klein4b-qat-b11L` (o b11 fica intacto). Passo 0 reproduziu o fim do b11 (holdout_rel 0,5572, sens 0,681).
Parado pelo dono no passo 2000 da continuação (4000 no total) por cota do Colab; ainda caindo.

| passo (cont.) | holdout_rel | sens |
|---|---|---|
| 0 | 0,5572 | 0,681 |
| 500 | 0,5493 | 0,704 |
| 1000 | 0,5405 | 0,717 |
| 1250 | 0,5411 | 0,706 |
| 1500 | 0,5385 | 0,725 |
| 1750 | 0,5349 | 0,737 |
| 2000 | **0,5339** | **0,737** |

Melhor = p2000, exportado e no HF (`melhor/aluno_ternario_diffusers.safetensors`). −4,2% no holdout_rel e +8% no sens
sobre o b11; 0 códigos mudados (só escalas). Sem render nem epsilon ainda — holdout_rel já se mostrou pouco confiável.
Dataset do professor `JoaoZaokk/klein4b-bf16` completo: 3556 amostras (holdout 24, fixa 10, grade 10, Parti 1756×2).

### b11L — render e epsilon (25/09 ~20:35 local, 3090)
Epsilon fora da amostra, 8 sementes (`bench/qat_klein/agrega_eps_b11L.txt`): **b11L 0,5305**, b11 0,5491, b6 0,5516,
PTQ 1,0592. b11L abaixo do b11 nas 8/8 sementes (−3,4% na média) e abaixo do b6 nas 8/8. Folhas
`bench/qat_klein/avaliacao_fixa/render_b11L/folha_fixa.png` e `bench/qat_klein/render_b11L/folha_grade.png`
(BF16 | b11 | b11L | b6): mesmas composições do b11, um pouco mais limpo no bonde, letreiro SALE, garça, maçã s11 e
vila s11 (leitura visual minha; julgamento do dono pendente). Zebra continua errada.
