# Critério — acaso casado por limiar, interseção de flips e transplantes (24/09, escrito antes de rodar)

Passo 1 da análise das leituras externas (GLM/Fable/Mimo/Grok/consolidação). Só CPU e disco; nenhum
render. Arquivos: os BFL de `ComfyUI/models/diffusion_models/` (fusões do mapa só concatenam LINHAS, então
os grupos de 128 no eixo K ficam intactos).

## A — acaso casado por limiar

`acerto_da_mudanca` (fração dos nossos flips que caem no código do Bonsai) deu 46% contra acaso 5,4%
(acaso = flip uniforme). Mas nossos flips e os do Bonsai se concentram perto do limiar r = |w|/d ≈ 0,5.
Nulo novo: para cada faixa de r (largura 0,02), a probabilidade de o Bonsai ter feito o flip "natural"
naquela faixa (0 → sinal de w; ±1 → 0), ponderada pela contagem de flips do braço em cada faixa.

- **A1 [JULGAMENTO]:** a razão acerto observado / acerto nulo fica entre 1,0 e 2,0 — o nulo casado explica
  a maior parte dos 46%. Se ≤ 1,1: nossos flips acham só "peso instável", nada de estrutura do Bonsai.
  Se ≥ 2: há seleção de posição além da distância ao limiar.
- **A2:** ≥ 95% dos flips dos braços são "naturais" (vão para o vizinho que r aponta). Se não, o nulo
  (que supõe flip natural) não vale e é dito.

## B — interseção de flips entre braços

Par b4d × b5 (mesmo dado, mesma receita, render igual), b4d × QAT local (mesmo dado, otimizador
diferente), b4d × controle p3024 (render oposto). Nulo: flips independentes casados por faixa de r.

- **B1:** b4d × b5 com sobreposição ≫ nulo (mesmo início, mesmo dado, mesma ordem). Leitura: alta
  sobreposição NÃO separa causal de epifenomenal (os dois compartilham tudo); só baixa sobreposição
  seria informativa.
- **B2 (a que decide algo):** se controle × b4d sobrepõe tanto quanto b4d × QAT local, os MESMOS
  flips aparecem em cena e em colagem — a identidade dos flips não separa o render.

## C — transplantes (arquivos para o passo 2, render)

T1 = corpo do PTQ + 69 tensores não-ternários do b4d; T2 = códigos do PTQ × escalas por grupo do b4d +
69 do b4d; T3 = corpo do b4d + 69 originais.
- **C0 (controle que tem de passar):** o construtor, pedido "corpo do b4d + resto do b4d" e "códigos do
  b4d × escalas do b4d", reproduz o b4d byte a byte; pedido "tudo do PTQ", reproduz o PTQ. Qualquer
  diferença invalida os três.
- Nenhuma previsão de render aqui: isso é o passo 2.

Não coberto: um único acaso (faixa de r), sem condicionar por camada no nulo; T2 usa a escala do b4d,
que é ótima para os códigos do b4d, não para os do PTQ.

## Resultado (24/09, ~11:00; `bench/qat_klein/flips_nulo_limiar.{log,json}`, `transplante_b4d.log`)

PTQ do arquivo = receita absmean em 100,0000% das posições. Bonsai: 396 M flips, 99,31% naturais.

    braco   flips   natural  acerto  nulo_lim  razao   recall  rec_nulo  razao
    local    76 M   100,00%  46,06%   44,07%   1,05    8,84%    8,46%   1,05
    p859     29 M   100,00%  48,29%   47,26%   1,02    3,50%    3,43%   1,02
    ctrl     68 M   100,00%  46,55%   45,02%   1,03    7,99%    7,72%   1,03
    p9191   124 M    99,99%  44,15%   41,62%   1,06   13,76%   12,97%   1,06
    b2      143 M    99,97%  41,59%   39,85%   1,04   15,01%   14,38%   1,04
    b3      170 M    99,98%  47,74%   46,16%   1,03   20,52%   19,84%   1,03
    b4d      80 M   100,00%  46,13%   44,22%   1,04    9,27%    8,88%   1,04
    b5       76 M   100,00%  46,28%   44,54%   1,04    8,83%    8,49%   1,04

- **A1: razão 1,02–1,06 em TODOS os braços → abaixo de 1,1.** Os 46% de acerto são quase inteiros
  explicados pela distância ao limiar: logo acima de r = 0,5 o Bonsai rebaixou 62% dos pesos para 0, e é
  ali que nossos flips caem. Nossos flips acham "peso instável", não estrutura do Bonsai. O "8,5× o acaso"
  (contra o uniforme de 5,4%) era o acaso errado. Nenhum braço — cena ou colagem — se distingue.
- **A2 confirmada:** ≥ 99,97% dos flips são naturais; o nulo vale.
- **B1 refutada, e é o caso informativo:** b4d × b5 (mesmo início, dado e receita; render igual) sobrepõem
  só 1,4× o nulo, Jaccard 0,24 — três quartos dos flips de um não estão no outro, e a imagem é a mesma.
  Quando coincidem, vão para o mesmo código (100%).
- **B2 confirmada:** b4d × controle 1,3× (Jaccard 0,21) = b4d × QAT local 1,3× (0,21) ≈ local × controle
  1,2×. Os mesmos flips aparecem em cena e em colagem na mesma proporção.
  Leitura: a identidade dos ~2% de flips é majoritariamente ruído de fronteira e não separa o render.
  Isso FAVORECE a hipótese dos contínuos, sem provar (o transplante decide).
- **C0 passou** (depois de corrigir o construtor: o PTQ grava −0,0 em ~metade dos zeros; sign() reescrevia
  como +0,0 — mesmo valor, byte diferente; nada tinha sido escrito na primeira tentativa).
  T1/T2/T3 escritos (7,75 GB cada). No b4d contra o PTQ: **escala por grupo quase não mudou** (b4d/PTQ
  mediana 1,0000, p5 0,980, p95 1,023, |log| médio 1,0%); **norms 0,3–0,6%**; **os 9 densos 2,7–8,8%**
  (time_in e modulação img ~8,7%, txt_in 5,6%). Se a recuperação é contínua, ela mora quase toda nos 9
  densos e nas escalas — e o braço antigo "só densos" (9 densos, sem norms, só 4 prompts) já tinha dado
  borrão/nítido-errado, então T1 é o teste que diz se o b4d achou densos diferentes daqueles.

## Previsões do passo 2 (render), escritas às 11:20, antes de qualquer imagem dos transplantes

Base: A1/B1/B2 dizem que os flips são ruído de fronteira; as escalas do b4d mudaram ~1% e os 9 densos 3–9%.
Protocolo: 5 prompts × sementes 11 12 (mesmo das grades), 10 prompts de generalização (semente 11), epsilon
fora da amostra (4 prompts × 2 sementes, trajetória BF16 imposta).
- **R1 [JULGAMENTO]:** T1 (corpo PTQ + resto b4d) faz cena reconhecível em ≥ 6/10 células, com as mesmas
  composições do b4d. Refutada se ≤ 2/10.
- **R2:** T2 (códigos PTQ × escalas b4d + resto b4d) ≥ T1 no render, e epsilon a ≤ 5% do b4d.
- **R3:** T3 (corpo b4d + resto original) ≤ 2/10 e epsilon > 0,9 (perto do PTQ, 1,059).
- **R4 (off-genre):** se R1 valer, T1/T2 repetem o vazamento do atrator do b4d nos prompts fora do gênero.
Leitura se R1 cair e R3 também: interação — nem corpo nem resto sozinhos carregam a cena.

## Resultado do passo 2 (12:45; `render_transp/grade_transp.png`, `generaliza_transp/grade_generaliza_transp.png`, `agrega_eps_transp.txt`)

    epsilon fora da amostra (8 sementes)   PTQ 1,059   T1 1,027   T2 1,008   T3 0,619   b4d 0,551
    fração da distância PTQ→Bonsai           0%        4,6%       7,3%      62,5%      72,1%

- **R1 REFUTADA:** T1 (corpo PTQ + resto b4d) = bege liso em 10/10, como o PTQ.
- **R2 REFUTADA:** T2 (+ escalas do b4d) = bege liso em 10/10; epsilon 7% do caminho. As escalas do b4d
  sozinhas não fazem nada.
- **R3 REFUTADA, na direção oposta:** T3 (corpo b4d + 69 ORIGINAIS) faz cena em 10/10 da grade principal
  (filhote, placa com letras, vila, jarro de vidro com folhas, maçã vermelha), com uma textura de mosaico
  granulada que o b4d não tem.
- **R4 não se aplica (R1 caiu); o achado off-genre é outro:** na grade de 10 prompts o T3 **não tem o
  atrator** (forma branca curva, tecido, tigela) que domina o b4d. Leitura minha, célula a célula contra o
  BF16: T3 reconhecível em SALE, sopa, bonde, quadrados, olho, torre (6) e parcial em garça, guitarrista,
  "zebra" que virou felino pintado bebendo, coração (4); o b4d fica em ~2 + 2. Tudo com o grão de mosaico.
- **Epsilon errou de novo:** T3 tem epsilon PIOR que o b4d (0,619 × 0,551) e imagem melhor fora do gênero.

**Leitura.** A recuperação mora no CORPO TERNÁRIO (os ~2% de flips junto com as escalas; escalas sozinhas
não bastam), não nos contínuos — o oposto da hipótese de consenso das leituras externas e da minha
previsão. E o **atrator mora no RESTO treinado** (9 densos + norms): tirando-o (voltando aos originais) o
atrator some. Os flips parecem ruído de fronteira pela estatística de posição (A1/B1), mas coletivamente
carregam a cena — conjuntos diferentes de flips (b4d × b5, Jaccard 0,24) funcionam igual.
Não coberto: um único braço doador (b4d), 2 sementes na grade principal, 1 na de generalização; julgamento
de "reconhecível" meu.

## Passo 3 — T3 do controle (escrito às 12:50, antes de rodar)

Se o atrator mora no resto treinado, a colagem do controle (dado Parti) também pode morar lá.
T3_ctrl = corpo do controle p3024 + resto original; T3_p9191 = idem com o p9191 (colapso).
- **S1 [JULGAMENTO]:** T3_ctrl faz cena em ≥ 5/10 células da grade principal. Refutada se ≤ 2/10 —
  aí o dado Parti estragou o CORPO também, e "o eixo é o dado" volta a valer para o corpo.
- **S2:** T3_p9191 ≤ T3_ctrl (mais passos no dado Parti não melhora o corpo).

## Resultado do passo 3 (13:20; `render_transp_ctrl/`, `generaliza_transp_ctrl/`; C0 ok nos dois)

Escala por grupo quase parada também no controle (mediana 1,0000 no p3024, 1,0041 no p9191).
Julgamento meu, célula a célula contra o BF16:
- **T3_ctrl (corpo do controle p3024 + resto original):** a colagem SOME e a cena aparece por baixo, com um
  estilhaçado de mosaico muito mais forte que o do T3_b4d. Grade principal: vila ×2, jarro de vidro com
  chá ×2, maçã ×2 reconhecíveis; filhote s11 fraco, s12 não; placa não → **6/10 com defeito pesado.**
  **S1 confirmada no limite** (≥ 5/10), mas a qualidade é muito abaixo do T3_b4d. Generalização: sopa e olho
  reconhecíveis, garça (brejo), bonde (estação) e quadrados parciais; o resto é estilhaço (controle inteiro: ~0).
- **T3_p9191:** estilhaço quase puro, 0–1/10 (vila com luzes, textura de folhas). **S2 confirmada.**

**Leitura.** O dado Parti mexe nas duas partes, de formas diferentes: (1) o RESTO treinado aprende o
atrator (a colagem do controle some quando ele sai — a mesma coisa que no b4d, só que mais forte); (2) o
CORPO do controle carrega cena, mas pior que o do b4d, e piora com mais passos no Parti (p9191 quase só
estilhaço). O "colapso" do controle é, na maior parte, o resto; a degradação lenta é o corpo.
Consequência de desenho: treinar o corpo com o resto CONGELADO (lr_denso = 0) — no dado curado e no Parti —
é o braço que separa as duas coisas durante o treino, e o T3 prevê que o curado dá cena sem atrator.
