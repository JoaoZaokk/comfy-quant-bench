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
