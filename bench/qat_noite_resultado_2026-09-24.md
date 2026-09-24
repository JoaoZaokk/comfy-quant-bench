# QAT ternário do klein — resultado da noite 23→24/09

Plano, critério, previsões escritas antes e o diário com hora: `bench/plano_qat_noite_2026-09-23.md`.
Aqui só o que ficou medido. Braços da fila: BF16 → ternário, 8-bit-sr, lote 1, na A100 `qat-a100b`
(o controle rodou na `qat-a100`). **Revisado às 08:05 de 24/09 por verificação independente** (entrada
08:05 do diário): os números batem com as fontes; quatro leituras estavam erradas e foram corrigidas aqui.

## A resposta

**A receita do b4d renderiza muito melhor que o controle, mas a causa NÃO está isolada.** Os dois diferem
em quatro eixos ao mesmo tempo:

| | b4d | controle A100 #1 |
|---|---|---|
| conteúdo | 124 prompts curados | 1.756 = **os mesmos 124** + 1.632 PartiPrompts |
| sementes | 2 | 1 |
| repetição | ~2 épocas (cada prompt ~32 vezes) | ~0,2 época (cada prompt ~1,7 vez) |
| passos | 4000 | 3024 |
| professor | gravado pelo próprio braço | o do controle |

A frase anterior, "o eixo é o dado", não se sustenta: pode ser conteúdo, repetição, passos ou professor.
O braço que separa os dois primeiros está proposto no fim.

| braço | dado | lr corpo / denso | extra | render (5 prompts × 2 sementes) | epsilon fora da amostra |
|---|---|---|---|---|---|
| controle A100 #1 p3024 | 1.756 × 1 | 1e-5 / 1e-5 | — | colagem com rastro do prompt | 0,557 |
| b1 p3000 | 1.756 × 1 | 3e-5 / 3e-5 | — | colagem marrom, igual para todo prompt | 0,615 |
| b2 p3000 | 1.756 × 1 | 3e-5 / 3e-6 | — | colagem, um pouco melhor que b1 | 0,615 |
| b3 p3000 | 1.756 × 1 | 3e-5 / 3e-6 | L1 λ=1 | colagem como b2 | 0,606 |
| **b4d p4000** | **124 × 2** | 1e-5 / 1e-5 | — | 4 nítidas, 5 parciais, 1 fraca | 0,551 |
| QAT local p4000 (referência) | 124 × 2 | 1e-5 (4-bit Adam, 3090) | — | quase o nível do b4d na mesma grade | 0,551 |
| b5 | 124 × 2 | 1e-5 / 1e-5 | condição cruzada 0,5 | (treino terminando; render depois) | — |

Nenhum dos 5 prompts da grade está em treino algum (conferido por grep exato). Grades:
`bench/qat_klein/render_b1/`, `render_b2/`, `render_b3/`, `render_b4d/grade_b4d.png`.

## Ressalvas que mudam a leitura

- **Generalização.** `generaliza/grade_generaliza.png` tem 10 prompts: 6 do holdout curado (mesmo estilo
  dos 124) e 4 PartiPrompts que **estão no treino do controle**. O b4d acerta ~4/10, no nível do QAT local;
  o controle falha em quase todos, inclusive nos que viu no treino. Não é teste de "outro estilo nunca visto".
- **Epsilon empata b4d, controle e QAT local** (0,551 × 0,557 × 0,551; diferença pareada b4d − controle
  −1%, b4d menor em 3 de 8). É amostra equivalente: medido localmente, 4 prompts × 2 sementes fora dos dois
  treinos, trajetória BF16 imposta. O epsilon é cego a esse colapso; **não** é cego a b1-b3, que ficam pior
  que o controle como no render.
- **MSE de holdout não foi comparado no par b4d × controle**: os holdouts são diferentes (b4d: sementes 1+2,
  professor próprio). Em p3000, b4d 0,533 × b1-b3 0,533-0,537 é outro conjunto, não uma comparação.
- **Os códigos ternários não separam o render.** Controle (colagem) 1,85%, QAT local 2,07%, b4d 2,16%
  mudados contra o ingênuo, com acerto e concordância com o Bonsai praticamente iguais.
- **O mesmo artefato em todo QAT desta família** (forma branca curva, montes de neve/tecido), mais fraco
  no b4d, presente em todos.

## O que cada previsão deu

- P1 (lr 3x muda ≥ 2x os códigos, colapsa antes e holdout do b1 abaixo do controle): **parcial** — códigos
  3,90% contra 1,84% do controle em p3000 e render pior, confirmados; a cláusula do holdout foi
  **refutada** (MSE do b1 acima do controle em todos os pontos pareados; epsilon 0,615 × 0,557).
- P2 (prender o denso preserva o sens): 5 de 6 avaliações a favor — sinal fraco (teste de sinal p≈0,11,
  sens oscila 1,3-1,4x entre avaliações); na imagem, ganho leve.
- P3 (L1 esparsifica, rebaixa/promove > 1,5): **confirmada** — 4,0 (b2: 1,06; Bonsai: 2,5). O L1 acerta a
  direção da mudança; NÃO reproduz as outras duas assinaturas do Bonsai (mexer longe do limiar e mais nos
  blocos single profundos).
- P4 (lr 1e-4 colapsa antes de 2000): **não testada** — trocada pelo b4d, desvio registrado antes de rodar.
- P5 (b4d perto do QAT local ⇒ o eixo é o dado): a condição se cumpriu (b4d ≈ QAT local), mas a conclusão
  não segue, porque b4d e controle diferem em quatro eixos (tabela acima). **Causa não isolada.**
- P6 (b5 cruzado: sens ≥ b4d em ≥ 5 de 8): **refutada no sens** — abaixo do b4d nas 6 avaliações até p3000.
  Render pendente.
- Repetição (critério de 23/09): R1 no MSE; o ganho visual do replay em +893 não durou até p6000.

## Instrumentos novos (commitados)

- `sens` (guarda de condicionamento): 12 pares (24 no b4d/b5); oscila 1,3-1,4x (máx/mín) entre avaliações.
  Na calibração, p3024 × p9191 separam por 1,1%, menos que esse ruído — "ordena como o render" não está
  demonstrado. No b4d subiu nas 4 primeiras avaliações e depois oscilou (0,51 → 0,61 → 0,58 → 0,60).
- `holdout_rel`, `holdout_cruz_rel`, `melhor`, parada antecipada, lr por grupo, L1 proximal com SR.
- Condição cruzada (`--cruzado`): o professor grava a saída com a condição do prompt vizinho.
- Fila de braços no Colab que mantém o supervisor vivo entre corridas.

## Próximo passo sugerido (decisão do dono)

1. **Separar conteúdo de repetição** antes de escalar dado: receita do b4d (124 × 2 sementes, 4000 passos,
   lr 1e-5, professor próprio) com 124 PartiPrompts sorteados (fora do holdout e da grade) no lugar dos
   curados. Se renderizar como o b4d, o eixo é repetição/épocas; se fizer colagem, é conteúdo.
2. Conjunto de avaliação fixo que esteja fora de TODOS os treinos (conferido por grep contra cada arquivo de
   treino), de estilos variados, julgado por render célula a célula contra o BF16.
3. Só depois: dado curado maior, lr/L1/profundidade.
