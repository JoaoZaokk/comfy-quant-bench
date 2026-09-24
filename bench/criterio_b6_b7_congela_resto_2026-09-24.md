# Critério — treinar só o corpo ternário, resto congelado (b6, b7) — escrito 24/09 ~13:50, antes de lançar

Motivo: transplantes de 24/09 (`bench/criterio_transplante_flips_2026-09-24.md`). A cena mora no corpo
ternário; o atrator/colagem mora no resto treinado (9 densos + 60 norms). Corpo do b4d + resto original
= cena sem atrator, com grão de mosaico; corpo do controle + resto original = colagem some, cena
estilhaçada; corpo do p9191 = estilhaço.

`--congela-resto` (novo): o resto sai do otimizador (requires_grad False) e fica byte a byte o do
professor. Mesma receita do b4d/controle no resto: 8-bit + SR, lote 1, lr 1e-5, holdout a cada 500,
sens-tol 0,3, paciência 3, cada braço grava o próprio professor (VM nova `qat-a100c`).

| braço | dado | passos | compara com |
|---|---|---|---|
| b6 | 124 curados × sementes 1 2 | 4000 | b4d (mesmo dado, resto treinado) e T3_b4d (corpo b4d + resto original) |
| b7 | 1.756 (124 + Parti) × semente 1 | 3000 | controle p3024 e T3_ctrl |

Avaliação: grade principal (5 prompts × 11 12), `bench/qat_klein/avaliacao_fixa/prompts.txt` (10 prompts,
semente 11; prompt exato fora de TODO treino, conferido por grep), epsilon fora da amostra, e a conferência
de que o resto exportado é byte a byte o do BF16 original (se não for, a flag não fez o que diz e nada vale).

## Previsões

- **K0 (tem de passar):** os 69 tensores do resto no exportado de b6 e b7 são idênticos aos do BF16.
- **K1 [JULGAMENTO]:** b6 faz cena na grade principal em ≥ 8/10 e NÃO tem o atrator (forma branca
  curva, tecido, tigela) na avaliação fixa; grão de mosaico menor que o do T3_b4d (o corpo se adapta ao
  resto original durante o treino, em vez de ser colado nele depois). Refutada se ≤ 5/10 ou se o atrator
  aparecer em ≥ 3 das 10 células da avaliação fixa.
- **K2 [JULGAMENTO]:** b7 fica acima do controle p3024 no render (sem colagem), mas abaixo do b6: o dado
  Parti degrada o corpo (T3_ctrl estilhaçado). Refutada para "o Parti é inocente" se b7 ≈ b6; refutada
  para "o corpo também colapsa sozinho" se b7 fizer colagem/estilhaço ≤ 2/10.
- **K3:** epsilon de b6 entre o T3_b4d (0,619) e o b4d (0,551). Registrado para ver se o epsilon mente de
  novo, não para decidir.

Não coberto: um braço por dado, uma semente de embaralhamento, 2 sementes no render; "reconhecível" julgado
por mim; A100 (8-bit) contra o QAT local (4-bit) não é comparado aqui.

## Resultado b6 — medido 24/09 19:07 (render na 3090, `.scratch/render_b6.sh`)

Folhas: `bench/qat_klein/render_b6/folha_grade.png` (BF16 | b6 | b4d | T3_b4d, 5 prompts x 11 12) e
`bench/qat_klein/avaliacao_fixa/render_b6/folha_fixa.png` (mesmas colunas, 10 prompts, semente 11).

- **K0 passou** (69/69 do resto identicos ao BF16, conferido antes do render) e foi reconferido por
  outro caminho: amostra do resto b6 == original, b4d != original; no corpo, b6 difere do b4d em
  1,8-2,6% dos codigos, o mesmo tanto que cada um difere do PTQ (1,4-2,5%). O b6 e outro modelo.
- **K1 refutada.** O b6 renderiza quase igual ao b4d, composicao a composicao (mesmo "OPAL",
  mesmo filhote-raposa, mesma xicara-maca), e o atrator (forma branca curva, bone, tecido) aparece em
  ~8 das 10 celulas da avaliacao fixa (garca, guitarrista, zebra, bonde, raposa, pavao, navio,
  moinho; contado por mim), como no b4d -- muito acima do limite de 3. Sem o grao de
  mosaico do T3_b4d (esta parte da previsao valeu).
- **K3 refutada.** Epsilon fora da amostra, 8 sementes: b6 0,5516, b4d 0,5510, T3_b4d 0,6190,
  braco0_ptq 1,0592. O b6 nao fica entre b4d e T3: empata com o b4d.

**O que isso muda.** A leitura dos transplantes ("o atrator mora no resto treinado") era sobre o
PAR corpo+resto, nao sobre o resto sozinho: congelado o resto, o corpo aprende o mesmo atrator e o
mesmo epsilon. Com codigos diferentes (sobreposicao b6 x b4d no nivel do nulo, como no passo 1), o
comportamento e o mesmo -- o atrator e da receita (124 prompts x 2 sementes, 4000 passos), nao de
um conjunto de parametros. O T3 sem atrator e o efeito de DESCASAR um par co-adaptado (vem com
grao e epsilon pior), nao de remover a sede do atrator. K2 (b7) segue valendo como teste do dado.

Nao coberto: uma semente de embaralhamento; atrator contado por mim nas folhas; b6 x b4d nao sao
renders pareados de mesma execucao (b4d vem de `render_b4d`, mesmo protocolo e sementes).
