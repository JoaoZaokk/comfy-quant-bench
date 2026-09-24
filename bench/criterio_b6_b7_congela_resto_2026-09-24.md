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
