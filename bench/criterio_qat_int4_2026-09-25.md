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
