# Critério — repetir os MESMOS exemplos consolida, ou não faz nada? (2026-09-23)

Escrito antes de lançar a segunda A100. Proposta do dono: "pegar o checkpoint atual, lançar uma
segunda sessão de A100 e repetir os prompts do zero, e ver o que ele responde vendo EXATAMENTE as
mesmas ações pela segunda vez".

## Desenho

Ponto de partida comum: o checkpoint da A100 #1 no HF (passo P, ~8.100), mestre + AdamW8bit.

    ramo NOVO     A100 #1, continua a época 1: exemplos nunca vistos (P -> 14.000)
    ramo REPETE   A100 #2 (40 GB), `--inicia-de` o mesmo checkpoint, ordem zerada: revê os exemplos
                  0..k na MESMA ordem em que a #1 os viu da primeira vez

Mesmo lr (1e-5), mesmo otimizador, mesmo holdout (12 prompts, semente 1, 96 exemplos). O professor
da #2 é regravado (mesmos prompts e sementes; determinístico a menos de ruído numérico de GPU).
Comparação: holdout nos mesmos k passos a partir de P (a #1 mede a cada 1.000 passos ABSOLUTOS, então
os pontos ficam defasados em até ~100 passos — dito aqui para não virar surpresa).

## Previsões

- **R1 [JULGAMENTO meu]:** diferença de holdout entre os ramos < 3% em k = 1.000..4.000 — o platô é
  ruído do lr constante, e dado novo ou repetido dá no mesmo.
- **R2 (a hipótese dele):** o ramo REPETE fica ≥ 3% abaixo do NOVO (consolida).
- **R3:** o ramo REPETE fica ≥ 3% ACIMA do NOVO (repetir sem dado novo não ajuda e o novo ajuda).

Exatamente uma das três vale; a que valer é o resultado.

Não coberto: uma semente de embaralhamento, um lr, uma rodada; holdout é MSE na trajetória do
professor, não imagem.

## Resultado parcial (2026-09-23, ~19:45) — replay em +893

- **Holdout (a metrica do criterio):** repete 0,2875 contra novo 0,2863 em ~+1.000 (0,4% de
  diferenca) -> **R1** neste ponto. Em ~+2.000: 0,2809 contra 0,2773 (1,3%), ainda R1.
- **Render (fora do criterio, registrado como observacao):** `render_replay1/grade_origem_repete_novos.png`.
  A origem 8159 ja e colagem com o prompt fraco; o ramo REPETE (+893) traz o prompt de volta em
  varias celulas (maca vermelha, placa OPEN, telhado da vila, folhas do cha); o ramo NOVO (p9191,
  +1.032) fica MAIS generico que a origem. Na imagem, repetir melhorou e dado novo piorou.
- Leitura: o MSE nao separa os ramos e a imagem separa -- mais uma vez o MSE do holdout e cego ao
  colapso de condicionamento. Nao muda o veredito do criterio (R1), que foi escrito sobre o MSE.
  Uma semente de embaralhamento; o replay segue ate 6.000.
