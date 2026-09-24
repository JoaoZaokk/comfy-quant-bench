# Plano + critério — QAT do klein, noite de 2026-09-23 → 24

Arquivo-âncora: se o contexto compactar, **ler este arquivo primeiro**. O estado ao vivo fica na
seção ESTADO no fim (atualizada a cada marco, com hora).

Autorização do dono (23/09, ~19:50 local): "vai fundo, pode seguir autonomamente". Vale para: rodar os
braços abaixo na VM `qat-a100b` (que já tem o token de escrita dele), subir checkpoints em repos
**públicos** `JoaoZaokk/klein4b-qat-*`, avaliar localmente na 3090 (só INFERÊNCIA: render e
epsilon; treino na 3090 ele vetou: "3090 não pelo amor"). NÃO vale para: criar VM nova (precisaria
de ele subir token), mexer em modelo original, WSL, pagefile.

## O que já está medido (não re-derivar)

| fato | valor | fonte |
|---|---|---|
| QAT 4-bit local 1e-5 | epsilon 0,551 (72,1% da distância), render ~6/8 reconhecível | `render_braco1_resultado_2026-09-22.md` |
| A100 #1, lr 1e-5, 8-bit-sr | p3024 epsilon empata com o local, render pior; p9191 colapso (colagem igual para todo prompt) com MSE ainda caindo; eps rel 0,557→0,582 | idem, seção 6-7 |
| MSE do holdout | **cego ao colapso de condicionamento** (medido 2x: A100 #1 e replay) | `criterio_repeticao_qat_2026-09-23.md` |
| replay +893 (mesmos exemplos) | MSE empata (R1), imagem MELHOR que a origem 8159; dados novos p9191 PIOR | idem |
| nossos QAT vs Bonsai | mudam 0,8-3,4% dos códigos vs 10,77% do Bonsai; acerto da mudança 44-48% (acaso 5,4%) | `codigos_vs_bonsai.log` |
| Bonsai esparsifica | rebaixa 7,70%, promove 3,06%; mexe longe do limiar; mais nos single profundos | `mudancas_vs_bonsai.log` |
| denso | A100 inflou modulação 0,068 vs 0,027 do Bonsai | idem |
| mestre bf16+SR ≡ fp32 | A/B | commit 377f70e |

## Correção de desenho feita ANTES de rodar (registrar, não repetir o erro)

A proposta de "weight decay no mestre para esparsificar" **não funciona com absmean**: o decay é
multiplicativo e uniforme no grupo, então `r = |w|/d` não muda e nenhum código troca. O decay só
age como aumento de lr efetivo (invariância de escala). O que esparsifica é **L1 proximal
relativo ao grupo**: `w ← sign(w)·max(|w| − lr·λ·d_g, 0)`. Um peso em r = 0,5 vai para
(0,5−δ)/(1−δ) < 0,5 e é rebaixado; o gradiente segura os importantes. E o passo precisa de
**arredondamento estocástico** (em bf16, lr·λ·d ≈ 6e-7 contra ulp ≈ 1,2e-4 seria zerado, o mesmo
congelamento do braço 1).

## Instrumentos novos no `tools/qat_ternario_klein.py`

- `holdout_rel`: média de ‖s−a‖/‖a‖ por exemplo (o MSE agregado é dominado pelos passos de sigma
  alto e premia a média).
- `sens` (guarda de condicionamento DENTRO do treino): para 12 prompts × passos {0,3,6} do
  holdout, saída com a própria condição vs com a condição do prompt vizinho, no mesmo x_t:
  `d = ‖s(x,c_i) − s(x,c_j)‖/‖s(x,c_i)‖`. Referência = o mesmo com o mestre ainda BF16 e o STE
  desligado (medida antes do primeiro passo e gravada em `ref_sens.json`). `sens = d_aluno/d_ref`.
  Colapso = sens caindo.
- `melhor`: a cada holdout, se `holdout_rel` melhora e `sens ≥ piso`, exporta o ternário e sobe
  (fila de envio, sem parar o treino). **piso = (1 − `--sens-tol` 0,1) × maior sens visto no braço,
  a partir do passo > 0** (o piso absoluto 0,8 da primeira versão foi derrubado pelo smoke: o
  ternário ingênuo mede sens 0,156 e caiu para 0,066 em 2 passos — um piso absoluto pararia tudo,
  e o passo 0 é reação errática de modelo quebrado, não condicionamento).
- `--parada-paciencia N`: para o braço após N avaliações seguidas sem melhorar `holdout_rel` OU com
  sens abaixo do piso.
- `--avalia-ckpt`: calibração do instrumento em checkpoints de desfecho conhecido, sem treinar.
- `--lr-denso`, `--l1-corpo` (L1 proximal relativo com SR).
- Exportado final sobe junto (7,2 GB em vez do ckpt de 15,6) para a avaliação local.

## Braços (um eixo por vez; início do BF16; 1.756 prompts PartiPrompts — o mesmo `prompts_treino.txt` da VM que a A100 #1 usou —, semente 1, lote 1, 8-bit-sr; 3000 passos; holdout a cada 500)

| braço | lr corpo | lr denso | L1 corpo λ | pergunta |
|---|---|---|---|---|
| (controle) A100 #1 | 1e-5 | 1e-5 | 0 | já medido em p3024: epsilon ~0,55, render pior que o local |
| B1 | 3e-5 | 3e-5 | 0 | força: 3x lr move mais códigos e melhora, ou colapsa antes? |
| B2 | 3e-5 | 3e-6 | 0 | prender o denso muda o colapso? |
| B3 | 3e-5 | 3e-6 | 1,0 | esparsificar aproxima do Bonsai (rebaixar > promover)? |
| B4 | 1e-4 | 1e-5 | 0 | força alta: onde fica o teto antes da overcorrection? |

O controle NÃO é re-rodado: A100 #1 usou os mesmos prompts, sementes, ordem (semente de
embaralhamento 20260922) e lote — conferir o `args.json` dela antes de comparar. Os instrumentos
novos (holdout_rel, sens) não existem para ele; a comparação com o controle é por MSE do holdout,
epsilon local e render.

## Previsões (escritas antes de rodar)

- **P1 [JULGAMENTO]:** B1 muda ≥ 2x mais códigos que o controle no mesmo passo (controle p3000:
  ~2%) e o holdout_rel melhor do B1 fica abaixo do controle; sens de B1 cai mais cedo que o
  controle (colapso chega antes com lr maior).
- **P2:** B2 tem sens ≥ B1 no mesmo passo (o denso livre é parte do colapso). Refutada se sens(B2)
  < sens(B1) em ≥ 3 das 6 avaliações.
- **P3:** B3 rebaixa mais do que promove (razão rebaixa/promove > 1,5; Bonsai 2,5), medido com
  `tools/analisa_mudancas_bonsai.py` no exportado. Refutada se ≤ 1,2.
- **P4 [JULGAMENTO]:** B4 colapsa (sens < 0,7) antes do passo 2000.
- **Decisão final por render** (5 prompts × 2 sementes, protocolo `quality_ladder.py` do
  render_replay1): o braço que manter o prompt reconhecível em mais células vence; epsilon e MSE
  desempatam, nunca decidem sozinhos.

Não coberto: uma semente de embaralhamento por braço; 3000 passos (curto); holdout = trajetória
do professor; nenhuma imagem real.

## Execução (quem faz o quê)

- VM `qat-a100b`: `tools/colab_qat/fila_qat_ternario_klein.py` (config `fila.json`, lançada por `celula_lanca_fila.py`) roda B1→B4 em sequência num processo só, esperando o replay terminar.
  O log da fila só escreve `FIM` no final de tudo — assim o supervisor (probe a cada 4 min, que
  mantém a VM viva) não para entre braços. Cada braço: `/content/qat_bN`, professor por symlink
  de `/content/qat_run/professor*`, push para `JoaoZaokk/klein4b-qat-bN` (público).
- Local: `.scratch/avalia_braco.sh bN` (background, um por braço) espera cada exportado no HF, baixa, remapeia para
  BFL e renderiza + epsilon; também renderiza o replay final (p6000) para fechar R1/R2/R3.
- Fim: `colab stop qat-a100b` quando a fila terminar; conferir `colab sessions`.

## Fase 2 (não roda esta noite; depende do resultado)

1. Perda que preserva condicionamento (termo contrastivo entre prompts) se o colapso persistir.
2. lr com warmup + cosseno no vencedor, rodada longa (10-14k) com parada por sens/holdout_rel.
3. Escala aprendida por grupo (LSQ).
4. Qwen-Image: destilação bloco a bloco (BRECQ) — cabe em 24 GB; depois ajuste curto FSDP.
5. 3 bits: LSQ + GPTQ antes de tocar o corpo.

## ESTADO

Horários: o log da VM é UTC; a máquina local é UTC−3. Abaixo, hora LOCAL.
- 19:50 replay (qat-a100b) no passo 2700/6000, 1,54 s/passo, fim previsto ~21:15 local.
- 19:57 smoke local (3090, 4-bit, 4 passos, 3 prompts de holdout): referência d_ref 0,603; passo 0
  holdout_rel 1,107 sens 0,156; passo 2 holdout_rel 0,916 sens 0,066, `melhor` exportado.
- 20:00 liberados 51 GB no F: (exportados diffusers refazíveis + origem que está no HF).
- 20:15 CALIBRAÇÃO do sens (holdout local, 12 prompts, semente 1; `F:/qat_klein/calib_sens/avalia_ckpt.jsonl`):

  | checkpoint | render conhecido | holdout MSE | holdout_rel | sens |
  |---|---|---|---|---|
  | QAT local 4-bit p4000 | o melhor (~6/8) | 0,334 | 0,525 | 0,624 |
  | replay p893 | melhor que a origem | 0,377 | 0,558 | 0,579 |
  | A100 #1 p3024 | pior que o local | 0,365 | 0,551 | 0,549 |
  | A100 #1 p9191 | colapso | 0,403 | 0,577 | 0,543 |

  Previsão confirmada (sens(p9191) é o menor), mas a separação p3024-p9191 é pequena (0,006).
  sens e holdout_rel ordenam igual ao render, salvo replay × p3024 (sens diz replay melhor, rel diz
  p3024). Decisão: fica a guarda, com `--sens-tol` 0,05 (a faixa inteira observada é 13%).
- 20:16 fila lançada na VM (pid 45951), esperando o replay; supervisor religado (`bqbxkgfyo`),
  agora observando `/content/qat_fila`. Disco da VM: 107 GiB livres.
- 20:20 HIPÓTESE MORTA: "o professor da VM condiciona diferente do ComfyUI". Mesmo prompt,
  encoder_hidden_states: VM × local 1,12%, ComfyUI × local 1,36%, ComfyUI × VM 1,48% — o ComfyUI
  está igualmente longe dos dois, e a diferença mora nos tokens de padding (primeiros 8-32 tokens:
  0,03-0,16%). `.scratch/compara_cond_comfy.py`. O que DIFERE é o ruído inicial: mesma semente, x_t
  com rel √2 = 1,41 (diffusers 0.40 sorteia diferente do 0.38) — por isso MSE de holdout NÃO se
  compara entre VM e local (conjuntos de trajetórias diferentes). Em aberto: no holdout local o MSE
  do p9191 SOBE (0,365→0,403) enquanto no da VM caía (0,319→0,286); 12 prompts × 1 sorteio cada.
- 20:22 cadeia local `bmlszjveo` (`.scratch/avalia_noite.sh`): replay p6000 → b1 → b2 → b3 → b4,
  cada etapa espera o artefato no HF. Logs `.scratch/avalia_*.log`, renders `bench/qat_klein/render_<b>`.
- Previsão de horário (local): replay termina ~21:15; b1 ~22:45; b2 ~00:15; b3 ~01:50; b4 ~03:20
  (sem parada antecipada). Fim da fila → `colab stop qat-a100b`.
- 20:35 FASE 2 implementada (não roda na fila atual): **condição cruzada** (`--cruzado`,
  `--cruzado-frac`). O professor grava, no mesmo (x_t, t) do shard i, a saída com a condição do
  vizinho j; o aluno treina uma fração dos passos nesses exemplos. Métrica nova `holdout_cruz_rel`
  (erro relativo do aluno contra o professor nos cruzados do holdout, passos {0,3,6}).
  PREVISÃO antes de calibrar: ordena como o render (4-bit < replay ≈ p3024 < p9191) e separa
  p3024 × p9191 por ≥ 3% relativo (o sens separou por 1,1%). Refutada se p9191 não for o pior.
- 20:41 CALIBRAÇÃO do `holdout_cruz_rel` (`F:/qat_klein/calib_sens/avalia_ckpt.jsonl`):
  4-bit 0,556 < p3024 0,572 < replay 0,581 < p9191 0,587. **p9191 é o pior: confirmado.
  Separação p3024 × p9191 = 2,6% relativo, abaixo dos 3% previstos: essa parte REFUTADA.** Ordem
  idêntica à do `holdout_rel` (0,525 / 0,551 / 0,558 / 0,577) — a métrica cruzada, como MÉTRICA,
  quase não acrescenta informação ao erro relativo comum. Ela continua valendo como SINAL DE TREINO
  (é o que a fase 2 testa), não como instrumento de parada.
- 20:47 smoke da fase 2 passou (3090, 4 passos, frac 0,5; 32 exemplos cruzados indexados; gravar
  cruzado custa 5,2 s/shard na 3090). Commit e3dc928.
- DECISÃO PENDENTE (~03:20, quando a fila acabar): braço b5 = melhor config de lr entre b1-b4 +
  `--cruzado --cruzado-frac 0.5`, na mesma VM, se ainda houver tempo antes das ~07:00. Gravar o
  cruzado dos 1.756 shards na A100 estimado em ~50 min [ESTIMATIVA: 3x mais rápido que a 3090].
  Previsão para b5: sens e holdout_cruz_rel melhores que o mesmo lr sem cruzado, holdout_rel igual
  ou pior (±2%).
- 21:29 b1 MORREU logo após o passo 0: `RuntimeError: lr was changed to a non-Tensor object` no
  `step()` do torchao. O torchao da VM não converte o lr de grupo passado em dict (o 0.18 local
  converte, por isso o smoke local passou — mesmo eixo que o smoke segurou fixo: a versão da lib).
  Conserto: lr como tensor no grupo (micro-teste: corpo andou 9e-5 e denso 1,5e-5 em 3 passos, bate
  com 3e-5/3e-6). Fila parada (b2 já tinha começado), dirs sem checkpoint limpos, relançada 21:36.
  Replay terminou: passo 6000, holdout 0,2768. Novo horário previsto: b1 ~23:05, b2 ~00:35,
  b3 ~02:10, b4 ~03:40.
- 21:49 b1 p500: holdout 0,403, holdout_rel 0,584, sens 0,407, códigos mudados 1,46% (controle: 1,01% em p1000, 1,84% em p3000). melhor exportado.
- 21:52 replay p6000 renderizado: ganho do +893 NÃO se sustentou (ver criterio_repeticao, seção final). R1 fechado.
- 22:05 b1 p1000: holdout 0,377 (CONTROLE p1000: 0,350 — b1 PIOR no MSE), holdout_rel 0,563, sens 0,391 (caindo de 0,407; piso 0,386), códigos 2,10%. Nota: sens da VM (0,39-0,41) não compara com a calibração local (0,54-0,62): holdouts com ruído diferente.
- 22:20 b1 p1500: holdout 0,374, holdout_rel 0,559, sens 0,518 (SUBIU de 0,391 — sens oscila ±30% entre avaliações; o piso relativo passa a 0,492), códigos 2,65%.
- 22:35 b1 p2000: holdout 0,357 (controle p2000 0,322), holdout_rel 0,548 (melhor até aqui), sens 0,437 < piso 0,492 -> melhor NÃO exportado, 1ª avaliação ruim. Códigos 3,12% (controle p3000: 1,84%). O piso relativo bloqueou o melhor holdout_rel por causa do pico ruidoso de sens em p1500 — defeito da guarda, registrado.
- 22:37 DESVIO do plano (antes de ver b2-b4): `--sens-tol` 0,05 -> 0,3 para b2-b4. Motivo medido em
  b1: o sens oscila ±30% entre avaliações (0,407 / 0,391 / 0,518 / 0,437) com 12 pares × 3 passos,
  e o piso relativo de 5% bloqueou o `melhor` em p2000 (holdout_rel 0,548, o melhor do braço). A fila
  será relançada na troca b1->b2 só com b2-b4. b1 fica como rodou (tol 0,05).
- 22:50 b1 p2500: holdout 0,335, holdout_rel 0,529 (segue melhorando), sens 0,407 (<piso 0,492, 2ª ruim; melhor bloqueado de novo), códigos 3,50%.
- 23:05 b1 p3000 (fim): holdout 0,347 (controle p3000 0,319), holdout_rel 0,537 (pior que p2500 0,529), sens 0,403, códigos 3,90% (2,1x o controle — P1 parte 1 CONFIRMADA: >=2x). melhor = p1500 (0,559); final = p3000. MSE de holdout do b1 fica ACIMA do controle em todos os pontos pareados.
- 23:20 b1 terminou rc=0 (100 min; ckpt+exportado no HF, cópias locais apagadas, VM com 100 GiB livres). Fila trocada: b2-b4 com sens-tol 0,3 (pid 93239), supervisor brzrzyfq4. Previsão: b2 ~00:55, b3 ~02:30, b4 ~04:05. Cadeia local deve estar avaliando b1 agora.
- 23:32 b2 p500: holdout 0,425 (b1 0,403), holdout_rel 0,600 (b1 0,584), sens 0,576 (b1 0,407 — P2 1/6 a favor), códigos 1,49%.
- 23:40 RENDER b1 (`render_b1/grade_b1.png`): b1 p1500 = colagem turva com rastro do prompt; b1 p3000
  = colagem marrom genérica, igual para todo prompt. **Pior que o controle p3024 e muito pior que o
  QAT local.** P1 parte 2 CONFIRMADA: lr 3x colapsa ANTES. O MSE do b1 também ficou acima do controle.
- 23:42 CONFUNDIDOR identificado olhando a grade: o único QAT bom (local 4-bit) treinou nos 124
  prompts curados × 2 sementes; TODOS os da A100 (controle, replay, b1) treinaram nos 1.756 PartiPrompts
  — e também em outro otimizador (8-bit) e outra versão do diffusers. Os 5 prompts de avaliação são
  fotográficos e descritivos, como os 124.
  DESVIO (antes de rodar): **b4 (lr 1e-4) sai** — b1 já mostrou que lr maior colapsa mais cedo, então
  b4 seria colapso mais rápido, pouca informação. Entra **b4d = eixo DADOS**: receita do controle
  (lr 1e-5 corpo e denso, 8-bit-sr) na VM com os 124 prompts × sementes 1 2, 4000 passos (= o
  checkpoint local p4000). Professor gravado pela própria corrida (248 shards, ~8 min).
  Previsão P5: se b4d renderizar perto do QAT local (prompt reconhecível na maioria das células), o
  eixo é o DADO; se ficar como o controle, é otimizador/versão/ruído. Relançamento na troca b2->b3.
