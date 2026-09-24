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
- 23:48 b2 p1000: holdout 0,378 (b1 0,377), holdout_rel 0,564 (b1 0,563), sens 0,469 (b1 0,391 — P2 2/2 a favor), códigos 2,11% (b1 2,10%). Prender o denso não muda o corpo; muda o sens.
- 23:50 EPSILON b1 (8 sementes, ComfyUI, fora da amostra): melhor 0,613, final 0,615, controle p3024 0,557, ptq 1,059. b1 PIOR que o controle no epsilon também — epsilon, MSE e render concordam.
- 00:03 b2 p1500: holdout 0,373 (b1 0,374), holdout_rel 0,559 (b1 0,559), sens 0,422 (b1 0,518 — P2 1 contra), códigos 2,61%.
- 00:18 b2 p2000: holdout 0,352 (b1 0,357), holdout_rel 0,543 (b1 0,548), sens 0,501 (b1 0,437 — P2 3/4 a favor), códigos 3,07%.
- 00:33 b2 p2500: holdout 0,342 (b1 0,335), holdout_rel 0,535 (b1 0,529), sens 0,456 (b1 0,407 — P2 4/5 a favor), códigos 3,47%.
- 00:50 b2 p3000 (fim): holdout 0,344 (b1 0,347), holdout_rel 0,536 (b1 0,537), sens 0,478 (b1 0,403 — P2 5/6 a favor: CONFIRMADA), códigos 3,89% (= b1). melhor = p2500 (0,535).
- 00:55 fila relançada com b3 + b4d (pid 120895), supervisor b7k2le9sf, avaliador local do b4d b7nh68hjs (a cadeia avalia_noite trava esperando 'b4', que não existe — matar quando chegar lá). Previsão: b3 ~02:30, b4d ~04:40.
- 01:05 b3 com L1 roda a 1,81 s/passo (b1/b2 1,53: L1 custa +18% na A100). b3 termina ~02:40, b4d ~04:55.
- 01:10 RENDER b2 (`render_b2/grade_b2.png`): um pouco melhor que o b1 (volta o azul do cobertor e o cachorro, a maçã vermelha), mas ainda é colagem genérica, pior que o controle p3024 e muito pior que o QAT local. Prender o denso ajuda de leve (sens e imagem concordam) e NÃO evita o colapso com lr 3e-5. Conclusão parcial: lr maior acelera o colapso nos dados Parti; a esperança fica no eixo dados (b4d).
- 01:16 b3 p500: holdout 0,405 (b2 0,425), holdout_rel 0,586 (b2 0,600), sens 0,476 (b2 0,576), códigos 1,57% (b2 1,49%).
- 01:34 b3 p1000: holdout 0,382 (b2 0,378), holdout_rel 0,566 (b2 0,564), sens 0,455 (b2 0,469), códigos 2,31% (b2 2,11%: L1 +0,2 pp).
- 01:35 EPSILON b2: melhor 0,611, final 0,615 (b1 0,613/0,615; controle 0,557). Epsilon não separa b1 de b2; a imagem separou de leve.
- 01:54 b3 p1500: holdout 0,372 (b2 0,373), holdout_rel 0,558 (b2 0,559), sens 0,488 (b2 0,422), códigos 2,92% (b2 2,61%). (3,06 s/passo naquela janela = salvamento síncrono do ckpt de 54 s; voltou a 1,85.)
- 02:11 b3 p2000: holdout 0,351 (b2 0,352), holdout_rel 0,542 (b2 0,543), sens 0,484 (b2 0,501), códigos 3,52% (b2 3,07%).
- 02:29 b3 p2500: holdout 0,344 (b2 0,342), holdout_rel 0,536 (b2 0,535), sens 0,479 (b2 0,456), códigos 4,08% (b2 3,47%).
- 02:48 b3 p3000 (fim): holdout 0,340 (b2 0,344), holdout_rel 0,533 (b2 0,536 — o melhor dos 3 braços em p3000), sens 0,430, códigos 4,63% (b2 3,89%). P3 (razão rebaixa/promove) a medir no exportado.
- 03:00 F: chegou a 46 GB livres com 3 downloads simultâneos; apagados BFL do b1 (melhor/final), do replay p6000 e o ckpt local do replay p6000 (todos refazíveis do HF, renders e epsilon já feitos).
- 03:05 **P3 CONFIRMADA** (`bench/qat_klein/mudancas_b2_b3.log`): b3 (L1 λ=1) rebaixa 3,70% e promove
  0,93% -> razão **4,0** (b2 sem L1: 2,00/1,89 = 1,06; Bonsai 7,70/3,06 = 2,5). O L1 corrige a
  DIREÇÃO da mudança (esparsifica, até mais que o Bonsai). NÃO corrige as outras duas assinaturas:
  (a) distância do limiar — 92-95% das mudanças do b3 têm r em [0,35; 0,65] contra 69-75% do Bonsai,
  que mexe em peso longe do limiar; (b) profundidade — b3 plano (4,3-5,0% em todas as camadas),
  Bonsai sobe de 7,9% a 19,9% nos single profundos. Essas duas são as que ainda faltam na receita.
- 03:25 b4d p500: holdout 0,400, holdout_rel 0,582, sens 0,506, códigos 0,70% (lr 1e-5: metade dos braços 3e-5, como esperado). Holdout do b4d é OUTRO conjunto (sementes 1+2, professor próprio): não comparar número a número com b1-b3.
- 03:28 RENDER b3 (`render_b3/grade_b3.png`): melhor = final (p3000, o melhor foi o último). Colagem genérica como b2, sem ganho visível — perde até a maçã vermelha que o b2 mantinha. O L1 muda a assinatura dos pesos (P3) e NÃO muda a imagem em 3000 passos nestes dados. Até aqui, nenhum braço da noite ficou melhor que o controle p3024; todos pioram com lr 3e-5 nos Parti.
- 03:35 PREPARADO b5 = receita do b4d (124 prompts × sementes 1 2, lr 1e-5, 4000 passos) +
  `--cruzado --cruzado-frac 0.5`, professor do b4d por symlink (cruzado gravado no dir do b5, ~7 min).
  Um eixo contra o b4d. Entra na troca b4d->b5 (~05:05), termina ~07:00.
  Previsão P6 (escrita antes): b5 com sens ≥ b4d em ≥ 5 de 8 avaliações e holdout_cruz_rel menor;
  holdout_rel dentro de ±2% do b4d; render com o prompt reconhecível em ≥ tantas células quanto o b4d.
- 03:42 b4d p1000: holdout 0,358, holdout_rel 0,551, sens 0,532 (subindo), códigos 1,00% (controle p1000: 1,01% — mesmo lr, mesmo ritmo de códigos).
- 03:45 cadeia avalia_noite parada (chegou em 'b4', que não existe). EPSILON b3:  media 1.0592e+00 6.0589e-01 6.0589e-01 5.5669e-01 (ordem: ptq, b3 melhor, b3 final, controle).
- 03:58 b4d p1500: holdout 0,351, holdout_rel 0,545, sens 0,575 (subindo 3 avaliações seguidas: 0,506/0,532/0,575 — diferente de b1-b3, onde oscilou/caiu), códigos 1,25%.
- 04:15 b4d p2000: holdout 0,345, holdout_rel 0,539, sens 0,607 (4ª alta seguida), códigos 1,47%.
- 04:32 b4d p2500: holdout 0,341, holdout_rel 0,536, sens 0,604 (estável), códigos 1,66%.
- 04:48 b4d p3000: holdout 0,338, holdout_rel 0,533, sens 0,577, códigos 1,84% (controle p3000: 1,84% — idêntico).
- 05:29 b4d p4000 (fim): holdout 0,330, holdout_rel 0,526, sens 0,598, códigos 2,16%. Fila acabou (FIM); supervisor saiu DONE.
- 05:34 b5 (cruzado) lançado na mesma VM (pid 195898), supervisor br0s0qxz1, avaliador local b549b7d90. Previsão de fim ~07:40 (7 min de cruzado + 4000 passos). Avaliador local do b4d rodando.
- 05:50 [LEITURA CORRIGIDA ÀS 08:05, ver abaixo] **RENDER b4d — RESULTADO DA NOITE** (`render_b4d/grade_b4d.png`): 10/10 células com o prompt
  reconhecível e cena coerente — filhote no cobertor azul, placa quase legível (OPAL/ONAL), vila na
  neve com montanhas, vidro com folhas verdes, maçã vermelha na mesa de madeira. **Melhor que o QAT
  local p4000 e muito acima do controle A100 p3024.** melhor = final (p4000).
  **P5 CONFIRMADA: o eixo é o DADO.** Mesma receita (lr 1e-5, 8-bit-sr), mesma VM, mesmo diffusers:
  1.756 PartiPrompts × 1 semente colapsa; 124 prompts curados × 2 sementes não. O "colapso de
  condicionamento" que atribuí ao lr/objetivo era, antes de tudo, o conjunto de treino. b1-b3 (lr,
  denso, L1) testaram eixos secundários em cima de um dado que já colapsava.
  Ressalva: os 5 prompts de avaliação são do mesmo estilo dos 124 (fotográficos e descritivos). Falta
  medir num conjunto de avaliação de OUTRO estilo antes de dizer que generaliza.
  Correção da ressalva acima: dos 5 prompts da grade, F0-F3 (filhote, placa, vila, bule) são do
  HOLDOUT — o b4d nunca os viu; só a maçã (D0) está no treino. Então a grade já mostra generalização
  em 4 prompts fora da amostra, do mesmo estilo. O teste de OUTRO estilo está rodando:
  `bench/qat_klein/generaliza/` (6 do holdout curado + 4 PartiPrompts curtos/abstratos, semente 11,
  b4d × QAT local × controle), depois do epsilon do b4d.
- 06:00 [CORRIGIDO 08:05] EPSILON b4d: 0,551 — **igual ao controle (0,557) e ao QAT local (0,551)**. O epsilon por passo
  (entrada casada na trajetória do BF16) NÃO separa o b4d, que reconstrói as cenas, do controle, que
  faz colagem. Terceiro instrumento cego ao colapso depois do MSE do holdout e da divergência de
  latente. O que separou foi o RENDER (e, fracamente, o sens: b4d subiu 0,51->0,61 enquanto b1-b3
  oscilavam/caíam). Regra para daqui em diante: nenhum braço de QAT se aceita ou recusa sem render.
- 06:05 [CORRIGIDO 08:05] **GENERALIZAÇÃO do b4d** (`generaliza/grade_generaliza.png`, 10 prompts nunca vistos, semente 11):
  b4d reconhecível em ~4/10 (sopa na tigela, bonde na neve, vitrine com letreiro "SAL…", pintura a
  óleo com torre), parcial em 3 (garça virou pessoa no brejo, olho sem planeta, quadrados só nas
  cores), falha em 3 (zebra, coração verde, guitarrista). **O QAT local fica no mesmo nível; o controle
  falha em quase todas.** Os três compartilham os MESMOS artefatos recorrentes (a forma branca curva,
  montes de neve/tecido) — um atrator comum a todo QAT desta família, mais fraco no b4d.
  Leitura corrigida: o dado curado tira o modelo do colapso (b4d ≫ controle, aqui também), mas o
  resultado dos 5 prompts da grade principal SUPERESTIMA a qualidade: fora do estilo dos 124, b4d ≈
  QAT local, e ainda longe do BF16. O próximo eixo é cobertura de dados COM qualidade (mais prompts
  curados e variados, várias sementes), não mais passos em PartiPrompts.
- 06:05 b5 p500: holdout 0,372 (b4d 0,400), holdout_rel 0,563 (b4d 0,582), sens 0,468 (b4d 0,506 — P6 1 contra), holdout_cruz_rel 0,577, códigos 0,62% (b4d 0,70%).
- 06:23 b5 p1000: holdout 0,363 (b4d 0,358), holdout_rel 0,555 (b4d 0,551), sens 0,464 (b4d 0,532 — P6 2 contra), holdout_cruz_rel 0,574, códigos 0,90% (b4d 1,00%).
- 06:40 b5 p1500: holdout 0,352 (b4d 0,351), holdout_rel 0,546 (b4d 0,545), sens 0,461 (b4d 0,575 — P6 3 contra), holdout_cruz_rel 0,568, códigos 1,15%.
- 06:57 b5 p2000: holdout 0,343 (b4d 0,345), holdout_rel 0,538 (b4d 0,539), sens 0,525 (b4d 0,607 — P6 4 contra: REFUTADA em sens, >=5/8 a favor já impossível), holdout_cruz_rel 0,560, códigos 1,37%.
- 07:15 b5 p2500: holdout 0,342 (b4d 0,341), holdout_rel 0,536 (b4d 0,536), sens 0,441 (b4d 0,604), holdout_cruz_rel 0,560.
- 07:25 [CORRIGIDO 08:05] PESOS DA NOITE × BONSAI (`codigos_noite_vs_bonsai.log`, `mudancas_noite_vs_bonsai.log`):
  | braço | render | mudou vs ingênuo | acerto da mudança | recall Bonsai | concorda Bonsai | rebaixa/promove | mudanças perto do limiar |
  | Bonsai | — | 10,77% | 100% | 100% | 100% | 2,5 | 69-75% |
  | ingênuo | destruído | 0 | — | 0 | 89,23% | — | — |
  | QAT local p4000 | bom | 2,07% | 46,1% | 8,8% | 89,07% | 1,05 | 99,3% |
  | b1 | colagem | 3,90% | 41,6% | 15,1% | 88,58% | 1,06 | 94% |
  | b2 | colagem | 3,89% | 41,6% | 15,0% | 88,58% | 1,06 | 94% |
  | b3 (L1) | colagem | 4,63% | 47,7% | 20,5% | 89,02% | 4,0 | 92-95% |
  | b4d | cenas | 2,16% | 46,1% | 9,3% | 89,06% | 1,05 | 99,5% |
  **Achado: o braço que renderiza (b4d) é o que MENOS mexeu nos códigos, gêmeo do QAT local; o que
  mais se aproximou do Bonsai (b3: recall 20,5%, rebaixa/promove 4,0) faz colagem.** Chegar perto dos
  códigos do Bonsai NÃO é o que produz imagem — o que separou foi o dado. E nenhum braço passou a
  concordância do ingênuo com o Bonsai (89,23%): b1/b2 ficam ABAIXO dele (88,58%). Perfil por
  profundidade plano em todos (o Bonsai sobe até 19,9% nos single profundos).
- 07:31 b5 p3000: holdout 0,333 (b4d 0,338), holdout_rel 0,530 (b4d 0,533), sens 0,523 (b4d 0,577), holdout_cruz_rel 0,552, códigos 1,73%.
- 08:05 **VERIFICAÇÃO INDEPENDENTE (agente verificador, só CPU, fontes primárias).** Os NÚMEROS batem
  (>100 valores do diário contra o log da VM, os agregados de epsilon e os JSON do HF). A LEITURA não.
  Correções, cada uma conferida na fonte:
  1. **O "dado Parti" CONTÉM os 124 curados** — `prompts_treino_parti.txt` começa pelos mesmos 124, na
     mesma ordem (cabeçalho do arquivo; `cmp` das 124 primeiras linhas: idênticas). Então b4d × controle
     NÃO é "mesma receita, só o dado muda". Diferem em QUATRO eixos ao mesmo tempo: conteúdo (124 × 1.756),
     repetição (b4d ~2 épocas, cada prompt ~32 vezes; controle ~0,2 época, cada prompt ~1,7 vez),
     passos (4000 × 3024) e professor (gravado pelo próprio b4d × o do controle). **P5 fica
     "b4d ≫ controle no render, causa NÃO isolada"**, não "o eixo é o dado". A leitura de 05:50 e a do
     commit a99091c estão erradas nesse ponto.
  2. **A maçã (D0) não está em NENHUM treino** (grep exato: 0 nos 124 e nos 1.756). "Só a maçã está no
     treino" (05:50) vinha do critério de 22/09, que é de outro treino. As 10 células da grade são fora
     da amostra para todos os braços.
  3. **A grade de generalização não é "10 nunca vistos, de outro estilo"**: 6 são do holdout curado
     (mesmo estilo) e os 4 PartiPrompts ESTÃO no treino do controle (linhas 1093, 1244, 1539, 1599 de
     `prompts_treino_parti.txt`). O controle falhar neles é pior para o controle, não melhor.
  4. **b4d "10/10" exagera.** Julgado célula a célula contra o BF16: 4 nítidas (filhote ×2, vila ×2),
     5 parciais (placa s11 "OPYAL", bule virou jarro de vidro ×2, maçã como objeto vermelho deformado ×2),
     1 fraca (placa s12). O QAT local na MESMA grade fica quase no mesmo nível — o "~6/8" era outra amostra.
  5. **sens do b4d não foi monotônico**: 0,506 → 0,532 → 0,575 → 0,607 → 0,604 → 0,577 → 0,590 → 0,598.
     Subiu nas 4 primeiras e oscilou depois. No b4d/b5 o sens usa 24 pares, não 12.
  6. **Tabela de pesos (07:25) omitia o controle p3024**, que faz colagem: mudou 1,85%, acerto 46,55%,
     recall 7,99%, concorda 89,10% (`codigos_vs_bonsai.log:14`) — gêmeo do b4d e do QAT local nos
     códigos. **Os códigos não separam o render** (colagem e cenas com a mesma assinatura). "b4d é o que
     menos mexeu" é falso: o QAT local mexeu 2,07%, o b4d 2,16%, o controle 1,85%.
  7. **P1 tinha uma cláusula refutada que o resumo omitiu**: previa holdout do b1 abaixo do controle; o
     MSE do b1 ficou ACIMA em todos os pontos pareados (0,377/0,350; 0,357/0,322; 0,347/0,319) e o
     epsilon também (0,615 × 0,557). O "1,84%" da P1 é o controle em p3000.
  8. **holdout_rel do b4d (0,526) não se compara com b1-b3 (0,533-0,537)**: outro holdout (sementes 1+2,
     professor próprio, 24 pares) e outro passo (4000 × 3000). Em p3000 o b4d dá 0,533.
     Pelo mesmo motivo, "MSE de holdout cego ao colapso" NÃO foi medido no par b4d × controle.
  9. **"sens ordena como o render" não está demonstrado**: p3024 × p9191 separam por 0,006 (1,1%), bem
     abaixo da oscilação de 1,3-1,4x (máx/mín) medida depois. P2 5/6 é sinal fraco (teste de sinal
     p≈0,11). O epsilon não é cego a b1-b3: põe os três pior que o controle, como o render.
  Continua de pé: epsilon b4d = controle = QAT local (diferença pareada −1%, amostra equivalente: 4
  prompts × 2 sementes, trajetória BF16 imposta, fora de ambos os treinos); lr 3e-5 nos dados Parti piora
  (b1-b3 < controle no render e no epsilon); L1 corrige a direção das mudanças (P3) e não a imagem;
  nenhum braço passa a concordância do ingênuo com o Bonsai; controle rodou em `qat-a100`, não na `qat-a100b`.
  Não verificável sem GPU: ComfyUI × local 1,36% e × VM 1,48% (saída não gravada); VM × local 1,12%
  reproduzido na CPU.
  **Braço que separa os eixos (proposto, NÃO lançado — o dono pediu parada):** receita do b4d
  (124 prompts × 2 sementes, 4000 passos, lr 1e-5, professor próprio) com 124 PartiPrompts SORTEADOS
  (fora do holdout e da grade) no lugar dos curados. Renderiza como b4d ⇒ o eixo é repetição/épocas;
  colagem ⇒ o eixo é conteúdo. Um segundo, mais caro: os 1.756 com 2 sementes até ~2 épocas (~56 mil passos).
- 08:30 b5 p4000 (fim): holdout 0,331 (b4d 0,330), holdout_rel 0,526 (b4d 0,526, mesmo holdout), sens 0,562 (b4d 0,598 — abaixo nas 8 de 8 avaliações), holdout_cruz_rel 0,548, códigos 2,05%. Fila FIM 11:22 UTC; final e melhor confirmados no HF; `colab stop -s qat-a100b` e `colab sessions` vazio. Render/epsilon do b5 esperam a 3090 (o dono está testando outros modelos; gate `.scratch/gpu_livre`). Relatório 22→24/09: https://claude.ai/artifact/JXVGDMQcRSB4ED6SJWeoVM
- 08:45 PENDENTE (dono reiniciando o Claude; rodar só com a 3090 livre e quando ele pedir): `bash .scratch/render_b5.sh` — render + epsilon do b5 a partir de `klein4b_qat_b5_final_bfl.safetensors` já no disco (melhor = final = p4000; NÃO rodar `avalia_braco.sh b5`, que rebaixaria). Depois: grade BF16 | QAT local | b4d | b5 com `.scratch/grade_colunas.py`, julgar P6 no render, linha do b5 no resultado e no relatório HTML (`bench/relatorio_qat_klein_2026-09-24.html`, artifact JXVGDMQcRSB4ED6SJWeoVM).
- 09:25 RENDER + EPSILON b5 (`render_b5/grade_b5.png`, `agrega_eps_b5.txt`; 3090 livre, lock tomado pelas ferramentas): epsilon 0,544 (b4d 0,551, controle 0,557; b5 − b4d pareado −1,3%, b5 menor em 5 de 8 — dentro do ruído). Render **empata com o b4d**: mesmas composições por semente, placa um pouco pior (s12 "ROMERL" mais poluída), filhote um pouco melhor, vila/bule/maçã iguais. **P6 fechada: refutada no sens (8/8 abaixo), holdout_rel igual, render igual — a condição cruzada 0,5 não muda nada visível em 4000 passos.** Pendência de 08:45 cumprida.
