# 07 - Subgraph expansion is unimplemented, and it blocks 27 of the owner's 46 workflows

Type: grilling
Status: resolved
Blocked by: -
Provenance: EXECUTED 2026-08-23 against a RUNNING ComfyUI 0.33 on the 3090

## What was measured

ComfyUI was started for the first time in this project's life and every one of the 46 real workflow
files was converted against the **live** `/object_info` (3366 node classes):

```
conversion over 46 real workflow files, live object_info:
  fatal              32
  CONVERTS           14

why the fatals (grouped):
   27x  node N is an instance of subgraph '<uuid>'
    4x  widget count mismatch (ByteDance2TextToVideoNode, PreviewAny, ...)
    1x  other
```

**59% of the owner's collection cannot run.** Every Z-Image, Qwen-edit and Flux template on this bench
is a subgraph instance — those are exactly the light, fast, everyday workflows, so the ones that DO
convert are the heavy video ones.

The converter refuses loudly rather than silently mangling, which is right:

```
node 57 is an instance of subgraph 'f2fdebf6-...'; expanding subgraphs is not implemented here,
so its contents never reach the server
```

## Why this is not a small fix

A subgraph instance is a node whose `type` is a uuid naming an entry in the file's own
`definitions.subgraphs`. Expanding it means inlining that definition's nodes with fresh ids, rewiring
its inputs and outputs to the instance's links, and doing it recursively — a subgraph may contain
another. Widget promotion makes it worse: a subgraph exposes selected inner widgets on its instance,
so `widgets_values` on the instance maps to inner nodes by a table, not by position.

That last part is where the danger is, and it is the same danger this project keeps meeting: a
half-correct expansion produces a graph that RUNS and returns a plausible wrong image.

## Two related bugs, both found by running, both already fixed

**A. `SaveVideo.codec` was dropped, and the cost was 97.7 s of GPU.** ComfyUI 0.33 types that input
`COMFY_DYNAMICCOMBO_V3` with no `widgetType` option, so `is_widget()`'s string branch classified it as
a link. `widget_names` returned `['filename_prefix', 'format']` for three saved values, `codec` never
reached the prompt, and the run died at the last node on
`SaveVideo.execute() missing 1 required positional argument: 'codec'` — **after** SDXL's 15 steps and
SVD's 20. Fixed two ways: `is_widget` now treats any type whose name contains `COMBO` as a widget
(substring, not a list — the suffix already moved V1→V3), and a required WIDGET input with no value
and no default is now a **fatal** note instead of a silent `continue`.

The first version of that fatal was too broad and failed 16 tests by firing on unwired `MODEL` /
`CONDITIONING` inputs in single-node fixtures. **The tests were right and the code was wrong**;
narrowing it to widget-typed inputs is the fix.

**B. `/history` is empty on this install, so a successful run reported zero outputs.** EXECUTED: a
prompt finished `done` in 93.4 s and wrote `ComfyUI/output/video/ComfyUI_00023_.mp4`, while
`/history/{prompt_id}` **and** `/history?max_items=3` both returned `{}`. Output discovery went only
through history, so the UI had nothing to show for a run that worked. Fixed by also collecting from
the `executed` WebSocket frame, which carries the filenames (TRACED,
`ComfyUI/execution.py:436` and `:578`). Both sources are unioned, history first — neither is trusted
alone.

After both fixes, run 4: **`done` in 224.1 s, 2 outputs**, a 1,371,698-byte PNG (magic
`89504e470d0a1a0a`) and a 468,571-byte MP4 (magic `...66747970`), both fetched back through
ComfyLite's own `/api/output` proxy.

## A third finding, NOT fixed

`probe()`'s 4 s default timeout turned a **busy** ComfyUI into `503 timed out after 4s -- a socket
opened but nothing answered`, refusing to queue. Measured moments later, `/system_stats` answered in
1.09 s; the slow reading was transient. ComfyUI has a queue precisely so work can be submitted while
it is busy, so refusing to submit because a status probe was slow is the wrong call. Decide whether
`/api/generate` should submit anyway when the last known state was reachable.

## Closing criterion

Closed when the owner decides between:

- **A. Implement subgraph expansion** — the only path that makes the everyday workflows usable. Needs
  recursion, id remapping, and the widget-promotion table, and it needs a test that proves a converted
  subgraph produces the SAME prompt ComfyUI's own frontend would submit, not merely a prompt that runs.
- **B. Leave it refused** — 14 workflows work, the rest say why. Honest, and useless for the light
  ones.
- **C. Convert through ComfyUI's frontend instead** — ask a running ComfyUI to flatten the graph rather
  than reimplementing its expansion. Not investigated; there may be no such endpoint.

Recommendation: **A**, and treat "same prompt the frontend would submit" as the acceptance test rather
than "it ran".

---

## VERIFICACAO, 2026-09-01. A opcao C esta MORTA, e o numero do titulo mudou.

Pedido pelo dono: verificar este ticket antes de decidir, porque nao precisa de GPU. Tudo abaixo
foi **EXECUTADO**, nao lido -- as duas primeiras perguntas rodam sem servidor e sem placa.

### C ("pedir para o ComfyUI achatar") nao existe, e a mensagem dele seria PIOR que a nossa

Listadas **todas** as rotas HTTP do servidor (`server.py`, `app/`, `comfy_api/`, `api_server/`):
nao ha nenhuma que receba um workflow em formato UI e devolva prompt em formato API. O que existe
com "subgraph" no nome e o `app/subgraph_manager.py`, e ele serve um **catalogo** de subgrafos de
custom nodes e templates (`/global_subgraphs`, `/global_subgraphs/{id}`) -- nao achata nada. O
`subgraph` do `execution.py` e outro conceito: expansao dinamica em runtime, quando um no devolve
um `GraphBuilder`.

Ausencia numa lista nao e ausencia no sistema, entao a `validate_prompt` foi chamada **de verdade**,
em processo, com 854 classes carregadas e uma instancia de subgrafo:

    aceita?   False
    tipo      missing_node_type
    mensagem  Node 'instancia' not found. The custom node may not be installed.

**Isso e um argumento a favor de recusar no conversor**, nao contra. Se o ComfyLite parasse de
recusar e simplesmente postasse, o dono receberia *"o custom node pode nao estar instalado"* para um
workflow cujos nos estao **todos** instalados -- diagnostico errado, e ele iria caçar um pacote que
nao falta. A recusa atual nomeia a causa real.

### O numero: nao sao mais 27 de 46

Contagem estatica hoje, sem servidor (um workflow tem instancia de subgrafo quando algum no tem
`type` igual a um id de `definitions.subgraphs`, ou um uuid):

    .json em ComfyUI/user/default/workflows/   79
    COM instancia de subgrafo                  39
    sem                                        39
    ilegivel                                   1   <- ver abaixo, e virou conserto

O ticket dizia **27 de 46 (59%)**; hoje sao **39 de 79 (49%)**. A colecao cresceu 72% em nove dias.
A conclusao qualitativa **nao muda** -- metade da colecao continua bloqueada, e os bloqueados
continuam sendo os leves do dia a dia (Z-Image, Qwen-edit, Flux) -- mas o numero especifico nao
sobrevive a uma semana, e este repo ja tem regra para isso: contar, nao citar.

Ressalva: os 46 do ticket eram "workflow files reais" e podem ter passado por um filtro que esta
contagem nao aplica; e a contagem estatica so reproduz a causa DOMINANTE (27 dos 32 fatais). Os
outros fatais (widget count mismatch) dependem do `/object_info` vivo e nao estao aqui.

### Um defeito achado no caminho, ja consertado: BOM

O 1 arquivo ilegivel acima e `SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_safe_720p_Q8.json`, cujos tres
primeiros bytes sao `ef bb bf` -- BOM de UTF-8. O `tools/comfy_run_workflow.py`, que o ticket 05
chama de "the real asset", lia com `encoding="utf-8"` e **nao conseguia abrir esse workflow**:

    utf-8      FALHA  Unexpected UTF-8 BOM (decode using utf-8-sig)
    utf-8-sig  OK     11 nos

Nao e arquivo de teste nem caso hipotetico: e um workflow real do dono, e editor do Windows grava
BOM sem perguntar. Consertado para `utf-8-sig`, que le os dois casos (sem BOM os codecs sao
identicos). `--object-info-file` levou o mesmo tratamento por consistencia, e esta dito no codigo
que ali foi prevencao e nao medicao.

O teste novo (`test_main_reads_a_workflow_that_carries_a_utf8_bom`) roda o MESMO workflow com e sem
BOM e exige o mesmo codigo de saida -- testar so o caso com BOM provaria que ele nao explode, nao
que o codec novo deixou o caso comum intacto. **Provado que pega o defeito**: revertendo o codec de
proposito, o teste falha com o `Unexpected UTF-8 BOM` exato; com o conserto, passa.

### O que isso faz com a decisao

Restam **A** (implementar a expansao) e **B** (deixar recusado). C sai da mesa por medicao, nao por
opiniao. A recomendacao do ticket continua **A**, com o criterio de aceitacao sendo "o MESMO prompt
que o frontend submeteria", nao "rodou" -- e a verificacao acima reforca isso: uma expansao
meio-certa produz um grafo que RODA e devolve imagem plausivel errada, e o servidor nao vai avisar.

**Nao coberto por esta verificacao:** nada foi expandido; nenhum prompt foi submetido; a opcao C na
leitura "dirigir o frontend em navegador headless" nao foi investigada, so a leitura "existe
endpoint". `validate_prompt` e validacao, nao execucao.

---

## DECISAO DO DONO: **A**. Implementado e verificado contra o frontend, 2026-09-01.

Palavras dele: *"A 7; emenda depois no 2"*.

### O criterio que o ticket exigiu foi cumprido, e ele foi cumprido literalmente

O ticket pedia *"um teste que prove que um subgrafo convertido produz o MESMO prompt que o proprio
frontend do ComfyUI submeteria, nao meramente um prompt que roda"*. As referencias vieram do
frontend de verdade: ComfyUI subido em 127.0.0.1:8199, navegador aberto nele, e
`window.comfyAPI.app.app.graphToPrompt()` chamado depois de `loadGraphData` em workflows reais do
dono. As saidas estao em `tests/fixtures/subgrafo_referencia/`.

    5 referencias COM subgrafo + 2 CONTROLES sem subgrafo nenhum
    tests/test_subgrafo_contra_frontend.py   8 passaram, 0 falharam

**Identidade do conjunto de nos: alcancada.** Mesmos ids, mesmas classes, em todos -- inclusive no
`LTX25_VIDEO_INPAINT_TWO_STAGE_DUALGPU`, que tem **69 nos expandidos de 76**. O unico no fora da
conta e um cuja classe o servidor genuinamente nao tem instalada (`Image To Mask`), que o frontend
emite com `class_type` nulo.

### O numero que o ticket existe para mover

    antes    14 de 46 convertiam    (nenhum com subgrafo)
    agora    52 de 79 convertem     (22 deles COM subgrafo)

Os 27 fatais que restam sao **todos** de contagem de widget (`N widget values for M widget slots`),
que e outro buraco, ja listado neste ticket como categoria separada.

### A convencao de id veio do frontend, nao foi inventada

`<id_da_instancia>:<id_interno>` -- `175:164`, `175:165`. Ids frescos e sequenciais teriam passado
em qualquer teste que so exigisse "roda", e por isso o criterio do ticket foi escrito como foi.

### Eu conclui o oposto sobre widget promovido, e a comparacao derrubou

Primeira leitura, apoiada no `Seedance_2_Extend_Video`: os nos internos ja carregam os proprios
`widgets_values`, logo a instancia e so espelho de UI e **nao ha tabela de promocao a implementar**.
Isso contrariava o ticket, que chama a promocao de "onde mora o perigo".

**O ticket estava certo e eu estava errado.** Aquele subgrafo nao tem NENHUMA entrada
widget-tipada, entao instancia e interno nao podiam discordar -- conferi exatamente o caso em que o
eixo estava segurado. Onde ha entrada widget, a instancia VENCE:

    instancia 5407   widgets_values  ltx-2.5-22b-distilled-transformer-bf16.safetensors
    no interno 5602  widgets_values  ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
    frontend usa     ...-bf16

Usar o interno carregaria um **modelo diferente do que a UI mostra**, e o grafo rodaria -- que e
palavra por palavra o risco que este ticket descreve. Consertado: o valor da instancia e gravado no
no interno, e a posicao dentro de `widgets_values` sai de `widget_slots()`, a mesma funcao que o
conversor usa para ler, importada e nao reescrita.

### Um segundo defeito, achado pela mesma comparacao

Instancia em `mode=4` (bypass) estava sendo expandida. O frontend descarta o conteudo inteiro: no
`image_qwen_image_layered` ele emite 18 nos e eu emitia 28. Instancias mudas/bypass agora ficam
intactas, e o tratamento de mute que o conversor **ja tem** cuida delas -- nao ha uma segunda regra
de mute para manter em dia.

### Os CONTROLES sao o que impede a conclusao errada

Duas das sete referencias sao workflows **sem subgrafo nenhum**, e elas TAMBEM divergem do frontend
em valor (10 diferencas somadas). Sem elas, este ticket teria atribuido a expansao diferencas que
sao buracos anteriores do conversor: widget so-de-UI (`tokens`), widget composto
(`model.duration`), e o no virtual `Reroute` que o frontend resolve ATRAVES. O teste afirma
identidade de CONJUNTO DE NOS e diz, no proprio corpo, por que nao afirma identidade de valor.

### Onde o codigo mora

`worker/comfylite/subgrafo.py`, passe separado que roda ANTES do conversor: `ui_to_api` nao aprendeu
o que e um subgrafo, ele recebe um grafo em que instancia de subgrafo ja nao existe.

Uma definicao com `definitions.subgraphs[].widgets` nao vazia e **recusada**, nao expandida:
mecanismo separado, zero das 65 definicoes do dono o usa, logo nunca foi verificado contra o
frontend -- e palpite ali produz exatamente o grafo que roda e esta errado.

**Status: resolved.**

### Nao coberto

Subgrafo ANINHADO: o caminho existe (ponto-fixo com limite) e **nenhuma referencia tem um** --
nenhuma chave do frontend veio com dois `:`. Existe e nao esta provado. Um frontend (1.49.6), um
dia, sete workflows. E nenhuma imagem foi gerada: prompt identico ao do frontend diz que submetemos
o mesmo, nao que a imagem presta.
