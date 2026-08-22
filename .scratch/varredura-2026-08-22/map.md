# Varredura 2026-08-22 -- endpoints, kernels, entradas, caches, regressoes

## Destination

A written, verified account of what is actually broken across `F:/COMFY_PORTABLE`'s own code and its
coupling to the ComfyUI checkout -- with every claim carrying its file, its line, and whether it was
**traced** or **executed** -- plus a ticket per cluster the owner can take or refuse. The destination
is the account and the tickets, not the fixes: nothing here was repaired, by the owner's instruction.

## Notes

- **Domain**: a live ComfyUI Portable install being used as a quantization bench. Read `CLAUDE.md`
  and `AGENTS.md` before touching anything; both carry hard rules that override defaults.
- **Skills every session should consult**: `docs/agents/domain.md` names them.
- **Standing preference**: an unverified finding stays labelled unverified all the way into the file
  that acts on it. Every ticket below carries a `Provenance:` line for that reason.
- **Nothing in this effort touches the GPU.** The audit itself was read-only across 14 agents; the one
  execution performed (ticket 01) ran against a copy of a script pointed at a throwaway path.

## How this was produced

Six finder agents (endpoints, ComfyUI integration, kernels/dispatch, inputs, caches, regressions), each
piped into an adversarial verifier told to **refute** rather than confirm and to default to refuted
when uncertain. 92 findings claimed, 5 refuted outright, and the rest corrected downward in severity --
zero critical, one high. Two more agents produced the structural review and the ComfyLite feasibility
study.

The low refutation count is worth stating plainly rather than presenting as a quality signal: the
verifiers killed findings mostly by **correcting severity**, not by rejecting them. Read the
`Severity:` line on each ticket as the verifier's number, not the finder's.

## Decisions so far

<!-- one line per closed ticket, and per sub-question a run settled -->

Janela de GPU de 2026-08-22, 3090 pinada com `CUDA_VISIBLE_DEVICES=0`, lock tomado em formato
PowerShell **de proposito**: dado o ticket 01, esse e o unico formato que os dois leitores
respeitam. Solto ao fim, placa de volta a 758 MiB / 0%%, sem heartbeat orfao. O ComfyUI nunca foi
iniciado, entao a instancia zumbi que o dono avisou nao entrou na medicao.

### Rodada 2, 2026-08-22 -- 13 de 16 fechados

Quatro tickets, quatro agentes, cada um num revisor adversarial. Os quatro vereditos voltaram
SHIP WITH FIXES de novo. `09` e `10` foram para o **mesmo** agente de proposito: os dois mexiam em
`nunchaku_compare.py` e em todas as ferramentas de tempo, e dois agentes brigando pelo mesmo
arquivo e o que a particao existe para evitar.

**Os revisores provaram duas coisas que os implementadores nao reivindicaram:**

- **O `clip_name2` que eu adicionei a mao na rodada 1 era INERTE.** Um revisor extraiu o pacote do
  HEAD e rodou os dois contra o mesmo registry falso: com um arquivo quantizado em nome diffusers
  apontado para o slot 2, `RESULT[HEAD]: PASSES` / `RESULT[NEW]: BLOCKS`. Ganho real de recusa,
  nao log mais alto.
- O probe antigo do `verify_w4a4`, num checkpoint com `layers` vazio, montava snippet vazio,
  obtinha `resolved = {}`, e `all([])` fazia `native_ready = True`. Agora levanta.

**Tres defeitos no meu proprio conserto do lock da rodada 1, achados usando ele:**

- `Take-GpuLock` dormia **2 s fixos** esperando o heartbeat. Perdeu a corrida aqui, matou o beat,
  devolveu false -- **e deixou o arquivo de lock no disco**, nomeando o pid que tinha acabado de
  matar. A placa ficou trancada para todo mundo ate eu remover o arquivo a mao. Agora faz polling
  ate 20 s, e na falha remove o resto **so** se ele nomear o beat que acabou de matar. Medido
  depois numa maquina ociosa: o beat sobe em 0,4 s -- que e exatamente por que sleep fixo e a
  forma errada, ele passa toda vez que voce testa.
- `_pid_alive` marcava **todo pid morto como vivo** no Windows. Pid morto nao levanta
  `ProcessLookupError` aqui; levanta `OSError` com `winerror=87`, que caia no meu
  `except OSError: return True`. Medido: pid 30692 morto -> `winerror=87`, `psutil.pid_exists`
  False, e o `Get-Process` do PowerShell concordava com o psutil e nao comigo. Direcao segura --
  lock velho gruda em vez de ser roubado -- mas a mensagem mentia, e colocava as duas metades do
  lock de volta na mesma discordancia que o ticket 01 era.
- `test_gpu_lock` foi de 23 para **31** checks: take que falha nao pode deixar a placa trancada,
  nao pode apagar lock alheio, e as duas metades tem que concordar sobre liveness.

**E uma coisa que eu consertei em cima da rodada 2, do mesmo formato da rodada 1:** o `scope_line`
do preflight dizia `opened N file widget(s)` -- afirmacao sobre arquivos. O validador abre **so**
`.safetensors`; o `MODEL_FILE_SUFFIXES` reconhece sete. Seis dos sete chegavam no balde `checked`.
**35 dos 159 arquivos do inventario nao sao safetensors.** O teste que devia pegar variava o
**nome** do widget e mantinha a extensao fixa -- e o da rodada 1 variava a **forma** do tensor
enquanto a colisao vivia em formas identicas. **Teste que varia o eixo em que o bug nao esta** ja
custou tres fixtures.

**Janela de GPU**, 3090 pinada, lock tomado com o codigo consertado (o heartbeat subiu como pid
**30692** -- o mesmo numero do orfao morto, que e por que `hb` importa ao lado de liveness):

- Os tres formatos continuam verificando depois da mudanca do codebook: `int8_tensorwise` 0,0129,
  `asym_w4a8_int8` 0,0824, `convrot_w4a4` 0,2220.
- **Ticket 15 item 4 fechado por execucao**: `grep -c ComfyUI-Models quantization_inventory.json`
  = **72**, nao 0. Schema 3, duas raizes, 199 arquivos, 1,01 TiB.
- E a regeneracao revelou que **`D:` nao e um disco**: `net use` diz `D: -> \\192.168.3.68\estoque`.
  408 GiB do conjunto de modelos chegam por SMB, de um NAS que nao e o do ERP (`.40`). Por isso
  "a raiz D: esta offline" e estado normal, nao quebra.

Abertos: `04` e `08` (decisao do dono) e `16` (que ganhou itens novos das revisoes da rodada 2 --
o mais valioso e verificar que o `m_crossover` reescrito sobre a primitiva produz os numeros que
produzia antes, e isso precisa da placa).

### Rodada 1, 2026-08-22 -- 10 de 16 fechados

Oito tickets, oito agentes particionados por arquivo, cada um jogado num revisor adversarial.
**Todos os oito vereditos voltaram SHIP WITH FIXES**, e essa uniformidade era exatamente o que
desconfiar: rodei as seis suites eu mesmo e conferi os diffs contra os criterios, nao contra os
relatorios.

**Cinco correcoes foram regressoes que a propria rodada introduziu**, e quatro delas nas categorias
que este repo define como defeito:

- `quant_mixed` trocou comparacao de basename por sha256 do header, alegando que "o header sozinho
  identifica um checkpoint". Verdade sobre layout, falso sobre conteudo: **quatro grupos colidindo,
  nove arquivos** -- entre eles os tres Z-Image que o `--foreign-analysis` nomeia como modelos
  diferentes, as duas metades de um par Wan, e os dois passes de uma conversao. Recusa virou
  aceitacao. O digest agora amostra o corpo (45/45 distintos) e e ORado com o basename, entao o
  sinal mais fraco so pode acrescentar recusa.
- `verify_w4a4` passou a exigir `quant_group_size` do metadata. `ComfyUI/comfy/ops.py:1201` e um
  **literal 64** -- o verificador ficou mais estrito que o runtime que ele verifica e recusou 100%
  dos checkpoints existentes, inclusive o comando impresso no CLAUDE.md.
- `quant_audit` transformou recusa em aviso: um `--models-root` com typo sobrescrevia o inventario
  rastreado com uma varredura parcial e saia 0.
- `comfy-quant-preflight` listava so `clip_name1` do `DualCLIPLoader`, entao um encoder no slot 2
  passava sem check **com a classe reportada como coberta**.
- E a `cudart64_12.dll` **nao sumiu**: esta em `torch/lib/cudart64_12.dll.disabled`, 556.544 bytes,
  sha256 `d954ca54...cf9dad`. O comando citado como prova nos dois documentos nao consegue casar um
  nome terminado em `.disabled`. Um deles e o arquivo que enuncia a regra *ausencia num grep nunca
  vira ausencia no sistema*.

Janela de GPU depois disso: **os tres formatos executam**, primeira vez para w4a8 e int8 --
`int8_tensorwise` 0,0127, `asym_w4a8_int8` 0,0701, `convrot_w4a4` 0,2333, um layer cada, entrada
aleatoria em M=2. Escada monotona na direcao que os formatos preveem, o que e uma conferencia
independente barata sobre a atribuicao do `quant_mixed`.

Abertos: `09` (primitiva de medicao) e `10` (BenchGuard, que depende dela), `05` (consolidar seis
probes num), `04` e `08` (decisao do dono), e `16` (o que os revisores pediram e eu nao fiz).

- [01](issues/01-lock-ps-rouba-lock-python.md) **RESOLVIDO**: o conserto foi no lado Python, nao no PowerShell -- mudar o `.ps1` para JSON teria reproduzido o mesmo bug apontando para a sessao irma, que pode estar rodando codigo velho. `key=value` ja tinha dois leitores; agora tem tres. `tools/test_gpu_lock.py`, 23 checks, 23 passaram, nas duas direcoes. Junto foram CACHE-02 (pid comparado como substring) e CACHE-03 (heartbeat orfao sobrescrevia lock alheio para sempre).
- [05](issues/05-preflight-nao-prova-kernel.md): **minha alegacao de groupsize foi refutada.** As
  quatro combinacoes (cg 64/256 x tensor dummy/real) resolvem para `comfy_kitchen.backends.cuda`,
  e as duas chamadas reais funcionam. O preflight 64/64 do `quant_w4a4.py` e desleixado, nao
  errado. E `_native_probe.native_backend_ready()` **rodou pela primeira vez** e funciona.
- [06](issues/06-verify-cobre-1-de-5.md): mismatch de groupsize **falha alto** (RMSE 1.023 contra
  teto 0.9), nao produz PASS falso -- mas a folga e de 14%%. As partes (b) e (c) confirmadas em
  checkpoint real: o `zimage-v2-mixed.safetensors` (115 w4a4 + 55 w4a8) e rejeitado camada por
  camada, e a saida do `verify_w4a4.py` termina em JSON sem nenhuma ressalva.
- [09](issues/09-medicao-sem-spread.md): **o 1,4x nao reproduz em attention.** Spread de 1,04x em
  cinco rajadas intercaladas. Mas a primeira rajada e a outlier (1,676x contra cluster 1,610-1,625)
  e o `attn_bench` roda exatamente uma -- entao ele erra ~3%% **na mesma direcao**, sempre. Vies,
  nao ruido.
- [08](issues/08-converter-core.md): a divergencia de dtype do `quant_mixed.py` e **latente, nao
  viva** neste build -- `quantize_w4a8_int8_weight` devolve int8/fp8_e4m3fn/f32/None/f32, nada que
  o `.numpy()` recuse.
- [02](issues/02-claude-md-206-errado-na-direcao-insegura.md): a alegacao de que o
  `verify_w4a4.py` imprime o que nao cobriu e **falsa, agora medida**. E a divida do proprio
  CLAUDE.md sobre o `_check_accel.py` "was not run" esta **fechada**: rodou na 3090, Triton
  compila, Sage `mean|d|=0.0006`, FlashAttention `0.0000`, ALL GOOD. A `cudart64_12.dll` e
  medidamente desnecessaria, o que torna o `W4A4_HANDOFF.md:22` inequivocamente errado.

## Not yet specified

- Whether the eight `fbcache_*.py` probes should share a harness. The structural review rates it
  *Worth exploring* and explicitly declines to overstate it: the duplication has not been shown to
  produce a wrong answer, and a parameterized block factory could easily end up an interface as wide as
  the six bodies it replaces. Revisit when a ninth probe is needed.
- Whether `verify_w4a4.py`'s smoke RMSE ceiling of 0.9 is the right number. It is documented as a
  liveness signal, not a quality metric, and nothing has tested what it would fail to catch.
- The ComfyUI `assets` table as a hash source. Measured 2026-08-22: the table exists with a `hash`
  column and holds **zero rows**, so the "ComfyUI already hashed your models" shortcut is not available
  today. Whether the background seeder can be pointed at both roots is unexplored.

## Out of scope

- Fixing anything. The owner's instruction for this effort was report and tickets, no code.
- ComfyUI's own defects, except where our code depends on them. It is an upstream checkout we do not
  own.
- GitHub issues and PRs. There is no remote; see `docs/agents/issue-tracker.md`.

## Tickets

- [01-lock-ps-rouba-lock-python.md](issues/01-lock-ps-rouba-lock-python.md)
- [02-claude-md-206-errado-na-direcao-insegura.md](issues/02-claude-md-206-errado-na-direcao-insegura.md)
- [03-ltx-studio-endpoint.md](issues/03-ltx-studio-endpoint.md)
- [04-launchers-bind-0000.md](issues/04-launchers-bind-0000.md)
- [05-preflight-nao-prova-kernel.md](issues/05-preflight-nao-prova-kernel.md)
- [06-verify-cobre-1-de-5.md](issues/06-verify-cobre-1-de-5.md)
- [07-svdq-write-sem-contrato.md](issues/07-svdq-write-sem-contrato.md)
- [08-converter-core.md](issues/08-converter-core.md)
- [09-medicao-sem-spread.md](issues/09-medicao-sem-spread.md)
- [10-benchguard-optin.md](issues/10-benchguard-optin.md)
- [11-hf-download-sem-integridade.md](issues/11-hf-download-sem-integridade.md)
- [12-quant-mixed-procedencia.md](issues/12-quant-mixed-procedencia.md)
- [13-docs-desatualizados.md](issues/13-docs-desatualizados.md)
- [14-preflight-node-fail-open.md](issues/14-preflight-node-fail-open.md)
- [15-quant-audit-uma-raiz.md](issues/15-quant-audit-uma-raiz.md)
