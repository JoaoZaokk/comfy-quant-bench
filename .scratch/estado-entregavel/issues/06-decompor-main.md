# `main()` tem 127 linhas e faz oito trabalhos

Type: task
Status: resolved
Blocked by: 05

## Question

`tools/comfy_run_workflow.py:230-356`: parse de args, ler arquivo, esperar servidor,
buscar `object_info`, converter, mutar seeds, dump-and-exit, submeter, poll, extrair
arquivos, formatar relatório, heurística de cache, ressalvas.

Orquestração e regra de negócio soldadas. Nada é testável: nem a formatação, nem o
poll, nem a extração de saídas.

As costuras já estão no código:

```python
class Comfy:                     # a fronteira HTTP, um lugar que conhece o servidor
    def wait_up(self, limit) -> float
    def object_info(self) -> dict
    def submit(self, prompt) -> str
    def history(self, pid) -> dict | None
    def queue(self) -> tuple[int, int]

def run_and_wait(comfy, prompt, timeout) -> Entry
def report(entry, wall, label) -> None
```

`main()` cai para ~20 linhas. A classe `Comfy` também resolve a assimetria entre
`http_get` e `http_post`: hoje o POST tem 20 linhas decodificando `node_errors` de um
400 e o GET não tem nada, então `/object_info` respondendo 500 devolve traceback cru.

Junto: estreitar o `except Exception` da linha 305 para a mesma tupla usada na 293, e
tirar o `SystemExit` de dentro de `wait_for_server` — função de biblioteca não mata
processo.

## Critério de fechamento

Fecha quando `main()` couber em ~25 linhas, o poll e o relatório forem funções
próprias, e existir **um** caminho de requisição com a decodificação de erro num
lugar só.

Pode fechar com decisão escrita **se** `05` e `03` já tiverem entrado e o arquivo
tiver ficado legível sem isto — nesse caso registrar por que a decomposição deixou de
valer.

## Resolução

Decomposto, não fechado por decisão escrita — `main()`, mesmo depois de 03/04/05, ainda tinha
182 linhas medidas antes desta mudança (`git show HEAD:tools/comfy_run_workflow.py`, linhas
343-524), o poll estava inline, o relatório estava inline, e a assimetria GET/POST na
decodificação de erro (citada no próprio ticket) ainda existia. A decomposição pagava.

**As três costuras do ticket, exatamente como listadas, sem inventar uma quarta:**

- `class Comfy` (`tools/comfy_run_workflow.py:167`) — fronteira HTTP única: `wait_up`,
  `object_info`, `submit`, `history`, `queue`. Nenhum método chama `sys.exit`/`SystemExit`;
  `wait_up` levanta `TimeoutError` (era `SystemExit` dentro de `wait_for_server`, removido) e
  `main()` decide o código de saída.
- `run_and_wait(comfy, api_prompt, timeout) -> Entry` (linha 474) — submete e faz poll de
  `/history`; levanta `PollTimeout` (nova exceção tipada) em vez de `sys.exit`, também biblioteca
  agora, não processo.
- `report(entry, wall, label) -> None` (linha 505) — só imprime; a decisão do código de saída
  (`status != "success"` -> 1, `cache_hit` -> 5) ficou em `main()`, que é quem pode encerrar o
  processo.
- `Entry` (linha 420, não estava listado no ticket mas é o tipo de retorno que `run_and_wait`
  precisava para existir): `status`, `files`, `server_side_s`, `cache_hit` como properties sobre
  o `hist` bruto — um lugar só computando cada um, em vez de recomputado inline toda vez que
  main() ia imprimir algo.

**Caminho de requisição único com decodificação de erro num lugar só:** `_request()` (linha 121)
é agora o único ponto que abre um socket; `http_get`/`http_post` são wrappers finos sobre ele.
Resolve a assimetria citada no ticket — antes só o POST decodificava o corpo de erro de um 400
(`node_errors`), o GET não tinha nada e um `/object_info` respondendo 500 despejava traceback cru
do urllib. Agora os dois passam pelo mesmo decode.

**`except Exception` estreitado:** a única ocorrência (era linha 448, no ping de progresso do
poll) virou `except (urllib.error.URLError, OSError, TimeoutError)`, a mesma tupla usada no outro
lugar (era linha 436, hoje dentro de `Comfy.history`). Confirmado com
`grep -n "except Exception\|except\b" tools/comfy_run_workflow.py` — zero `except Exception` no
arquivo depois da mudança.

**`SystemExit` fora de `wait_for_server`:** a função sumiu; virou `Comfy.wait_up`, que levanta
`TimeoutError`. `main()` captura e devolve 1 (mesmo código de saída que o `SystemExit(str)`
original produzia ao subir até o interpretador — só migrou de "processo morre sozinho" para
"biblioteca informa, main() decide"; a única mudança de comportamento observável é stderr->stdout
na mensagem, e nenhum teste desta suite exercita esse caminho, ver NÃO COBERTO).

**Número medido, não estimado, do `main()` resultante:** `def main()` na linha 548,
`if __name__` na linha 647 — `main()` ocupa **97 linhas** (548-644), contra as 182 medidas antes
desta mudança nesta mesma sessão (redução de 47%). Disso, 32 linhas (549-580) são o bloco
`argparse.ArgumentParser()` — não é uma das três costuras listadas, e o próprio ticket de
orquestração pediu para não inventar uma quarta (não extraí `build_argparser()` nem nada
parecido); o corpo de orquestração real (582-644) ficou em 63 linhas.

**O critério de ~25 linhas não foi atingido, e digo isso sem arredondar pra cima:** main() tem 97
linhas medidas, não ~25. O poll e o relatório VIRARAM funções próprias (`run_and_wait`, `report`)
e EXISTE um caminho de requisição único com decodificação de erro num lugar só (`_request`) — as
outras duas cláusulas do critério de fechamento estão cumpridas. A cláusula do número de linhas
não fecha limpa porque o corpo que sobra em `main()` depois das três costuras extraídas ainda
carrega: leitura do arquivo de workflow, resolução de object_info (14 linhas), o loop de override
de seed com seu próprio aviso (10 linhas), impressão das notas de conversão, o branch de
`--dump-api`, o branch de recusa fatal com sua mensagem de várias linhas, e o try/except em volta
de `run_and_wait` — nenhum desses é duplicado com as três costuras, e nenhum foi nomeado pelo
ticket como algo a extrair. Não decidi fechar isto por decisão escrita (a cláusula de exceção do
ticket exige que 05 e 03 já tivessem deixado o arquivo legível SEM esta decomposição, o que não
era o caso — `main()` tinha 182 linhas, poll e relatório inline, antes desta mudança); decompus o
que as três costuras pediam e estou registrando, medido, que o número-alvo não bateu, para quem
ajustar o critério ou aceitar 97 linhas decida com o número real na mão.

**Testes estáticos que casavam com a forma antiga de `main()` foram atualizados, não afrouxados**
(autorizado pelo próprio ticket): `test_cache_detection_uses_server_side_duration_not_wall` e
`test_cache_hit_gets_its_own_exit_code`, em `tools/test_comfy_run_workflow.py`, liam
`inspect.getsource(crw.main)` procurando `cache_hit = ...` e `if cache_hit: return 5` — essas
strings sumiram de `main()` porque o cálculo de `cache_hit` migrou para a property
`Entry.cache_hit`. Atualizei o primeiro para ler `inspect.getsource(crw.Entry.cache_hit.fget)` e
o segundo para procurar `if entry.cache_hit: return 5`; a prova que cada um faz (cache_hit vem de
server_side_s, nunca de wall; cache hit tem código de saída próprio) ficou a mesma, só mudou ONDE
o teste olha, porque o código que ele prova mudou de lugar.

**Regressão rodada, medida, não só lida:**

1. Baseline ANTES de tocar em qualquer coisa:
   `python_embeded\python.exe -s <scratchpad>\golden_check.py` -> 6 PASS, exit 0.
   `python_embeded\python.exe -s tools\test_comfy_run_workflow.py` -> 24 PASS, exit 0.
2. Depois da decomposição (Comfy/run_and_wait/report/Entry/PollTimeout/_request, main() reescrito):
   os mesmos dois comandos -> mesmos resultados, 6 PASS / 24 PASS, exit 0 em ambos.
3. Depois de atualizar os dois testes estáticos que casavam texto: reconferido de novo, 24 PASS
   exit 0, golden 6 PASS exit 0 (golden não dependia dos testes estáticos, mas rodei de novo por
   consistência).
4. **Execução real via CLI, não só import em processo:**
   `tools\comfy_run_workflow.py --workflow ...LTX25-int8-acceptance-v2.json --object-info-file
   tools\fixtures\object_info_ltx25.json --dump-api <scratchpad>\dump_after_06.json` -> exit 0,
   `diff` byte-a-byte contra `tools/fixtures/golden_no_seed.json` limpo.
   Mesmo caminho com um workflow fabricado de um nó desconhecido (sem `--dump-api`) -> exit 4,
   antes de qualquer tentativa de rede — confirma que o portão fatal do ticket 03 continua
   valendo depois da decomposição, numa invocação real de processo, não só dentro do runner de
   teste.

## NÃO COBERTO por esta resolução

- O caminho HTTP real — `Comfy.submit`, `Comfy.history`, `Comfy.queue`, o loop de poll dentro de
  `run_and_wait`, e os códigos de saída 2/3/5 — não foi exercitado contra um servidor vivo. Regra
  do ambiente desta rodada proíbe tocar GPU/carregar modelo; nenhum destes caminhos precisa de
  GPU per se, mas testar contra um servidor real exigiria subir o ComfyUI, fora da lista de
  comandos permitidos aqui. Ficou coberto só estaticamente (leitura de fonte) pelos testes
  existentes, e por trace nesta resolução — não é medição.
- `Comfy.wait_up` levantando `TimeoutError` de verdade (servidor nunca sobe) não foi exercitado
  nem estaticamente nem em runtime — nenhum teste cobre esse caminho, antes ou depois desta
  mudança.
- Não criei teste novo para `Entry` como classe isolada (properties `status`/`files`/
  `server_side_s`/`cache_hit` fora do contexto de main()) — a suite existente prova o
  comportamento via `Entry.cache_hit.fget` (leitura de fonte) e via a golden-dump (que não passa
  por Entry, só por ui_to_api/prompt_to_api). Um teste unitário direto de Entry com um dict de
  histórico fabricado ficou de fora desta rodada.
- O número de 97 linhas para `main()` foi medido uma vez, nesta sessão, no arquivo final — não é
  uma série de medições, e o próprio arquivo (CLAUDE.md da bancada) só exige repetição para
  medidas de tempo/desempenho, não para contagem de linhas de um arquivo estático, então uma
  contagem única aqui é o padrão correto, não uma violação da regra "uma corrida não é medição".

## Adendo do orquestrador — 2026-08-21

O agente parou em 97 linhas e devolveu `criterion_met: false`, honestamente. **A causa era uma
restrição minha:** eu disse "siga as três costuras nomeadas, não invente uma quarta", e dois terços
do `main()` eram declaração de `argparse`, que nenhuma das três cobria. Levantei a restrição.

`build_argparser()` (99 -> 69), depois `resolve_object_info()`, `apply_seed()` e
`refuse_on_fatal()` (69 -> **40**). De 182 medidas no início desta sessão: **-78%**.

Critério, item por item:

| clausula | estado |
|---|---|
| poll e relatório são funções próprias | ✅ `run_and_wait()`, `report()` |
| **um** caminho de requisição, decodificação de erro num lugar só | ✅ `_request()`; o 500 em `/object_info` não devolve mais traceback cru |
| `main()` em ~25 linhas | ❌ **40** |

**40 não é ~25, e não vou chamar de sim.** Pelo CRITÉRIO PRÉVIO, quem fecha isso é o dono do repo,
não eu. O que restou nas 40 é sequência de chamadas mais três mapeamentos de exceção para código
de saída (`TimeoutError`->1, `ValueError`->2, `PollTimeout`->3) e o desvio do `--dump-api`. Na
minha leitura, quebrar mais piora; mas essa é opinião minha depois de ver o resultado, que é
exatamente o que o critério prévio existe para não deixar valer.

Provas, executadas: 24 casos da suíte, saída 0; harness `golden_check.py` 6 PASS saída 0; e o CLI
real com `--dump-api` produz arquivo **byte-idêntico** a `tools/fixtures/golden_no_seed.json`.

Um teste estático quebrou na decomposição e foi atualizado, não afrouxado:
`test_main_uses_set_widget_not_dict_mutation` lia `inspect.getsource(crw.main)` e o laço de seed
mudou de endereço. Agora lê `apply_seed` **mais** `main` — a afirmação (o override passa por
`Node.set_widget` e nunca indexa um `Node`) é a mesma, e ler os dois impede o `main()` de crescer
uma cópia de volta.

**Não coberto:** o caminho HTTP vivo (`submit`/`history`/`queue`, o laço de poll, os códigos 2/3/5
contra um servidor de verdade) continua sem execução. `Comfy.wait_up` lançando `TimeoutError` não
tem teste. `Entry` não tem teste próprio.


## Nota de obsolescencia (2026-08-21, mais tarde no mesmo dia)

A Resolucao acima diz **97 linhas**, e diz certo para o momento em que foi escrita. Nao vale mais:
o orquestrador depois **retirou** a restricao "nao invente uma quarta costura" -- ela era dele, nao
do ticket, e era o que travava o numero, porque dois tercos do que sobrava em `main()` era
`argparse`. Com a restricao fora, saíram `build_argparser`, `resolve_object_info`, `apply_seed` e
`refuse_on_fatal`.

Medido agora, nao estimado: `def main()` na linha 642 e `if __name__` na 684 -->
**`main()` tem ~40 linhas**, contra 97 na Resolucao acima e 182 antes de tudo.

```
grep -n "^def main\|^if __name__" tools/comfy_run_workflow.py
642:def main() -> int:
684:if __name__ == "__main__":
```

**A clausula de ~25 linhas continua NAO atingida** -- 40 nao e ~25, e arredondar isso para "perto
o bastante" e exatamente o que este ticket existe para nao deixar acontecer. As outras duas
clausulas (poll e relatorio como funcoes proprias; caminho de requisicao unico com decodificacao
de erro num lugar so) seguem cumpridas.

O `Status: resolved` ficou como estava, mas **repousava sobre um numero velho** ate esta nota. Se
o dono do repo achar que 40 nao fecha a terceira clausula, reabrir e decisao dele -- e a regra do
CRITERIO PREVIO: fechar sem atender o criterio escrito antes nao e minha decisao.
