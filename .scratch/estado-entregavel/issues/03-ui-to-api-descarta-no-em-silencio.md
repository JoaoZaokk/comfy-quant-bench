# `ui_to_api` descarta nó desconhecido em silêncio e submete o grafo assim

Type: task
Status: resolved
Blocked by: 05

## Question

`tools/comfy_run_workflow.py:165-168`:

```python
defn = object_info.get(ctype)
if defn is None:
    warnings.append(f"node {nid}: server does not know class '{ctype}'")
    continue                      # o nó some do grafo
```

O nó desaparece da conversão, um aviso vai para uma lista de strings, e o prompt é
**submetido assim mesmo**. Um grafo a que falta um nó pode executar e "ter sucesso"
com saída errada.

Mesma classe de falha para o desalinhamento de widget: hoje o aviso existe, mas não
impede nada. E foi exatamente um desalinhamento que quase passou — o tipo de
`frame_rate` é a string `"FLOAT,INT"`, não bateu com nenhum escalar simples, o campo
foi descartado, e `[49, 25, 1]` virou *batch 25* em vez de *fps 25*.

Quatro avisos hoje, dois graves e dois informativos, todos como texto livre.

## Depende de 05

O conserto é um tipo, não um `if`:

```python
@dataclass
class Note:
    level: Literal["info", "warn", "fatal"]
    node: str
    text: str
```

`fatal` recusa submeter sem `--force`. Isso é trabalho de modelo, e por isso este
ticket espera `05`.

## Critério de fechamento

Fecha quando classe desconhecida e desalinhamento de mais de um widget forem `fatal`,
e um `fatal` impedir a submissão salvo `--force` explícito.

Não fecha com decisão escrita: submeter um grafo mutilado produz um resultado
plausível e errado, que é a falha que este arquivo inteiro existe para impedir.

## Resolução

`Note` (dataclass, `level: Literal["info","warn","fatal"]`, `node: str`, `text: str`) substitui
`list[str]` como retorno de `ui_to_api` (`tools/comfy_run_workflow.py`). Classificação:

- classe desconhecida (`defn is None`) -> `fatal`
- widget faltando por mais de um (`len(names) - len(vals) > 1`) -> `fatal`
- widget sobrando por mais de um (`len(vals) - len(names) > 1`) -> `fatal` (já existia o aviso;
  só ganhou nível)
- widget faltando por exatamente um -> `info` (caso CLIPLoader `device`, citado no ticket)
- widget sobrando por exatamente um -> sem Note, como antes (control_after_generate)
- nó mudo/bypassed na UI -> `info`
- required input preenchido com default do servidor -> `warn`

`main()` ganhou `--force` (store_true). Depois de converter e antes de montar `client_id`/POST:
`fatal = [n for n in notes if n.level == "fatal"]`; se não vazio e `--force` não foi passado,
imprime a lista de fatais e `return 4` -- **antes** de qualquer chamada de rede. `--dump-api`
continua funcionando mesmo com fatal presente (ele já não executa nada; a doc do próprio flag
diz "not executed" -- só a submissão real é bloqueada).

Comando exato que provou (roda sozinho, sem servidor):

```
.\python_embeded\python.exe -s .\tools\test_comfy_run_workflow.py
```

21 PASS, 0 FAIL, exit 0 (12 pré-existentes + 9 novos: `test_note_shape`,
`test_unknown_class_is_fatal`, `test_muted_node_is_info_not_fatal`,
`test_widget_off_by_one_missing_is_info`, `test_widget_off_by_two_missing_is_fatal`,
`test_widget_extra_by_two_is_fatal`, `test_widget_extra_by_one_produces_no_note`,
`test_required_input_filled_from_default_is_warn`,
`test_main_refuses_fatal_without_force_and_does_not_touch_network` -- este último chama
`crw.main()` de verdade via `--object-info-file` sem `--dump-api` e sem `--force`, confirmando
`return 4` **antes** de qualquer POST, sem tocar rede).

Regressão exigida pelo orquestrador, rodada antes e depois da mudança
(`golden_check.py` no scratchpad da sessão): baseline 6 PASS/exit 0 confirmado ANTES de tocar o
arquivo. Depois: `prompt convertido identico (18 nos)` continua PASS; `avisos identicos` quebra
como esperado (str vs `Note`), mas o único aviso do workflow de aceitação LTX 2.5
(`node 2 (CLIPLoader): 2 widget values for 3 widget inputs [...] -- trailing ones left at server
default`) sai como `Note(level='info', node='2', text=...)` -- mesmo texto, não bloqueia,
confirmado também por invocação real de CLI com `--dump-api` (exit 0, "converted 18 nodes").

**Não coberto:** o caminho `--force` de fato permitindo a submissão contra um servidor vivo --
exigiria tocar rede/servidor, fora do permitido nesta rodada. `main()` além do ponto de recusa
(POST real, poll de `/history`, heurística de cache) segue não coberto por este arquivo, como já
dizia o `NOT_COVERED` da suite -- só o texto foi atualizado para não afirmar mais que a questão de
severidade do ticket 03 estava fora de escopo. Tickets 04 (heurística de cache) e 06 (decompor
`main()`) intocados, como instruído.
