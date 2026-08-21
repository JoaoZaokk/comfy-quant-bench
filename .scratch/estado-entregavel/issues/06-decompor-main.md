# `main()` tem 127 linhas e faz oito trabalhos

Type: task
Status: open
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
