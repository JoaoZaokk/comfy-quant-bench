# `hf_parallel_get.py` resolve o problema que `fetch_*` descrevem, mas nenhum dos dois o usa

Type: task
Status: resolved

## Question

`tools/hf_parallel_get.py` existe especificamente para um problema nomeado no próprio docstring:

> `hf_hub_download` uses a single connection, which on a 300-500 Mbit link tops out well below the
> line rate, and its resume is per-file: a dropped connection here left three separate 6.9 / 2.9 /
> 18.3 GiB `.incomplete` files for the same blob, none of which continued the others.

Isso é exatamente o mecanismo que `tools/fetch_ltx25.py` e `tools/fetch_minimax_h3.py` usam:

```
$ grep -n "^from\|^import" tools/fetch_ltx25.py
from huggingface_hub import hf_hub_download

$ grep -n "^from\|^import" tools/fetch_minimax_h3.py
from huggingface_hub import hf_hub_download
```

Nenhum dos dois importa `hf_parallel_get`. E:

```
$ git grep -w hf_parallel_get
(nada — nem import, nem menção em doc, nem em .bat)
```

Zero sítios de chamada no repo inteiro, incluindo documentação. Tem `argparse` + `def main()`
(não é morto pela regra "chamado só pelo usuário"), mas os dois downloaders que existem para o
mesmo tipo de arquivo grande (LTX-2.5 e MiniMax H3, ambos dezenas de GiB) usam o caminho de
conexão única que este arquivo foi escrito para substituir.

Duas explicações possíveis, e a triagem não decide qual:
1. `hf_parallel_get.py` foi escrito depois dos dois `fetch_*` e nunca foi retrofitado neles.
2. `hf_parallel_get.py` foi tentado e abandonado por algum motivo não registrado (a resumabilidade
   por chunk tem mais partes móveis — `.parts.json` sidecar, header `Authorization` só para
   huggingface.co e não para o CDN assinado — que podem ter mordido em algum teste não
   documentado).

## Critério de fechamento

Fecha com decisão escrita, não necessariamente com código: alguém compara os dois caminhos
(single-connection via `hf_hub_download` vs. `hf_parallel_get`) e decide:

- migrar `fetch_ltx25.py` / `fetch_minimax_h3.py` para `hf_parallel_get.py` (fecha o problema que
  o próprio docstring descreve, com o resultado sendo o download realmente mais robusto); ou
- documentar por que os dois `fetch_*` continuam em `hf_hub_download` apesar do problema descrito
  (ex.: `hf_parallel_get.py` nunca foi testado ponta-a-ponta, ou tem um defeito conhecido) — e
  nesse caso `hf_parallel_get.py` vira candidato a `morto` de verdade numa próxima triagem, não
  "morto?" como está agora.

Não fecha deixando os três arquivos como estão: ferramenta escrita para o problema, problema
descrito nos dois lugares que a teriam, e nenhuma conexão entre eles.

## Resolução

Status: resolved. Escolhida a saída (a) — ligar os dois `fetch_*` a `hf_parallel_get.py`.

**Ausência de chamador, reconfirmada (LIDO + EXECUTADO um grep, não copiado do ticket):**
`git -C F:\COMFY_PORTABLE grep -w hf_parallel_get` e `git -C F:\COMFY_PORTABLE\ComfyUI grep -w
hf_parallel_get` (repo separado, HEAD dele) rodados de novo nesta rodada — a segunda deu zero
hits, a primeira só as menções em `.scratch/**/*.md`. Estendido além do que o ticket pedia: `find`
+ `grep` direto em todos os `.bat`/`.ps1` da raiz (7 arquivos), que `git grep` teria pulado por
estarem fora do allowlist do `.gitignore` — zero hits também. Não cobri `F:\cortiq-cmf` (repo
Rust separado, fora do escopo de arquivos permitidos deste ticket) nem `D:\` fora deste repo.

**Por que (a) e não (b):** a API de `hf_parallel_get.py` já cobria o que os dois `fetch_*`
precisavam sem precisar reescrever a forma como cada um decide onde cada arquivo cai
(`ROOT/filename` num, `local_dir/filename` variável no outro) — `download(repo, file, dest_dir,
expected_size=...)` calcula `dest = (dest_dir / file).resolve()`, que é exatamente a mesma conta
que `hf_hub_download(repo_id=repo, filename=file, local_dir=dest_dir)` fazia. Ligar exigiu menos
reescrita do que documentar-e-matar exigiria.

**Mudança de código** (as três permitidas):
- `tools/hf_parallel_get.py`: extraído o corpo de `main()` (idêntico, só `args.x` → `x` nos nomes)
  para `download(repo, file, dest_dir, *, revision="main", connections=8, chunk_mb=256, retries=6,
  expected_size=None) -> int`. `main()` virou wrapper fino: `parse_args()` + chamada a `download()`.
  Contrato explícito no docstring da função: retorna 130 em vez de levantar em Ctrl-C (`except
  KeyboardInterrupt` interno, inalterado), e levanta `SystemExit` (não `Exception`) em mismatch de
  tamanho — os dois pontos que um chamador ingênuo erraria.
- `tools/fetch_ltx25.py`: troca `from huggingface_hub import hf_hub_download` por `from
  hf_parallel_get import download as parallel_download`; o `except Exception` em volta da chamada
  virou `except (SystemExit, Exception)` (SystemExit não é subclasse de Exception — não seria
  capturado sem isso); `rc == 130` agora aborta o lote inteiro (`return 130`) em vez de deixar o
  loop seguir pro próximo arquivo como aconteceria se só o `except` fosse mantido.
- `tools/fetch_minimax_h3.py`: mesma troca de import; `except SystemExit` novo em volta da
  chamada (o `hf_hub_download` direto de antes não podia levantar `SystemExit`, então não havia
  necessidade); `rc == 130` retorna 130 em vez de seguir para o próximo arquivo.

**Prova, EXECUTADA nesta rodada (nenhum download disparado — todo código que toca rede mora
dentro de `download()` ou atrás de `if __name__ == "__main__"`, nenhum dos dois roda ao importar):**
```
.\python_embeded\python.exe -s -m py_compile tools\hf_parallel_get.py    -> COMPILE_OK
.\python_embeded\python.exe -s -m py_compile tools\fetch_ltx25.py         -> COMPILE_OK
.\python_embeded\python.exe -s -m py_compile tools\fetch_minimax_h3.py    -> COMPILE_OK
```
E, além do que o ticket pedia (leitura de assinatura lado a lado), a wiring foi confirmada em
tempo de execução — importar os dois `fetch_*` e checar identidade de objeto contra
`hf_parallel_get.download`, sem chamar nada:
```python
import fetch_ltx25, fetch_minimax_h3, hf_parallel_get
fetch_ltx25.parallel_download is hf_parallel_get.download        # True
fetch_minimax_h3.parallel_download is hf_parallel_get.download   # True
inspect.signature(hf_parallel_get.download)
# (repo: str, file: str, dest_dir: Path, *, revision: str = 'main', connections: int = 8,
#  chunk_mb: int = 256, retries: int = 6, expected_size: int | None = None) -> int
```
Sítios de chamada, lidos: `parallel_download(REPO, filename, ROOT, expected_size=expected)`
(`fetch_ltx25.py`) e `parallel_download(repo, filename, local_dir, expected_size=expected)`
(`fetch_minimax_h3.py`) — posicional `repo, file, dest_dir` em ambos, batendo com a assinatura.

**O que NÃO foi coberto:**
- Nenhum download real rodou — nem o caminho antigo nem o novo foram exercitados ponta-a-ponta
  nesta rodada (proibido: rede/GPU fora de escopo, e os arquivos são de dezenas de GiB). A
  segunda explicação que o ticket levantava ("tentado e abandonado por defeito não documentado")
  segue não-testada em produção; só a leitura de código e a resolução de import foram verificadas.
- `tools/test_comfy_run_workflow.py` (a única suite que a lista de comandos permite rodar) não
  cobre `fetch_*`/`hf_parallel_get` — não rodada, por ser irrelevante ao que mudou aqui, não por
  omissão.
- Comportamento de resume entre chunks, de re-resolução de URL assinada em 401/403, e de
  `.parts.json` sob interrupção real — tudo isso já existia em `hf_parallel_get.py` antes deste
  ticket e não foi re-verificado; só a extração para `download()` foi (mesmo corpo, nomes
  trocados, comparação linha a linha ao editar, não um teste rodado).
