# `hf_parallel_get.py` resolve o problema que `fetch_*` descrevem, mas nenhum dos dois o usa

Type: task
Status: open

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
