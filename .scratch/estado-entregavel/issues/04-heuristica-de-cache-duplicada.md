# A trava anti-cache é um chute, e está duplicada em duas linguagens

Type: task
Status: resolved

## Question

`tools/comfy_run_workflow.py:344` decide se o ComfyUI realmente renderizou usando
`if wall < 5.0 and files:` — constante mágica, sem nome, dentro do bloco de
impressão, sem efeito no código de saída.

Dez linhas acima, o mesmo `main()` já calcula
`execution_start → execution_success` a partir do que o servidor reporta, e joga o
valor fora num `print`. **O servidor afirma a duração; o código a estima.**

E `F:\cortiq-cmf\run_e2e_comfy.ps1:79` reimplementa a mesma heurística, com o mesmo
`5` cravado, em PowerShell. A mesma trava de segurança duplicada através de uma
fronteira de linguagem, sem nada que mantenha os dois números iguais.

## Por que isso é correção e não estilo

Esta checagem existe porque uma passada com prompt idêntico voltou em 0,6 s com as
49 saídas listadas e `status success` — o cache de resultado por nó do ComfyUI. Lido
de fora, isso é "ComfyUI faz em 0,6 s o que o cortiq faz em 510 s", 850×, e teria
sobrevivido a qualquer conferência superficial. O servidor confirma:
`Prompt executed in 0.01 seconds`.

## Independente do ticket 05

O modelo tipado do formato API (`05`) **não** dissolve este achado: aqui não se trata
da forma do prompt, e sim de usar o dado do servidor em vez de um limiar. Fica na
fronteira, tomável em paralelo.

## Critério de fechamento

Fecha quando as três forem verdade:

1. A duração do lado servidor é valor de primeira classe no código, não um `print`.
2. A detecção de cache usa esse valor, não `wall`.
3. Um cache-hit sai por **código de saída próprio**, e `run_e2e_comfy.ps1` apaga a
   cópia dele passando a olhar só o código de saída.

Não fecha com decisão escrita: é a única trava do arquivo contra reportar um cache
como render, e ela já quase falhou uma vez.

## Resolução

Os três itens do critério, um por um, contra `tools/comfy_run_workflow.py`:

1. **Duração do servidor é valor de primeira classe.** `main()` agora calcula
   `server_side_s: float | None` a partir de `execution_start`/`execution_success`
   ANTES de qualquer print (era `print(f"server-side {...}")` direto dentro do loop,
   valor descartado em seguida). `server_side_s` é usado duas vezes depois: no print
   e na detecção de cache.
2. **Detecção usa esse valor, não `wall`.** `cache_hit = bool(files) and
   server_side_s is not None and server_side_s < CACHE_HIT_THRESHOLD_S` — o `5.0`
   mágico virou `CACHE_HIT_THRESHOLD_S`, uma constante nomeada e documentada no
   módulo. `wall` continua existindo (linha de resumo `WALL ...`), mas não entra
   mais na decisão. Quando o servidor não manda essas mensagens (`server_side_s is
   None`), o código NÃO cai de volta pra `wall` — imprime um WARN dizendo que a
   detecção não rodou, e a lista "NOT covered by this run" ganha uma linha dizendo
   que um cache-hit pode escapar como exit 0 nesse caso.
3. **Cache-hit sai por código de saída próprio.** `main()` agora retorna `1` se
   `status != "success"`, `5` se `cache_hit`, senão `0` — antes o print do banner
   "THIS TIMED NOTHING" não tinha nenhum efeito no retorno (`return 0 if status ==
   "success" else 1`, sempre). `F:\cortiq-cmf\run_e2e_comfy.ps1` teve sua cópia da
   heurística apagada: o bloco que fazia
   `if ($rc -ne 0 -or $sw.Elapsed.TotalSeconds -lt 5)` virou `if ($rc -ne 0)`, lendo
   só o código de saída do processo Python (que agora cobre o caso cache-hit via
   `exit 5`). O stopwatch do PowerShell continua rodando e sendo logado (linha
   `CLIENT-SIDE WALL`), mas só como informação — não decide mais nada.

Comando exato que provou (LIDO+EXECUTADO, offline, sem GPU nem servidor — a rota
`/prompt`+`/history` real onde `cache_hit` de fato dispara não foi exercitada nesta
rodada, ver abaixo):

```
python_embeded\python.exe -s tools\test_comfy_run_workflow.py
```

24/24 PASS (21 anteriores + 3 novos: `test_cache_hit_threshold_is_a_named_module_constant`,
`test_cache_detection_uses_server_side_duration_not_wall`,
`test_cache_hit_gets_its_own_exit_code` — provas ESTÁTICAS sobre o texto-fonte de
`main()`, não uma corrida contra servidor vivo).

Regressão obrigatória, antes e depois da mudança:

```
python_embeded\python.exe -s <scratchpad>\golden_check.py
```

Ambas as corridas: `PASS  prompt convertido identico (18 nos antes, 18 depois)`,
`FAIL  avisos identicos` (esperado desde o ticket 03 — `str` virou `Note` — sem
mudança entre antes/depois desta rodada).

## O que ficou sem cobertura

- **A rota real `/prompt` + `/history`.** `server_side_s < CACHE_HIT_THRESHOLD_S` e
  o `return 5` nunca foram exercitados contra um servidor vivo devolvendo um
  cache-hit de verdade (nem contra um render de verdade) — sem GPU nesta rodada,
  por regra da orquestração. As três provas novas são leitura de padrão no código
  fonte via `inspect.getsource`, não execução do caminho HTTP.
- **`run_e2e_comfy.ps1` não foi executado.** A edição nele (troca da condição do
  `if`) foi LIDA e revisada, não rodada — rodar precisa do servidor ComfyUI vivo em
  8190, fora da lista de comandos permitidos desta rodada. O defeito conhecido e
  separado do ticket 29 (`ErrorActionPreference 'Stop'` + `Tee-Object` matando o
  script) não foi tocado, como instruído.
- **`main()` continua um bloco só.** A decomposição é o ticket 06, explicitamente
  fora do escopo desta rodada.
