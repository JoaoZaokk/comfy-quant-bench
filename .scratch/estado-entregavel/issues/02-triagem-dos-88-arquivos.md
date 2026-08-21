# Triagem barata dos 93 arquivos rastreados

Type: task
Status: resolved

## Question

O repo da bancada rastreia **93** arquivos (recontado em 2026-08-21 com
`git ls-files | wc -l`; o título antigo dizia 88, já estava errado antes mesmo do dia da
recontagem oficial do `CLAUDE.md`): 62 Python + 3 fixtures JSON + `benchmark_prompt.txt` + 2
`.ps1` em `tools/` (68 no total), 8 JSON em `calib/`, 3 em `custom_nodes/`, e 14 documentos e
scripts na raiz.

Levar todos à barra da revisão termonuclear é programa, não mapa. Fazer uma passada
**barata** que classifique cada arquivo em uma linha:

- `saudável` — nada a fazer
- `suspeito` — há motivo concreto para olhar de perto; **gradua para ticket próprio**
- `morto` — não é chamado por nada; candidato a apagar

## O que NÃO é

Não é revisão. Um `suspeito` aqui não diz qual é o problema, só que há motivo. O
motivo sai no ticket que graduar.

## Por que barato primeiro

É literalmente o que a névoa de guerra existe para fazer: limpar barato o que ainda
não dá para ticketar. Precedente do mesmo dia: a triagem por estágio do decoder do
VAE evitou escrever um kernel de convolução que não precisava existir, porque
mostrou que três blocos carregavam 81,6% do tempo.

## Critério de fechamento

Fecha quando os 93 arquivos tiverem uma linha cada, e todo `suspeito` tiver virado
ticket na fronteira. Um arquivo sem classificação é ticket aberto, não fechado.

Fecha **sem** mudança de código por definição: triagem não conserta nada.

## Resolução

EXECUTADO (leitura + alguns comandos read-only; nenhuma mudança de código, por definição do
ticket). Tabela completa em
[triagem-89.md](../triagem-89.md) (nome do arquivo mantido porque foi o pedido; o número no nome
está desatualizado igual o título antigo deste ticket estava — o conteúdo usa 93).

Contagem reconferida:

```
$ git -C F:\COMFY_PORTABLE ls-files | wc -l
93
```

Método: cabeçalho/docstring + imports + `def`s + `git grep -w <nome>` no repo inteiro para achar
sítio de chamada, por arquivo. Onde `AUDITORIA_2026-08-18.md` já tinha um achado com linha exata,
reconferido contra o código de hoje em vez de copiado — a seção 6 daquele documento já registra
uma leva grande de correções aplicadas no mesmo dia, e boa parte do que o auditor listou como
achado já está corrigido. Tratar aquele documento como still-open sem reconferir teria inflado a
contagem de `suspeito` com coisa já resolvida.

Resultado: 93/93 arquivos classificados, nenhum sem linha.
- **saudável**: a maioria — 14 raiz, 8 `calib/`, 3 `custom_nodes/`, e a maior parte de `tools/`
  (incluindo ~15 arquivos onde o achado do `AUDITORIA_2026-08-18.md` foi reconferido como já
  corrigido: `quant_w4a4.py`, `quant_w4a4_smooth.py`, `quant_mixed.py`, `verify_w4a4.py`,
  `check_w4a8.py`, `m_crossover.py`, `gpu_lock.py`, `model_audit.py`, `precheck_workflows.py`,
  `quality_battery.py`, `inspect_quant.py`, `attn_dtype_ab.py`, `convrot_ops_probe.py`, entre
  outros — cada um com a linha exata do código que prova a correção, na tabela).
- **8 arquivos já cobertos** por ticket existente (01, 03, 04, 05, 06, 12) — apontados, não
  duplicados.
- **14 linhas `suspeito`**, em **9 arquivos distintos de motivo** mas agrupadas onde o defeito é
  o mesmo cluster, graduaram para 9 tickets novos (18 a 26):
  - 18 — `activation_balance.py` + `plot_weight_balance.py`: crest medido pré-rotação
  - 19 — `core_patch.py`: `revert` sobrescreve sem checar hash do arquivo atual
  - 20 — sonda de "backend nativo pronto" / `instrument()` duplicada e já divergente
    (`gemma_chat.py`, `diffusion_smoke.py`, `stage_probe.py`, `convrot_ops_probe.py`,
    `te_smoke.py`)
  - 21 — `nunchaku_compare.py`: pico de VRAM só sobe; `--attention sage` sem verificação
  - 22 — `svdq_to_bf16.py`: 3 achados de baixa confiança do auditor seguem sem fechar
  - 23 — `test_svdq_verify.py`: não chama `recover_weight`, a função real
  - 24 — `quant_w4a8.py` (`as_bytes` morto) + `quant_int8.py` (sem preflight com `--device cuda`)
  - 25 — `fbcache_visual.py`: único `fbcache_*` sem resultado registrado em `FBCACHE_FINDINGS.md`
  - 26 — `hf_parallel_get.py`: resolve o problema que `fetch_ltx25.py`/`fetch_minimax_h3.py`
    descrevem, mas nenhum dos dois o importa

- **0 `morto`** confirmado; **0 `morto?`** — todo arquivo sem sítio de chamada externo achado
  (`_check_accel.py`, `fetch_ltx25.py`, `fetch_minimax_h3.py`, `hf_parallel_get.py`,
  `ltx25_queue.py`, `svdquant_probe.py`, `weight_dtype_probe.py`) tinha `argparse`/`__main__`
  confirmado por leitura, então caiu na regra "chamado só pelo usuário na linha de comando não é
  morto" em vez de `morto?`. `hf_parallel_get.py` é o caso mais perto da linha — o ticket 26
  deixa em aberto que ele pode virar `morto` de verdade numa próxima rodada se a decisão for não
  migrar os `fetch_*` para ele.

Comandos executados (todos leitura, nenhum mudou estado):
```
git -C F:\COMFY_PORTABLE ls-files | wc -l
git -C F:\COMFY_PORTABLE ls-files | sort | uniq -d | wc -l
git -C F:\COMFY_PORTABLE grep -n -w -- "<basename>" -- ':!<path>'   # por arquivo .py
grep -n "..." <arquivo>                                             # dezenas, para reconferir cada achado do AUDITORIA
.\python_embeded\python.exe -s -c "..."                             # 1x, para reconfirmar os 2 predicados de quality_battery.py (leitura pura, sem GPU, sem modelo)
```

O que ficou sem cobertura: não editei nenhum arquivo de código (fora do escopo do ticket, por
definição). Não abri os 8 arquivos `calib/*.json` quanto a frescor de conteúdo (se ainda batem
com os checkpoints atuais) — isso é outro escopo, anotado na tabela. Não reconferi por execução
nenhum achado que precisasse de GPU ou modelo — os que restam `[GPU]` em
`AUDITORIA_2026-08-18.md` continuam abertos e não entraram nesta triagem como novo achado, só
como contexto. Dois outros agentes estavam editando `tools/comfy_run_workflow.py` e lendo
`tools/quant_audit.py` durante esta rodada — não toquei nenhum dos dois além de apontar para os
tickets que já os cobrem.
