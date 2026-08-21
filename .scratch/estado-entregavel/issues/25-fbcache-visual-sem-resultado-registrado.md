# `fbcache_visual.py` é o único `fbcache_*` sem resultado registrado

Type: task
Status: open

## Question

`tools/fbcache_visual.py` existe (docstring: "End-to-end FBCache A/B on HunyuanVideo 1.5 with real
conditioning and decoded frames") e tem `argparse` + `main()`, então não é morto pela regra do
ticket ("chamado só pelo usuário na linha de comando não é morto"). Mas é o único dos 8 arquivos
`fbcache_*` que não aparece em `FBCACHE_FINDINGS.md`:

```
$ grep -n "fbcache_visual" FBCACHE_FINDINGS.md
(nada)
```

Comparar com a tabela de cobertura do mesmo documento (`FBCACHE_FINDINGS.md:302-312`), que lista
os outros 7:

| arquivo | listado em FBCACHE_FINDINGS.md |
|---|---|
| `fbcache_routing_test.py` | sim, linha 304 |
| `fbcache_catorder_test.py` | sim, linha 305 |
| `fbcache_tuple_test.py` | sim, linha 306 |
| `fbcache_branch_test.py` | sim, linha 307 |
| `fbcache_kwargs_test.py` | sim, linha 308 |
| `fbcache_clone_test.py` | sim, linha 309 |
| `fbcache_probe.py` | sim, linha 311 |
| `fbcache_audit.py` | sim, linha 311 |
| **`fbcache_visual.py`** | **não** |

`fbcache_visual.py` só é mencionado de um lado — o próprio `fbcache_probe.py` cita "The threshold
sweep in fbcache_probe.py..." dentro do docstring de `fbcache_visual.py`, mas nada aponta o
inverso. Não há evidência de que já rodou, nem resultado (quality/relL2/tempo) registrado em
lugar nenhum do repo.

Isto não é "achado de bug" — é achado de **cobertura**: uma ferramenta cara (carrega Qwen2.5-VL +
ByT5 + o transformer, decodifica pixels reais) que existe há tempo suficiente para os 7 irmãos
terem virado uma tabela de achados, e ela não.

## Critério de fechamento

Fecha com uma decisão escrita — ao contrário da maioria dos tickets deste lote, este não é
código quebrado, é lacuna de execução:

- ou alguém roda `fbcache_visual.py` (precisa de GPU e do checkpoint HunyuanVideo 1.5) e o
  resultado entra em `FBCACHE_FINDINGS.md` junto dos outros 7;
- ou fica documentado explicitamente por que não vale a pena rodar (redundante com
  `fbcache_probe.py` + `fbcache_audit.py`, por exemplo) e a decisão fica escrita no próprio
  docstring do arquivo.

Qualquer uma das duas fecha o ticket. Ficar como está — sem rodar e sem decisão — não fecha.
