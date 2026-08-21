# Teste atrás de `cfg(feature)` reporta `ok` rodando zero testes

Type: task
Status: open

## Question

`crates/cortiq-engine/tests/gpu_q4tp_batch.rs:11` tem `#![cfg(feature = "gpu")]`. Sem
`--features gpu`:

```
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured
```

`ok`. Verde. Zero cobertura. Foi assim que a recusa do ticket `14` ficou registrada como
propriedade do lote por dias — ninguém tinha rodado o teste, e a saída não dizia isso.

Isto é o mesmo defeito que o `CLAUDE.md` da bancada descreve: "uma coluna de PASS sem dizer o
que não cobriu lê como verificado". Aqui é pior, porque não há nem a coluna.

Quantos arquivos de teste da crate estão nessa condição não foi levantado — `gpu_q4tp_parity.rs`
já teve o mesmo sintoma.

## Critério de fechamento

Fecha quando rodar a suíte sem `--features gpu` produzir, para cada arquivo assim, uma linha
dizendo que foi pulado e por quê — não `ok`. Um `#[ignore]` com mensagem, um teste sentinela que
imprime e passa, ou o que a crate já usar em outro lugar.

Não fecha com decisão escrita: `ok` com zero testes é o defeito, e ele não some sozinho.
