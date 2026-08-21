# `std::env::var` dentro da função que o patch existe para acelerar

Type: task
Status: open

## Question

`crates/cortiq-engine/src/ltxdit.rs:745`, dentro de `Attn::forward`:

```rust
let pv_nt = std::env::var("CMF_LTX_PV_NT").as_deref() == Ok("1");
```

`Attn::forward` roda ~288 vezes por passo de denoise (48 blocos × ~6 attentions),
~864 por render. Cada uma aloca uma `String` e pega o lock global do env.

Em tempo absoluto é ruído. Mas está na **única função que este patch existe para
medir**, e é inconsistente com `im2col_fast()` no mesmo branch, que usa `OnceLock`
corretamente. Não é nit: é a diferença entre medir a função e medir a função mais o
instrumento.

Junto, do mesmo diff: a escolha entre os dois gathers de `v` deveria sair do laço de
cabeças e virar duas funções nomeadas (`gather_v` / `gather_v_transposed`), para o
corpo de `forward` não crescer com um `if` de implementação.

## Critério de fechamento

Fecha quando a leitura do env acontecer uma vez por processo, como em
`ltxvae.rs::im2col_fast`.

Pode fechar **sem** mudança se o ticket `09` levar a apagar a flag `CMF_LTX_PV_NT`
também — nesse caso o problema deixa de existir, e isso é decisão registrável.
