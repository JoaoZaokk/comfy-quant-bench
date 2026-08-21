# `std::env::var` dentro da função que o patch existe para acelerar

Type: task
Status: resolved

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

## Resolução

LIDO primeiro: `git grep CMF_LTX_PV_NT` em `crates/cortiq-engine/src` mostrou que a flag ainda
existe (2 leituras) -- ticket 09 NÃO a apagou. Fechar exigia mudança.

LIDO segundo: a referência pedida pelo ticket, `ltxvae.rs::im2col_fast`, não existe mais --
`grep im2col_fast` em todo `crates/cortiq-engine/src` não bate em nada, e `ltxvae.rs` não tem
nenhum `OnceLock`. Confirmado que o ticket 09 mexeu ali (comentário deixado no próprio código,
linhas 695-698 de `ltxdit.rs`, registra isso). Usei como referência o idioma que já é o padrão
no resto da crate: `fcd_ops.rs` e `dsv4.rs`, ambos com `static ON: OnceLock<bool>` dentro de fn.

Achado ao abrir o arquivo: `ltxdit.rs` já tinha `pv_nt_enabled()` (linhas 701-704, com
`OnceLock`) e as duas funções nomeadas `gather_v` / `gather_v_transposed` (linhas 708-724) --
exatamente o que o ticket pede -- mas nenhuma das três era chamada. `Attn::forward` continuava
com `std::env::var("CMF_LTX_PV_NT")` cru na linha 961 e o `if pv_nt { ... } else { ... }` inline
nas linhas 969-981 fazendo o gather com um loop escrito na mão. Trabalho parcial de sessão
anterior, não fiado -- as três funções ficavam com warning `never used` (confirmado, ver EXECUTADO
abaixo, antes da mudança essas três apareceriam ali; não guardei o log de antes, mas o comentário
do próprio código nas linhas 691-700 já documentava a intenção de que `forward` as usasse).

Mudança (`ltxdit.rs`, dentro de `Attn::forward`):
- linha 961: `let pv_nt = std::env::var(...)` -> `let pv_nt = pv_nt_enabled();`
- linhas 969-981: os dois loops manuais de gather (14 linhas) -> `gather_v_transposed(&v, &mut vh, h, m, inner, dh)` / `gather_v(&v, &mut vh, h, m, inner, dh)` (2 linhas), mesmo corpo que já existia como função solta.

Comportamento idêntico: mesma condição (`CMF_LTX_PV_NT=1`), mesmo corpo de cada braço, só ligado
ao código que já existia e estava morto. Nenhuma outra leitura de `CMF_LTX_PV_NT` sobrou --
confirmado por `grep CMF_LTX_PV_NT ltxdit.rs`, resta só a de dentro de `pv_nt_enabled`.

EXECUTADO (não só lido): os dois comandos exigidos pelo ticket, ambos concluindo com
`Finished \`release\` profile [optimized] target(s)` e zero erros:

```
cargo check --release -p cortiq-engine              -> Finished em 7.44s, 269 warnings (pré-existentes, nenhum novo em ltxdit.rs)
cargo check --release -p cortiq-engine --features gpu -> Finished em 40.54s, 59 warnings (pré-existentes, nenhum novo em ltxdit.rs)
```

Conferi especificamente por `ltxdit` na saída de cada build: a única linha que bate é um warning
pré-existente e não relacionado, de import não usado na linha 191 (`use crate::ltxdit::{...}` em
outro arquivo), nada perto da mudança.

NÃO coberto: `cargo test --features gpu` não rodou (proibido nesta rodada -- executaria kernel,
GPU em uso por outro agente). Não há verificação de runtime de que os dois caminhos (`pv_nt`
ligado/desligado) ainda produzem o mesmo resultado numérico -- só o `cargo check` de tipos, que
não roda código. `tests/` não foi tocado (instrução explícita: outro agente lá agora), então não
sei se algum teste existente exercita `pv_nt_enabled`/`gather_v*` diretamente.
