# O laço im2col existe três vezes, e a flag que o duplica não precisa existir

Type: task
Status: resolved

## Question

Em `F:\cortiq-cmf`, o mesmo algoritmo aparece três vezes:

1. `ltxvae.rs`, ramo `else` de `Conv3d::forward` — o original
2. `ltxvae.rs`, `fill_patches_fast` — o novo, provado bit-a-bit idêntico
3. `tests/ltxvae_im2col_parity.rs::fill_patches_reference` — transcrição do #1

A justificativa escrita no teste ("um teste que chegasse ao caminho rápido pela flag
compararia o motor com ele mesmo") acerta o **risco** e erra o **remédio**. A correção
é extrair, não copiar:

```rust
pub fn fill_patches_reference(x: &Vol, patches: &mut [f32], p0, n, k, c_in);
pub fn fill_patches_fast(     x: &Vol, patches: &mut [f32], p0, n, k, c_in, npos);
```

`forward` chama uma; o teste compara as duas. Some a terceira cópia, e o teste passa
a exercitar o código que roda de verdade.

E aí cai de graça a flag `CMF_LTXVAE_IM2COL`: ela só servia para alcançar o ramo
antigo. Com a referência sendo função nomeada, `forward` chama a rápida sempre.
Somem: um `OnceLock`, uma env var, um `if` dentro do laço de chunks.

## O que a medição sustenta

O caminho rápido é **bit-a-bit idêntico** ao original, provado em 6 formas incluindo
`1x1` e `2x2` (sem posição interior nenhuma), com buffer pré-preenchido com sentinela
para pegar slot que nenhum dos dois escreve. Risco de trocar o padrão: zero.

O ganho de tempo é fraco: `im2col` de 74,4-87,0 s (4 corridas) para 68,0-74,3 s (3
corridas), ~10 s de média, e o total do decode não resolve porque o GEMM sozinho
varia 139,3-160,8 s. **Não vender como ganho de decode.**

## Critério de fechamento

Fecha quando existir uma cópia da referência (não três), o teste comparar duas
funções do motor, e a env var ter sumido.

Não fecha com decisão escrita: a duplicação tripla é o defeito, e ela não some
sozinha.

## Resolução

EXECUTADO em `F:\cortiq-cmf` (branch `pv-nt-audit-local`).

O que mudou, em `crates/cortiq-engine/src/ltxvae.rs`:
- O laço original (antigo ramo `else` de `Conv3d::forward`) virou a função
  nomeada `pub fn fill_patches_reference(x, patches, p0, n, k, c_in)`, com
  docstring dizendo que `forward` não a chama mais e que ela só existe para o
  teste de paridade comparar contra `fill_patches_fast`.
- `Conv3d::forward` agora chama `fill_patches_fast` incondicionalmente. O
  `if im2col_fast() { ... } else { ... }` sumiu junto com o ramo `else`
  inteiro (a segunda cópia do laço).
- `fn im2col_fast()`, o `OnceLock<bool>` que ele usava e a leitura de
  `std::env::var("CMF_LTXVAE_IM2COL")` foram removidos. `grep -rn
  "CMF_LTXVAE_IM2COL" F:\cortiq-cmf` (fora de `target/`) só acha a menção na
  docstring do teste dizendo que a flag não existe mais.

O que mudou, em `crates/cortiq-engine/tests/ltxvae_im2col_parity.rs`:
- A terceira cópia (`fn fill_patches_reference` local, transcrição do
  original) foi apagada. O teste agora importa
  `cortiq_engine::ltxvae::{fill_patches_fast, fill_patches_reference}` e
  compara as duas funções do motor, não uma cópia local contra uma função do
  motor.
- As 6 formas (incluindo `1x1` e `2x2`), o `p0` variando, e o buffer
  pré-preenchido com sentinela `-7.5` ficaram exatamente como estavam —
  nenhuma dessas partes foi tocada.

Comando que provou (EXECUTADO, não lido):

```
cd F:\cortiq-cmf
cargo check --release -p cortiq-engine
cargo test --release -p cortiq-engine --test ltxvae_im2col_parity -- --nocapture
```

`cargo check` compilou limpo (só os 269 warnings pré-existentes do crate,
nenhum novo, nenhum erro). O teste rodou e passou nas 6 formas:

```
ok  [c=3 f=5 h=8 w=8]  ordinary: interior and all four spatial borders
ok  [c=2 f=1 h=4 w=4]  single frame: time replicate collapses to one source
ok  [c=2 f=3 h=1 w=4]  height 1: every kh tap but the middle is padding
ok  [c=2 f=3 h=4 w=1]  width 1: every kw tap but the middle is padding
ok  [c=2 f=3 h=1 w=1]  1x1: only the middle tap of nine survives
ok  [c=5 f=2 h=2 w=2]  2x2: no interior position at all
test fast_im2col_is_bit_identical_to_the_original ... ok
```

`git status --porcelain` em `F:\cortiq-cmf` confirma só dois arquivos
modificados (`ltxvae.rs`, `ltxvae_im2col_parity.rs`); `git diff --stat --
crates/cortiq-engine/src/ltxdit.rs` não mostra nada — o arquivo do outro
agente não foi tocado.

Critério de fechamento batido nos três pontos: uma cópia da referência (não
três — `fill_patches_reference` só existe em `ltxvae.rs` agora, o teste a
importa), o teste compara duas funções do motor (`fill_patches_fast` e
`fill_patches_reference`, ambas de `cortiq_engine::ltxvae`), e a env var
`CMF_LTXVAE_IM2COL` sumiu do código (só resta a menção histórica na
docstring do teste e um binário compilado antigo em `target/`, que não é
código-fonte).

O que ficou sem cobertura: nenhum benchmark de tempo foi rodado nesta
resolução — o ticket já trazia a medição (`im2col` 74,4-87,0 s → 68,0-74,3 s,
GEMM 139,3-160,8 s dominando) e não pedia repeti-la, só pedia a extração e a
remoção da flag. Não toquei em `ltxdit.rs` nem em nada fora dos três
arquivos permitidos. Não commitei nada (orquestrador commita).
