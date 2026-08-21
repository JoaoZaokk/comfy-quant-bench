# `add_shaped` tem 8 parâmetros porque falta um tipo

Type: task
Status: resolved

## Question

Em `crates/cortiq-engine/src/ltxdit.rs`, o sítio de chamada:

```rust
t = attn_prof::add_shaped(
    &attn_prof::SCORE, &attn_prof::SCORE_N, &attn_prof::SCORE_FLOP,
    0, n, dh, m, t,
);
```

Três statics **e** um `u8` mágico que significa exatamente a mesma coisa que os três
statics: `kind == 0` ⟺ `(SCORE, SCORE_N, SCORE_FLOP)`. A mesma informação passa duas
vezes, por dois mecanismos, e nada impede passar `1` com os statics de scores.

Não é "muitos parâmetros" — é um tipo que não existe:

```rust
#[derive(Clone, Copy)]
pub enum Gemm { Scores = 0, Values = 1 }

struct Counters { us: AtomicU64, calls: AtomicU64, flops: AtomicU64 }
static GEMMS: [Counters; 2] = [Counters::NEW, Counters::NEW];

pub fn record(g: Gemm, n: usize, k: usize, m: usize, t: Instant) -> Instant
```

Sítio de chamada vira `attn_prof::record(Gemm::Scores, n, dh, m, t)`. Somem: o `u8`
mágico, três statics do sítio de chamada, o `#[allow(clippy::too_many_arguments)]`, e
a possibilidade de descasar `kind` dos contadores.

Junto: `report()` faz três trabalhos e **limpa ao ler** sem dizer isso no nome —
drena nove atômicos com `swap`, trava e limpa o `SHAPES`, e formata. Separar
`drain() -> Snapshot` de `render(&Snapshot) -> String` torna a formatação testável,
o que hoje ela não é.

## Critério de fechamento

Fecha quando o sítio de chamada não passar statics nem `u8`, e `drain`/`render`
estiverem separados.

Pode fechar **sem** mudança de código se `09` acabar removendo a instrumentação
inteira — mas então registrar isso explicitamente, não deixar implícito.

## Resolução

EXECUTADO (não só lido) — `cargo check --release -p cortiq-engine` e
`cargo test --release -p cortiq-engine --lib attn_prof` rodaram e passaram
(comandos exatos abaixo). `09` não removeu a instrumentação (mexe em
`ltxvae.rs`, não em `attn_prof`), então a saída por decisão escrita não
estava disponível — implementei.

Em `crates/cortiq-engine/src/ltxdit.rs`, dentro de `mod attn_prof`:

- Os três statics `SCORE`/`SCORE_N`/`SCORE_FLOP` (e os equivalentes
  `VALUE*`) saíram. No lugar: `enum Gemm { Scores = 0, Values = 1 }` e
  `static GEMMS: [Counters; 2]` (struct `Counters { us, calls, flops }`),
  indexado por `Gemm as usize`. `SHAPES` passou a ter `Gemm` na chave da
  `BTreeMap` (era `u8`).
- `add_shaped(&SCORE, &SCORE_N, &SCORE_FLOP, 0, n, k, m, t)` virou
  `record(Gemm::Scores, n, k, m, t)` — sumiu o `u8` mágico, sumiram os três
  statics do sítio de chamada, sumiu o `#[allow(clippy::too_many_arguments)]`
  (a assinatura nova tem 5 parâmetros), e não há mais como `kind` descasar
  dos contadores: os dois vêm do mesmo `g`.
  - Sítios de chamada (linhas ~769 e ~803 do arquivo, dentro do loop de
    atenção): `attn_prof::record(attn_prof::Gemm::Scores, n, dh, m, t)` e
    `attn_prof::record(attn_prof::Gemm::Values, n, m, dh, t)`.
- `report()` (fazia drenar + formatar, sem dizer no nome) virou duas
  funções: `drain() -> Snapshot` (lê e zera os nove atômicos + `SHAPES`,
  sem formatar nada) e `render(&Snapshot) -> String` (só formata, sem tocar
  atômico ou lock). O único chamador (`println!` do `CMF_LTX_PROF=1`, linha
  ~1132) virou `attn_prof::render(&attn_prof::drain())`. A string produzida
  é byte-a-byte a mesma que `report()` produzia antes (mesmo `format!`,
  mesma ordem de argumentos) — não é uma reformatação, é a mesma saída
  vinda de duas funções em vez de uma.
- Teste novo (`mod tests` dentro de `attn_prof`, `#[cfg(test)]`): dois
  casos que montam um `Snapshot` à mão e chamam `render()` direto — a única
  parte do módulo testável sem GPU, sem atômico e sem uma passada real de
  atenção. Um cobre o formato agregado e a ordenação "mais pesado primeiro"
  das linhas de shape; o outro cobre o `Snapshot` vazio (guardas
  `n>0`/`t>0.0` em `rate`/`per_call`, e a seção de shapes ausente). Marcados
  no próprio doc-comment como "EXECUTED, not traced" com o comando exato.

Comando exato que provou (rodado nesta sessão, sem GPU — só compila e roda
teste de CPU puro, sem alocar VRAM nem subir o ComfyUI):

```
cd F:\cortiq-cmf
cargo check --release -p cortiq-engine
  -> "Finished `release` profile [optimized] target(s) in 7.39s" (269
     warnings pré-existentes, nenhum novo nos símbolos tocados aqui —
     verificado por grep no output por "ltxdit|attn_prof|Gemm|Snapshot|
     ShapeRow|GemmTotals": só ruído de outro código do arquivo, nada dos
     símbolos novos)

cargo test --release -p cortiq-engine --lib attn_prof -- --nocapture
  -> "test ltxdit::attn_prof::tests::render_handles_empty_snapshot ... ok"
  -> "test ltxdit::attn_prof::tests::render_formats_aggregates_and_shape_rows_heaviest_first ... ok"
  -> "test result: ok. 2 passed; 0 failed"
  (build em release, 1m24s — uma corrida só, não citada como tempo de
  referência, só para dizer que compilou e passou)
```

O que ficou sem cobertura: `record()` e `drain()` em si (os atômicos e o
lock) não têm teste automatizado — teriam que rodar dentro de uma atenção
real (GPU), fora do escopo desta sessão por regra dura ("NAO usar a GPU").
A equivalência byte-a-byte entre a saída antiga de `report()` e a nova
`render(&drain())` foi conferida por LEITURA (mesmo `format!`, mesmos
argumentos na mesma ordem), não por um teste de regressão automatizado —
não existia snapshot da saída antiga para comparar contra. `clippy`/`ruff`
não estão disponíveis neste ambiente (regra do projeto), então o
`#[allow(clippy::too_many_arguments)]` some por não ser mais necessário
(5 parâmetros), mas isso não foi confirmado rodando `cargo clippy`.
