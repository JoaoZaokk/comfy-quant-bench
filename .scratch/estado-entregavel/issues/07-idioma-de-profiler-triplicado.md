# `CONV3D_PROF` é a terceira cópia de um idioma que já existia na crate

Type: task
Status: resolved

## Question

Em `F:\cortiq-cmf`, branch `pv-nt-audit-local`:

```
vae3d.rs:26   VAE3D_PROF  [AtomicU64; 8] + vae3d_prof_on() + vprof() + vae3d_prof_report()
mmh3.rs:43    MMH3_PROF   [AtomicU64; 9] + mmh3_prof_on()  + ...
ltxvae.rs:49  CONV3D_PROF [AtomicU64; 3] + conv3d_prof_on()+ cprof() + conv3d_prof_report()
```

Idêntico em forma: array de tamanho fixo, `OnceLock` lendo env, `xprof(slot, t)`, e um
relatório que divide por 1e6 e imprime percentuais.

O agravante: o de `vae3d.rs` foi **lido** durante este trabalho — para descobrir que
não servia, porque é o VAE de outro modelo — e mesmo assim a terceira cópia foi
escrita em vez de extrair.

```rust
pub struct PhaseProf<const N: usize> {
    slots: [AtomicU64; N],
    names: [&'static str; N],
    env: &'static str,
}
```

Três declarações substituem três blocos de ~30 linhas.

## Critério de fechamento

Fecha quando `CONV3D_PROF` for uma declaração de `PhaseProf` e não uma cópia do
idioma. Migrar `vae3d` e `mmh3` é bônus, não requisito — são código do autor upstream
e mexer neles amplia o diff de um fork.

Pode fechar **sem** mudança de código se a extração provar que os três não têm forma
comum de verdade (por exemplo, se os relatórios divergirem de um jeito que a
abstração não cobre). Registrar a divergência concreta, não a impressão.

## Resolução — 2026-08-21, commit `188b926`

Feito por agente sonnet interrompido antes de reportar. **Verificado pelo orquestrador,
por execução.**

`CONV3D_PROF` é agora `crate::phase_prof::PhaseProf<3>::new("CMF_CONV3D_PROF")`
(`ltxvae.rs:49`), `lib.rs` ganhou `pub mod phase_prof;`, e os sítios de chamada usam
`.record(0|1|2, tp)`. `cargo check --release -p cortiq-engine` limpo em 11,9 s, só os 269
warnings pré-existentes.

A string do `format!` do relatório é **byte-idêntica** à anterior — as duas foram
comparadas lado a lado contra `git show HEAD`.

**`vae3d` e `mmh3` não foram migrados, e o motivo é concreto, não impressão:** os dois
leem os contadores com `.load()` e nunca limpam, então o relatório deles é total corrido
desde o início do processo; `CONV3D_PROF` lê com `.swap(0, ..)`, então cobre só o
intervalo desde o relatório anterior. Dobrar os dois sobre um `take_seconds` que zera
mudaria em silêncio o que a saída deles significa. Registrado no docstring do módulo
`phase_prof.rs`, marcado **LIDO** — lido, não re-executado.

**Não coberto:** o profiler não foi exercitado. `CMF_CONV3D_PROF=1` num decode real
carrega o container de 22 GB e precisa da placa, que está com outra sessão. O que está
provado é que compila e que o formato não mudou — não que o número sai certo.
