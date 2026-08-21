# Teste atrás de `cfg(feature)` reporta `ok` rodando zero testes

Type: task
Status: resolved

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

## Resolução

**Levantamento (EXECUTADO, não lido de documento nenhum)** — `grep -rn 'cfg(feature' *.rs` em
`crates/cortiq-engine/tests/` e contagem manual de `#[test]` por arquivo, 2026-08-21:

- **20 arquivos** de teste da crate estavam na condição descrita: rodar a suíte sem `--features
  gpu` compilava e reportava `running 0 tests ... ok` para o arquivo inteiro, porque TODO `#[test]`
  do arquivo estava atrás de `cfg(feature = "gpu")` — sem sobrar nenhum teste para "não passar".
  - **15** com o gate em nível de arquivo (`#![cfg(feature = "gpu")]` como atributo interno,
    cobrindo o arquivo todo): `attn_gemm_fixed_cost.rs`, `bake_nt_coop.rs`, `gpu_axpy_parity.rs`,
    `gpu_bt_route_parity.rs`, `gpu_compress_parity.rs`, `gpu_dsv4_frame.rs`,
    `gpu_gemm_scratch.rs`, `gpu_mv4b_parity.rs`, `gpu_q4tp_batch.rs` (o do ticket),
    `gpu_route_parity.rs`, `gpu_spec_buf_diff.rs`, `gpu_spec_txn.rs`, `q4tp_roofline.rs`,
    `vram_bandwidth.rs`, `wgpu_q4tp_matmat_parity.rs`.
  - **5** com o gate por item (`#[cfg(feature = "gpu")]` no único `#[test]` do arquivo — mesmo
    efeito, mesmo defeito): `gpu_attend_parity.rs`, `gpu_hc_parity.rs`, `gpu_olora_parity.rs`,
    `gpu_q4tp_parity.rs` (o outro citado no ticket), `gpu_rope_parity.rs`.
- **1 arquivo teve o mesmo padrão descartado por não se qualificar**: `gpu_q4tp.rs` tem 8
  `#[test]`, dos quais só 4 estão atrás de `cfg(feature = "gpu")`; os outros 4 rodam normalmente
  sem a feature (`running 4 tests ... ok`), então não é o defeito deste ticket. Não mexido.
- Nenhum outro arquivo em `tests/` usa `cfg(feature` em qualquer forma — grep cobriu os 64 `.rs`
  do diretório (contados agora: `ls tests/*.rs | wc -l`), não é número de documento.

**Idioma escolhido** — a crate não tinha um idioma pronto pra "o arquivo inteiro evapora sob uma
feature": o único precedente (`#[cfg(not(feature = "gpu"))]` em `src/dsv4.rs`, `src/gpu.rs`,
`src/pipeline.rs`) é braço CPU alternativo dentro de uma função, não um teste sentinela; e o
`#[ignore]` bare já usado em `gpu_micro.rs`, `gpu_q4tp.rs`, `metal_mm_bench.rs`, `vae_parity.rs`
marca testes que SEMPRE compilam (a feature `gpu` não apaga a função) e só faltam um recurso em
runtime — não serve aqui porque muitos destes 20 arquivos chamam `cortiq_engine::gpu_wgpu::*`, e
esse módulo é `#[cfg(feature = "gpu")] pub mod gpu_wgpu;` em `src/lib.rs` — sem a feature ele não
existe, então a função não compilaria só com `#[ignore]`.

Escolhido: **teste sentinela que imprime e passa** (uma das opções que o próprio critério aceita).
Padrão aplicado:
- Nos 15 arquivos com gate de arquivo: `#![cfg(feature = "gpu")]` virou `#[cfg(feature = "gpu")]`
  em cima de `mod gpu_only { ... todo o conteúdo original, sem reindentar ... }`, e um
  `#[cfg(not(feature = "gpu"))] #[test] fn skipped_without_gpu_feature()` foi acrescentado fora do
  módulo, com `eprintln!` dizendo quantos testes reais existem no arquivo e por que não rodaram.
- Nos 5 arquivos com gate por item: só o sentinela foi acrescentado no fim do arquivo — o
  `#[test]` original já não precisava de wrapping.
- Nome do teste é igual (`skipped_without_gpu_feature`) em todos os 20 arquivos — nomes de teste só
  precisam ser únicos dentro do binário, e cada arquivo de `tests/` é seu próprio binário.

**PROVA (EXECUTADO)** — `cargo test --release -p cortiq-engine --test <nome>` SEM `--features gpu`,
para cada um dos 20 arquivos. Todos produzem `running 1 test`, um `SKIPPED: N test(s) in <arquivo>
need(s) --features gpu (not enabled here) — 0 ran` (via `--nocapture`), e `1 passed` — nunca mais
`running 0 tests`. Dois exemplos completos (os citados no ticket):

```
$ cargo test --release -p cortiq-engine --test gpu_q4tp_batch -- --nocapture
running 1 test
SKIPPED: 2 tests in gpu_q4tp_batch.rs need --features gpu (not enabled here) — 0 ran
test skipped_without_gpu_feature ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test gpu_q4tp_parity -- --nocapture
running 1 test
SKIPPED: 1 test in gpu_q4tp_parity.rs needs --features gpu (not enabled here) — 0 ran
test skipped_without_gpu_feature ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Os outros 18 rodados o mesmo jeito, mesmo padrão de saída (contagem de testes reais varia por
arquivo: 1, exceto `bake_nt_coop.rs`=5 e `gpu_compress_parity.rs`=6).

**Compilação sob `--features gpu` (EXECUTADO, não rodado — GPU ocupada pelo orquestrador nesta
rodada)**: `cargo check --release -p cortiq-engine --features gpu --tests` — `Finished` em 9.58s,
zero erros, zero warnings mencionando qualquer um dos 20 arquivos (grep no log completo por cada
nome de arquivo não bateu nada). Isso confirma que os testes reais (os que ficaram dentro de
`mod gpu_only { ... }` ou atrás do `#[cfg(feature = "gpu")]` original) continuam sendo
COMPILADOS sob a feature — não apagados nem quebrados pelo wrapping. `cargo test --features gpu`
NÃO foi rodado (executaria kernel, GPU em uso por outro ticket nesta rodada) — comando proibido
pela lista desta rodada, não um esquecimento.

**O que NÃO foi coberto:**
- Não roda os testes reais sob `--features gpu` (proibido nesta rodada — precisaria da GPU).
  `cargo check --features gpu --tests` prova compilação, não execução nem correção do conteúdo.
- Não toquei `gpu_q4tp.rs` — não se qualifica (tem testes que já rodam sem a feature).
- Não toquei nada em `src/` — fora do escopo e outro agente estava em `src/ltxdit.rs` durante esta
  rodada.
- O nome do teste sentinela é idêntico nos 20 arquivos (`skipped_without_gpu_feature`); isso é
  válido porque cada arquivo de `tests/` compila para um binário próprio, mas quem grepar
  `cargo test` output por nome de teste sem o `--test <arquivo>` não vai distinguir qual arquivo
  gerou qual linha — a contagem (`N tests in <arquivo>`) na mensagem `eprintln!` é o que distingue.
- Não recontei se existem arquivos de teste com o mesmo defeito atrás de OUTRA feature flag
  (por exemplo alguma feature específica de plataforma) — o levantamento cobriu só `feature =
  "gpu"`, que é a única citada no ticket.

## Resolução, segunda passada

A ressalva acima ("outra feature flag") era exatamente o buraco: `cfg(feature =` não pega
`cfg(target_os =`. A causa dos 7 restantes é **gate de plataforma**, não de feature.

**Levantamento (EXECUTADO)** — rodei a suíte inteira sem `--features gpu`
(`cargo test --release -p cortiq-engine --tests`, log completo em
`cortiq_suite_nogpu.log` no scratchpad desta sessão) e depois li, arquivo por arquivo, os 7 que
ainda imprimiam `running 0 tests`. Recontei `#[test]` por arquivo à mão (não `grep -c`, que conta
ocorrências e já tinha errado antes neste mesmo ticket) — os números batem com a tabela que abriu
esta rodada:

| arquivo | gate real (lido no arquivo) | testes reais | o que falta NESTA máquina (Windows, sem `--features gpu`) |
|---|---|---|---|
| `gpu_q4t_bench.rs` | `#![cfg(target_os = "macos")]`, nível de arquivo | 3 | `target_os = "macos"` |
| `gpu_q4t_mm.rs` | `#![cfg(target_os = "macos")]`, nível de arquivo | 5 | `target_os = "macos"` |
| `metal_mm_bench.rs` | `#![cfg(target_os = "macos")]`, nível de arquivo | 1 (já `#[ignore]` por dentro, pendente de `CMF_BENCH_MODEL`) | `target_os = "macos"` |
| `gpu_q4t_mm_wgpu.rs` | `#![cfg(all(feature = "gpu", not(target_os = "macos")))]` | 1 | só `--features gpu` — o `not(target_os="macos")` já vale em Windows |
| `vk_coop.rs` | `#![cfg(all(feature = "gpu", any(target_os = "linux", "windows", "android")))]`, multi-linha | 1 | só `--features gpu` — o `any(...)` já vale em Windows |
| `gpu_micro.rs` | sem gate de arquivo; `#[cfg(target_os = "macos")]` em CADA um dos 7 `#[test]` | 7 | `target_os = "macos"`, todos os 7 |
| `gpu_q4tp.rs` | sem gate de arquivo; por item, MISTO — 4 testes com `target_os = "macos"`, 4 com `feature = "gpu"` | 8 (4+4) | as duas coisas, em grupos separados |

`gpu_q4tp.rs` merece nota: a resolução da primeira passada dizia que ele "não se qualifica"
porque sem a feature rodava `running 4 tests ... ok` — verdade **numa máquina macOS**, onde os 4
testes com `target_os = "macos"` sobrevivem. Nesta máquina Windows, os mesmos 4 morrem pelo gate
de plataforma, e os outros 4 morrem pela feature ausente — sobra zero, e é exatamente o defeito
do ticket. A contagem de um agente não generaliza entre plataformas; isso não invalida a decisão
anterior de excluir o arquivo, invalida só a suposição de que "não se qualifica" era uma
propriedade do arquivo em vez de uma propriedade arquivo+plataforma.

**Idioma usado** — o mesmo dos 20 anteriores, sem inventar um segundo: teste sentinela
`skipped_without_gpu_feature`, mesmo nome em todo arquivo (cada arquivo de `tests/` é seu próprio
binário). Nos 5 arquivos com gate de nível de arquivo (`gpu_q4t_bench.rs`, `gpu_q4t_mm.rs`,
`metal_mm_bench.rs`, `gpu_q4t_mm_wgpu.rs`, `vk_coop.rs`): `#![cfg(...)]` virou `#[cfg(...)]` em
cima de `mod gpu_only { ... conteúdo original, sem reindentar, doc `//!` movida para dentro do
módulo ... }`, sentinela fora do módulo com `#[cfg(not(<mesma-expressão>))]` — a negação é sempre
o `not()` envolvendo a expressão original inteira (não uma reescrita De Morgan manual), para não
arriscar inverter errado uma condição composta. Em `gpu_micro.rs` (gate por item, uniforme):
sentinela solto no fim do arquivo, `#[cfg(not(target_os = "macos"))]`. Em `gpu_q4tp.rs` (gate por
item, misto): sentinela com `#[cfg(not(any(target_os = "macos", feature = "gpu")))]` — dispara só
quando NENHUM dos dois grupos de 4 compilaria, que é a definição exata de "arquivo com zero
testes".

A mensagem de cada sentinela diz o motivo real lido no arquivo, não um genérico
"--features gpu": os 3 arquivos só-macOS dizem `target_os = "macos"`; os dois arquivos
`feature=gpu AND <plataforma>` dizem que falta `--features gpu` E nomeiam a condição de
plataforma ao lado (mesmo ela já valendo em Windows — a mensagem descreve o que o `cfg` do
arquivo exige, não só o que falta nesta máquina específica); `gpu_q4tp.rs` diz quantos de cada
tipo.

**PROVA (EXECUTADO, sem `--nocapture`)** — `cargo test --release -p cortiq-engine --test <nome>`
para cada um dos 7, sem `--features gpu`:

```
$ cargo test --release -p cortiq-engine --test gpu_q4t_bench
running 1 test
test skipped_without_gpu_feature ... ignored, 3 tests in gpu_q4t_bench.rs need target_os = "macos" (not enabled here) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test gpu_q4t_mm
running 1 test
test skipped_without_gpu_feature ... ignored, 5 tests in gpu_q4t_mm.rs need target_os = "macos" (not enabled here) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test metal_mm_bench
running 1 test
test skipped_without_gpu_feature ... ignored, 1 test in metal_mm_bench.rs needs target_os = "macos" (not enabled here) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test gpu_q4t_mm_wgpu
running 1 test
test skipped_without_gpu_feature ... ignored, 1 test in gpu_q4t_mm_wgpu.rs needs --features gpu on a non-macOS target (not enabled here) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test vk_coop
running 1 test
test skipped_without_gpu_feature ... ignored, 1 test in vk_coop.rs needs --features gpu on linux/windows/android (not enabled here) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test gpu_micro
running 1 test
test skipped_without_gpu_feature ... ignored, 7 tests in gpu_micro.rs need target_os = "macos" (not enabled here) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo test --release -p cortiq-engine --test gpu_q4tp
running 1 test
test skipped_without_gpu_feature ... ignored, 8 tests in gpu_q4tp.rs (4 need target_os = "macos", 4 need --features gpu) — 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Nunca mais `running 0 tests` nestes 7 — a linha `ignored, <motivo>` aparece sem `--nocapture`,
como o critério pede.

**PROVA (EXECUTADO) — suíte inteira**, `cargo test --release -p cortiq-engine --tests` sem a
feature (log completo em `cortiq_suite_nogpu_v2.log` no scratchpad desta sessão):

```
running 0 tests count: 0
ignored, count: 27
FAILED count: 0
```

27 = os 20 da primeira passada + estes 7. Todas as 65 linhas `test result:` do log terminam em
`ok.` — nenhum `FAILED`, nenhum `0 tests`. Os 27 nomes de sentinela conferidos um a um batem com
a tabela acima (contagem e motivo, inclusive o `gpu_q4tp.rs` "(4 need target_os..., 4 need
--features gpu)").

**Compilação sob `--features gpu` (EXECUTADO, não rodado — GPU fora de escopo desta rodada por
regra do orquestrador)**: `cargo check --release -p cortiq-engine --features gpu --tests` —
`Finished` sem erro. `grep` no log completo por cada um dos 7 nomes de arquivo não bateu nada —
zero warning citando qualquer um deles, ou seja o conteúdo original (dentro de `mod gpu_only {
... }` ou atrás do `cfg` por item) continua COMPILANDO sob a feature, não foi apagado nem quebrado
pelo wrapping. `cargo test --features gpu` NÃO foi rodado — proibido nesta rodada, GPU sendo usada
por outra tarefa.

**O que NÃO foi coberto:**
- **macOS não pôde ser verificado nesta máquina** (Windows). Os 15 arquivos que dependem de
  `target_os = "macos"` (3 novos aqui + os que a primeira passada não tocou) tiveram seu lado
  `not(macos)` provado — que o sentinela dispara e cobre a ausência — mas o lado `macos` (que os
  testes reais existem e compilam lá) não foi e não pode ser provado por este ambiente. Ausência
  de erro aqui não é prova de compilação lá.
- **Vulkan/linux/android não pôde ser verificado além da compilação.** `vk_coop.rs` está coberto
  neste Windows (a condição de plataforma do `any(linux, windows, android)` já vale aqui), mas o
  caminho `target_os = "android"` nunca foi exercitado, nem a extensão Vulkan real (o teste real
  em si não rodou — só compilou sob `--features gpu` via `cargo check`).
- **Os testes reais dos 7 arquivos não foram executados sob `--features gpu`** nesta rodada —
  proibido pela lista de comandos, GPU em uso por outro ticket. `cargo check --features gpu
  --tests` prova compilação, não correção do conteúdo nem execução.
- Não toquei nada fora dos 7 arquivos listados nem fora deste ticket.
- Não recontei se existem arquivos com o mesmo defeito atrás de uma feature flag que não seja
  `gpu` nem de um `cfg(target_os)` diferente de macos/linux/windows/android — o levantamento desta
  passada cobriu só os 7 nomeados na tarefa desta rodada.


## Nota do orquestrador: por que o sentinela virou `#[ignore]`

A primeira passada entregou o sentinela como teste que **passa** e imprime por `eprintln!`. Rodei
e a saida padrao ficou assim:

```
running 1 test
test skipped_without_gpu_feature ... ok
test result: ok. 1 passed; 0 failed; 0 ignored
```

Melhor que `running 0 tests`, mas o criterio pede uma linha dizendo que foi pulado **e por que**, e
o "por que" ficava so em `--nocapture` -- `libtest` captura o stderr de teste que passa. Quem roda
`cargo test` sem flag extra continuava vendo um `ok`.

Troquei por `#[ignore = "<motivo>"]` no mesmo sentinela, mantendo o corpo com `eprintln!` (que
aparece em `--include-ignored --nocapture`). EXECUTADO:

```
$ cargo test --release -p cortiq-engine --test vram_bandwidth
running 1 test
test skipped_without_gpu_feature ... ignored, 1 test in vram_bandwidth.rs needs --features gpu; 0 ran
test result: ok. 0 passed; 0 failed; 1 ignored
```

O motivo agora sai **sem flag nenhuma**, que era o que faltava. Aplicado aos 20 da primeira passada
por script (`<scratch>/add_ignore_reason.py` e `add_ignore_reason2.py`, o segundo para os quatro
cujo `eprintln!` estava quebrado em varias linhas); os 7 da segunda passada ja nasceram assim.

Isto **nao contradiz** o raciocinio da primeira passada sobre `#[ignore]` bare -- aquele era sobre
ignorar os testes REAIS, que sem a feature nao compilam porque `gpu_wgpu` nem existe. Aqui o
`#[ignore]` esta no sentinela, que so existe quando a feature esta ausente. Sao coisas diferentes,
e a leitura da primeira passada estava certa sobre a dela.
