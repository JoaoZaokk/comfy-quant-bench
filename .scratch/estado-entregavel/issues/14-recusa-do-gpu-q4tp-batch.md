# `gpu_q4tp_batch` recusa `b=375` e `b=1879`: é o lote ou a forma?

Type: task
Status: resolved
Blocked by: 01

## Question

O teste `gpu_q4tp_batch` do cortiq imprime `device declined` para `b=375` e `b=1879`
e **passa mesmo assim**. Isso ficou registrado como propriedade do lote.

`wgpu_q4tp_matmat_parity` (escrito neste trabalho) rodou **os dois lotes**, em três
planos `dit.*` reais, com **zero recusas** e pior L2 relativo de 2,850e-4 contra a
barra de 4e-3 do `vk_coop`. Então a recusa não é do lote.

Estreitado por leitura, **não confirmado**: `gpu_q4tp_batch` usa formas muito
maiores — `mlp.fc1 = 28672 x 5376`, cuja saída em `b=1879` é **215 MB**, e o
comentário do próprio teste diz que é aí que um limite de binding ou uma largura de
índice morderia primeiro. O teste de paridade foi até `16384 x 4096`, saída de
123 MB.

Confirmar exige a placa.

## Critério de fechamento

Fecha quando estiver medido qual condição dispara a recusa — tamanho de saída, limite
de binding, largura de índice ou outra — com a forma exata onde ela começa.

Pode fechar **sem** mudança de código: se a recusa for um limite legítimo do
dispositivo, o correto é o teste dizer isso em vez de passar em silêncio, e isso vira
ticket separado em vez de virar conserto aqui.

## Resolução — 2026-08-21, janela limpa (3090 ociosa, 34 MiB, 0%), lock `claude:rodada3-gpu`

**A recusa não é do lote, não é da forma, e não é do q4tp.** Medido, não lido:

```
cargo test --release -p cortiq-engine --features gpu --test gpu_q4tp_batch -- --nocapture
```

O teste estava sendo pulado em silêncio: `#![cfg(feature = "gpu")]` na linha 11, e sem
`--features gpu` o binário roda **0 testes** e reporta `ok`. Era esse o motivo do registro
antigo, não uma recusa observada.

Com a feature ligada, `the_device_gemm_agrees_with_the_host_at_every_batch` (q4tp) rodou as
**cinco formas nos dois lotes, com zero recusas**:

| forma | b=375 | b=1879 | saída em b=1879 |
|---|---|---|---|
| 21504 x 5376 | 0.000e0 | 5.008e-3 | 162 MB |
| 5376 x 7168 | 0.000e0 | 5.360e-3 | 40 MB |
| **28672 x 5376** | 0.000e0 | 5.569e-3 | **215 MB** |
| 5376 x 14336 | 0.000e0 | 8.352e-3 | 40 MB |
| 4096 x 5376 | 0.000e0 | 4.501e-3 | 31 MB |

`non-finite 0` em todas. **A hipótese do ticket — limite de binding ou largura de índice
mordendo na saída de 215 MB de `mlp.fc1 = 28672 x 5376` — está refutada.** Essa forma exata,
nesse lote exato, passou.

As quatro linhas `device declined` vêm do **outro** teste do arquivo,
`the_two_bit_device_gemm_agrees_with_the_host` (linha 118; o `eprintln` é a linha 205, dentro
dele). É **q2tp — 2 bits**, que declina nos dois lotes igualmente. Não há braço de device para
q2tp; a recusa é ausência de implementação, não limite de dispositivo.

Fecha **sem mudança de código**, e o critério previa isso: "se a recusa for um limite legítimo
do dispositivo, o correto é o teste dizer isso em vez de passar em silêncio, e isso vira ticket
separado". Não é limite legítimo — é braço inexistente — mas o **passar em silêncio** continua
valendo como defeito, em dobro: o `cfg(feature)` faz o arquivo inteiro reportar `ok` com zero
testes. Graduado no ticket 27.

**Não coberto:** por que q2tp não tem braço de device (não investigado). O erro absoluto de
5e-3 em b=1879 contra 0.000e0 em b=375 não foi explicado — é consistente com acumulação em f16
num lote 5x maior, mas isso é hipótese, não medição.
