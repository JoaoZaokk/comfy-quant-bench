# `gpu_q4tp_batch` recusa `b=375` e `b=1879`: é o lote ou a forma?

Type: task
Status: open
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
