# Convolução direta no VAE, sem materializar o im2col

Type: task
Status: open
Blocked by: 13

## Question

O ticket `13` mediu e o resultado apontou para cá: o gargalo do decode não é o FLOP, é o volume
que se cria para alimentá-lo.

`Conv3d::forward` reduz convolução a im2col + `gemm_nt`. Cada voxel aparece em 27 patches, então
209 MB de entrada viram 5,7 GB de linhas materializadas, e o buffer de patches é
`8192 x 13824 x 4 = 453 MB` por chamada.

Medido em 2026-08-21 (ticket `13`, 4 corridas alternadas, janela limpa): mandar o GEMM para a
placa piora o total de 241,6–246,8 s para 282,8–286,6 s. Por bloco o sinal troca — `block_4`
(52,4 Mvoxels) fica 2,4x mais rápido na placa, `block_8` (102,8 Mvoxels) fica 2,9x mais lento.
O bloco de maior volume é o que a placa piora.

Um kernel que lê `x` uma vez e reconstrói os 27 taps em registrador não paga esse transporte.

## Critério de fechamento

Fecha com uma implementação de convolução direta medida contra o caminho im2col+GEMM atual, nas
formas que o decoder realmente emite, sob janela limpa, com repetição alternada e a dispersão
citada — **e com comparação de saída**, não só de tempo. Para mudança de kernel, tempo sem
comparação de saída não é resultado.

Fecha **sem** implementar se a medição de um protótipo mostrar que não ganha; nesse caso registrar
os números, não a impressão.

## Alternativa mais barata a considerar primeiro

Dispatch por volume por chamada: usar a placa onde ela ganhou (`block_4` e menores) e o host onde
perdeu (`block_8`). Não foi testado. Se der a maior parte do ganho por uma fração do trabalho, o
kernel direto pode não valer.
