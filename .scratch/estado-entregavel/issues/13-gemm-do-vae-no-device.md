# O GEMM do VAE vale ir para a placa?

Type: task
Status: open
Blocked by: 01

## Question

O decode do VAE de vídeo é **255,6 s de um render de 510 s — 46%**, medido isolado
via `ltx-decode`, e portanto não é artefato de dividir processo com o DiT. Dentro
dele, três blocos `res_x` carregam 81,6%.

`Conv3d::forward` **já** reduz convolução a im2col + `gemm_nt`, e `gemm_nt` tem braço
de device. Medido no host: **im2col 74,4 s (33%), gemm 139,3 s (61%), scatter 14,1 s
(6%)**. O GEMM é o pedaço grande, e roda a ~43 GFLOP/s num Ryzen de 6 núcleos.

Duas coisas puxam em direções opostas, e é isso que este ticket resolve:

- **A favor**: 139,3 s de host contra uma 3090. Se render como o q4tp rende, é ~20 s,
  ou seja ~119 s de um render de 510 — **23%**.
- **Contra**: o im2col materializa `8192 x 13824 x 4 = 453 MB` de patches por chamada,
  12,5 chamadas por conv. O volume de entrada é 209 MB e vira 5,7 GB de linhas
  materializadas, porque cada voxel aparece em 27 patches. Mandar isso para a placa
  paga upload de 453 MB por chamada.

Se o segundo dominar, a resposta não é "manda o GEMM para o device" e sim "não
materializa o im2col" — um kernel de convolução direto lê `x` uma vez e reconstrói os
27 taps em registrador.

## Sinal lateral, NÃO controlado

`ltx-decode` com `CMF_GPU=wgpu` deu 255,6 s; o mesmo decode sem GPU deu 269,7 s.
**5%.** As duas corridas não foram controladas para carga. Consistente com o GEMM não
estar indo para a placa hoje — o que também é hipótese, não medição.

## Critério de fechamento

Fecha quando houver medição, sob janela limpa (ticket `01`), de quanto o GEMM do VAE
custa no device contra no host **nas formas que o decoder realmente emite**, e uma
recomendação escrita entre as três saídas: mandar para a placa, escrever conv direto,
ou deixar como está.

Fecha **sem** mudança de código: é um veredito, não uma implementação. Implementar
gradua daqui como ticket novo.
