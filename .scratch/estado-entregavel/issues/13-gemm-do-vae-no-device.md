# O GEMM do VAE vale ir para a placa?

Type: task
Status: resolved
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

## Resolução — 2026-08-21, janela limpa sob lock `claude:rodada3-gpu` (3090 ociosa, 34 MiB, 0%)

**Veredito: NÃO mandar o GEMM do VAE para a placa como está. Medido, e o efeito é o oposto do
esperado — o braço de device deixa o decode MAIS LENTO.**

Quatro corridas do mesmo teste, alternadas GPU/host na mesma janela:

```
CMF_LTXVAE_STAGES=F:/cortiq/ltx25-q4tp.cmf CMF_CONV3D_PROF=1 \
  cargo test --release -p cortiq-engine [--features gpu] --test ltxvae_stage_cost -- --nocapture
```

| | corrida 1 | corrida 2 | faixa |
|---|---|---|---|
| **total, com `--features gpu`** | 286,6 s | 282,8 s | **282,8–286,6** |
| **total, host** | 246,8 s | 241,6 s | **241,6–246,8** |
| gemm, gpu | 184,7 s | 180,1 s | 180,1–184,7 |
| gemm, host | 134,2 s | 137,2 s | 134,2–137,2 |

As faixas **não se sobrepõem** em nenhuma das duas linhas. O host ganha por ~40 s (~14%) no
total, e o GEMM do braço de device é ~45 s **mais lento**.

Por bloco, o efeito se separa e troca de sinal:

| bloco | saída | gpu | host | |
|---|---|---|---|---|
| `after_block_4` | [512,25,64,64] 52,4 Mvox | 37,9 / 39,2 | 90,7 / 94,7 | gpu **2,4x mais rápido** |
| `after_block_6` | [256,49,64,64] 51,4 Mvox | 55,9 / 51,3 | 56,2 / 51,2 | empate |
| `after_block_8` | [128,49,128,128] **102,8 Mvox** | 147,3 / 156,6 | 54,4 / 50,5 | gpu **~2,9x mais lento** |

**O bloco que a placa piora é o de maior volume.** Isso é exatamente o argumento "contra" escrito
neste ticket antes da medição: o im2col materializa os patches e o upload por chamada cresce com o
volume. Em `block_4` o cálculo domina e a placa ganha; em `block_8` o transporte domina e a placa
perde por três vezes.

**Recomendação, entre as três saídas que o ticket pediu:**

1. ~~mandar o GEMM para a placa~~ — medido, perde. Não fazer.
2. **Escrever convolução direta que não materializa o im2col** — é para onde a evidência aponta.
   O gargalo não é o FLOP, é o volume que se cria para alimentá-lo. Um kernel que lê `x` uma vez e
   reconstrói os 27 taps em registrador ataca a causa. **Gradua como ticket próprio.**
3. Deixar como está — aceitável no curto prazo; o host já é o braço mais rápido dos dois.

Isto também **corrige o sinal lateral não controlado** registrado antes neste ticket (255,6 s com
`CMF_GPU=wgpu` contra 269,7 s sem, "5%", duas corridas não controladas). Sob janela limpa e com o
braço selecionado em tempo de compilação, a diferença é 14% e vai na direção contrária.

**Não coberto:** POR QUE `block_8` regride 2,9x não foi diagnosticado — upload, ocupação e
sincronização são um número só aqui. O encoder e o VAE de áudio não foram tocados. O teste ranqueia
estágios, não explica nenhum. E `--features gpu` é escolha de compilação, não de dispatch por
forma: um braço que escolhesse por volume por chamada não foi testado e é justamente o que o
resultado sugere valer.
