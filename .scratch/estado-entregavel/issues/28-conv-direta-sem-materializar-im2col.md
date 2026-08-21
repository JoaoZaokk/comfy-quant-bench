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

## Antes de medir: o eixo da alternativa barata esta errado (2026-08-21)

A "Alternativa mais barata" acima diz **dispatch por volume por chamada**. A tabela por bloco do
proprio ticket `13` ja diz que volume nao e o que separa os casos. LIDO, nao medido -- a leitura e
da tabela que ja existe, e das assinaturas do codigo:

| bloco | voxels | canais da saida do estagio | placa |
|---|---|---|---|
| `after_block_4` | 52,4 M | 512 | **2,4x mais rapida** |
| `after_block_6` | 51,4 M | 256 | empate |
| `after_block_8` | 102,8 M | 128 | **2,9x mais lenta** |

Os blocos 4 e 6 tem **praticamente o mesmo volume** (52,4 contra 51,4 Mvox) e resultados
**diferentes**. Ja a largura de canal -- 512, 256, 128 -- e monotonica com o sinal do efeito, na
ordem certa.

**Ressalva que viaja com isso:** 512/256/128 sao os canais da *saida do estagio*, nao o `m` de uma
chamada de `gemm_nt` -- um bloco tem mais de uma conv, e so a ultima necessariamente emite essa
largura. A hipotese e que `m` acompanha; **isso nao foi medido**, e a primeira coisa que a
implementacao tem que fazer e tabelar os `(n, k, m)` reais por chamada numa corrida de verdade,
antes de escolher o corte.

E ha um motivo mecanico, do proprio `Conv3d::forward`: cada chamada leva `n = CHUNK.min(npos - p0)`
posicoes, ou seja **`n` e `CHUNK` em todas menos a ultima de cada conv**, e o volume por chamada e
`CHUNK * k * m`. O que sobe com o numero de
voxels nao e o tamanho da chamada, e a **quantidade** de chamadas. Por chamada:

    upload   = n*k (os patches materializados) + download n*m
    computo  = n*k*m
    razao    = m / (1 + m/k)  ~=  m   enquanto k >> m

Ou seja, o trabalho aritmetico por byte transportado escala com **`m = c_out`**. `block_8` tem
`m=128` contra `m=512` do `block_4`: quatro vezes menos computo por byte, e por isso o transporte
domina exatamente la. Dispatch por volume por chamada mandaria `block_8` (chamadas MENORES, porque
`m` e menor) para a placa e `block_4` para o host -- **o inverso do que a medicao pede**.

**Correcao da alternativa barata: o discriminante e `m`, nao o volume.**

## E o mecanismo para isso ja existe nesta crate

`gemm_nt` (`fcd_ops.rs:302`) manda para a placa quando `n*k*m >= (1<<22)` **e**
`gpu::probe_arm(OpClass::GemmNt)` concorda. E `probe_arm` decide **por OpClass, uma vez por
processo**, a partir de amostras medias -- nao por chamada.

O docstring do proprio `OpClass::GemmNt` (`gpu.rs:303-309`) diz que a classe cobre "attention's
QKᵀ and AV, **and the VAE decoders' projections**". Sao populacoes diferentes com respostas
diferentes, num veredito so. E a crate **ja nomeou esse modo de falha duas vezes**, no mesmo enum:

- `MatmatWide` (`gpu.rs:290-295`): *"one imagegen process runs BOTH populations ... a single
  shared verdict locks the wrong arm for whichever population samples second."*
- `MatvecHead` (`gpu.rs:296-302`): mesma razao, e diz onde mordeu.

E ha o seletor por tamanho ja escrito: `matvec_class(rows, cols)` (`gpu.rs:315-321`), que separa
`MatvecHead` de `Matvec` por um limiar.

Entao a alternativa barata nao e codigo novo: e **aplicar ao `GemmNt` o mesmo split que
`MatmatWide` e `MatvecHead` ja receberam**, com um `gemm_nt_class(n, k, m)` cortando por `m`.

## Como isto muda o plano do ticket

O criterio de fechamento continua valendo como escrito (medicao alternada, dispersao, comparacao
de saida). O que muda e a ordem: **medir o split por `m` antes de escrever kernel de convolucao
direta**, porque agora ha razao para achar que ele recupera o ganho do `block_4` (2,4x) sem pagar
a perda do `block_8`, por uma mudanca de dez linhas num idioma que a crate ja usa.

Se o split por `m` entregar a maior parte do ganho, o kernel direto pode nao valer -- que e
exatamente o que a secao "Alternativa mais barata" ja previa, so que pelo eixo errado.
