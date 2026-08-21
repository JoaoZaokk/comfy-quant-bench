# Convolução direta no VAE, sem materializar o im2col

Type: task
Status: resolved
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

## Resolucao — 2026-08-21, 3090 pinada e sob lock `bench:ticket28_gemm_nt_split`

**Veredito: a alternativa barata funciona, e por larga margem. 1,43-1,52x mais rapido que o
host no decode inteiro, e o host era o braco recomendado pelo ticket 13.** Nao foi preciso
escrever convolucao direta.

### A mudanca

`OpClass::GemmNtNarrow`, `gemm_nt_class(m)` e as tres chamadas de `fcd_ops::gemm_nt` passando a
classe em vez de `OpClass::GemmNt` fixo. E o mesmo split que a crate ja fez duas vezes no mesmo
enum (`MatmatWide` de `Matmat`, `MatvecHead` de `Matvec`) pela mesma razao escrita nos dois
docstrings: um processo roda duas populacoes e elas nao tem a mesma resposta.

`CMF_GEMM_NT_M` move o corte; **`0` devolve o comportamento anterior**, que e como as duas
disposicoes foram comparadas sem recompilar.

### O eixo estava errado no ticket, e a medicao confirmou

`CMF_GEMM_SHAPES=1` (instrumentacao nova em `Conv3d::forward`) mostra o decoder emitindo
**1242 chamadas, seis larguras**:

    m=48     98 chamadas   1,4 G MAC/chamada
    m=128   784 chamadas   3,6 G
    m=256   300 chamadas  14,5 G
    m=512   150 chamadas  29-58 G
    m=1024    5 chamadas
    m=4096    3 chamadas

As chamadas que a placa perde (ticket 13: o estagio `m=128`/`m=48`, 2,9x mais lento) sao as
**menores**. Dispatch por volume por chamada — a "alternativa mais barata" como o ticket a
escreveu — mandaria a populacao perdedora para a placa e manteria a vencedora no host. O
discriminante e `m`, porque por chamada o buffer de patches custa `n*k` para subir e compra
`n*k*m` MACs: trabalho por byte transportado escala com `m`.

Corte em 256 (o empate medido no ticket 13) separa 882 chamadas estreitas de 458 largas.

### Medicao

Todas as corridas: `CMF_LTXVAE_STAGES=F:/cortiq/ltx25-q4tp.cmf`, `CMF_CONV3D_PROF=1`,
`CMF_PROBE_CACHE=0`, `CMF_GPU_ADAPTER=3090`, alternadas dentro da janela.

```
cargo test --release -p cortiq-engine [--features gpu] --test ltxvae_stage_cost -- --nocapture
```

**host contra split, alternado na mesma janela:**

| | rodada 1 | rodada 2 | faixa | GEMM |
|---|---|---|---|---|
| host | 261,4 s | 274,0 s | 261,4-274,0 | 152,1 / 156,4 |
| **split** | **182,6 s** | **180,8 s** | **180,8-182,6** | **73,9 / 74,7** |

Faixas **nao se sobrepoem**. 1,43-1,52x no total, 2,06-2,09x no GEMM.

**flat (o comportamento de hoje) contra split, outra janela alternada, mesma placa:**

| | rodada 1 | rodada 2 | faixa |
|---|---|---|---|
| flat (`CMF_GEMM_NT_M=0`) | 332,9 s | 335,9 s | 332,9-335,9 |
| split | 177,2 s | 171,1 s | 171,1-177,2 |

1,88-1,96x. E o flat perde para o host (332,9-335,9 contra 261,4-274,0), o que **reproduz o
veredito do ticket 13 por caminho independente**.

### O probe escolheu o que o corte previa

Apontando `CMF_PROBE_CACHE` para um arquivo e lendo o veredito de volta — nao inferido do
tempo:

```
0.5.95  NVIDIA GeForce RTX 3090/Vulkan  gemm-nt         gpu     <- 458 chamadas largas
0.5.95  NVIDIA GeForce RTX 3090/Vulkan  gemm-nt-narrow  cpu     <- 882 chamadas estreitas
```

Sem o split, a classe unica decide `gpu` e leva as 882 perdedoras junto.

### Comparacao de saida — o criterio exigia, e a saida NAO e identica

`ltxvae_stage_cost` agora imprime `OUTPUT n= mean= l2= amax=` sobre os pixels decodificados; antes
descartava `out`.

| | mean | l2 | amax |
|---|---|---|---|
| host | -0,344200471 | 2235,355818 | 1,003083348 |
| split | -0,344202516 | 2235,366866 | 1,002965808 |

Diferenca: **4,9e-6 relativo no l2**, 2,0e-6 absoluto na media, **1,2e-4 no maior valor absoluto**.
Vem de o kernel `gemm_nt_coop` carregar A e B como **f16** em cooperative matrices e acumular em
**f32** (`coop_mat16x16<f32, C>`, `gpu_wgpu.rs:26252, no shader gemm_nt_coop`), nao de um erro de roteamento.

Referencia para julgar isso: o proprio teste de paridade da crate para este kernel,
`tests/gpu_gemm_scratch.rs:72`, aceita **`rel < 5e-3`** contra referencia f64. A divergencia
medida esta **tres ordens de grandeza dentro** desse orcamento. Em pixel de 8 bits sobre faixa
[-1,1], 1,2e-4 e 0,015 de um nivel — abaixo da quantizacao. **Nao e "identico", e esta escrito
que nao e.**

Efeito lateral que vale registrar: o split e **reprodutivel** (as duas corridas dao byte a byte o
mesmo `OUTPUT`), enquanto o flat **nao e** — as tres corridas de flat deram tres agregados
diferentes, porque a classe unica alterna os bracos de forma diferente a cada processo.

### O erro que esta medicao encontrou em si mesma

**As primeiras nove corridas rodaram na placa errada.** O `cortiq` escolhe adaptador com
`request_adapter(HighPerformance)` quando `CMF_GPU_ADAPTER` nao esta setado, e neste host isso da
a **RTX 3080 Ti**, nao a 3090. Eu segurei o lock da 3090 — ociosa — enquanto o trabalho ia para a
`cuda:1`, que esteve entre 9% e 26% de utilizacao em toda amostra de `nvidia-smi` da sessao. Quem
denunciou foi o proprio arquivo de cache do probe, que carimba o nome do adaptador em cada linha.

Na placa certa o efeito e **maior**, nao menor (split 171-183 s contra 188-238 s na 3080 Ti), mas
isso e sorte: podia ter sido ao contrario. Registrado tambem no `CLAUDE.md` da bancada, secao
*The GPU window*, porque e falha de protocolo e nao deste ticket.

**O ticket `13` esta sob a mesma sombra** — mesma selecao automatica, mesma epoca. O veredito dele
("nao mandar o GEMM para a placa") **se sustenta**: reproduzi `flat` perdendo para `host` na 3090
pinada, 332,9-335,9 contra 261,4-274,0. Mas os numeros por bloco daquele ticket foram medidos na
3080 Ti ocupada e nao devem ser citados como propriedade da 3090.

### O que ficou sem cobertura

- **A convolucao direta nao foi escrita nem medida.** O ticket fecha porque a alternativa barata
  entregou; se alguem quiser o kernel direto, o alvo agora e o `im2col`, que passou de 27% para
  **40%** do decode justamente porque o GEMM encolheu.
- **Um modelo, uma resolucao, um decoder.** Nada aqui diz o que o split faz com o `gemm_nt` da
  atencao, que divide a mesma `OpClass` e cujas larguras nao foram levantadas.
- **O corte 256 nao foi varrido.** Saiu do empate medido no ticket 13 e das seis larguras que este
  decoder emite. `CMF_GEMM_NT_M` existe para varrer; ninguem varreu.
- **Duas rodadas por braco**, nao tres. As faixas nao se sobrepoem e o spread do split e 1,0%, mas
  sao duas.
- **`vmmemWSL` segurava 26,5 GB durante tudo isso** (trabalho nao relacionado do dono, regra dura:
  nao mexer). O decode e CPU-bound; entre a primeira e a segunda rodada da sessao a maquina
  derivou 28%. Por isso as corridas sao alternadas e por isso **so as razoes dentro da janela
  valem** — os absolutos de hoje nao sao comparaveis aos do ticket 13.
- **`cargo test --features gpu` nao foi rodado na suite inteira** apos a mudanca; so este teste.
