# Mapa: Estado entregável dos dois repos

## Destination

Os dois repos em **estado entregável**: todo fio aberto tem dono, estado e próximo
passo escritos, de modo que outra sessão (ou o autor) retoma qualquer um deles sem
precisar perguntar nada.

Dentro desse destino, duas faixas com artefato próprio:

- **Veredito de performance** — uma resposta medida sobre se o cortiq tem caminho
  nesta placa, ou o registro de que não tem.
- **Ferramental confiável** — as ferramentas que produziram os números são boas o
  bastante para outra pessoa reproduzi-los sem cair nas armadilhas que este dia
  encontrou.

O mapa fecha quando as três fecham.

## Notes

### Domínio

Dois repos, e eles não são iguais:

| repo | o que é | git |
|---|---|---|
| `F:\COMFY_PORTABLE` | bancada viva de ComfyUI Portable; nosso pipeline W4A4 mora em `tools/` | repo local, **sem remote**, 88 arquivos rastreados |
| `F:\cortiq-cmf` | clone de `github.com/infosave2007/cmf`, motor Rust de terceiro | fork em `JoaoZaokk/cmf`; branch `pv-nt-audit-local` é nosso trabalho |

`ComfyUI/` dentro da bancada é **checkout git separado** (0.33.0) e não é rastreado
por este repo. Os 66 pacotes de custom nodes também não — exceto três arquivos
nossos, que estão nos 88.

### ESTE MAPA CARREGA EXECUÇÃO

Wayfinder é planejamento por padrão: tickets resolvem decisões, não fatias de
build. **Este esforço sobrescreve isso.** O destino é um estado do código, não uma
especificação, então tickets aqui mudam arquivos. Onde um ticket for só decisão,
ele diz isso no corpo.

### Regra de fechamento: CRITÉRIO PRÉVIO

Um ticket de dívida só fecha **sem o código mudar** quando o critério de fechamento
já estava escrito no ticket **antes de alguém olhar o resultado**. Sem critério
prévio, ou com critério que não bate, o fechamento é decisão do dono do repo.

Por que essa regra e não "eu decido" nem "tu decide": em 2026-08-19/20 o
autojulgamento acertou quatro vezes (o 414 GFLOP/s de docstring, o "patch está
errado", uma extrapolação de 9,0 s dentro de uma fase de 6,5 s, e "48 threads" numa
máquina de 12) — **todas as quatro porque existia critério escrito antes**. E falhou
uma vez, na violação do lock da GPU, onde não havia critério: quem pegou foi a outra
sessão, não eu.

Todo ticket de dívida abaixo carrega `## Critério de fechamento`.

### Protocolo de GPU

A RTX 3090 é dividida com outras sessões. Quando um ticket precisar dela:

1. Tomar o lock com `Assert-GpuLock` (`F:\COMFY_PORTABLE\tools\gpu_lock.ps1`), que
   **lança** em vez de devolver `false` — nunca `Take-GpuLock | Out-Null`.
2. Recusado? **Medir a placa de verdade** (`nvidia-smi`/NVML) por ~1 minuto. O
   arquivo de lock não é a placa.
3. Placa ociosa **e** lock mantido: isso é problema. **Não esperar.** Escrever no
   chat "esperei X, lock de `<dono>` me prendeu" e seguir para o que não precisa de
   placa.
4. **Nunca** tomar um lock vivo em silêncio. Isso já aconteceu aqui, contra um
   `agent-delta` com heartbeat de 2 s, e pode ter contaminado a medição dele.

Por que isso importa para o destino: contenção move número. O mesmo código, no
mesmo dia, deu `im2col 74,4 s` com a máquina quieta e `87,0 s` carregada — 17%,
contra um efeito de 10 s que se tentava medir. Sem janela limpa, a faixa de GPU não
produz veredito.

### Disciplina de medição

Herdada do `CLAUDE.md` da bancada e reforçada por este dia:

- **Uma corrida não é medição.** Repetir alternado e citar a dispersão, não a média.
- **Para mudança de kernel, tempo sem comparação de saída não é resultado** — é
  ruído com unidade de segundo.
- Todo artefato diz o que **não** cobriu, na saída, em toda corrida.
- Número citado não é número medido: se veio de docstring, de comentário ou de outra
  máquina, isso vai colado no número.

### Skills a consultar

`grilling` e `domain-modeling` em qualquer ticket de decisão. `research` para os
`wayfinder:research`.

## Decisions so far

<!-- índice: uma linha por ticket fechado, com link para o detalhe -->

_(nenhuma ainda)_

## Not yet specified

Névoa em escopo, ainda sem nitidez para virar ticket:

- **O que a triagem dos 88 arquivos achar.** Cada suspeito pode graduar em um ou
  vários tickets; só dá para dizer depois que `02` rodar.
- **Se o container q8 vale ser tentado noutra máquina.** Aqui morreu por memória: o
  DiT sozinho quer ~41,5 GiB contra fonte de 39,13 GiB, e o watchdog cortou em
  5,93 GiB livres. Depende de haver outra máquina, o que ainda não foi levantado.
- **Dívida própria do pipeline W4A4** (`quant_*.py`, `verify_*`, `calibrate_*`,
  `to_native.py` — ~4.000 linhas). Sabe-se que existe; a forma dela sai da triagem.
- **Se o cortiq vale continuar depois do veredito.** Pergunta real, mas ela é
  *sobre* o resultado da faixa B, então não dá para formulá-la antes dele.
- **O que fazer com os três dialetos do lock** (`hb=` do controlador, `pid=` do bash
  antigo, `gpu_lock.py` atômico). Convergir pode ser certo, mas depende do que `01`
  acordar com a outra sessão.

## Out of scope

Trabalho conscientemente colocado além do destino. Não gradua; se voltar, volta como
esforço novo.

- **PR #1 e issues #2–#5 em `infosave2007/cmf`.** Resposta de terceiro não pode ser
  passo desta rota — travaria o mapa indefinidamente. O que foi enviado já foi; o
  mapa registra o que **nós** fizemos, não o julgamento de quem recebe.
- **Os 66 pacotes de custom nodes de terceiros.** Código que não é nosso e que não
  mantemos.
- **O checkout upstream do ComfyUI** (`ComfyUI/`, 0.33.0). Repo separado, código de
  terceiro. Artefatos **nossos** que moram lá dentro são exceção e estão no ticket
  `16`.
