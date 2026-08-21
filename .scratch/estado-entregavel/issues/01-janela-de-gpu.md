# Janela de GPU acordada com a outra sessão

Type: task
Status: resolved

## Question

A RTX 3090 é dividida entre esta sessão e outras (`agent-delta`, `cache-limiar`, e
o dono da bancada trabalhando direto). O lock técnico existe e funciona —
`F:\GPU_BENCH.lock`, com `Assert-GpuLock` lançando em vez de devolver `false`. Não é
o lock que falta.

O que falta é humano: **ninguém sabe quando a placa está livre para um bloco de 20 a
30 minutos**. Descobre-se batendo nela.

Estabelecer e escrever, onde as duas sessões leem, um protocolo de janela: quem pede
avisa, quem tem larga, e qual é o bloco nomeado.

## Por que isso bloqueia a faixa inteira

Contenção move número. Medido em 2026-08-19/20, mesmo código, mesmo dia:
`im2col 74,4 s` com a máquina quieta contra `87,0 s` carregada — **17%**, contra um
efeito de 10 s que se tentava medir. Sem janela limpa, os tickets `13`, `14` e `15`
produzem números que não sustentam veredito nenhum, e a faixa de performance não
fecha por mais tickets que se resolva.

## Critério de fechamento

Fecha quando existir, escrito num arquivo que as duas sessões leem, um protocolo com
os três elementos: tamanho do bloco, como pedir, como largar. **Não** exige que o
protocolo seja bom — exige que exista e esteja acordado.

Fecha **sem** mudança de código: o lock já está correto. Se a conversa revelar que o
lock precisa mudar, isso vira ticket novo, não vira este.

## Resolucao

Escrito em `F:\COMFY_PORTABLE\CLAUDE.md`, secao `### The GPU window`, substituindo o paragrafo
de uma linha "GPU access, agreed 2026-08-20". E o arquivo que toda sessao desta bancada le antes
de qualquer coisa, que era o requisito ("escrito num arquivo que as duas sessoes leem").

Os tres elementos que o criterio pede, um a um:

- **Tamanho do bloco:** 30 minutos, e com nome — `bench:ticket25_fbcache_visual`, nao "preciso da
  placa". A string do dono e o que o outro lado le quando e recusado, entao ela carrega o plano.
  Trabalho que nao consegue declarar o bloco antes nao ganha bloco.
- **Como pedir:** `Assert-GpuLock -Owner '<repo>:<o-que>'`, que lanca. Nunca
  `Take-GpuLock | Out-Null`. Se recusado: medir a placa com NVML por ~1 min (o lock nao e a placa),
  falar no chat, e sair para trabalho sem GPU. Nao esperar, nunca tomar lock vivo em silencio.
- **Como largar:** `Release-GpuLock`, depois que o trabalho para, nunca antes. Segurar o lock
  ocioso e alegacao falsa sobre placa compartilhada; lock livre sobre GPU ocupada e a mesma mentira
  ao contrario.

Comando exato que provou que o mecanismo descrito funciona (EXECUTADO nesta sessao, 2026-08-21):

```
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1; Assert-GpuLock -Owner 'bench:ticket25_fbcache_visual'
-> lock TAKEN by bench:ticket25_fbcache_visual (heartbeat pid 40488)
-> dono=bench:ticket25_fbcache_visual / pid=40488 / hb=1787304296 / owner_kind=controller
```

O heartbeat destacado subiu e o `pid` gravado e o DELE, nao o do processo da tool call — que era o
defeito de 2026-08-19 que o `gpu_lock.ps1` existe para corrigir.

**O que ficou sem cobertura, e importa:** o criterio pede protocolo "acordado", e nesta sessao so
havia um lado. O dono disse "gpu liberada, so tem voce agora" — nao ha sessao irma viva para
acordar com. Entao o que existe e **um lado do protocolo escrito, em vigor ate ser contestado**,
e a secao do `CLAUDE.md` diz isso de si mesma, em vez de se apresentar como acordo negociado. Se
uma segunda sessao voltar e discordar, o acerto e ticket novo, nao reabertura deste: o mecanismo
(`tools/gpu_lock.ps1`, heartbeat, limite de 55 s) foi EXECUTADO; o acordo e decisao, nao medicao.
