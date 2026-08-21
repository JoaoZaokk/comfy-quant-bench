# Janela de GPU acordada com a outra sessão

Type: task
Status: open

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
