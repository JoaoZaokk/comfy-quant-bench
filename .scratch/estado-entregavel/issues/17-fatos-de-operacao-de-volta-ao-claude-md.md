# Fatos de operação da bancada voltam para o `CLAUDE.md`

Type: task
Status: resolved (os tres fatos escritos; fato 3 entrou quando o 01 fechou)
Blocked by: 01

## Question

Este trabalho mediu fatos sobre como **esta bancada** opera, e eles ficaram escritos
no lugar errado: dentro de scripts e do ledger no repo do **cortiq**, que é fork de
terceiro. A casa canônica é `F:\COMFY_PORTABLE\CLAUDE.md`, que é o que toda sessão lê.

Três fatos, todos executados:

**1. `--disable-dynamic-vram` é necessário para o workflow LTX 2.5 desta bancada.**
Hoje o `CLAUDE.md` diz que a flag serve para loaders Nunchaku, e o ledger do cortiq
chegou a afirmar que este workflow *não* devia recebê-la — leitura, não execução. Sem
ela o render morre:

```
Model LTXAVTEModel_ prepared for dynamic VRAM loading. 14612MB Staged.
Model LTXAV        prepared for dynamic VRAM loading. 20484MB Staged.
aimdo: src/hostbuf.c:283:ERROR:hostbuf_read_file_slice: device copy failed
RuntimeError: HostBuffer.read_file_slice failed
torch.AcceleratorError: CUDA error: out of memory
```

14612 + 20484 = 35 GB encenados numa placa de 24. O resumo do próprio PyTorch,
impresso ao lado, diz `Allocated memory 141211 KiB` e `CUDA OOMs: 0` — quem estourou
foi o alocador do dynamic-vram, não o do torch. **A mensagem não aponta para o
subsistema culpado em lugar nenhum**, e manda o leitor direto para "o modelo é grande
demais", numa placa com 18 GB livres no momento da falha.

**2. `main.py --windows-standalone-build` re-executa como processo filho.** Matar o
pid que `Start-Process` devolve deixa um órfão. Aqui ficou segurando **20.578 MiB da
3090** e 16,67 GB de RAM, invisível: `Get-Process -Id` dizia morto, `nvidia-smi` dizia
ocupado. Matar a árvore, não o pid.

**3. O protocolo de GPU acordado no ticket `01`.**

## Depende de 01

Os dois primeiros fatos não dependem de nada. O terceiro é o resultado do `01`, e
escrever o documento uma vez só é melhor que escrever duas.

## Critério de fechamento

Fecha quando os três fatos estiverem no `CLAUDE.md` da bancada, **com o erro literal
junto** — a mensagem enganosa é metade do valor do registro.

Não fecha com decisão escrita: o fato já foi medido, e deixar de escrevê-lo é
exatamente a falha que este ticket corrige.

## Resolução

O que mudou: fatos 1 e 2 foram escritos em `F:\COMFY_PORTABLE\CLAUDE.md`, cada um com
o bloco de erro/número literal ao lado, colado sem paráfrase, e marcados no próprio
texto como `verified 2026-08-21` / "executed, not traced" (a disciplina "diga qual foi:
traçado ou executado" do próprio `CLAUDE.md`, aplicada ao registro que a introduz).

- Fato 1: parágrafo novo logo após o parágrafo Nunchaku existente, seção `## Commands`
  (`CLAUDE.md` linhas ~94-104). Deixa explícito que é um segundo motivo independente
  para `--disable-dynamic-vram` — não reescreve a frase Nunchaku, acrescenta ao lado —
  e nomeia que o ledger do cortiq afirmou o oposto por leitura, não execução. O bloco
  de erro (`Model LTXAVTEModel_...` até `torch.AcceleratorError: CUDA error: out of
  memory`) foi colado verbatim.
- Fato 2: parágrafo novo após "Launcher notes" na mesma seção (`CLAUDE.md` linhas
  ~110-112), com os números literais (20.578 MiB de VRAM, 16,67 GB de RAM, o par
  `Get-Process -Id` morto / `nvidia-smi` ocupado) colados como estavam no ticket.

Comando exato que provou: nenhum — esta tarefa é transcrição de uma medição já feita
em sessão anterior (registrada no corpo deste ticket antes de eu olhar), não uma
reprodução. Os dois fatos exigem a GPU, e a RTX 3090 está com outra sessão nesta
rodada (regra dura do orquestrador: não usar a GPU, não subir o ComfyUI). Não
reproduzi nenhum dos dois números; copiei-os do ticket para o `CLAUDE.md` preservando
o texto do erro e citando a data de verificação já registrada (2026-08-21). A prova
em si é anterior a esta sessão.

O que ficou sem cobertura: o fato 3 (protocolo de GPU acordado no ticket `01`) **não**
foi escrito, por instrução explícita de quem lançou esta tarefa — o ticket `01`
(janela de GPU, acordo humano) ainda está em aberto, e escrever o fato 3 agora seria
documentar um acordo que ainda não existe. Status fica `blocked`, não `resolved`,
porque o critério de fechamento deste ticket pede os três fatos juntos; só dois foram
escritos. Reabrir/fechar de verdade quando o `01` estiver resolvido e o fato 3 puder
ser escrito na mesma passada — como o próprio ticket já recomendava ("escrever o
documento uma vez só é melhor que escrever duas").


## Resolucao, parte 2 — o fato 3 (2026-08-21)

O ticket `01` fechou, entao o fato 3 pode ser escrito, e foi: `CLAUDE.md`, secao
`### The GPU window`, substituindo o paragrafo de uma linha que existia antes. Os tres elementos
(bloco de 30 min nomeado / `Assert-GpuLock` / `Release-GpuLock` depois do trabalho parar) estao la,
com os dois erros literais que o criterio pedia ao lado: o `im2col 74.4 s` contra `87.0 s` (17% de
contencao contra um efeito de 10 s) e o lock de 2026-08-19 que nomeava um pid morto no instante em
que foi escrito, com a placa ociosa atras dele por ~40 min.

Como o ticket recomendava, escrito **numa passada so** junto dos fatos 1 e 2, e nao em duas.

Criterio agora atendido inteiro: os tres fatos no `CLAUDE.md`, cada um com o erro/numero literal
junto — que era metade do valor do registro.

**Sem cobertura:** a mesma coisa que o `01` diz de si — o fato 3 e um protocolo de um lado so, e a
secao do `CLAUDE.md` declara isso. Os fatos 1 e 2 continuam sendo transcricao de medicao anterior,
nao reproducao (ver parte 1).
