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

<!-- indice: uma linha por ticket fechado, com link para o detalhe -->

**21 de 29 fechados** em 2026-08-21. Um por linha, na ordem do numero.

- [02 — Triagem barata dos 93 arquivos rastreados](issues/02-triagem-dos-88-arquivos.md)
- [03 — `ui_to_api` descarta nó desconhecido em silêncio e submete o grafo assim](issues/03-ui-to-api-descarta-no-em-silencio.md)
- [04 — A trava anti-cache é um chute, e está duplicada em duas linguagens](issues/04-heuristica-de-cache-duplicada.md)
- [05 — Não existe modelo do formato API, e três achados são sintoma disso](issues/05-modelo-tipado-do-formato-api.md)
- [06 — `main()` tem 127 linhas e faz oito trabalhos](issues/06-decompor-main.md)
- [07 — `CONV3D_PROF` é a terceira cópia de um idioma que já existia na crate](issues/07-idioma-de-profiler-triplicado.md)
- [08 — `add_shaped` tem 8 parâmetros porque falta um tipo](issues/08-add-shaped-oito-parametros.md)
- [09 — O laço im2col existe três vezes, e a flag que o duplica não precisa existir](issues/09-tres-copias-do-im2col.md)
- [11 — O que acontece com o branch `pv-nt-audit-local`](issues/11-destino-do-branch-de-auditoria.md)
- [12 — Inventário auditável de env, nodes e modelos convertidos](issues/12-inventario-da-bancada.md)
- [13 — O GEMM do VAE vale ir para a placa?](issues/13-gemm-do-vae-no-device.md)
- [14 — `gpu_q4tp_batch` recusa `b=375` e `b=1879`: é o lote ou a forma?](issues/14-recusa-do-gpu-q4tp-batch.md)
- [15 — Re-medir o PV-NT sob janela limpa](issues/15-remedir-pv-nt-com-janela-limpa.md)
- [16 — Artefatos nossos que moram dentro de `ComfyUI/`](issues/16-artefatos-nossos-dentro-de-comfyui.md)
- [18 — `activation_balance.py` e `plot_weight_balance.py` medem crest no tensor errado](issues/18-crest-medido-antes-da-rotacao.md)
- [19 — `core_patch.py revert` sobrescreve o arquivo atual sem checar o hash dele antes](issues/19-core-patch-revert-sem-checar-hash-atual.md)
- [20 — A sonda de "backend nativo pronto" segue duplicada, e já divergiu](issues/20-sonda-de-backend-nativo-duplicada.md)
- [21 — `nunchaku_compare.py`: pico de VRAM nunca desconta memória liberada, e `--attention sage` não verifica nada](issues/21-nunchaku-compare-pico-de-vram-so-sobe.md)
- [24 — `quant_w4a8.py`: `as_bytes()` morto. `quant_int8.py`: sem preflight de backend com `--device cuda`](issues/24-quant-w4a8-as-bytes-morto-quant-int8-sem-preflight.md)
- [26 — `hf_parallel_get.py` resolve o problema que `fetch_*` descrevem, mas nenhum dos dois o usa](issues/26-hf-parallel-get-sem-chamador.md)
- [29 — Cinco scripts de medição morrem quando o binário escreve no stderr](issues/29-scripts-de-medicao-morrem-com-o-proprio-log.md)

**Em aberto:**

- [01 — Janela de GPU acordada com a outra sessão](issues/01-janela-de-gpu.md) — `open`
- [10 — `std::env::var` dentro da função que o patch existe para acelerar](issues/10-env-var-no-caminho-quente.md) — `open`
- [17 — Fatos de operação da bancada voltam para o `CLAUDE.md`](issues/17-fatos-de-operacao-de-volta-ao-claude-md.md) — `blocked`
- [22 — `svdq_to_bf16.py`: três achados de baixa confiança seguem sem fechar](issues/22-svdq-to-bf16-achados-baixa-confianca-abertos.md) — `open`
- [23 — `test_svdq_verify.py` não chama `recover_weight`, a função que ele diz testar](issues/23-test-svdq-verify-nao-chama-recover-weight.md) — `open`
- [25 — `fbcache_visual.py` é o único `fbcache_*` sem resultado registrado](issues/25-fbcache-visual-sem-resultado-registrado.md) — `open`
- [27 — Teste atrás de `cfg(feature)` reporta `ok` rodando zero testes](issues/27-teste-com-cfg-feature-reporta-ok-com-zero-testes.md) — `open`
- [28 — Convolução direta no VAE, sem materializar o im2col](issues/28-conv-direta-sem-materializar-im2col.md) — `open`

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
