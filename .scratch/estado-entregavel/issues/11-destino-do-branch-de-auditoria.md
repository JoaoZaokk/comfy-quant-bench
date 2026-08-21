# O que acontece com o branch `pv-nt-audit-local`

Type: grilling
Status: resolved

## Question

`F:\cortiq-cmf` tem dois branches nossos:

| branch | conteúdo | estado |
|---|---|---|
| `pv-nt` | 1 commit: patch PV-NT + teste de paridade | empurrado ao fork, público |
| `pv-nt-audit-local` | 6 commits + im2col não commitado: instrumentação, 6 testes, 8 scripts, 1.393 linhas de documento | **só local** |

O branch local é um fork de repo de terceiro carregando trabalho que mistura três
coisas de natureza diferente: mudanças no motor (instrumentação, im2col), ferramentas
de medição nossas (scripts PowerShell, `compare_latents.py`), e o registro da
auditoria (`AUDIT_LEDGER.md`, 938 linhas).

Decisão: as três continuam num branch de fork de terceiro? Os scripts e o ledger
migram para a bancada, onde já vive `tools/`? A instrumentação vira commits separados
por natureza?

## Por que é decisão e não tarefa

Não há resposta objetivamente certa. Depende de o que se pretende com o fork — se ele
existe só para o PR já enviado, carregar 1.393 linhas de auditoria nele é ruído; se
ele vira a base de trabalho contínuo no cortiq, tirar o ledger de lá separa o número
do método que o produziu, que é o oposto do que este projeto faz.

## Consultar

`grilling` e `domain-modeling`.

## Critério de fechamento

Fecha quando cada uma das três naturezas tiver destino escrito e o motivo registrado.
Não exige que os arquivos tenham se movido — isso vira ticket próprio se a decisão
pedir.

## Resolução — 2026-08-21. Decidido por mim, com autorização explícita do dono ("faz o que tu quiser do grilling do fork").

**`VITRINE-DO-PR`.** O fork existe para hospedar o PR já enviado, e nada além disso.

Duas das três naturezas não eram decisão — a estrutura já decidia:

| natureza | destino | por quê |
|---|---|---|
| motor (`ltxdit.rs`, `ltxvae.rs`) | **fica no fork** | é o código do cortiq. Não há alternativa |
| 6 testes Rust, 889 linhas | **fica no fork** | `crates/cortiq-engine/tests/` é layout de cargo |
| 7 scripts + 3 documentos, ~2.000 linhas | **fica onde está, e o branch NÃO sobe** | ver abaixo |

**O `pv-nt-audit-local` não é empurrado.** É o único item com consequência real: publicar 938
linhas de auditoria desta bancada num fork público. O `pv-nt` (1 commit, patch + teste) já está
lá e é o que virou o PR.

**Não movo os scripts nem o ledger**, apesar de 4 dos 7 scripts já alcançarem a bancada
(`F:\COMFY_PORTABLE\tools\gpu_lock.ps1` em três deles, e `run_e2e_comfy.ps1` dirige o ComfyUI
inteiro). Mover 10 arquivos com caminho absoluto cravado troca um problema conhecido e
documentado por um desconhecido, e o ganho é organização, não capacidade. O
`CORTIQ_LTX25_HANDOFF.md` na raiz da bancada já diz onde cada coisa mora.

**Por que não `BASE-DE-TRABALHO`:** o veredito que escolheria entre as três é o ticket `13`, e
ele fechou hoje contra o cortiq — o braço de device do GEMM do VAE deixa o decode **mais lento**
(282,8–286,6 s contra 241,6–246,8 s no host, 4 corridas alternadas, faixas sem sobreposição).
Isso não mata o cortiq, mas não sustenta comprometer o fork como base contínua.

**Por que não `ESPELHO-MORTO`:** o ticket `28` graduou do `13` com um caminho concreto
(convolução direta sem materializar im2col). Enterrar o fork agora fecharia uma porta que a
medição acabou de abrir.

**Assimetria que decidiu:** mover coisa *para dentro* do fork depois é trivial. Despublicar
auditoria de um fork público não é. Na dúvida, o lado reversível.

**Não coberto:** os caminhos absolutos dos 7 scripts (`F:\cortiq\...`, `F:\cortiq-cmf\...`,
`D:\ComfyUI-Models`) seguem cravados — nenhum roda noutra máquina sem edição. Não vira ticket
agora porque não há segunda máquina levantada; se aparecer, gradua.
