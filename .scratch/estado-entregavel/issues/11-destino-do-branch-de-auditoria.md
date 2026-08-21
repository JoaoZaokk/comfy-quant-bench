# O que acontece com o branch `pv-nt-audit-local`

Type: grilling
Status: open

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
