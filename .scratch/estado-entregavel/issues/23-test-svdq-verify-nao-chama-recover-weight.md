# `test_svdq_verify.py` não chama `recover_weight`, a função que ele diz testar

Type: task
Status: open

## Question

`AUDITORIA_2026-08-18.md` item 16 (fila de prioridade) e a seção final "Não corrigido, e por quê"
dizem, ao pé da letra:

> `test_svdq_verify.py` fica verde com 3 mutações (w3/w1 trocado, `.T` removido, q/k/v invertido)
> porque reimplementa a sonda em vez de chamar `recover_weight`. Consertar é reescrever a suíte
> contra o módulo real, não um ajuste — e a confirmação da mutação `.T` pede GPU.

Reconferido em 2026-08-21: o arquivo **já melhorou** desde a auditoria — `split_fused` agora tem
4 testes que chamam a função real (`svdq.split_fused`, linhas 190-243), cobrindo a ordem w3/w1 e a
ordem q/k/v. O que continua exatamente como o auditor descreveu é `recover_weight` em si: o
próprio arquivo documenta a lacuna nas linhas 250-263 --- `recover_weight() is NOT covered by the
CPU tests: it builds its layer from a state dict via a reimplementation _recover, so a bug in the
shipped recover_weight passes them` --- o que é exatamente a disciplina que o `CLAUDE.md` pede
(achado não confirmado rotulado como tal, dentro do próprio artefato). A lacuna está honestamente
rotulada; não está fechada.

`W4A4_HANDOFF.md:214` já rastreia isto na prosa ("`test_svdq_verify` cobre `split_fused` agora
(provado por mutação) mas **não** `recover_weight`"), mas nenhum ticket do `.scratch` cobria.

## Critério de fechamento

Fecha quando existir um teste que **chame `svdq_to_bf16.recover_weight`** diretamente (não uma
reimplementação `_recover`) e que a mutação `.T` (transposição removida) faça esse teste falhar —
mesmo padrão que os testes de `split_fused` já seguem para w3/w1 e q/k/v.

Se a confirmação numérica exata da mutação `.T` exigir pesar contra um kernel real (GPU), o teste
ainda pode existir em CPU comparando `recover_weight` contra uma referência matemática simples
(ex.: reconstrução manual de uma matriz pequena conhecida) — não precisa da GPU para chamar a
função real, só precisa parar de reimplementá-la.

Não fecha só melhorando o comentário — a lacuna já está bem documentada; falta fechar o código.
