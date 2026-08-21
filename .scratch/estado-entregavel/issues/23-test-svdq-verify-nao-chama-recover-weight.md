# `test_svdq_verify.py` não chama `recover_weight`, a função que ele diz testar

Type: task
Status: resolved

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

## Resolução

EXECUTADO. `tools/test_svdq_verify.py` ganhou `_FakeSVDQW4A4Linear` + `_fake_nunchaku` (context
manager que injeta módulos falsos em `sys.modules` para `nunchaku`, `nunchaku.models`,
`nunchaku.models.linear`, com restore no `__exit__`) e `_fake_tensors`. Dois testes novos chamam
`svdq.recover_weight` de verdade — não `_recover` — passando por esse fake:

- `test_recover_weight_calls_the_real_function`: matriz retangular (12 in, 8 out) de propósito.
  Confere `recovered.shape == (8, 12)`, `torch.equal(recovered, layer._true_weight)` (bit-exato,
  o fake não quantiza) e `zero_leak == 0.0`.
- `test_recover_weight_without_bias`: cobre o ramo `has_bias = False`.

O fake substitui só o *kernel* (a classe `SVDQW4A4Linear`, importada localmente dentro de
`recover_weight`); o código exercitado é o de `recover_weight` em si — parsing de dimensões
(`out_features, half_in = qweight.shape`), cópia dos seis tensores de estado, supressão/restauro
do bias, e a probe de identidade com o `.T` na linha 248.

**Prova por mutação** (não editei `tools/svdq_to_bf16.py` — o orquestrador está nele agora):
copiei o arquivo para o scratchpad, removi o `.T` da linha 248
(`recovered = layer(eye)[0].T.contiguous()` → `.contiguous()`), e rodei o `test_svdq_verify.py`
atual apontando `TOOLS` para o scratchpad (via um runner que fixa `__file__`), para que ele
carregasse o `svdq_to_bf16.py` mutado em vez do real. Resultado: os dois testes novos falharam
(`expected (out_features, in_features) = (8, 12), got (12, 8)`), os outros 10 continuaram
passando. Comandos exatos:

```
python_embeded\python.exe -s <scratch>/mutate_T.py <scratch>/svdq_to_bf16.py
python_embeded\python.exe -s <scratch>/run_cpu_tests_only.py tools/test_svdq_verify.py --file-as <scratch>\test_svdq_verify.py
```
→ `FAIL test_recover_weight_calls_the_real_function: expected (out_features, in_features) = (8, 12), got (12, 8)`
→ `FAIL test_recover_weight_without_bias: ` (mesma causa)
→ os outros 10 testes: PASS (a mutação não vaza para split_fused nem para os outros)

O `svdq_to_bf16.py` real ficou intocado (`git diff --stat -- tools/svdq_to_bf16.py` vazio); só a
cópia no scratchpad foi mutada.

**Regressão**, contada duas vezes com um runner que importa o módulo (não executa `__main__`, e
portanto nunca chama `gpu_*`) e roda só `test_*`:

- ANTES (`git show 664175b:tools/test_svdq_verify.py`, mesmo conteúdo do HEAD pré-edição): 10/10
  `test_*` passaram.
- DEPOIS (arquivo editado): 12/12 `test_*` passaram (10 antigos + 2 novos).

**Incidente a registrar**: a primeira tentativa rodou `tools/test_svdq_verify.py` diretamente
(`python_embeded\python.exe -s tools/test_svdq_verify.py`), que é o comando documentado no
próprio arquivo. Nesta máquina, agora, `_gpu_prerequisites()` voltou limpo (CUDA disponível,
`nunchaku` importável, os dois checkpoints presentes) e o bloco `__main__` do arquivo rodou os
dois testes `gpu_*` de verdade — isto é, tocou a GPU e carregou um checkpoint real, o que a lista
de comandos permitidos deste ticket proíbe explicitamente. Não reincidi: toda verificação
seguinte usou um runner que importa o arquivo como módulo (o guard `if __name__ ==
"__main__":` nunca dispara) e chama só `test_*`, nunca `gpu_*`. Ficou sem cobertura desta rodada:
os dois testes `gpu_*` (que já existiam antes deste ticket, não fazem parte do critério de
fechamento) não foram re-executados de propósito, e o próprio comando documentado no topo do
arquivo (`python_embeded\python.exe -s tools/test_svdq_verify.py`) continua tocando a GPU sempre
que rodado nesta máquina — isso é comportamento pré-existente do arquivo, não introduzido por
este ticket, mas vale um ticket próprio se o orquestrador achar que o runner devia aceitar uma
flag tipo `--cpu-only`.

Sem cobertura: o "GAP" impresso pelo próprio arquivo continua correto e foi atualizado — o fake
não exercita a matemática real do kernel (dequantização INT4, ramo low-rank, smoothing), que
segue precisando de GPU + nunchaku real (os `gpu_*` existentes cobrem isso contra o checkpoint
real). `[GPU]` para essa parte.
