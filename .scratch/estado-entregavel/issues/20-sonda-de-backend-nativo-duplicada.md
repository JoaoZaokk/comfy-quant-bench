# A sonda de "backend nativo pronto" segue duplicada, e já divergiu

Type: task
Status: resolved

## Question

`AUDITORIA_2026-08-18.md` (item 17/18) achou 8 cópias divergentes da pergunta "o backend nativo
está pronto?" e 6 cópias de uma função `instrument()` que conta chamadas nativas vs
dequantizadas. O projeto já resolveu o problema equivalente duas vezes — `tools/_bench_guard.py`
para o guard NVML (4 instrumentos) e `tools/_ram_guard.py` para o guard de RAM (3 conversores) —
mas não para este cluster.

**`instrument()`, 6 cópias**, confirmado por `grep -rln "def instrument" tools/`:
`compile_w4a4_probe.py` (superado, mantido de propósito — fora deste ticket),
`convrot_ops_probe.py`, `diffusion_smoke.py`, `gemma_chat.py`, `stage_probe.py`, `te_smoke.py`.

Duas divergências concretas, reconferidas em 2026-08-21 (ainda presentes):

1. **`gemma_chat.py:105`** rotula o segundo argumento posicional de `convrot_w4a4_linear` como
   `"qweight"`:
   ```python
   wrap_linear(convrot, "convrot_w4a4_linear", ("x", "qweight", "wscales", "bias"))
   ```
   A assinatura real do kernel usa `qdata` (`comfy_kitchen/tensor/w4a8_int8.py:93-96`, citado por
   `AUDITORIA:174`). Como o dicionário é montado por `dict(zip(arg_names, args))`, o valor
   capturado continua correto posicionalmente — o que diverge é a **chave**. Não confirmado se
   algum consumidor lê `probe["qweight"]` esperando outra coisa; é leitura de assinatura
   divergente, não uma falha medida.

2. **`diffusion_smoke.py:173`, `stage_probe.py:163`, `convrot_ops_probe.py:116`** contam
   chamadas ao **despachante** `convrot_w4a4_linear` como prova de "nativo". O despachante roteia
   para CUDA ou para eager conforme `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` e conforme o
   registry resolve a implementação — contar a chamada ao despachante não distingue os dois casos,
   nem distingue W4A4 de W4A8. `quality_battery.py` teve exatamente este defeito e já foi
   corrigido (agora reporta `kernel_impls`, não só um booleano `native`) — os outros três não.

## Critério de fechamento

Fecha quando existir **um módulo** (`tools/_native_probe.py` ou nome equivalente) com:
- uma função de "backend nativo pronto" que resolve `quantize_convrot_w4a4_weight` E
  `convrot_w4a4_linear` (o padrão mais estrito, já usado por `quant_w4a4.py`), com kwargs reais
  para não pular a validação de constraints (o mesmo bug que `quant_mixed.py` já corrigiu);
- uma função `instrument()` que reporta a implementação real resolvida (`kernel_impls`), não só
  contagem de chamadas ao despachante;

e os 5 arquivos hoje com cópia própria (`convrot_ops_probe.py`, `diffusion_smoke.py`,
`gemma_chat.py`, `stage_probe.py`, `te_smoke.py`) importando do módulo em vez de reimplementar.
`compile_w4a4_probe.py`/`probe2.py` ficam de fora — já são superados e mantidos só por
irreversibilidade do apagar, não vale a pena migrá-los.

Não fecha só documentando a divergência — o `gemma_chat.py:105` já está documentado
(`AUDITORIA:174`) e a duplicação continua.

## Resolucao

Criado `tools/_native_probe.py` com as duas funções do critério: `native_backend_ready(portable_root)`
(resolve `quantize_convrot_w4a4_weight` E `convrot_w4a4_linear`, com kwargs reais de uma
quantização real, no padrão que `quant_mixed.py` já corrigiu -- não o de tensores dummy que
`quant_w4a4.py` ainda usa) e `instrument()` (conta chamadas nativas vs dequantizações e registra
`impls`, os módulos que `registry.get_implementation` de fato resolveu, para ConvRot W4A4, W4A8 e
INT8 tensor-wise -- união do que `diffusion_smoke.py` já cobria, estendida aos outros 4). Os 5
arquivos do critério (`convrot_ops_probe.py`, `diffusion_smoke.py`, `gemma_chat.py`,
`stage_probe.py`, `te_smoke.py`) agora importam `instrument` do módulo em vez de reimplementá-la;
`compile_w4a4_probe.py`/`probe2.py` ficaram de fora, como o critério pede.

**Item 1 (qweight vs qdata) -- LIDO, e a conclusão do ticket estava invertida.** Reli a assinatura
real de `convrot_w4a4_linear` em `comfy_kitchen/tensor/convrot_w4a4.py:80-89` e nas quatro
implementações instaladas (`backends/{cuda,eager}/`): o segundo argumento é `qweight` em todas.
`gemma_chat.py:105` (a linha citada pelo ticket) já estava certa. O bug real -- confirmado pela
mesma leitura, mais `backends/{cuda,eager,hip,triton}/w4a8_int8.py` e o probe já corrigido de
`quant_mixed.py:114-120` que usa os dois nomes certos -- estava na linha seguinte,
`gemma_chat.py:106`, no wrap de `w4a8_int8_linear`: rotulava o argumento `qdata` como `qweight`.
`AUDITORIA:174` citou o arquivo/linha certos (`w4a8_int8.py:93-96`) mas o ticket read a partir dali
anexou a citação à função errada. Não mudei nada por causa disso sozinho -- o `_native_probe.py`
novo já nasce com `qweight` para ConvRot e `qdata` para W4A8 (as duas funções corretas), e
`gemma_chat.py` herda o valor certo ao importar `instrument` do módulo em vez de manter sua cópia
com o argumento trocado. NÃO EXECUTADO contra a GPU nesta sessão -- sem placa disponível (regra da
rodada) -- então não foi medido se a mistura de rótulos chegava a mudar qual implementação
`registry.get_implementation` resolvia para uma chamada real; só a leitura de assinatura confirma
que a chave estava errada.

**Item 2 (contar despachante como "nativo") -- corrigido nos 3 pontos citados.**
`convrot_ops_probe.py`, `diffusion_smoke.py` e `stage_probe.py` computavam `"native"` (ou o texto
`native=`) a partir de `native_calls > 0 and dequant_calls == 0` -- conta a chamada ao
despachante, que roteia para CUDA ou eager conforme constraints e
`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, não prova qual dos dois rodou. Troquei os 3 por
`any(".backends.cuda" in impl for impl in counters["impls"])`, lendo o path do módulo que
`instrument()` já registra -- o mesmo padrão que `quality_battery.py` (que importa `instrument` de
`gemma_chat.py`) já usava ao reportar `kernel_impls` em vez de um booleano.

Provado por: `py_compile` dos 6 arquivos (`.\python_embeded\python.exe -s -m py_compile
tools\_native_probe.py tools\convrot_ops_probe.py tools\diffusion_smoke.py tools\gemma_chat.py
tools\stage_probe.py tools\te_smoke.py`, todos OK) e um import real de cada um dos 6 módulos
(`sys.path` com `ComfyUI/` e `tools/`, `import comfy.quant_ops` seguido de
`importlib.import_module(...)` para cada) que executou `_native_probe.instrument()` de fato --
monta os wraps em `comfy_kitchen.tensor.{convrot_w4a4,w4a8_int8,int8}` e no
`_LAYOUT_DISPATCH_TABLE`, sem tocar CUDA -- e confirmou que os 5 arquivos migrados expõem
`instrument` importado, sem erro. Isso é EXECUTADO, mas só a montagem dos monkeypatches: nenhuma
chamada real a `convrot_w4a4_linear`/`w4a8_int8_linear` aconteceu (precisaria de tensor CUDA), e
`native_backend_ready()` nunca foi chamada nesta sessão -- ela mesma levanta `SystemExit` sem CUDA,
então rodá-la sem GPU só provaria que ela falha do jeito certo, o que não fiz por não valer o
custo. Grep no repo inteiro por `native_linear_calls`, `convrot_linear_calls`,
`weight_dequant_calls`, `kernel_impls` -- únicos consumidores fora de `tools/` são documentação
(`.md`), nenhum script fora dos 5 tocados lê esses campos.

NÃO COBERTO: (1) execução real contra a GPU -- nem `instrument()` contando uma chamada real, nem
`native_backend_ready()`, nem confirmação de que a correção do rótulo `qdata` muda alguma
resolução observável; fica para uma rodada com a placa livre. (2) `te_smoke.py` só exercitava
ConvRot antes; agora herda os wraps de W4A8/INT8 também, mas os nomes de campo no relatório
(`convrot_linear_calls` etc.) continuam com prefixo `convrot_` -- não renomeei porque nenhum
checkpoint que passa por esse script hoje mistura formato, então o campo continua correto para o
uso atual, mas o nome ficaria enganoso no dia em que misturar. (3) não toquei
`quant_w4a4.py`/`verify_w4a4.py`/`quant_mixed.py`/`quality_battery.py` -- fora da lista de arquivos
permitidos desta rodada; `quality_battery.py` continua importando `instrument` de `gemma_chat.py`
(reexportado do módulo novo) e não foi alterado nem precisou ser.

## Verificação com GPU — 2026-08-21, pelo orquestrador, lock `claude:rodada3-gpu-lane`

O agente que resolveu este ticket **não pôde executar** `native_backend_ready()` nem
`instrument()`: a placa estava em uso noutro trilho. Fechei a lacuna em vez de aceitar o
"não coberto".

**`native_backend_ready()` — EXECUTADO com CUDA:**

```
quantize_convrot_w4a4_weight: comfy_kitchen.backends.cuda
convrot_w4a4_linear:          comfy_kitchen.backends.cuda
native_ready: True
```

Resolve os **dois** ops, como o ticket exigia — não só o `linear`, que era o defeito do
`verify_w4a4.py` registrado no `CLAUDE.md`.

**`instrument()` — EXECUTADO com uma chamada CUDA real:**

```
native_calls  = 1
dequant_calls = 0
impls = ['comfy_kitchen.backends.cuda.convrot_w4a4_linear']
```

**Uma armadilha que vale registrar, porque quase virou um falso defeito:** a primeira tentativa
chamou `comfy_kitchen.convrot_w4a4_linear` (topo do pacote) e o contador deu **zero**. Não é bug
— `instrument()` instrumenta a camada `comfy_kitchen.tensor.*`, que é por onde o ComfyUI passa.
Chamando `comfy_kitchen.tensor.convrot_w4a4.convrot_w4a4_linear`, conta certo. Quem for auditar
esta função de novo: chamar pelo topo e ver zero **não** prova que a instrumentação está quebrada.

**Efeito colateral que responde outro aviso:** o `CLAUDE.md` diz, desde hoje, que comfy-kitchen
subiu de 0.2.23 para **0.2.31** e manda reconferir o preflight antes de confiar em conversão
antiga. Reconferido: **0.2.31 não quebrou a resolução nativa.**

`tools/quant_int8.py` (ticket `24`) também foi verificado na mesma janela —
`normal_comfy_backend(root, 256)` devolve `{'quantizer': 'comfy_kitchen.backends.cuda',
'native_ready': True}`. Executado, não lido.

**Não coberto:** `w4a8_int8_linear` e o ramo INT8 do `_LAYOUT_DISPATCH_TABLE` que `instrument()`
também embrulha não foram exercitados — só o caminho ConvRot W4A4. Nenhuma conversão de modelo
foi rodada.
