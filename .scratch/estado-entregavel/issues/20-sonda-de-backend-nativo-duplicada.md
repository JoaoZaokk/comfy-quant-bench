# A sonda de "backend nativo pronto" segue duplicada, e já divergiu

Type: task
Status: open

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
