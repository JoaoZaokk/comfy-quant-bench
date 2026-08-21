# `quant_w4a8.py`: `as_bytes()` morto. `quant_int8.py`: sem preflight de backend com `--device cuda`

Type: task
Status: open

## Question

Dois achados independentes, mesmo par de arquivos, ambos em `AUDITORIA_2026-08-18.md` seção 5
(baixa confiança) e nenhum na lista de correções de 2026-08-18.

**1. `tools/quant_w4a8.py:177` — `as_bytes()` morto.**

```
$ git grep -n "as_bytes" tools/
tools/quant_w4a8.py:177:def as_bytes(tensor: torch.Tensor) -> memoryview:
```

Zero sítios de chamada em qualquer arquivo de `tools/`. Escrita exatamente para BF16
(`.view(torch.int16)`) e float8_e5m2 (`.view(torch.uint8)`); o caminho real do conversor
(linhas ~320, ~347 segundo `AUDITORIA:163`) cobre só float8_e4m3fn. Ou o código usa `as_bytes`
para os outros dois dtypes, ou a função é lixo e o buraco de bf16/e5m2 que ela existia para tapar
segue aberto de outro jeito.

**2. `tools/quant_int8.py` — sem preflight de backend quando `--device cuda`.**

```python
if args.device == "cuda" and not torch.cuda.is_available():   # linha 113-114
    raise SystemExit("CUDA requested but unavailable")
```

Isso confirma que a placa existe, não que `comfy_kitchen` resolve o quantizador int8 para a
implementação CUDA nativa. `quant_w4a4.py`, `quant_w4a4_smooth.py`, `quant_w4a8.py` e
`quant_mixed.py` chamam `normal_comfy_backend()` antes de converter qualquer coisa
(`grep -rln "normal_comfy_backend" tools/` → os 4). `quant_int8.py` não está na lista. O
docstring do arquivo explica que o desenho é CPU-first ("Unlike quant_w4a8.py this does not
require CUDA... resolves to comfy-kitchen's eager backend"), o que é razão suficiente para não
recusar por padrão — mas não é razão para pular a checagem quando o usuário passa
explicitamente `--device cuda` pedindo o caminho nativo.

Achado relacionado, mesma seção do `AUDITORIA` (item junto ao de `as_bytes`): nenhum dos dois
conversores valida o dtype/shape do retorno do quantizador além de um acordo tácito de shape com
`comfy_kitchen` (`quantize_int8_rowwise` devolve `[N,1]`, o conversor confia nisso sem checar).

## Critério de fechamento

Fecha em duas partes independentes:

1. `as_bytes()` em `quant_w4a8.py`: ou passa a ser chamado no caminho real para bf16/float8_e5m2
   (fechando o buraco que motivou escrevê-la), ou é removida com uma nota dizendo que os dois
   dtypes não são suportados por este conversor hoje. Não fica como está — função morta com um
   propósito declarado e nunca exercitado.
2. `quant_int8.py --device cuda` chama `normal_comfy_backend()` (importado de `quant_w4a4.py`,
   como os outros 4 conversores já fazem) antes de converter, e recusa se o backend nativo não
   resolver.

Não fecha com decisão escrita: os dois são código, não hipótese.
