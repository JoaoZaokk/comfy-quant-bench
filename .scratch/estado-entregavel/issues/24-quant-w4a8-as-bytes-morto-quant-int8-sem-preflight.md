# `quant_w4a8.py`: `as_bytes()` morto. `quant_int8.py`: sem preflight de backend com `--device cuda`

Type: task
Status: resolved

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

## Resolução

**1. `as_bytes()` — wired in, não removida.** Confirmado morto ANTES da mudança por grep de
`as_bytes\(` em `*.py` no repo inteiro, não só `tools/` — inclusive dentro de `ComfyUI/` e
`ComfyUI/custom_nodes/` (66 repos), que o `.gitignore` da raiz esconde de um `git grep` normal
(`.gitignore:5: /* -> ComfyUI`, `-> ComfyUI/custom_nodes`; confirmado com
`git check-ignore -v ComfyUI ComfyUI/custom_nodes`). Um grep raso teria simplesmente pulado essas
duas árvores em silêncio — não teria dado zero resultados por *não achar nada*, teria dado zero
por *não olhar*. Provei que o grep de fato varreu essas árvores rodando o mesmo grep com um termo
comum ("import torch") antes de confiar no negativo:

```
$ rg --glob '*.py' -c 'import torch' ComfyUI/custom_nodes | wc -l
1886 arquivos, 3146 ocorrências        # prova que a árvore FOI varrida
$ rg --glob '*.py' 'as_bytes\(' ComfyUI ComfyUI/custom_nodes
(vazio)                                 # zero call sites, agora sim uma leitura válida
$ rg --glob '*.py' 'as_bytes\(' .        # raiz allowlist (tools/ etc.)
tools/quant_w4a8.py:177:def as_bytes...  # só a definição, zero chamadas
```

Zero sítios de chamada confirmados em toda a instalação antes da mudança. Escolhi fechar o buraco
em vez de apagar: `torch.Tensor.numpy()` falha para `bfloat16` e para os dois dtypes float8 —
EXECUTADO no torch embarcado (CPU, sem tocar GPU) para confirmar, não inferido:

```
$ python_embeded\python.exe -s -c "import torch; torch.zeros(4, dtype=torch.bfloat16).numpy()"
TypeError: Got unsupported ScalarType BFloat16
```

`as_bytes()` já resolvia isso via `.view()` antes do `.numpy()`, só nunca era chamada — o caminho
real de escrita (`quant_w4a8.py`, laço de gravação final) chamava `item.numpy()` direto. Se
`quantize_w4a8_int8_weight` algum dia devolver `s_channel` ou `codebook` em bf16 (dtypes não
verificáveis sem CUDA — não executado, sem placa nesta sessão), o conversor crashava nesse ponto.
Também um `float8_e5m2` teria batido em `SAFETENSORS_DTYPE[torch.float8_e5m2]` → `KeyError`, um
passo antes, porque só `float8_e4m3fn` tinha caso especial. Separei a responsabilidade em
`header_dtype()` (nome do dtype no header — fp8 vira "U8", igual já era pra `e4m3fn`) e `as_bytes()`
(bytes crus, agora chamada de fato no laço de escrita). `quant_int8.py` não foi tocado nesta parte
— suas próprias saídas (`qdata` int8, `scale` já forçada a `.float()`) não passam por esse gap.

**2. `quant_int8.py --device cuda` agora chama preflight.** Adicionei `normal_comfy_backend()`
local (não importada de `quant_w4a4.py`) porque este conversor resolve um op diferente
(`quantize_int8_convrot_weight`, não `quantize_convrot_w4a4_weight`/`convrot_w4a4_linear` de
w4a4 nem `quantize_w4a8_int8_weight` de w4a8) — reusar a checagem de outro arquivo verbatim
resolveria o op errado. Mesmo idioma de subprocesso que `quant_w4a4.py:89` e `quant_w4a8.py:97`
(interpretador novo, `ck.registry.get_implementation`, `native_ready` só se o módulo resolvido
contém `.backends.cuda`), aplicado ao op certo. Chamado só quando `--device cuda` e `--convrot`:
`--no-convrot` chama `eager.quantize_int8_rowwise` direto, sem passar pelo registry, então não há
"backend nativo" a resolver ali por desenho declarado no próprio docstring do arquivo (isso
imprime uma nota em vez de fingir uma checagem sem sentido). Recusa com `SystemExit` se
`native_ready` for falso, como os outros 4 conversores.

**Não executado (sem GPU nesta sessão, proibido pelo orquestrador):** a resolução real do
backend via `ck.registry.get_implementation` — nem para `quantize_int8_convrot_weight` (novo)
nem para o `quantize_w4a8_int8_weight` já existente que passou a ser exercitado pelo caminho de
escrita mudado. Verificado por `py_compile` dos dois arquivos e leitura lado a lado com
`quant_w4a4.py:89-109` e `quant_w4a8.py:97-114` (mesma estrutura: subprocess, tensor dummy,
`get_implementation`, checagem de `.backends.cuda`, `native_ready`). Também não executado: uma
conversão real ponta-a-ponta com `--device cuda --convrot` para confirmar que o preflight de fato
recusa/aceita corretamente em condições reais, e nenhuma verificação de que
`quantize_w4a8_int8_weight` de fato devolve algum dia bf16/e5m2 para `s_channel`/`codebook` — a
correção em `as_bytes()` é uma blindagem provada contra um `TypeError` genérico do torch, não uma
prova de que esse dtype específico ocorre na prática.

**Achado relacionado, fora do critério de fechamento (não corrigido):** o ticket também nota que
nenhum dos dois conversores valida dtype/shape do retorno do quantizador além de um acordo tácito
de shape com `comfy_kitchen`. Não tratado — o critério de fechamento lista só as duas partes acima
como fechamento, e o texto do achado relacionado não está dentro de "## Critério de fechamento".
