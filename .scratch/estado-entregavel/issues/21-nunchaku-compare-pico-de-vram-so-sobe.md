# `nunchaku_compare.py`: pico de VRAM nunca desconta memória liberada, e `--attention sage` não verifica nada

Type: task
Status: open

## Question

Dois achados de `AUDITORIA_2026-08-18.md` (seção 5, baixa confiança) sobre `tools/nunchaku_compare.py`
que **não** entraram na leva de correções de 2026-08-18 (a única correção aplicada nesse arquivo
foi o rótulo `speedup (A/B)` invertido, confirmado corrigido nesta triagem). Reconferido por
leitura em 2026-08-21:

1. **Pico de VRAM só sobe** (`AUDITORIA:88`):
   ```python
   self.baseline = self.peak = self._used()      # linha 88-89, 115
   ...
   self.peak = max(self.peak, self._used())       # linha 137 — nunca desce
   ```
   Memória liberada durante o run vira desconto até `0.00 GiB` no relatório, e `report()` trata
   `0.0` como falsy, então a anotação some em vez de aparecer como zero explícito. O auditor não
   mediu se isso muda algum número publicado — a premissa contestada (se o CLIP ainda está
   residente no baseline) depende de como o ComfyUI libera memória (`del clip` + `empty_cache`
   sem `gc.collect()`, grafo com ciclos via `weakref.finalize`), que não foi observado.

2. **`--attention sage` sem verificação** (`AUDITORIA:203`): confirmado por leitura que os ramos
   `sparge` e `flash` checam disponibilidade e retornam 1 se ausente; o ramo `sage` só imprime
   texto, sem checar se o backend de fato está instalado ou se caiu para `attention_pytorch` com
   máscara (o que `attention.py:646,676-680` faz silenciosamente por chamada).

## Critério de fechamento

Fecha em duas partes independentes:

1. `DevicePeak` (ou equivalente) passa a rastrear um mínimo entre amostras, não só um máximo — e
   o relatório distingue "0 GiB liberado, medido" de "não medido". Não precisa provar quanto muda
   numericamente (isso é `[GPU]`); precisa parar de descartar silenciosamente memória liberada.
2. `--attention sage` ganha a mesma checagem de disponibilidade que `sparge` e `flash` já têm —
   retorna 1 se o backend não estiver de fato disponível, em vez de só imprimir texto.

Não fecha com decisão escrita: os dois são bugs de código, não hipóteses a validar.
