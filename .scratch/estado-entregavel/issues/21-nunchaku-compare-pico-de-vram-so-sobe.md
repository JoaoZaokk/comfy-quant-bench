# `nunchaku_compare.py`: pico de VRAM nunca desconta memória liberada, e `--attention sage` não verifica nada

Type: task
Status: resolved

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

## Resolução

Ambos consertados em `tools/nunchaku_compare.py`. LIDO e EXECUTADO como indicado abaixo — nenhum
número de VRAM real foi medido nesta rodada (GPU em uso pelo orquestrador); a prova é
`py_compile` mais leitura lado a lado, não uma corrida real.

1. **Pico só sobe** -- `DevicePeak` ganhou `self.min_after_peak`, atualizado em `_run()`: quando a
   amostra iguala ou supera o pico atual, o pico *e* o mínimo são resetados para esse valor;
   caso contrário só o mínimo desce. `summary()` agora expõe `device_released_gib =
   peak - min_after_peak`, sempre um número quando NVML está disponível (nunca `None` só porque
   deu zero) e `None` só quando NVML não está disponível. Os dois pontos de impressão --
   o resumo de uma corrida em `main()` e a comparação de duas em `report()` -- foram trocados de
   checagem "truthy" (`if released:`) para `is not None`, e ambos agora imprimem "not measured"
   explicitamente no caso `None` em vez de omitir a linha. EXECUTADO: validei o algoritmo de
   estado (peak/min_after_peak) isolado, fora do arquivo, com uma sequência sintética de amostras
   (`10,15,20,12,8,18,5` -> peak=20, min_after_peak=5, released=15), sem GPU e sem tocar o
   arquivo real -- confirma a lógica, não confirma o número que o NVML real produziria.
   A premissa contestada no achado original (se o CLIP segue residente no baseline do ComfyUI)
   continua NÃO observada -- não afirmo nada sobre ela.
2. **`--attention sage` sem checagem** -- o ramo `sage` agora importa
   `comfy.ldm.modules.attention` e confere `SAGE_ATTENTION_IS_AVAILABLE` antes de imprimir
   "attention: SageAttention", devolvendo 1 com mensagem se ausente -- mesmo idioma que os ramos
   `sparge` (`SAGE_ATTENTION_IS_AVAILABLE`) e `flash` (`FLASH_ATTENTION_IS_AVAILABLE`) já usavam
   logo acima/abaixo dele no arquivo.

Comando que provou (LIDO/EXECUTADO conforme acima, sem GPU):

```
F:\COMFY_PORTABLE\python_embeded\python.exe -s -m py_compile F:\COMFY_PORTABLE\tools\nunchaku_compare.py
```
-> `COMPILE_OK`, sem alterar nenhum outro arquivo.

**O que ficou sem cobertura**: nenhum número de VRAM real (NVML) foi coletado -- a GPU estava em
uso por outro ticket. O comportamento do `sage` branch (retorno 1 quando SageAttention de fato
está ausente, e execução normal quando está presente) não foi exercitado ao vivo, só verificado
por leitura lado a lado com os ramos `sparge`/`flash` que já faziam o mesmo. Não existe suíte de
teste dedicada a `nunchaku_compare.py` neste repo (só `test_comfy_run_workflow.py`, que cobre
outro arquivo) -- nenhuma foi rodada nem criada aqui, por estar fora dos arquivos permitidos deste
ticket.
