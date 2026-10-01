"""Perfis de arquitetura: QUAIS camadas de um checkpoint cada ferramenta quantiza. Um lugar so.

Ate 2026-09-29 esta tabela existia em QUATRO copias que ja divergiam (revisao de 2026-09-29,
achado 6): `quant_w4a4.PROFILE_PATTERNS` (com `EXCLUSIONS`), `quant_w4a8.PROFILE_PATTERNS`,
`calibrate_activations.PROFILE_PATTERNS` (nomes de MODULO) e `PROFILE_FILE_PATTERNS`. As regex de
LTX, Qwen-Image-2.1 e HunyuanVideo eram copiadas literalmente de um arquivo para o outro ("the same
regex quant_w4a8.py uses"); `detect_profile` tambem, e a copia do w4a4 nao tinha o ramo estrutural
de LTX -- `quant_w4a4 --profile auto` numa LTX levantava ValueError enquanto o w4a8 detectava; e a
segunda rede (`EXCLUSIONS`) so existia no w4a4.

Tres vocabularios, porque sao tres perguntas diferentes, agora DERIVADOS um do outro em vez de
copiados:

    MODULE_PATTERNS   nome de MODULO no modelo carregado -- o que `calibrate_activations` engancha.
    FILE_PATTERNS     nome da CHAVE no checkpoint, sem o `.weight` -- o que `quant_mixed` casa. Igual
                      a MODULE_PATTERNS exceto onde o ComfyUI renomeia ao carregar (MODULE_TO_FILE).
    WEIGHT_PATTERNS   FILE_PATTERNS + `\\.weight$`, mais os text encoders (que so existem do lado do
                      conversor) -- o que w4a4/w4a8/int8/awq/gguf casam, tensor a tensor.

`PROFILE_PATTERNS` (o nome antigo, importado por weight_balance, quant_gguf...) continua sendo o
subconjunto de WEIGHT_PATTERNS que os conversores de formato unico oferecem em `--profile`, com as
MESMAS seis chaves de antes. Conferido na migracao: as regex derivadas sao, caractere a caractere,
as que as copias antigas carregavam, e a selecao nos headers reais do disco nao mudou (ver
`.scratch/revisao_2026-09-29/resultado_conversao.md`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

HIGH_PRECISION_DTYPES = frozenset({"BF16", "F16", "F32"})

# ---------------------------------------------------------------- nomes de MODULO (calibracao)

# Kept identical to the converter's, and imported by it, so a layer can never be calibrated under
# one definition and quantized under another.
