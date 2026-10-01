# Os perfis moraram aqui ate 2026-09-29; agora moram em `_profiles.py`, junto com os dos
# conversores, e sao DERIVADOS um do outro em vez de copiados (revisao 2026-09-29, achado 6). Os
# nomes antigos continuam valendo: `PROFILE_PATTERNS` aqui e o vocabulario de MODULO (o que este
# arquivo engancha) e `PROFILE_FILE_PATTERNS` o de CHAVE do checkpoint (o que `quant_mixed` casa).
from _profiles import FILE_PATTERNS as PROFILE_FILE_PATTERNS  # noqa: E402,F401
from _profiles import MODULE_PATTERNS as PROFILE_PATTERNS  # noqa: E402
from _profiles import MODULE_TO_FILE, to_file_name  # noqa: E402,F401
