"""splice.py <arquivo> <inicio_literal> <fim_literal|EOF> <novo_arquivo>: troca o trecho [inicio, fim) pelo conteudo do novo."""
import sys
from pathlib import Path
alvo, ini, fim, novo = sys.argv[1:5]
s = Path(alvo).read_text(encoding="utf-8")
i = s.index(ini)
j = len(s) if fim == "EOF" else s.index(fim, i)
Path(alvo).write_text(s[:i] + Path(novo).read_text(encoding="utf-8") + s[j:], encoding="utf-8", newline="\n")
print("splice ok", alvo)
