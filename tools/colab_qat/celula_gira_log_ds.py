"""Celula (colab exec): renomeia /content/ds/ds.log (que termina em FIM) antes de relancar o gerador, senao o probe
le o FIM velho e declara DONE na hora."""
import time
from pathlib import Path

log = Path("/content/ds/ds.log")
if log.is_file():
    log.rename(log.with_name(f"ds.{time.strftime('%H%M%S')}.log"))
    print("log girado")
