"""Celula (colab exec): para a fila e o braco em curso, e mostra o fim do log de cada braco que ja' rodou.
Usa pgrep + os.kill (um `pkill -f` dentro de `bash -lc` mata o proprio shell -- registrado)."""
import os
import signal
import subprocess
import time
from pathlib import Path

r = subprocess.run(["pgrep", "-f", "qat_ternario_klein.py"], capture_output=True, text=True)
pids = [int(x) for x in r.stdout.split() if int(x) != os.getpid()]
for p in pids:
    try:
        os.kill(p, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(5)
vivos = subprocess.run(["pgrep", "-f", "qat_ternario_klein.py"], capture_output=True, text=True).stdout.split()
print(f"parados {pids}; ainda vivos {vivos}")
for d in sorted(Path("/content").glob("qat_b*")):
    log = d / "qat.log"
    if log.is_file():
        linhas = log.read_text(errors="replace").splitlines()
        print(f"===== {log} ({len(linhas)} linhas) =====")
        print("\n".join(linhas[-45:]))
