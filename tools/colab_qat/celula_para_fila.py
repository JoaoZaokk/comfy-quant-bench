"""Celula (colab exec): para a fila de QAT e o braco em curso (SIGTERM, depois SIGKILL), sem apagar nada."""
import subprocess
import time
for sinal in ("-TERM", "-KILL"):
    for padrao in ("fila_qat_ternario_klein.py", "qat_ternario_klein.py"):
        subprocess.run(["pkill", sinal, "-f", padrao])
    time.sleep(8)
vivos = subprocess.run(["pgrep", "-af", "qat_ternario_klein"], capture_output=True, text=True).stdout.strip()
print("vivos:", vivos or "nenhum")
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
