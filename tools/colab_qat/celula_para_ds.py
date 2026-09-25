"""Celula (colab exec): para o gerador do dataset (SIGTERM, depois SIGKILL). O que ja' subiu fica no HF."""
import subprocess
import time
for sinal in ("-TERM", "-KILL"):
    subprocess.run(["pkill", sinal, "-f", "gera_dataset_klein.py"])
    time.sleep(8)
print("vivos:", subprocess.run(["pgrep", "-af", "gera_dataset"], capture_output=True, text=True).stdout.strip() or "nenhum")
