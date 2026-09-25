"""Probe do supervisor para o GERADOR DO DATASET do klein (contrato do `probe_background_job.py` do frankestein).

Executado por `colab exec` a cada 4 min pelo `colab_job_supervisor.py`: alem de observar, e' a
execucao REAL de kernel que impede o Colab de podar a VM ociosa. Fail-closed: processo morto sem
`FIM HH:MM:SS` no log e' FAILED, nunca RUNNING.
"""
import json, re, subprocess
from pathlib import Path

STATUS_PREFIX = "COLAB_JOB_STATUS="
FALHA = re.compile(r"Traceback|CUDA (?:error|out of memory)|OutOfMemory|Killed|MemoryError|"
                   r"ABORTANDO|FALHOU|AssertionError|RuntimeError|RECUSADO|pip rc=[1-9]", re.I)
FIM = re.compile(r"FIM\s+\d{2}:\d{2}:\d{2}\b")

d = Path("/content/ds")
log = d / "ds.log"
tail = ""
if log.is_file():
    with log.open("rb") as h:
        h.seek(0, 2); n = h.tell(); h.seek(max(0, n - 32768))
        tail = h.read().decode("utf-8", "replace")
vivo = subprocess.run(["pgrep", "-f", "[g]era_dataset_klein.py"], stdout=subprocess.DEVNULL).returncode == 0
jr = d / "journal.jsonl"
ult = jr.read_text().strip().splitlines()[-1] if jr.is_file() and jr.stat().st_size else ""
p = {"alive": vivo, "ultimo_journal": ult, "log_tail": "\n".join(tail.splitlines()[-15:])}
try:
    # uso da placa na hora do probe: decide se cabe um segundo processo (pedido do dono, 25/09)
    p["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception as e:  # noqa: BLE001
    p["gpu"] = f"erro {type(e).__name__}"
ck = d / "ckpt" / "ultimo.pt"
p["ckpt_bytes"] = ck.stat().st_size if ck.is_file() else 0
if FIM.search(tail):
    p.update(status="DONE", reason="marcador_fim")
elif (m := FALHA.search(tail)):
    p.update(status="FAILED", reason=f"log:{m.group(0)}")
elif vivo:
    p.update(status="RUNNING", reason="processo_vivo")
else:
    p.update(status="FAILED", reason="processo_morto_sem_fim")
print(p["log_tail"])
print(STATUS_PREFIX + json.dumps(p, ensure_ascii=False, sort_keys=True))
