"""Probe do supervisor (contrato do probe_background_job.py do frankestein) para o QAT Qwen 2.1. Fail-closed:
processo morto sem `FIM HH:MM:SS` no log e FAILED."""
import json, re, subprocess
from pathlib import Path
STATUS_PREFIX = "COLAB_JOB_STATUS="
FALHA = re.compile(r"Traceback|CUDA (?:error|out of memory)|OutOfMemory|Killed|MemoryError|RuntimeError|recusado", re.I)
FIM = re.compile(r"FIM\s+\d{2}:\d{2}:\d{2}\b")
d = Path(json.loads(Path("/content/qatq/config.json").read_text())["dir"])
log = d / "qat.log"
tail = ""
if log.is_file():
    with log.open("rb") as h:
        h.seek(0, 2); n = h.tell(); h.seek(max(0, n - 32768))
        tail = h.read().decode("utf-8", "replace")
# so a execucao atual: um relancamento anexa ao mesmo log, e o traceback da anterior nao pode reprovar esta
if "carregando" in tail:
    tail = tail[tail.rfind("carregando") - 12:]
vivo = subprocess.run(["pgrep", "-f", "[q]at_qwen21_blocos.py"], stdout=subprocess.DEVNULL).returncode == 0
p = {"alive": vivo, "log_tail": "\n".join(tail.splitlines()[-12:]),
     "blocos_prontos": len(list((d / "blocos").glob("bloco_*.pt"))) if (d / "blocos").is_dir() else 0}
try:
    p["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
except Exception as e:  # noqa: BLE001
    p["gpu"] = f"erro {type(e).__name__}"
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
