"""Probe do supervisor (contrato do `probe_background_job.py` do frankestein), UM para todos os jobs.

Executado por `colab exec` a cada 4 min pelo `colab_job_supervisor.py`: além de observar, é a execução REAL
de kernel que impede o Colab de podar a VM ociosa. `colab exec` não passa argumento, então o job vem de:
  * a linha `JOB = ...` abaixo -- os `probe_qat.py`, `probe_ds.py` e `probe_qat_qwen21.py` são GERADOS deste
    arquivo trocando só essa linha (`colab_ops.py monta_probes`; o teste confere que estão em dia);
  * senão `/content/qat/probe.json` (gravado pelo `colab_ops.lanca(..., probe=...)`).

Fonte de verdade: o `status.json` que o job grava de forma atômica (estado, fase, passo, pid, heartbeat).
Sem ele (run lançado com código antigo) cai no contrato antigo: palavras de falha / `FIM HH:MM:SS` no fim
do log e `pgrep -f`. Fail-closed nos dois: processo morto sem fim registrado é FAILED, nunca RUNNING.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

JOB = None

STATUS_PREFIX = "COLAB_JOB_STATUS="
FALHA = re.compile(r"Traceback|CUDA (?:error|out of memory)|OutOfMemory|Killed|MemoryError|"
                   r"ABORTANDO|FALHOU|AssertionError|RuntimeError|RECUSADO|pip rc=[1-9]", re.IGNORECASE)
FIM = re.compile(r"FIM\s+\d{2}:\d{2}:\d{2}\b")
HEARTBEAT_VELHO_S = 30 * 60


def job_atual() -> dict:
    if JOB:
        return dict(JOB)
    pj = Path("/content/qat/probe.json")
    if pj.is_file():
        return json.loads(pj.read_text())
    return {"config": "/content/qat/config.json", "log": "qat.log", "processo": "qat_ternario_klein.py"}


def pasta_do_job(job: dict) -> Path:
    if job.get("dir"):
        return Path(job["dir"])
    return Path(json.loads(Path(job["config"]).read_text())["dir"])


def cauda(log: Path, n: int = 32768) -> str:
    if not log.is_file():
        return ""
    with log.open("rb") as h:
        h.seek(0, 2)
        tam = h.tell()
        h.seek(max(0, tam - n))
        return h.read().decode("utf-8", "replace")


def processo_vivo(padrao: str, pid=None) -> bool:
    if pid and Path(f"/proc/{pid}").exists():
        try:
            return padrao.split("/")[-1] in Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            return False
    alvo = f"[{padrao[0]}]{padrao[1:]}"  # a classe [x] impede o pgrep de achar a si mesmo
    return subprocess.run(["pgrep", "-f", alvo], stdout=subprocess.DEVNULL, check=False).returncode == 0


def decide(st: dict | None, tail: str, vivo: bool, agora: float | None = None) -> tuple[str, str]:
    """(status, reason) do contrato. `st` = conteúdo do status.json ou None (run antigo)."""
    agora = time.time() if agora is None else agora
    if st is not None:
        estado = st.get("estado")
        if estado == "FIM":
            return "DONE", "status_fim"
        if estado == "PARADO":
            return "DONE", f"status_parado:{st.get('motivo', '')}"
        if estado in ("FALHOU", "RECUSADO"):
            return "FAILED", f"status_{estado.lower()}:{str(st.get('motivo', ''))[:200]}"
        if vivo:
            velho = agora - float(st.get("heartbeat", agora)) > HEARTBEAT_VELHO_S
            return "RUNNING", "heartbeat_velho" if velho else f"status_{st.get('fase', 'rodando')}"
        return "FAILED", "processo_morto_sem_fim"
    if FIM.search(tail):
        return "DONE", "marcador_fim"
    m = FALHA.search(tail)
    if m:
        return "FAILED", f"log:{m.group(0)}"
    if vivo:
        return "RUNNING", "processo_vivo"
    return "FAILED", "processo_morto_sem_fim"


def mede(job: dict) -> dict:
    d = pasta_do_job(job)
    tail = cauda(d / job.get("log", "qat.log"))
    if job.get("corta_em") and job["corta_em"] in tail:
        # so a execucao atual: um relancamento anexa ao mesmo log
        tail = tail[tail.rfind(job["corta_em"]) - 12:]
    st = None
    sj = d / "status.json"
    if sj.is_file():
        try:
            st = json.loads(sj.read_text())
        except ValueError:
            st = None  # gravacao atomica: nao deveria acontecer; cai no contrato antigo
    vivo = processo_vivo(job.get("processo", "qat_ternario_klein.py"), (st or {}).get("pid"))
    p = {"alive": vivo, "log_tail": "\n".join(tail.splitlines()[-15:])}
    if st is not None:
        p["job_status"] = {k: st.get(k) for k in ("estado", "fase", "passo", "heartbeat_utc", "rc")}
    jr = d / "journal.jsonl"
    p["ultimo_journal"] = jr.read_text().strip().splitlines()[-1] if jr.is_file() and jr.stat().st_size else ""
    if job.get("conta_glob"):
        p[job.get("conta_nome", "contagem")] = len(list(d.glob(job["conta_glob"])))
    try:
        # uso da placa na hora do probe: decide se cabe um segundo processo (pedido do dono, 25/09)
        p["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                                   "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False).stdout.strip()
    except Exception as e:  # noqa: BLE001
        p["gpu"] = f"erro {type(e).__name__}"
    ck = d / "ckpt" / "ultimo.pt"
    p["ckpt_bytes"] = ck.stat().st_size if ck.is_file() else 0
    p["status"], p["reason"] = decide(st, tail, vivo)
    return p


def main() -> None:
    p = mede(job_atual())
    print(p["log_tail"])
    print(STATUS_PREFIX + json.dumps(p, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__" and os.environ.get("PROBE_NAO_RODA") != "1":
    main()
