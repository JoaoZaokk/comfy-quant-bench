"""Fila de bracos do QAT numa VM do Colab: roda N corridas de `qat_ternario_klein.py` em sequencia num
processo so'. Lancada por `celula_lanca_fila.py` com `setsid nohup`.

POR QUE UMA FILA E NAO N LANCAMENTOS. O supervisor (`colab_job_supervisor.py` + `probe_qat.py`) para
no primeiro `FIM HH:MM:SS` do log; parado, o kernel fica ocioso e a VM e' podada em 22-26 min -- e esta
VM e' a unica com o token de escrita do dono. Entao:
  * o log da fila (`/content/qat_fila/qat.log`, o que o probe le) so' recebe `FIM` no final de TUDO;
  * a saida de cada braco vai inteira para o log DELE (`<dir>/qat.log`) e para o da fila so' passam as
    linhas com carimbo de hora, com o `FIM` do braco reescrito e qualquer palavra que o probe le como
    falha trocada por `[x]` -- um braco que morre nao derruba a VM, a fila segue para o proximo;
  * o nome deste arquivo contem `qat_ternario_klein.py`, entao o `pgrep -f` do probe ve a fila viva
    tambem entre dois bracos.

Antes do primeiro braco espera qualquer outro `qat_ternario_klein.py` (o replay) terminar.
Depois de cada braco, apaga o checkpoint e o exportado LOCAIS so' se o HF ja' tiver os dois.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

Q = Path("/content/qat")
CFG = json.loads((Q / "fila.json").read_text())
D = Path("/content/qat_fila")
D.mkdir(parents=True, exist_ok=True)
LOG = (D / "qat.log").open("a", encoding="utf-8")
FALHA = re.compile(r"Traceback|CUDA (?:error|out of memory)|OutOfMemory|Killed|MemoryError|"
                   r"ABORTANDO|FALHOU|AssertionError|RuntimeError|RECUSADO|pip rc=[1-9]", re.I)
FIM = re.compile(r"FIM(\s+\d{2}:\d{2}:\d{2})")


def diz(msg: str) -> None:
    LOG.write(f"[{time.strftime('%H:%M:%S')}] fila: {FALHA.sub('[x]', msg)}\n")
    LOG.flush()


def outros_vivos() -> bool:
    r = subprocess.run(["pgrep", "-af", "qat_ternario_klein.py"], capture_output=True, text=True)
    return any("fila_qat_ternario_klein" not in ln and str(os.getpid()) not in ln.split()[:1]
               for ln in r.stdout.splitlines() if ln.strip())


def no_hf(repo: str, caminhos: list[str]) -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        return all(api.file_exists(repo, c) for c in caminhos)
    except Exception as e:  # noqa: BLE001
        diz(f"nao consegui conferir o HF ({type(e).__name__}); nada apagado")
        return False


diz(f"{len(CFG['bracos'])} bracos: {[b['nome'] for b in CFG['bracos']]}")
while outros_vivos():
    diz("esperando a corrida anterior terminar")
    time.sleep(300)
diz("GPU livre, comecando")

for b in CFG["bracos"]:
    d = Path(b["dir"])
    d.mkdir(parents=True, exist_ok=True)
    # professor compartilhado por symlink, salvo se o braco pedir o proprio ("professor_de": null):
    # entao a corrida grava o professor dela (outro conjunto de prompts ou de sementes).
    prof_de = b.get("professor_de", CFG["professor_de"])
    for sub in ("professor", "professor_holdout"):
        if prof_de and not (d / sub).exists():
            (d / sub).symlink_to(Path(prof_de) / sub, target_is_directory=True)
    jl = D / "journal.jsonl"
    if jl.is_symlink() or jl.exists():
        jl.unlink()
    jl.symlink_to(d / "journal.jsonl")
    cmd = [sys.executable, "-u", str(Q / "qat_ternario_klein.py"), "--raiz", CFG["raiz"],
           "--professor", CFG["professor"], "--prompts", str(Q / b.get("prompts", "prompts_treino.txt")),
           "--prompts-holdout", str(Q / "prompts_holdout.txt"), "--dir", str(d), *b["args"]]
    diz(f"== braco {b['nome']} ==  {' '.join(b['args'])}")
    t0 = time.time()
    with (d / "qat.log").open("ab") as blog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(Q))
        for raw in p.stdout:
            blog.write(raw)
            blog.flush()
            ln = raw.decode("utf-8", "replace").rstrip()
            if ln.startswith("["):
                ln = FIM.sub(r"fim_do_braco\1", ln)
                LOG.write(f"{b['nome']} {FALHA.sub('[x]', ln)}\n")
                LOG.flush()
        rc = p.wait()
    diz(f"braco {b['nome']} terminou com rc={rc} em {(time.time() - t0) / 60:.0f} min")
    repo = next((b["args"][i + 1] for i, x in enumerate(b["args"]) if x == "--push-hf"), None)
    if rc == 0 and repo and no_hf(repo, ["ckpt/ultimo.pt", "final/aluno_ternario_diffusers.safetensors"]):
        for f in (d / "ckpt" / "ultimo.pt", d / "aluno_ternario_diffusers.safetensors"):
            if f.is_file():
                f.unlink()
        diz(f"braco {b['nome']}: HF tem ckpt e exportado; copias locais apagadas")
    st = os.statvfs("/content")
    diz(f"disco livre {st.f_bavail * st.f_frsize / 2**30:.0f} GiB")

diz("todos os bracos terminaram")
LOG.write(f"FIM {time.strftime('%H:%M:%S')}\n")
LOG.flush()
