"""Fila de bracos do QAT numa VM do Colab: roda N corridas de `qat_ternario_klein.py` em sequencia num
processo so'. Lancada por `celula_lanca_fila.py` com `setsid nohup`.

POR QUE UMA FILA E NAO N LANCAMENTOS. O supervisor (`colab_job_supervisor.py` + `probe_qat.py`) para
no primeiro fim registrado; parado, o kernel fica ocioso e a VM e' podada em 22-26 min. Entao:
  * o `status.json` da fila (`/content/qat_fila/status.json`, o que o probe le) so' vira FIM no final de
    TUDO; cada braco tem o seu `status.json` na pasta dele;
  * a saida de cada braco vai inteira para o log DELE (`<dir>/qat.log`); para o log da fila so' passam as
    linhas com carimbo de hora, com o `FIM` do braco reescrito e as palavras de falha trocadas por `[x]`
    (isso so' importa para um probe ANTIGO, que ainda raspa o log);
  * o nome deste arquivo contem `qat_ternario_klein.py`: o `pgrep -f` de um probe antigo ve a fila viva.

SIGTERM na fila (celula_para_fila): nao lanca o proximo braco, repassa o sinal ao braco em curso (que grava
checkpoint ao fim do passo) e espera ele sair antes de encerrar.

Antes do primeiro braco espera qualquer outro `qat_ternario_klein.py` (o replay) terminar.
Depois de cada braco, apaga o checkpoint e o exportado LOCAIS so' se o HF ja' tiver os dois.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

Q = Path("/content/qat")
sys.path.insert(0, str(Q))
from probe_job import FALHA
from qat_klein.status import Status

FIM = re.compile(r"FIM(\s+\d{2}:\d{2}:\d{2})")


class Fila:
    def __init__(self, cfg: dict, pasta: Path):
        self.cfg, self.d = cfg, pasta
        self.d.mkdir(parents=True, exist_ok=True)
        self.log = (self.d / "qat.log").open("a", encoding="utf-8")
        self.status = Status(self.d, "fila_qat_ternario_klein")
        self.parar = False
        self.filho = None

    def diz(self, msg: str) -> None:
        self.log.write(f"[{time.strftime('%H:%M:%S')}] fila: {FALHA.sub('[x]', msg)}\n")
        self.log.flush()

    def _sinal(self, signum, _frame):
        self.parar = True
        self.diz(f"sinal {signum}: nao lanco o proximo braco; esperando o braco em curso gravar e sair")
        if self.filho is not None and self.filho.poll() is None:
            self.filho.send_signal(signal.SIGTERM)

    def outros_vivos(self) -> bool:
        r = subprocess.run(["pgrep", "-af", "qat_ternario_klein.py"], capture_output=True, text=True, check=False)
        return any("fila_qat_ternario_klein" not in ln and str(os.getpid()) not in ln.split()[:1]
                   for ln in r.stdout.splitlines() if ln.strip())

    def no_hf(self, repo: str, caminhos: list[str]) -> bool:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            return all(api.file_exists(repo, c) for c in caminhos)
        except Exception as e:  # noqa: BLE001
            self.diz(f"nao consegui conferir o HF ({type(e).__name__}); nada apagado")
            return False

    def braco(self, b: dict) -> int:
        d = Path(b["dir"])
        d.mkdir(parents=True, exist_ok=True)
        # professor compartilhado por symlink, salvo se o braco pedir o proprio ("professor_de": null).
        # O QAT confere o prompt de dentro de cada shard (nao o nome) e recusa vazamento treino x holdout.
        prof_de = b.get("professor_de", self.cfg["professor_de"])
        for sub in ("professor", "professor_holdout"):
            if prof_de and not (d / sub).exists():
                (d / sub).symlink_to(Path(prof_de) / sub, target_is_directory=True)
        for nome in ("journal.jsonl",):
            ln = self.d / nome
            if ln.is_symlink() or ln.exists():
                ln.unlink()
            ln.symlink_to(d / nome)
        cmd = [sys.executable, "-u", str(Q / "qat_ternario_klein.py"), "--raiz", self.cfg["raiz"],
               "--professor", self.cfg["professor"], "--prompts", str(Q / b.get("prompts", "prompts_treino.txt")),
               "--prompts-holdout", str(Q / "prompts_holdout.txt"), "--dir", str(d), *b["args"]]
        self.diz(f"== braco {b['nome']} ==  {' '.join(b['args'])}")
        self.status.atualiza(fase=f"braco {b['nome']}", braco=b["nome"], braco_dir=str(d))
        with (d / "qat.log").open("ab") as blog:
            self.filho = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(Q))
            for raw in self.filho.stdout:
                blog.write(raw)
                blog.flush()
                ln = raw.decode("utf-8", "replace").rstrip()
                if ln.startswith("["):
                    ln = FIM.sub(r"fim_do_braco\1", ln)
                    self.log.write(f"{b['nome']} {FALHA.sub('[x]', ln)}\n")
                    self.log.flush()
            rc = self.filho.wait()
        self.filho = None
        return rc

    def roda(self) -> None:
        signal.signal(signal.SIGTERM, self._sinal)
        bracos = self.cfg["bracos"]
        self.diz(f"{len(bracos)} bracos: {[b['nome'] for b in bracos]}")
        while self.outros_vivos() and not self.parar:
            self.diz("esperando a corrida anterior terminar")
            self.status.atualiza(fase="esperando outra corrida")
            time.sleep(300)
        self.diz("GPU livre, comecando")
        feitos = {}
        for b in bracos:
            if self.parar:
                break
            t0 = time.time()
            rc = self.braco(b)
            feitos[b["nome"]] = rc
            self.status.atualiza(bracos_rc=feitos)
            self.diz(f"braco {b['nome']} terminou com rc={rc} em {(time.time() - t0) / 60:.0f} min")
            repo = next((b["args"][i + 1] for i, x in enumerate(b["args"]) if x == "--push-hf"), None)
            d = Path(b["dir"])
            if rc == 0 and repo and self.no_hf(repo, ["ckpt/ultimo.pt", "final/aluno_ternario_diffusers.safetensors"]):
                for f in (d / "ckpt" / "ultimo.pt", d / "aluno_ternario_diffusers.safetensors"):
                    if f.is_file():
                        f.unlink()
                self.diz(f"braco {b['nome']}: HF tem ckpt e exportado; copias locais apagadas")
            st = os.statvfs("/content")
            self.diz(f"disco livre {st.f_bavail * st.f_frsize / 2**30:.0f} GiB")
        if self.parar:
            self.diz("fila PARADA por sinal")
            self.status.fim("PARADO", 143, f"parada por sinal; bracos feitos {feitos}")
            return
        self.diz("todos os bracos terminaram")
        self.log.write(f"FIM {time.strftime('%H:%M:%S')}\n")
        self.log.flush()
        self.status.fim("FIM", 0, json.dumps(feitos))


if __name__ == "__main__":
    fila = Fila(json.loads((Q / "fila.json").read_text()), Path("/content/qat_fila"))
    try:
        fila.roda()
    except BaseException as e:
        fila.status.fim("FALHOU", 1, f"{type(e).__name__}: {e}")
        raise
