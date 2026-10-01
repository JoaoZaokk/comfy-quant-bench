"""Envio do checkpoint e dos exportados para o HF, numa thread, sem parar o treino.

Por que: no Colab o `/content` morre com a VM, `colab drivemount` trava sem TTY e `colab download` até a
máquina local mede ~5 MB/s. O token NÃO é deste código: `HfApi()` usa o do ambiente, e a permissão de
escrita é provada ANTES de treinar (sobe 1 byte e apaga; `whoami` responde igual para token de leitura).

Cada envio é um commit de ~14 GB de LFS; `super_squash_history` deixa só a revisão atual (e NÃO devolve a
cota na hora, medido 2026-09-23 -- por isso os repos passaram a públicos, decisão do dono).

Duas proteções novas contra perder o checkpoint remoto (o squash torna a sobrescrita irreversível):
  * `baixa_se_faltar` só começa do zero se o repo ou o arquivo NÃO EXISTEM (RepositoryNotFound /
    EntryNotFound); qualquer outro erro (rede, 401, 5xx) aborta a corrida em vez de recomeçar calado;
  * `envia` grava `ckpt/passo.json` NO MESMO COMMIT do checkpoint e recusa subir um passo menor que o
    remoto. Repo antigo sem `passo.json`: vale o passo do checkpoint baixado nesta execução, se houve.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .util import log

ARQ_CKPT = "ckpt/ultimo.pt"
ARQ_PASSO = "ckpt/passo.json"


def _erros_de_ausencia():
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    return (EntryNotFoundError, RepositoryNotFoundError)


class EmpurraHF:
    def __init__(self, repo: str, publico: bool = False, api=None):
        import queue
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api, self.repo, self.t = api, repo, None
        self.passo_remoto: int | None = None
        # Fila de envios AVULSOS (exportado `melhor`, exportado final, json): um worker, em ordem, para
        # dois commits nunca correrem juntos no mesmo repo. O checkpoint segue pelo caminho dele.
        self.fila: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        self.api.create_repo(repo, private=not publico, exist_ok=True)
        self.api.upload_file(path_or_fileobj=b"x", path_in_repo="_prova_escrita", repo_id=repo)
        self.api.delete_file("_prova_escrita", repo_id=repo)
        log(f"HF: escrita PROVADA em {repo} (subiu e apagou 1 byte)")

    # ------------------------------------------------------------------ leitura
    def _baixa(self, nome: str, local_dir: str | None = None) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, nome, local_dir=local_dir, force_download=local_dir is None)

    def le_passo_remoto(self) -> int | None:
        """Passo do `ckpt/passo.json` remoto; None se o repo ainda não tem (formato antigo ou repo novo)."""
        try:
            return int(json.loads(Path(self._baixa(ARQ_PASSO)).read_text(encoding="utf-8"))["passo"])
        except _erros_de_ausencia():
            return None

    def baixa_se_faltar(self, arq: Path) -> None:
        if arq.is_file():
            return
        try:
            p = self._baixa(ARQ_CKPT, local_dir=str(arq.parent.parent))
        except _erros_de_ausencia() as e:
            log(f"HF: sem checkpoint remoto ({type(e).__name__}); comecando do zero")
            return
        except Exception as e:  # noqa: BLE001 -- NAO comecar do zero: o proximo envio apagaria o remoto
            raise SystemExit(f"FALHOU: nao consegui consultar/baixar o checkpoint remoto de {self.repo} "
                             f"({type(e).__name__}: {str(e)[:500]}); recusando comecar do zero") from None
        import torch
        self.passo_remoto = int(torch.load(p, map_location="cpu", weights_only=False, mmap=True)["passo"])
        log(f"HF: checkpoint baixado para retomar ({Path(p).stat().st_size / 2**30:.2f} GiB, "
            f"passo {self.passo_remoto})")

    # ------------------------------------------------------------------ envio do checkpoint
    def _pode_enviar(self, passo: int) -> bool:
        try:
            remoto = self.le_passo_remoto()
        except Exception as e:  # noqa: BLE001
            log(f"HF: nao li {ARQ_PASSO} ({type(e).__name__}); este checkpoint fica so local")
            return False
        if remoto is None:
            remoto = self.passo_remoto
        if remoto is not None and passo < remoto:
            log(f"HF: RECUSADO envio do passo {passo}: o remoto ja tem o passo {remoto}")
            return False
        return True

    def _envia_agora(self, arq: Path, passo: int) -> None:
        from huggingface_hub import CommitOperationAdd
        t0 = time.perf_counter()
        try:
            if not self._pode_enviar(passo):
                return
            marca = json.dumps({"passo": passo, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            self.api.create_commit(repo_id=self.repo, commit_message=f"checkpoint passo {passo}", operations=[
                CommitOperationAdd(path_in_repo=ARQ_CKPT, path_or_fileobj=str(arq)),
                CommitOperationAdd(path_in_repo=ARQ_PASSO, path_or_fileobj=marca.encode())])
            self.passo_remoto = passo
            self.api.super_squash_history(repo_id=self.repo)
            dt = time.perf_counter() - t0
            tam = arq.stat().st_size
            log(f"HF: enviado passo {passo}, {tam / 2**30:.2f} GiB em {dt:.0f} s "
                f"({tam / 2**20 / max(dt, 1e-9):.0f} MB/s) [MEDIDO]")
        except Exception as e:  # noqa: BLE001 -- envio falho nao pode derrubar o treino
            log(f"HF: envio NAO completou ({type(e).__name__}: {str(e)[:2000]}); checkpoint local intacto")

    def envia(self, arq: Path, passo: int) -> None:
        if (self.t is not None and self.t.is_alive()) or self.fila.unfinished_tasks:
            log("HF: envio anterior ainda em curso; este checkpoint fica so local")
            return
        self.t = threading.Thread(target=self._envia_agora, args=(arq, passo), daemon=False)
        self.t.start()

    # ------------------------------------------------------------------ avulsos
    def _worker(self) -> None:
        while True:
            fonte, destino, apagar = self.fila.get()
            t0 = time.perf_counter()
            try:
                if self.t is not None and self.t.is_alive():
                    self.t.join()  # nao concorrer com o envio do checkpoint no mesmo repo
                self.api.upload_file(path_or_fileobj=fonte if isinstance(fonte, bytes) else str(fonte),
                                     path_in_repo=destino, repo_id=self.repo)
                tam = len(fonte) if isinstance(fonte, bytes) else Path(fonte).stat().st_size
                log(f"HF: {destino} enviado ({tam / 2**30:.2f} GiB, {time.perf_counter() - t0:.0f} s)")
                if apagar and not isinstance(fonte, bytes):
                    Path(fonte).unlink()
            except Exception as e:  # noqa: BLE001 -- envio falho nao derruba o treino
                log(f"HF: {destino} NAO enviado ({type(e).__name__}: {str(e)[:2000]})")
            finally:
                self.fila.task_done()

    def avulso(self, fonte, destino: str, apagar: bool = False) -> None:
        self.fila.put((fonte, destino, apagar))

    def espera(self) -> None:
        if self.t is not None:
            self.t.join()
        self.fila.join()
