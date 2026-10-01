"""`status.json` atômico: o contrato entre um job em background e o probe do supervisor.

Antes o probe raspava o log atrás de palavras de falha e de `FIM HH:MM:SS`, e achava o processo por nome
(`pgrep -f`). Agora o job escreve o próprio estado; o probe só lê (e cai no log para runs antigos).

    estado     RODANDO | FIM | PARADO | FALHOU | RECUSADO
    fase       professor | cruzado | referencia | treino | avaliando | exportando | ...
    pid, passo, heartbeat (epoch s), heartbeat_utc, rc, motivo
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .util import grava_json_atomico

ESTADOS = ("RODANDO", "FIM", "PARADO", "FALHOU", "RECUSADO")


class Status:
    def __init__(self, pasta: Path, job: str):
        self.arq = Path(pasta) / "status.json"
        self.dados = {"job": job, "pid": os.getpid(), "estado": "RODANDO", "fase": "inicio",
                      "inicio_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._grava()

    def _grava(self) -> None:
        self.dados["heartbeat"] = time.time()
        self.dados["heartbeat_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        grava_json_atomico(self.arq, self.dados)

    def atualiza(self, **campos) -> None:
        self.dados.update(campos)
        self._grava()

    def fim(self, estado: str, rc: int, motivo: str = "") -> None:
        if estado not in ESTADOS:
            raise ValueError(estado)
        self.atualiza(estado=estado, rc=rc, motivo=motivo[-2000:])
