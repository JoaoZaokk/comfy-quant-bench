"""Checkpoint atômico com o ESTADO INTEIRO da corrida, e o pedido de parada (SIGTERM).

Antes o checkpoint (a cada ~25 min) levava só passo, mestre, otimizador e RNGs, enquanto `melhor.json`,
`ref_sens.json` e o journal eram gravados a cada avaliação: retomar do passo P trazia o `melhor` de passos
> P (holdout_rel, sens_max, ruins_seguidas do futuro), a permutação da época em curso se perdia e o
`args.json` era sobrescrito sem comparar. Agora `EstadoCorrida` vai DENTRO do mesmo `torch.save` atômico;
os JSONs viram relatório. As chaves antigas continuam no dicionário (leitores antigos seguem lendo), e um
checkpoint antigo, sem `estado`, retoma com o comportamento antigo e um aviso.
"""
from __future__ import annotations

import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .util import log

VERSAO = 1
SEMENTE_ORDEM = 20260922
SEMENTE_CRUZ = 20260923


@dataclass
class EstadoCorrida:
    passo: int = 0
    ordem: list = field(default_factory=list)       # o que resta da permutação da época em curso
    ordem_cruz: list = field(default_factory=list)
    rng_ordem: tuple | None = None                   # random.Random(20260922).getstate()
    rng_cruz: tuple | None = None
    melhor: dict = field(default_factory=lambda: {"holdout_rel": float("inf"), "passo": None,
                                                  "ruins_seguidas": 0, "melhor_rel_qualquer": float("inf")})
    impressao: dict = field(default_factory=dict)
    versao: int = VERSAO


class Geradores:
    """Os dois sorteios da corrida, cada um no seu `random.Random` (antes: o `random` global + um próprio).
    `Random(20260922).shuffle` dá a mesma sequência que `random.seed(20260922); random.shuffle`."""

    def __init__(self):
        self.ordem = random.Random(SEMENTE_ORDEM)
        self.cruz = random.Random(SEMENTE_CRUZ)


def salva_ckpt(pasta: Path, treinaveis, opt, estado: EstadoCorrida, extra: dict, ger: Geradores) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / "ultimo.pt"
    tmp = pasta / "ultimo.pt.partial"
    estado.rng_ordem = ger.ordem.getstate()
    estado.rng_cruz = ger.cruz.getstate()
    dados = {"passo": estado.passo, "extra": extra,
             "mestre": {k: v.detach().to("cpu") for k, v in treinaveis},
             "otim": opt.state_dict(),
             # chaves antigas: o `rng_py` é o gerador da ordem (antes o `random` global)
             "rng_py": estado.rng_ordem, "rng_torch": torch.get_rng_state(),
             "estado": asdict(estado)}
    t0 = time.perf_counter()
    torch.save(dados, tmp)
    os.replace(tmp, arq)
    log(f"  checkpoint passo {estado.passo}: {arq.stat().st_size / 2**30:.2f} GiB em "
        f"{time.perf_counter() - t0:.0f} s")
    return arq


def carrega_mestre(est: dict, treinaveis, dev) -> None:
    with torch.no_grad():
        for k, v in treinaveis:
            v.copy_(est["mestre"][k].to(dev, v.dtype))


def retoma(est: dict, treinaveis, opt, dev, ger: Geradores, melhor_legado: dict | None) -> EstadoCorrida:
    """Restaura mestre, otimizador, RNGs e estado. Checkpoint antigo (sem `estado`): passo e RNG da ordem do
    arquivo, permutação da época re-sorteada, `melhor` do melhor.json (o comportamento de antes)."""
    carrega_mestre(est, treinaveis, dev)
    opt.load_state_dict(est["otim"])
    torch.set_rng_state(est["rng_torch"])
    if "estado" in est:
        e = EstadoCorrida(**est["estado"])
        ger.ordem.setstate(e.rng_ordem)
        if e.rng_cruz is not None:
            ger.cruz.setstate(e.rng_cruz)
        return e
    log("AVISO: checkpoint no formato antigo (sem estado da corrida): a permutacao da epoca em curso e o "
        "sorteio do cruzado recomecam, e o `melhor` vem do melhor.json (pode estar a frente do checkpoint)")
    ger.ordem.setstate(est["rng_py"])
    e = EstadoCorrida(passo=est["passo"])
    if melhor_legado:
        e.melhor = melhor_legado
    return e


class Parada:
    """Pedido de parada limpa: SIGTERM (ou `--para-no-passo`) marca; o laço grava o checkpoint ao fim do
    passo em curso e sai. Sem isto o SIGTERM matava na hora e perdia até `--ckpt-min` de treino."""

    def __init__(self, no_passo: int = 0):
        self.pedida = False
        self.motivo = ""
        self.no_passo = no_passo
        self._antigo = None

    def _sinal(self, signum, _frame):
        self.pedida, self.motivo = True, f"sinal {signum}"
        log(f"PARADA pedida (sinal {signum}): checkpoint ao fim do passo em curso")

    def instala(self) -> None:
        try:
            self._antigo = signal.signal(signal.SIGTERM, self._sinal)
        except ValueError:  # fora da thread principal (testes)
            self._antigo = None

    def restaura(self) -> None:
        if self._antigo is not None:
            signal.signal(signal.SIGTERM, self._antigo)
            self._antigo = None

    def confere(self, passo: int) -> bool:
        if self.no_passo and passo >= self.no_passo and not self.pedida:
            self.pedida, self.motivo = True, f"--para-no-passo {self.no_passo}"
        return self.pedida
