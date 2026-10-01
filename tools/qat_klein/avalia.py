"""Instrumentos de avaliação: holdout (MSE e relativo), guarda de condicionamento `sens`, erro cruzado e a
decisão de `melhor` / parada antecipada.

sens: pares de shards do holdout com a MESMA semente e prompts vizinhos; saída no mesmo x_t com a condição
do próprio prompt vs a do vizinho. d = ||s(x,c_i) - s(x,c_j)|| / ||s(x,c_i)||. A referência é o mesmo d
com o mestre AINDA igual ao BF16 e a quantização desligada -> sens = d_aluno / d_ref. Colapso (imagem igual
para todo prompt, visto na A100 #1 em p9191 com o MSE ainda caindo) = sens caindo.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .professor import forward_aluno, semente_do_nome
from .util import grava_json_atomico, log


class Instrumentos:
    def __init__(self, aluno, holdout, ho_cruz, sens_passos, dev, contexto):
        self.aluno, self.holdout, self.ho_cruz = aluno, holdout, ho_cruz
        self.sens_passos, self.dev, self.contexto = list(sens_passos), dev, contexto
        por_semente: dict[int, list[Path]] = {}
        for arq in holdout.arqs:
            por_semente.setdefault(semente_do_nome(arq.name), []).append(arq)
        self.pares = [(L[i], L[(i + 1) % len(L)]) for L in por_semente.values() if len(L) >= 2
                      for i in range(len(L))]

    def _fwd(self, comum, kw):
        return forward_aluno(self.aluno, comum, kw, self.dev, self.contexto).float()

    def mede_d(self) -> float:
        if not self.pares:
            raise SystemExit("RECUSADO: o holdout nao tem 2 shards da mesma semente; a guarda sens nao mede nada")
        ds = []
        with torch.no_grad():
            for ai, aj in self.pares:
                shi, shj = self.holdout._le(ai), self.holdout._le(aj)
                for pj in self.sens_passos:
                    kw = shi["passos"][pj]["kw"]
                    sii = self._fwd(shi["comum"], kw)
                    sij = self._fwd(shj["comum"], kw)
                    ds.append(float((sii - sij).norm() / sii.norm().clamp(min=1e-12)))
        return sum(ds) / len(ds)

    def erro_rel(self, conj) -> float:
        """Erro relativo medio do aluno contra o professor num conjunto (usado no cruzado do holdout)."""
        rs = []
        with torch.no_grad():
            for comum, kw, alvo in conj:
                s = self._fwd(comum, kw)
                al = alvo.to(self.dev, torch.float32)
                rs.append(float((s - al).norm() / al.norm().clamp(min=1e-12)))
        return sum(rs) / len(rs)

    def perda_holdout(self) -> tuple[float, float]:
        """(MSE medio, erro relativo medio ||s-a||/||a||). O MSE e' dominado pelos passos de sigma alto
        e ficou CEGO ao colapso duas vezes; o relativo pesa todo passo igual."""
        vs, rs = [], []
        with torch.no_grad():
            for comum, kw, alvo in self.holdout:
                s = self._fwd(comum, kw)
                al = alvo.to(self.dev, torch.float32)
                vs.append(float(F.mse_loss(s, al)))
                rs.append(float((s - al).norm() / al.norm().clamp(min=1e-12)))
        return sum(vs) / len(vs), sum(rs) / len(rs)

    def impressao_pares(self) -> str:
        txt = json.dumps([[a.name, b.name] for a, b in self.pares] + [self.sens_passos])
        return hashlib.sha256(txt.encode()).hexdigest()[:16]

    def referencia(self, arq: Path, chave) -> float:
        """d_ref, medido com o mestre ainda BF16 (chamar ANTES de carregar checkpoint). Reusa `ref_sens.json`
        só se foi medido nos MESMOS pares e passos; arquivo antigo (sem impressão) vale se contagem e passos
        batem."""
        imp = self.impressao_pares()
        if arq.is_file():
            r = json.loads(arq.read_text(encoding="utf-8"))
            if r.get("impressao") == imp:
                return r["d_ref"]
            if "impressao" not in r and r.get("pares") == len(self.pares) and r.get("passos") == self.sens_passos:
                log("AVISO: ref_sens.json antigo (sem impressao dos pares); reusado porque contagem e passos batem")
                return r["d_ref"]
            log("ref_sens.json foi medido em outros pares/passos; medindo de novo")
        with chave.desligada():
            d_ref = self.mede_d()
        grava_json_atomico(arq, {"d_ref": d_ref, "pares": len(self.pares), "passos": self.sens_passos,
                                 "impressao": imp})
        return d_ref


class Juiz:
    """Decide `melhor` e conta avaliações ruins para a parada. O estado (`melhor`) é do checkpoint; o
    `melhor.json` só espelha."""

    def __init__(self, cfg, melhor: dict, arq_espelho: Path):
        self.cfg, self.melhor, self.arq = cfg, melhor, arq_espelho

    def julga(self, passo: int, hr: float, sens: float) -> tuple[bool, float]:
        """(é o novo melhor?, piso). Limiar RELATIVO ao maior sens já visto: o ternário ingênuo mede sens
        0,156, então um piso absoluto de 0,8 pararia todo braço. O passo 0 (ternário ingênuo, imagem
        destruída) NÃO entra no sens_max: o d dele é reação errática de um modelo quebrado."""
        m = self.melhor
        if passo > 0:
            m["sens_max"] = max(m.get("sens_max", 0.0), sens)
        piso = max(self.cfg.sens_min, m.get("sens_max", 0.0) * (1 - self.cfg.sens_tol))
        novo = passo > 0 and hr < m["holdout_rel"] and sens >= piso
        if novo:
            m.update(holdout_rel=hr, passo=passo, sens=sens)
        ruim = not (hr < m["melhor_rel_qualquer"]) or sens < piso
        m["melhor_rel_qualquer"] = min(m["melhor_rel_qualquer"], hr)
        m["ruins_seguidas"] = m["ruins_seguidas"] + 1 if (ruim and passo > 0) else 0
        grava_json_atomico(self.arq, m)
        return novo, piso

    def deve_parar(self) -> bool:
        return bool(self.cfg.parada_paciencia) and self.melhor["ruins_seguidas"] >= self.cfg.parada_paciencia
