"""Exportado do aluno em nomes diffusers.

    desempacotado   corpo = código x escala em bf16 (o formato de sempre: é o que `aplica_mapa_diffusers_bfl`
                    + render leem); resto = mestre bf16
    lowbit          corpo em `lowbit_affine` (códigos empacotados + scale/zero), resto bf16; cada camada é
                    conferida antes de gravar: a desquantização do kernel do loader tem de dar, bit a bit, o
                    valor desempacotado (ternário: sempre; int4 pode recusar por arredondamento do zero)

LIMITE (medido 2026-09-29, sem editar o loader): o comfy-lowbit-loader funde q/k/v dos nomes diffusers do
FLUX.2 agrupando só pela letra; num arquivo `lowbit_affine` em nomes diffusers os `weight_scale`/
`weight_zeros`/`comfy_quant` de to_q/k/v se sobrescrevem. Por isso o default continua `desempacotado`.
"""
from __future__ import annotations

import os
from pathlib import Path

import lowbit_canon
import torch

from .util import log


def caminho_lowbit(saida: Path) -> Path:
    return saida.with_name(saida.stem + "_lowbit" + saida.suffix)


def _grava(sd: dict, saida: Path, meta: dict) -> None:
    from safetensors.torch import save_file
    tmp = saida.with_suffix(saida.suffix + ".partial")
    save_file(sd, str(tmp), metadata={k: str(v) for k, v in meta.items()})
    os.replace(tmp, saida)
    log(f"exportado {saida}  {saida.stat().st_size:,} B")


def exporta(aluno, corpo_mods, saida: Path, meta: dict, formato: str = "desempacotado") -> list[Path]:
    """Grava o(s) exportado(s) e devolve os caminhos. `corpo_mods`: [(nome, LinearQuant)]."""
    mods = dict(corpo_mods)
    denso, desemp, lowbit, ordem = {}, {}, {}, []
    with torch.no_grad():
        for k, v in aluno.state_dict().items():
            if k.endswith("._esc"):
                continue
            ordem.append(k)
            mod = k[:-len(".weight")] if k.endswith(".weight") else None
            if mod in mods:
                m = mods[mod]
                cod, esc = m.codigo_e_escala()
                ref = m.quant.reconstroi(cod, esc, torch.float32).to("cpu", torch.bfloat16).contiguous()
                desemp[k] = ref
                if formato in ("lowbit", "ambos"):
                    pack = m.quant.empacota(cod.cpu(), esc.cpu())
                    n, zeros = lowbit_canon.confere_exato(ref, pack)
                    if n - zeros:
                        raise SystemExit(f"RECUSADO: lowbit de {mod} difere do desempacotado em {n - zeros} "
                                         f"elementos; use --exporta desempacotado")
                    if zeros:
                        log(f"  {mod}: {zeros} zeros com sinal diferente (-0 x +0; mesmo valor)")
                    lowbit[mod] = pack
            else:
                denso[k] = v.detach().to("cpu", torch.bfloat16).contiguous()
    feitos = []
    if formato in ("desempacotado", "ambos"):
        todos = {**denso, **desemp}
        _grava({k: todos[k] for k in ordem}, saida, meta)  # a ordem do state_dict, como antes
        feitos.append(saida)
    if formato in ("lowbit", "ambos"):
        q = next(iter(mods.values())).quant
        s = caminho_lowbit(saida)
        _grava(lowbit_canon.state_dict_lowbit(denso, lowbit), s,
               {**meta, "lowbit_formato": lowbit_canon.FORMAT, "lowbit_bits": q.bits, "lowbit_grupo": q.grupo})
        feitos.append(s)
    return feitos
