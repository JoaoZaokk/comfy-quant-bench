"""Guardas de memoria dos conversores: RAM fisica disponivel E commit livre do Windows.

Dois recursos diferentes, e a regra do projeto e conferir os dois antes de carga grande
(AGENTS.md: "conferir commit livre: guardas de RAM/disco nao bastam"):

  - RAM fisica disponivel (`psutil.virtual_memory().available`) diz se o trabalho cabe sem trocar
    pagina. Faltando, a maquina para de responder.
  - Commit livre (`CommitLimit - CommitTotal`, de `GetPerformanceInfo`) diz se o Windows ainda
    aceita reservar memoria privada. Faltando, a alocacao FALHA mesmo com RAM fisica sobrando --
    e o commit e consumido por reservas que nao aparecem como RAM usada (o leitor normal de um
    safetensors pode comprometer 2x o arquivo, ver `.agent-reference/comfy/17-memoria-commit.md`).

Historico. Este arquivo ja teve formulas por formato (`w4a8_bytes`, `int8_bytes`,
`convrot_w4a4_bytes`) para conversores de DUAS passadas, que quantizavam o modelo inteiro num dict
antes de escrever; a heuristica de streaming (`maior tensor x 3`) subestimava esses conversores
por fatores medidos de 4,5x a 8,1x (auditoria de 2026-08-18). Desde 2026-09-29 todos transmitem
(`_conversion.plan_lazy` + `_formats`): o pico e um tensor por vez, a heuristica de streaming
volta a ser a correta para todos, e as formulas sairam junto com o acumulo que elas mediam.
"""

from __future__ import annotations

import ctypes
import sys

GIB = 1024 ** 3


def check(available_bytes: int, accumulated_bytes: int, headroom_gib: float = 2.0,
          label: str = "conversion") -> str | None:
    """Recusa (texto) ou None quando a RAM fisica disponivel cabe `accumulated_bytes` + folga.

    `headroom_gib` covers the one source tensor held during quantization plus interpreter
    overhead; it is deliberately generous because the failure mode on the other side is swapping,
    not an exception.
    """
    need = accumulated_bytes + int(headroom_gib * GIB)
    if available_bytes >= need:
        return None
    return (f"Insufficient RAM for {label}: it holds "
            f"{accumulated_bytes / GIB:.2f} GiB at its peak, and with "
            f"{headroom_gib:.1f} GiB of headroom needs {need / GIB:.2f} GiB, but only "
            f"{available_bytes / GIB:.2f} GiB is available. Close memory-heavy processes "
            "(vmmemWSL on this host is the usual one) and retry -- do not change the pagefile.")


class _PerformanceInformation(ctypes.Structure):
    # PERFORMANCE_INFORMATION (psapi.h). Os campos de pagina sao SIZE_T e contam PAGINAS.
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_uint32),
        ("ProcessCount", ctypes.c_uint32),
        ("ThreadCount", ctypes.c_uint32),
    ]


def commit_free_bytes() -> int | None:
    """Commit livre do sistema em bytes (`CommitLimit - CommitTotal`), ou None fora do Windows.

    None tambem quando a chamada falha: quem chama trata como "nao medido" e diz isso, nunca como
    "sobra".
    """
    if sys.platform != "win32":
        return None
    info = _PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    try:
        ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb)
    except (AttributeError, OSError):
        return None
    if not ok:
        return None
    return (info.CommitLimit - info.CommitTotal) * info.PageSize


def commit_gib() -> float | None:
    """Commit livre em GiB, para relatorio. None fora do Windows ou se a chamada falhar."""
    free = commit_free_bytes()
    return None if free is None else free / GIB


def check_commit(accumulated_bytes: int, headroom_gib: float = 2.0, label: str = "conversion",
                 free_bytes: int | None = None) -> str | None:
    """Recusa (texto) ou None quando o commit livre cabe `accumulated_bytes` + folga.

    `free_bytes` existe para teste; em uso normal vem de `commit_free_bytes()`. Sem medida (fora
    do Windows) devolve None -- a guarda de RAM fisica continua valendo sozinha nesse caso.
    """
    free = commit_free_bytes() if free_bytes is None else free_bytes
    if free is None:
        return None
    need = accumulated_bytes + int(headroom_gib * GIB)
    if free >= need:
        return None
    return (f"Insufficient commit for {label}: it reserves up to {accumulated_bytes / GIB:.2f} GiB "
            f"of private memory, and with {headroom_gib:.1f} GiB of headroom needs "
            f"{need / GIB:.2f} GiB of free commit, but only {free / GIB:.2f} GiB is free "
            "(CommitLimit - CommitTotal). Physical RAM being free does not help here: the "
            "allocation fails on commit, not on RAM. Close processes holding commit (vmmemWSL, "
            "another ComfyUI) and retry -- do not change the pagefile.")
