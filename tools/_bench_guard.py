"""One place that decides whether this machine is fit to be benchmarked right now.

Four GPU instruments in this directory each answered that question differently, and the audit of
2026-08-18 found the answers were not equivalent:

  * `m_crossover.py` read NVML but put the threshold test *outside* the try, so an exception after
    `nvmlInit()` leaked the handle and the script continued as if the card were empty
  * `w4a4_breakdown.py` had the same code with the test inside the try, and was the only one that
    also took the GPU lock
  * `check_w4a8.py` and `attn_dtype_ab.py` had no occupancy check at all

Both duplicated guards were fail-open: NVML missing meant "assume idle" and proceed. That is the
wrong default for a machine where a sibling project holds 23 of 24 GiB.

Two layers, because they fail differently:

  lock       a convention. Protects against the other side *when the other side takes it*.
  occupancy  a fact. Catches the case the lock cannot -- someone who forgot to take it. This has
             already happened here: 20.49 GiB resident with the lock file absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_lock import GpuLock, GpuLockBusy  # noqa: E402,F401  (re-exported)

# A desktop compositor plus a spare CUDA context fits well under this. Any model worth contending
# with is far above it.
IDLE_CEILING_GIB = 2.0


def gpu_occupancy_gib() -> float | None:
    """Resident device memory in GiB, or None if it could not be read.

    Read through NVML and **before** torch touches CUDA. Two reasons: creating the context costs
    ~270 MiB that would otherwise count against the caller's budget, and on this WDDM host
    `torch.cuda.mem_get_info()` reports 1292 MiB where NVML reports 548 for the same instant -- a
    744 MiB disagreement that once made a 1 GiB guard refuse on a completely idle card.
    """
    handle = None
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2 ** 30
    except Exception:
        return None
    finally:
        if handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def refuse_if_busy(ceiling_gib: float = IDLE_CEILING_GIB) -> str | None:
    """Return a refusal message, or None when the card is fit to measure.

    Fails **closed**: if occupancy cannot be read at all, that is a refusal, not a shrug. A
    benchmark that silently ran contended is worse than one that did not run.
    """
    resident = gpu_occupancy_gib()
    if resident is None:
        return ("could not read GPU occupancy through NVML, so there is no way to tell whether "
                "another job is on the card. Refusing rather than guessing. Install/repair "
                "pynvml, or set the ceiling explicitly if you know the card is idle.")
    if resident > ceiling_gib:
        return (f"{resident:.2f} GiB already resident on this GPU (ceiling {ceiling_gib:.2f}). "
                "Timings would be contended in both directions; refusing to run.")
    return None


class BenchGuard:
    """Take the lock, then check occupancy. Both must pass.

        with BenchGuard("m_crossover") as guard:
            if guard.refused:
                return 1
            ...
    """

    def __init__(self, owner: str, ceiling_gib: float = IDLE_CEILING_GIB):
        self.owner = owner
        self.ceiling_gib = ceiling_gib
        self.refused: str | None = None
        self._lock = GpuLock(owner)
        self._held = False

    def __enter__(self) -> "BenchGuard":
        try:
            self._lock.__enter__()
            self._held = True
        except GpuLockBusy as exc:
            self.refused = str(exc)
            return self
        self.refused = refuse_if_busy(self.ceiling_gib)
        return self

    def __exit__(self, *exc) -> None:
        if self._held:
            self._lock.__exit__(*exc)
            self._held = False
