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

Two things changed on 2026-08-22, both of them widening what this refuses rather than narrowing it:
occupancy is now read from **every** NVML device instead of index 0 (see `gpu_occupancy`), and an
entered guard registers itself so `_timing.compare()` can join it -- which is what lets `compare()`
take the lock unconditionally without a tool that already holds it deadlocking on its own file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_lock import GpuLock, GpuLockBusy  # noqa: E402,F401  (re-exported)

# A desktop compositor plus a spare CUDA context fits well under this. Any model worth contending
# with is far above it.
IDLE_CEILING_GIB = 2.0


def visible_nvml_indices(count: int) -> tuple[set[int], str]:
    """Which NVML indices this process could actually put work on, and how that was decided.

    `CUDA_VISIBLE_DEVICES` is what defines visibility, and it is the only honest way to narrow
    what this guard watches: a card excluded from the guard is a card the run cannot use either,
    so the exemption and the restriction are the same act. That matters because the alternative --
    a `--ignore-other-gpu` flag -- is a bypass, and a guard with a bypass is the opt-in problem
    with extra steps.

    Unparseable or unset means **all** devices are visible, which is the fail-closed direction:
    with no restriction in force the work can land anywhere, so anywhere busy is a problem. That
    includes the UUID form (`CUDA_VISIBLE_DEVICES=GPU-abc...`), which is not resolved here -- the
    guard watches everything rather than guess wrong about which card was meant.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return set(range(count)), "all devices (CUDA_VISIBLE_DEVICES unset)"
    try:
        # CUDA stops at the first invalid entry; matching that exactly is not worth it, so any
        # unparseable list falls back to watching everything.
        wanted = {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError:
        return set(range(count)), f"all devices (CUDA_VISIBLE_DEVICES={raw!r} not a plain index list)"
    inside = {i for i in wanted if 0 <= i < count}
    if not inside:
        return set(range(count)), f"all devices (CUDA_VISIBLE_DEVICES={raw!r} names no real device)"
    return inside, f"CUDA_VISIBLE_DEVICES={raw}"


def gpu_occupancy() -> list[tuple[int, str, float, bool]] | None:
    """`[(nvml_index, device_name, resident_gib, visible), ...]` for every device, or None.

    Read through NVML and **before** torch touches CUDA. Two reasons: creating the context costs
    ~270 MiB that would otherwise count against the caller's budget, and on this WDDM host
    `torch.cuda.mem_get_info()` reports 1292 MiB where NVML reports 548 for the same instant -- a
    744 MiB disagreement that once made a 1 GiB guard refuse on a completely idle card.

    **Every device, not index 0.** This used to read `nvmlDeviceGetHandleByIndex(0)` alone, so a
    job on `cuda:1` was invisible to the guard. That is not hypothetical here: CLAUDE.md records a
    whole measurement session running on the RTX 3080 Ti -- `F:\\cortiq-cmf` resolves
    `request_adapter(HighPerformance)` to it when `CMF_GPU_ADAPTER` is unset -- while the lock sat
    on an idle 3090, and the 3080 Ti was 9-26% busy in every `nvidia-smi` sample. A guard that
    watches one card of two guards the wrong half.

    The device *name* is returned, not just the index, because the index is what was ambiguous in
    that incident and the name is what settled it.

    `visible` is whether this process could put work on that device at all -- see
    `visible_nvml_indices`. Everything is reported; only the visible ones can refuse.
    """
    try:
        import pynvml
    except Exception:
        return None
    started = False
    try:
        pynvml.nvmlInit()
        started = True
        count = pynvml.nvmlDeviceGetCount()
        visible, _how = visible_nvml_indices(count)
        out = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            out.append((index, name, pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2 ** 30,
                        index in visible))
        # An empty list is not "idle", it is "NVML answered but found no devices", which on a
        # machine with two of them means the answer is wrong. Fail closed on it like any other
        # unreadable state.
        return out or None
    except Exception:
        return None
    finally:
        if started:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def gpu_occupancy_gib() -> float | None:
    """Resident GiB on the busiest **visible** device, or None. Kept for callers of the old
    single-device form; `gpu_occupancy()` is the one with the per-device detail."""
    devices = gpu_occupancy()
    if devices is None:
        return None
    usable = [gib for _, _, gib, visible in devices if visible]
    return max(usable) if usable else None


def occupancy_report(ceiling_gib: float = IDLE_CEILING_GIB) -> str:
    """One line per device, printed on **every** run, pass or fail.

    Printed even when nothing is busy, because a guard that only speaks when it refuses gives the
    reader no way to tell "checked, and the machine was quiet" from "did not check". The card a
    run landed on is precisely the fact that went missing in the incident CLAUDE.md records --
    nine runs on the 3080 Ti while the lock sat on an idle 3090 -- and it was the adapter *name*
    stamped into a cache file that eventually exposed it.
    """
    devices = gpu_occupancy()
    if devices is None:
        return "GPU occupancy: UNREADABLE (pynvml missing or failing)"
    lines = []
    for index, name, gib, visible in devices:
        state = "BUSY" if gib > ceiling_gib else "idle"
        where = "visible to this process" if visible else "NOT visible to this process"
        lines.append(f"  nvml{index} {name}: {gib:.2f} GiB {state} ({where})")
    return "GPU occupancy (ceiling %.2f GiB):\n%s" % (ceiling_gib, "\n".join(lines))


def refuse_if_busy(ceiling_gib: float = IDLE_CEILING_GIB) -> str | None:
    """Return a refusal message, or None when the machine is fit to measure.

    Fails **closed**: if occupancy cannot be read at all, that is a refusal, not a shrug. A
    benchmark that silently ran contended is worse than one that did not run.

    **Every device is read; the visible ones decide.** This used to read NVML index 0 alone, so a
    job on `cuda:1` was invisible to it -- and CLAUDE.md records a whole session whose work landed
    on the 3080 Ti while the lock sat on an idle 3090. Watching only device 0 guards the wrong
    half of that.

    Refusing on *every* busy device regardless of visibility was the first version of this fix and
    it was wrong in the other direction. MEASURED 2026-08-22 on this host: the 3080 Ti held 5.29
    GiB while the 3090 sat at 0.99 GiB, so that version refused every benchmark on the machine.
    A guard that blocks all work on a condition the operator cannot clear is a guard that gets
    commented out -- which is the failure CLAUDE.md names ("a check that blocks on a hypothesis
    teaches people to disable checks"). So a busy *invisible* device is reported loudly and does
    not refuse, and the way to make a card invisible is `CUDA_VISIBLE_DEVICES`, which excludes it
    from the run at the same time as from the guard. There is no flag that exempts a card the run
    can still use.
    """
    devices = gpu_occupancy()
    if devices is None:
        return ("could not read GPU occupancy through NVML, so there is no way to tell whether "
                "another job is on the card. Refusing rather than guessing. Install/repair "
                "pynvml, or set the ceiling explicitly if you know the card is idle.")
    busy = [(index, name, gib) for index, name, gib, visible in devices
            if visible and gib > ceiling_gib]
    if busy:
        which = "; ".join(f"nvml{index} {name}: {gib:.2f} GiB" for index, name, gib in busy)
        return (f"already resident above the {ceiling_gib:.2f} GiB ceiling on {which}. "
                "Timings would be contended in both directions; refusing to run.\n"
                + occupancy_report(ceiling_gib))
    return None


# The guard a tool has already entered, so `_timing.compare()` can join it instead of trying to
# take F:/GPU_BENCH.lock a second time from inside the same process -- which `GpuLock.__enter__`
# would correctly refuse with FileExistsError, turning "this tool already holds the lock" into
# "the card is busy". A list, not a single slot, because __exit__ must restore whatever was
# ambient before rather than clearing the field outright.
_ACTIVE: list["BenchGuard"] = []


def active_guard() -> "BenchGuard | None":
    """The innermost guard currently held in this process, or None."""
    return _ACTIVE[-1] if _ACTIVE else None


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

    @property
    def held(self) -> bool:
        """Public because `_timing.compare()` refuses to time anything under a guard that is not
        held, and asking through a private attribute would make that check look optional."""
        return self._held

    def __enter__(self) -> "BenchGuard":
        try:
            self._lock.__enter__()
            self._held = True
        except GpuLockBusy as exc:
            self.refused = str(exc)
            return self
        self.refused = refuse_if_busy(self.ceiling_gib)
        if self.refused is None:
            # Printed on the pass path too. A guard that is silent when it agrees leaves no
            # record of which card the run was on, and that is the fact the 3080 Ti incident
            # turned on: the lock named a card, the work used another, and nothing in either
            # tool's output said so.
            print(occupancy_report(self.ceiling_gib))
        # Registered only when the lock is genuinely ours AND occupancy passed. A refused guard
        # must not be joinable: `compare()` looking up an ambient guard that is refused would be
        # the opt-in problem back again, wearing a lock file.
        if self.refused is None:
            _ACTIVE.append(self)
        return self

    def __exit__(self, *exc) -> None:
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if self._held:
            self._lock.__exit__(*exc)
            self._held = False
