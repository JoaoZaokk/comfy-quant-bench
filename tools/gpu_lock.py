"""One lock file so two agents cannot benchmark the same GPU at the same time.

Both halves of this work run on one RTX 3090: this project's quantization sweeps and a sibling
project's vLLM batteries. Each had a "wait until the card is free, then start" watcher, which is
precisely the pattern that makes two jobs launch in the same second and hand each other contended
numbers -- and, worse, makes the second one OOM when the first is already holding 23 of 24 GiB.

The path is fixed at F:/GPU_BENCH.lock and is deliberately not configurable: a lock only works if
everyone names the same file. `open(path, "x")` is the atomic part -- it either creates the file
or raises, with no window between checking and creating.

    from gpu_lock import GpuLock
    with GpuLock("bench:m_crossover"):
        ...

Stale locks are not cleaned up automatically. A crashed run leaving a lock behind is a nuisance;
a lock that times itself out and lets a second job in while the first is still running is a
corrupted measurement, and the nuisance is the cheaper failure.

## The file format, and why this side changed rather than the other

MEASURED 2026-08-21, and it is the reason this module was rewritten. This file used to write
JSON. `gpu_lock.ps1:44-50` parses `key=value` lines with `Select-String "^$k=(.*)$"`. Against a
JSON lock every regex missed, so `Get-GpuLockState` returned an empty Owner, an empty Pid, and
`StaleFor = [int64]::MaxValue`; `Take-GpuLock:63-64` then evaluated `'' -and (...)` -> false and
`false -or (MaxValue -lt 55)` -> false, skipped the held branch, and **reclaimed a live lock**:

    lock is stale (pid  dead, hb 9223372036854775807s ago) -- reclaiming from
    lock TAKEN by other-session:steal-test

The theft was one-way, which is why it survived unnoticed: `open("x")` below meant the Python
side always refused a PowerShell-held lock correctly. One direction worked perfectly, and the
message the operator read named a blank owner -- indistinguishable from a leftover file.

**The fix is on this side, deliberately.** Switching the PowerShell side to JSON would have
reproduced the same bug pointing at the sibling: a session still running the old `.ps1` writes
`key=value`, and would in turn be robbed by anything new that only spoke JSON. `key=value` is the
dialect that already has two readers -- `gpu_lock.ps1`, and this module's own `__exit__`, which
already accepted `pid=<n>` alongside the JSON form. So this module now writes what everybody
already reads, and no sibling has to update anything for the theft to stop.

`_parse` still understands the old JSON, so a lock written by an out-of-date copy of this file is
recognised as held rather than run over.

## The heartbeat

`gpu_lock.ps1` judges a lock by `pid alive OR hb fresher than 55 s`. A long-running Python
conversion satisfies the first on its own -- `os.getpid()` here is the process actually doing the
work, not a tool call that dies on return (which is exactly the 2026-08-19 failure recorded in
`gpu_lock.ps1`'s header). The heartbeat thread below keeps the second true as well, so the entry
is honest under either reading, and it stops the moment the lock stops being ours.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

LOCK_PATH = Path("F:/GPU_BENCH.lock")
BEAT_INTERVAL_SEC = 15
STALE_LIMIT_SEC = 55  # must match gpu_lock.ps1's Take-GpuLock default


class GpuLockBusy(RuntimeError):
    pass


def _parse(text: str) -> dict[str, str]:
    """Both dialects. `key=value` is canonical; JSON is what this module used to write."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            raw = json.loads(stripped)
        except Exception:
            return {}
        return {
            "dono": str(raw.get("owner", "")),
            "pid": str(raw.get("pid", "")),
            "desde": str(raw.get("started", "")),
            "owner_kind": "python-legacy-json",
        }
    state: dict[str, str] = {}
    for line in stripped.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            state[key.strip()] = value.strip()
    return state


def _render(owner: str, pid: int, since: str) -> str:
    """Byte-for-byte the shape `gpu_lock_beat.ps1` writes, including key order."""
    return "\n".join([
        f"dono={owner}",
        f"pid={pid}",
        f"desde={since}",
        f"hb={int(time.time())}",
        "owner_kind=python",
    ]) + "\n"


def read_state(path: Path = LOCK_PATH) -> dict[str, str] | None:
    """Parsed lock, or None if there is no lock. Public so other tools can look without taking."""
    try:
        return _parse(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except OSError:
        return {}


def _pid_alive(pid: int) -> bool:
    """Windows liveness, and the POSIX intuition is wrong here in a way that matters.

    MEASURED 2026-08-22 on this host, CPython 3.13.12:

        pid   30692 (dead)   os.kill(pid, 0) -> OSError winerror=87 errno=22
        pid   25476 (alive)  os.kill(pid, 0) -> no raise
        pid  999999 (never)  os.kill(pid, 0) -> OSError winerror=87 errno=22

    **A dead pid does not raise `ProcessLookupError` on Windows.** It raises a bare `OSError` with
    `winerror = 87` (ERROR_INVALID_PARAMETER). The first version of this function caught
    `ProcessLookupError` for dead, `PermissionError` for alive, and fell through to
    `except OSError: return True` for everything else -- so **every dead pid read as alive.**

    The direction was safe (a stale lock stays stuck rather than being stolen, and this module
    never auto-reclaims anyway) but the *message* lied: `describe()` printed `alive=True` over a
    process that had been dead for minutes. And it put the two halves of this lock back into
    disagreement -- PowerShell's `Get-Process -Id` correctly said dead for the same pid, which is
    exactly the class of split that ticket 01 was about.

    `psutil.pid_exists` is what `Get-Process` effectively does and gets both cases right; it is
    already in the embedded interpreter and already imported by the benchmark tools that import
    this one. The `os.kill` path stays as the fallback for an interpreter without it, now reading
    `winerror` instead of assuming.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        pass
    else:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                     # alive, owned by someone else
    except OSError as error:
        if getattr(error, "winerror", None) == 87:
            return False                # ERROR_INVALID_PARAMETER: no such process
        return True                     # genuinely unknown -> assume held, the cheaper failure
    return True


def describe(state: dict[str, str]) -> str:
    pid_text = state.get("pid", "")
    try:
        alive = _pid_alive(int(pid_text))
    except (TypeError, ValueError):
        alive = False
    try:
        age = int(time.time()) - int(state.get("hb", ""))
        hb = f"{age}s ago"
    except (TypeError, ValueError):
        hb = "never stamped"
    return (f"owner={state.get('dono') or '(unnamed)'} pid={pid_text or '(none)'} "
            f"alive={alive} hb={hb} kind={state.get('owner_kind', '?')}")


class GpuLock:
    def __init__(self, owner: str, path: Path = LOCK_PATH):
        self.owner = owner
        self.path = path
        self.held = False
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    # -------------------------------------------------------------- heartbeat
    def _write_atomic(self, text: str) -> None:
        """Temp + replace, never a truncating rewrite in place.

        A reader that catches the file mid-write sees a truncated `pid=` and an absent `hb=`,
        which is precisely the state `Take-GpuLock` interprets as stale. The bug this module was
        rewritten to fix would come straight back through the heartbeat.
        """
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="ascii")
        os.replace(tmp, self.path)

    def _heartbeat(self, since: str) -> None:
        while not self._stop.wait(BEAT_INTERVAL_SEC):
            state = read_state(self.path)
            if state is None:
                return  # released, or someone deleted it -- do not resurrect it
            if state.get("pid") != str(os.getpid()):
                return  # no longer ours; never overwrite a lock somebody else took
            try:
                self._write_atomic(_render(self.owner, os.getpid(), since))
            except OSError:
                pass  # a reader holding it open for an instant; the next tick refreshes

    # -------------------------------------------------------------- context
    def __enter__(self) -> "GpuLock":
        since = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            with self.path.open("x", encoding="ascii") as handle:
                handle.write(_render(self.owner, os.getpid(), since))
        except FileExistsError:
            state = read_state(self.path) or {}
            raise GpuLockBusy(
                f"{self.path} is held: {describe(state)}\n"
                "Not reclaiming it. If that run is genuinely dead, delete the file by hand -- "
                "a lock that times itself out while the first job is still running is a "
                "corrupted measurement, and a stuck lock is the cheaper failure."
            ) from None
        self.held = True
        self._beat = threading.Thread(target=self._heartbeat, args=(since,), daemon=True)
        self._beat.start()
        print(f"GPU lock acquired: {self.path} ({self.owner}, pid {os.getpid()})")
        return self

    def __exit__(self, *exc) -> None:
        if not self.held:
            return
        self.held = False
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=2)
        # Check the file still says it is ours before deleting it. If this run was declared stale
        # and someone deleted the lock, a second job may already hold it -- unlinking blindly
        # would hand the GPU to a third while two are measuring. Releasing someone else's lock is
        # the one failure mode a lock must not have.
        state = read_state(self.path)
        if state is None:
            print(f"warning: {self.path} vanished before release; someone else deleted it")
            return
        # Compare as an INTEGER. The previous version tested `f"pid={os.getpid()}" not in current`,
        # a substring match: pid 424 matches a lock held by pid 4242, and this function's entire
        # job is to not do that.
        try:
            mine = int(state.get("pid", ""))
        except (TypeError, ValueError):
            mine = -1
        if mine != os.getpid():
            print(f"warning: refusing to release {self.path} -- it is no longer ours: "
                  f"{describe(state)}")
            return
        self.path.unlink()
        print(f"GPU lock released: {self.path}")


if __name__ == "__main__":  # pragma: no cover - a read-only look, takes nothing
    state = read_state()
    if state is None:
        print(f"{LOCK_PATH}: free")
    else:
        print(f"{LOCK_PATH}: {describe(state)}")
