"""One lock file so two agents cannot benchmark the same GPU at the same time.

Both halves of this work run on one RTX 3090: this project's quantization sweeps and a sibling
project's vLLM batteries. Each had a "wait until the card is free, then start" watcher, which is
precisely the pattern that makes two jobs launch in the same second and hand each other contended
numbers -- and, worse, makes the second one OOM when the first is already holding 23 of 24 GiB.

The path is fixed at F:/GPU_BENCH.lock and is deliberately not configurable: a lock only works if
everyone names the same file. `open(path, "x")` is the atomic part -- it either creates the file
or raises, with no window between checking and creating.

    from gpu_lock import GpuLock
    with GpuLock("m_crossover"):
        ...

Stale locks are not cleaned up automatically. A crashed run leaving a lock behind is a nuisance;
a lock that times itself out and lets a second job in while the first is still running is a
corrupted measurement, and the nuisance is the cheaper failure.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

LOCK_PATH = Path("F:/GPU_BENCH.lock")


class GpuLockBusy(RuntimeError):
    pass


class GpuLock:
    def __init__(self, owner: str, path: Path = LOCK_PATH):
        self.owner = owner
        self.path = path
        self.held = False

    def __enter__(self) -> "GpuLock":
        payload = json.dumps({
            "owner": self.owner,
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2)
        try:
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except FileExistsError:
            try:
                current = self.path.read_text(encoding="utf-8").strip()
            except Exception:
                current = "(unreadable)"
            raise GpuLockBusy(
                f"{self.path} is held:\n{current}\n"
                "Wait for it to be released, or delete it if you are sure that run is dead."
            ) from None
        self.held = True
        print(f"GPU lock acquired: {self.path} ({self.owner})")
        return self

    def __exit__(self, *exc) -> None:
        if not self.held:
            return
        self.held = False
        # Check the file still says it is ours before deleting it. If this run was declared stale
        # and someone deleted the lock, a second job may already hold it -- unlinking blindly
        # would hand the GPU to a third while two are measuring. Releasing someone else's lock is
        # the one failure mode a lock must not have.
        try:
            current = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"warning: {self.path} vanished before release; someone else deleted it")
            return
        if f'"pid": {os.getpid()}' not in current and f"pid={os.getpid()}" not in current:
            print(f"warning: refusing to release {self.path} -- it is no longer ours:\n{current}")
            return
        self.path.unlink()
        print(f"GPU lock released: {self.path}")
