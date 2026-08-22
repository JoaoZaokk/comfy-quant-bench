# 01 - The PowerShell lock steals a live Python-held GPU lock

Type: task
Status: ready-for-agent
Blocked by: -
Severity: high
Provenance: EXECUTED 2026-08-22

## Problem

`tools/gpu_lock.py` and `tools/gpu_lock.ps1` arbitrate the same file, `F:/GPU_BENCH.lock`, in two
formats that cannot read each other.

- `gpu_lock.py:42-46` writes JSON: `json.dumps({"owner":..., "pid":..., "started":...}, indent=2)`.
- `gpu_lock.ps1:44-50` parses `key=value` lines with `Select-String "^$k=(.*)$"` for `dono`, `pid`, `hb`.

Against a JSON lock every regex misses. `Owner` and `Pid` come back empty and `StaleFor` is set to
`[int64]::MaxValue`. `Take-GpuLock:63` then computes `$alive = '' -and (...)` -> false, and `:64`
computes `false -or (MaxValue -lt 55)` -> false, so the held-lock branch is skipped and execution
falls through to the reclaim branch.

The theft is one-way, which is why nobody noticed: `gpu_lock.py:48` uses `open(path, "x")`, which
fails on any existing file, so the Python side correctly refuses a PowerShell-held lock.

## Evidence -- EXECUTED, not traced

Run 2026-08-22 against a copy of `gpu_lock.ps1` pointed at a throwaway lock path (the real
`F:/GPU_BENCH.lock` was never touched, and was absent for the whole test). A JSON lock byte-identical
to what `gpu_lock.py` writes was placed at that path, then:

```
--- Get-GpuLockState reading a Python-written (JSON) lock ---
Owner    :
Pid      :
StaleFor : 9223372036854775807

--- Take-GpuLock: refuse, or steal? ---
lock is stale (pid  dead, hb 9223372036854775807s ago) -- reclaiming from
lock TAKEN by other-session:steal-test (heartbeat pid 42616)
Take-GpuLock returned: True
```

The heartbeat process spawned by the test was killed and the temp directory removed.

## Why this one matters more than its size

`CLAUDE.md`'s *The GPU window* section is built on `Assert-GpuLock` throwing. It does not throw here.
The message the operator reads -- "lock is stale (pid dead, hb ...s ago) -- reclaiming from " with a
blank owner -- reads exactly like a leftover file from a crashed run, which is the one situation
where reclaiming is correct. CLAUDE.md already records both halves of the resulting damage: a
2026-08-19 incident where a lock was taken against a live sibling with a 2-second-old heartbeat, and
the size of the effect -- `im2col 74.4 s` quiet against `87.0 s` loaded, 17%, against a 10 s effect
being measured.

Note this is a *shared-resource* defect, not a security one. There is no attacker; there are two
sessions and one card.

## Closing criterion (written before the fix)

Closed when, with `tools/gpu_lock.py` holding the lock in its own format:

1. `Get-GpuLockState` returns the real `Owner` and `Pid` from that file, and a `StaleFor` that is not
   `[int64]::MaxValue`;
2. `Take-GpuLock` returns `$false` and prints a HELD message naming the real owner;
3. `Assert-GpuLock` throws;
4. the reverse direction still holds -- `gpu_lock.py` still raises `GpuLockBusy` against a
   PowerShell-written lock;
5. the test above is captured as a script under `tools/` so the next format change fails loudly
   instead of silently.

One format wins; JSON is the obvious one since it is unambiguous and the Python side already emits
it. The PowerShell side keeps the `hb` heartbeat semantics as a JSON field.

## Related

`CACHE-02`: `GpuLock.__exit__` (`gpu_lock.py:76-85`) matches its own pid as a *substring* of the file
text, so pid `424` would match a lock held by pid `4242`. Same fix -- parse the JSON, compare
integers -- so do both here.
