# 01 - The PowerShell lock steals a live Python-held GPU lock

Type: task
Status: resolved
Blocked by: -
Severity: high
Provenance: EXECUTED 2026-08-22 -- found by run, fixed, and the fix run

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

## Answer -- FIXED and EXECUTED 2026-08-22

### The fix went on the Python side, and the direction is the whole point

The obvious move was to make `gpu_lock.ps1` read JSON. **That would have reproduced the same bug
pointing at the sibling.** A session still running the old `.ps1` writes `key=value`; anything new
that only spoke JSON would rob it in turn. **Format changes are not symmetric when the other party
may be running last week's code.**

`key=value` is the dialect that already has two readers: `gpu_lock.ps1`, and `gpu_lock.py`'s own
`__exit__`, which already accepted `pid=<n>` alongside the JSON form. So `gpu_lock.py` now writes
what everybody already reads, and **no sibling has to change anything for the theft to stop.**

Changes:

- **`tools/gpu_lock.py`** -- writes `key=value`, byte-compatible with `gpu_lock_beat.ps1` including
  key order. A daemon heartbeat thread refreshes `hb=` every 15 s and **stops the moment the lock
  stops naming our pid**. `_parse` still understands the old JSON, so a lock written by an
  out-of-date copy is recognised as held rather than run over. `__exit__` compares pid as an
  **integer** (CACHE-02, below). Writes go through temp + `os.replace`: a reader catching a
  truncating rewrite sees an empty `pid=` and no `hb=`, which is exactly the state that reads as
  stale -- the fix must not reintroduce its own bug through the heartbeat.
- **`tools/gpu_lock.ps1`** -- `Get-GpuLockState` parses both dialects, and reports an unparseable
  body as *held with an unknown owner* rather than as absent. `Take-GpuLock` now **refuses to
  reclaim a lock it cannot identify**: a blank owner is the signature of a parse failure, not of a
  crashed run, and a genuine leftover still names who wrote it. Second layer, aimed straight at the
  observed failure.
- **`tools/gpu_lock_beat.ps1`** -- CACHE-03. The loop rewrote the file unconditionally, forever. An
  orphaned heartbeat would stamp its own name over whatever lock somebody legitimately took
  afterwards, silently turning a released card into a stolen one. It now exits when the file is
  gone or no longer names its pid, and writes via temp + `Move-Item`.

### The test, which is the closing criterion

`tools/test_gpu_lock.py`, 23 checks, run 2026-08-22 against **copies** of both `.ps1` files pointed
at a throwaway path. The real `F:/GPU_BENCH.lock` was never touched and was absent before and
after. **23 passed, 0 failed.**

Criterion item by item, with what the run actually printed:

1. `Get-GpuLockState` returns the real Owner and Pid, and a `StaleFor` that is not
   `[int64]::MaxValue`:
   `OWNER=bench:long_sweep|PID=51748|STALE=1|KIND=python`
2. `Take-GpuLock` returns `$false` and names the real owner:
   `lock HELD by bench:long_sweep (pid 51748 alive=True, hb 1s ago, kind python) -- not taking`
   -- and it does **not** print "reclaiming".
3. `Assert-GpuLock` throws:
   `THREW: GPU lock held by bench:long_sweep (pid 51748, hb 2s ago) -- refusing to run GPU work.`
4. The reverse direction still holds: `gpu_lock.py` raises `GpuLockBusy` against a
   PowerShell-written lock, naming `controller:bench`.
5. Captured as a script under `tools/`, so the next format change fails loudly instead of silently.

Beyond the criterion, also proved by the run: the heartbeat refreshes (`1787391661 -> 1787391676`,
+15 s); a legacy-JSON lock with a **live** pid is refused; an unreadable lock file is treated as
held rather than free.

### CACHE-02, fixed in the same change

`__exit__` tested `f"pid={os.getpid()}" not in current` -- a substring match, so pid 424 matched a
lock held by pid 4242. Now parsed and compared as an integer. Proved: a `GpuLock` for this process,
told to release a lock owned by pid 4242, printed `warning: refusing to release ... it is no longer
ours` and the file survived.

### What this does NOT cover

The 55-second staleness path (needs a lock older than the limit with a genuinely dead pid), and two
PowerShell sessions racing each other -- `Start-Process` plus a 2 s settle is not atomic. The test
file says both in its own footer rather than leaving a reader to assume they were checked.
