# 04 - How ComfyLite takes the GPU without breaking the two sessions already sharing it

Type: task
Status: ready-for-human
Blocked by: 01
Provenance: TRACED, and blocked on a confirmed bench defect

## ComfyLite must participate, for four of the handoff's phases

Handoff sections 11 (attention autotuner), 12 (LoRA analyzer), 13 (multi-GPU planner) and 19
(ProbeRunner) all touch the card. Section 18's Compatibility Mode does **not** -- ComfyUI already holds
the card as one process.

## The mechanism, and the fact that it is currently broken

`tools/gpu_lock.py` is a Python context manager over `F:/GPU_BENCH.lock`. `open("x")` -- atomic
create-or-raise, no check-then-create window. `__exit__` re-reads the file and refuses to unlink if the
pid is no longer ours, because *"Releasing someone else's lock is the one failure mode a lock must not
have."* Stale locks are **deliberately** not auto-reclaimed.

`tools/gpu_lock.ps1` is the PowerShell side on the same path, with a 55-second heartbeat staleness
limit and `Assert-GpuLock`, which throws.

**They do not speak the same format, and the PowerShell side steals a live Python lock.** Confirmed by
execution 2026-08-22 -- see `varredura-2026-08-22/issues/01`. **This ticket is blocked on that one**:
a third participant writing a third format would make it worse, and ComfyLite must adopt whichever
format wins there.

## Three constraints that come with participating, all already paid for on this bench

1. **The owner string names the work, not the repo.** CLAUDE.md: *"`cortiq:bench` tells it nothing it
   can plan around."* So `comfylite:attn_autotune_ltx2_sm86`, not `comfylite:bench`.
2. **On refusal, do not wait.** Fall back to non-GPU work -- catalog scan, resolver, downloads -- report
   it in the UI, never silently reclaim.
3. **Holding the lock is not the same as using the locked card.** CLAUDE.md records, executed
   2026-08-21, that a whole measurement session ran on `cuda:1` (the 3080 Ti, 9-26% busy) while the lock
   sat on an idle 3090, because the adapter was chosen by `request_adapter(HighPerformance)`. The probe
   cache is what exposed it, because it stamps the adapter name into every line.

   **So ComfyLite's `DevicePlanner` must stamp the device it actually used into every benchmark record.**
   Without that, its persistent profiles attribute 3080 Ti numbers to the 3090 and the whole autotuner
   database is quietly wrong. The handoff does not say this; it should.

## And if ComfyLite ever manages a ComfyUI subprocess

Two things, both recorded on this bench:

- **`main.py --windows-standalone-build` re-executes itself as a child.** Killing only the PID that
  `Start-Process` returned left an orphan holding **20,578 MiB of the 3090 and 16.67 GB of RAM**,
  invisibly -- `Get-Process` said dead, `nvidia-smi` said occupied. **Kill the tree, not the PID.**
- **`--disable-dynamic-vram` is required** for Nunchaku SVDQuant loaders and for the LTX 2.5 workflow,
  and **zero of the nine `.bat` launchers passes it** (grepped 2026-08-22, 0 hits). Whoever starts the
  process owns that flag -- which is an argument for ComfyLite starting it rather than shelling to a
  launcher.

## Closing criterion

Closed when ComfyLite acquires the lock in the format that wins `varredura-2026-08-22/issues/01`, using
`tools/gpu_lock.py` directly rather than a reimplementation; a refusal degrades to non-GPU work and says
so in the UI; every benchmark record carries the device name actually used; and any ComfyUI process
ComfyLite spawns is killed as a tree.
