# 10 - The heaviest GPU consumers in the tree take no lock and no occupancy guard

Type: task
Status: ready-for-agent
Blocked by: 01, 09
Severity: medium
Provenance: TRACED

## Problem

`_bench_guard.py` is the best-designed module here: one context manager plus one boolean
(`:80-109`), it fails **closed** (`:63-77`), and it reads NVML before torch creates a context for a
documented reason (`:38-42`, the 744 MiB WDDM disagreement). High depth, small interface.

Grep for `BenchGuard` over `tools/` returns nine files: `quality_ladder`, `synthetic_vs_real`,
`w4a8_fallback_sweep`, `graph_capture_probe`, `w4a4_breakdown`, `m_crossover`, `attn_dtype_ab`,
`check_w4a8`, and `_bench_guard` itself.

Not among them, all of which time GPU work:

- `attn_bench.py` -- the entire file is a GPU benchmark (`:28-39`, `:70-114`). No lock, no occupancy check.
- `nunchaku_compare.py` -- 757 lines, loads two diffusion models and samples both (`:660-673`). The
  single largest GPU consumer in `tools/`, and it takes no lock. This reproduces exactly the scenario
  `_bench_guard.py:19` records: *"20.49 GiB resident with the lock file absent."*
- `compile_w4a4_probe.py`, `compile_w4a4_probe2.py`, `compile_w4a4_fix_poc.py`
- `fbcache_probe.py`, `fbcache_visual.py` -- real sampling loops, timed
- `svdquant_probe.py`, `weight_dtype_probe.py`, `stage_probe.py`, `dispatch_census.py`, `quality_battery.py`

Every number produced by that second list was measured with no evidence about whether the card was
contended -- and, worse for the other side, **with no lock, so a sibling's `Assert-GpuLock` would have
granted while they ran.** That compounds ticket 01.

## Two more defects in the guard itself

- `_bench_guard.py:51` reads occupancy from **NVML device index 0 only**. A job on `cuda:1` is
  invisible to it. Given CLAUDE.md's documented incident where nine runs went to the 3080 Ti while the
  lock sat on the 3090, a guard that only watches device 0 is guarding the wrong half of the problem.
- `nunchaku_compare.py:547` indexes NVML by **CUDA ordinal**. Under `CUDA_VISIBLE_DEVICES` the CUDA
  ordinal and the NVML index are different numbers, so the VRAM figure can describe the other card.

## The deepening

`BenchGuard` already *is* the deep module; the gap is that entering it is opt-in per file. Fold it
into the timing primitive from ticket 09 -- `_timing.compare()` acquires the guard and refuses without
it -- and the lock stops being something a tool author must remember. Every tool that times something
takes the lock, because taking the lock is what timing *is*.

## Closing criterion (written before the fix)

Closed when:

1. `_timing.compare()` acquires `BenchGuard` and cannot be called without it;
2. every tool in the "not among them" list either uses `_timing.compare()` or states in its own
   docstring why it is exempt (a converter is exempt; a benchmark is not);
3. `_bench_guard` reads occupancy from **every** visible device, and reports which one it found busy;
4. `nunchaku_compare.py` maps CUDA ordinal to NVML index via UUID, not by assuming they are equal, and
   stamps the device name into every record it writes.

Item 4 also fixes `PR-08` / `PR-09` / `PR-10` in the same file: load-seconds compared across
non-comparable paths, peak-VRAM and torch-peak measured over disjoint windows and printed adjacently,
and the dequantization counter that exists to prove the loader did not silently dequantize.
