# 09 - Four different aggregation rules across seven timing tools, and three cannot express uncertainty

Type: task
Status: ready-for-agent
Blocked by: -
Severity: medium
Provenance: TRACED

## Problem

CLAUDE.md is unambiguous: *"A single run is not a measurement... The tool now repeats interleaved and
prints the ratio's own min-max."* That discipline lives in three places:

- `m_crossover.py:163-217` -- interleaved repeats, paired ratio, `[lo-hi]`, `--repeats` default 3,
  with the 1.4x-disagreement evidence in its docstring at `:89-96`
- `w4a4_breakdown.py:132-153` -- `WALL_PASSES`, median, `[min-max]`, and an honest note at `:148-150`
  that the host figure's spread is a floor
- `graph_capture_probe.py` -- `PASSES`, median, min-max bracket

It is absent from:

- `attn_bench.py:16-39` -- `WARMUP=5`, `ITERS=20`, **mean** of CUDA events, one pass, no spread. Then
  prints `sage_speedup` to three decimals at `:101` and dumps JSON at `:117`. A number in exactly the
  shape that gets pasted into another session.
- `attn_dtype_ab.py:40-51` -- median of 20, one pass, no spread. Worse: it times **fp16 to completion
  before bf16**, which is the non-interleaved order `m_crossover`'s docstring argues against.
- `compile_w4a4_probe.py:94-105`, `compile_w4a4_fix_poc.py:113-123` -- mean of events, one pass
- `nunchaku_compare.py:669-690` -- repeats, but aggregates with **`min(passes)`** (`"best_s"`, printed
  at `:348` as "the fastest of N passes"). It does print every pass and warns above 1.15x spread at
  `:353-355`, the best behaviour outside the three above -- but `min` and `median` are different
  estimators and a table mixing tools cannot be read as one measurement.
- `fbcache_probe.py:117-125`, `fbcache_visual.py:156-160` -- single-shot wall time

Four aggregation rules -- mean-of-events, median-of-wall, median-of-medians, min-of-passes -- across
seven tools, three of which cannot express uncertainty at all.

Separately: the direction rule from the memory `razoes-na-direcao-certa` is implemented once, at
`m_crossover.py:205-209`. `attn_bench.py:109-111` prints `{sdpa_ms/sage_ms:.2f}x` unconditionally --
below 1.0 it emits exactly the form the rule forbids.

## The deepening

A `_timing.py` beside `_bench_guard.py`:

```python
def compare(paths: dict[str, Callable], *, repeats=3, iters, baseline=None) -> dict
```

Interleaves repeats across paths, returns per-path medians plus the **paired ratio's** min-max, and
refuses to format a bare two-decimal ratio without a bracket. `m_crossover.py:163-217` is the
reference implementation; it is ~55 lines and every one of them is general.

## Closing criterion (written before the fix)

Closed when:

1. `_timing.compare()` exists and `m_crossover.py` is rewritten on top of it, producing the same
   numbers it produces today on a re-run in a GPU window (this is the regression test);
2. `attn_bench.py`, `attn_dtype_ab.py` and the two `compile_*` probes use it;
3. no tool in `tools/` prints a ratio without a bracket -- checkable by grepping for `:.2f}x` and
   `:.3f}x` and finding every hit inside the primitive;
4. a ratio below 1.0 is emitted inverted with the direction named, per `razoes-na-direcao-certa`.

`nunchaku_compare.py`'s `min` may stay if the owner wants "fastest of N" -- but then the primitive
must offer it explicitly and the output must say which estimator it used.
