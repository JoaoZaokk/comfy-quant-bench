"""One way to time two implementations against each other, and the only place the GPU lock
stops being something a tool author has to remember.

Two problems, one primitive, because they turned out to be the same problem.

## 1. Four aggregation rules across seven tools, three of which could not express uncertainty

CLAUDE.md is unambiguous -- *"A single run is not a measurement"* -- and the discipline existed in
exactly three files (`m_crossover.py`, `w4a4_breakdown.py`, `graph_capture_probe.py`). The rest
reported mean-of-events, or median-of-wall, or `min(passes)`, each from a single burst, and a table
built out of them cannot be read as one measurement because the columns are different estimators.

**What one burst costs, MEASURED 2026-08-22 on the RTX 3090 with the lock held**
(`tools/probe_groupsize_and_spread.py`, five interleaved bursts, B=2 H=24 S=4096 D=64 bf16, median
of 20 per-iteration CUDA-event timings per burst):

    sdpa   3.971  4.007  3.972  3.970  3.974     median 3.972   spread 1.01x
    sage   2.370  2.466  2.466  2.446  2.446     median 2.446   spread 1.04x
    ratio  1.676x 1.625x 1.610x 1.623x 1.625x

The strong claim did not reproduce: **the 1.4x run-to-run disagreement in `m_crossover.py:89-96` is
a GEMM result and does not happen on attention.** Do not quote 1.4x for attention. Spread on the
attention ratio is 1.04x.

**The weak claim did reproduce, and it is worse, because it is bias rather than noise.** The
outlier is the *first* burst -- 1.676x against a 1.610-1.625 cluster -- and five warm-up iterations
did not settle it. `attn_bench.py` ran exactly one burst, the first one, so it did not draw
randomly from that distribution: it reported the draw that is systematically ~3% high, every time.
Repeating the tool cannot average that away. Hence `discard_first_burst=True` by default, and hence
the discarded value is *printed* rather than dropped silently -- a burst that is 3% off is evidence
about the machine, and a burst that is 40% off means the warm-up was not enough and the rest of the
table should not be believed either.

Concretely, on that shape: `attn_bench` printed `1.676x`. The honest answer is `1.63x [1.61-1.68]`.

## 2. The lock was opt-in, so the heaviest consumers in the tree skipped it

`_bench_guard.BenchGuard` is a good module and nine files used it. `attn_bench.py` and
`nunchaku_compare.py` -- the single largest GPU consumer in `tools/` at 757 lines and two diffusion
models -- did not, which means a sibling's `Assert-GpuLock` would have been *granted* while they
ran. Contention is not a rounding error here: the same code, same day, measured `im2col 74.4 s` on
a quiet machine and `87.0 s` on a loaded one -- 17%, against a 10 s effect being measured.

So `compare()` acquires the guard itself and there is no code path through this module that times
anything without one. Taking the lock is what timing *is*; it is not a step that can be forgotten.

## Why `Ratio` is a type and not a float

`razoes-na-direcao-certa`: a ratio below 1.0 is never reported as `0.29x`, it is reported as
`3.41x lighter`, and the fix belongs in the tool rather than in the prose. That rule was
implemented once in the whole tree (`m_crossover.py:205-209`) while `attn_bench.py:109` printed
`f"{sdpa_ms/sage_ms:.2f}x"` unconditionally -- below 1.0 that emits exactly the forbidden form.

A float cannot enforce either rule, because `f"{x:.2f}x"` always works on a float. `Ratio` renders
its own text: it inverts and names the direction below 1.0, and it carries its bracket. A numeric
format spec applied to it keeps the alignment and drops the precision, so the bracket survives a
careless `f"{ratio:>9.2f}x"` instead of being silently discarded by it.

Nothing here imports torch at module scope, so `tools/test_timing.py` runs on a machine with no
GPU and no CUDA context.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# The embedded interpreter ships a `python313._pth`, which suppresses the script-directory entry,
# so a sibling module in `tools/` is not importable without this. `m_crossover.py:49-52` records
# the same line being missing and `from _bench_guard import BenchGuard` dying with
# ModuleNotFoundError on the first real execution -- the guard had been written but never run.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_guard import BenchGuard, active_guard  # noqa: E402

# How the *bursts* are reduced. The reduction *within* a burst is fixed at median by the timer
# functions below and is not configurable: a median of per-iteration samples is what every tool in
# this tree that got it right already did, and making it an option would let two tables disagree
# again in a way the header cannot show.
ESTIMATORS: dict[str, Callable[[list[float]], float]] = {
    "median": statistics.median,
    # Kept because `nunchaku_compare.py` wants "fastest of N passes" -- a whole-model sampling run
    # has a floor (nothing on the card) and no ceiling (anything else on the card), so the minimum
    # is the closest thing to an uncontended measurement. That is a defensible choice; mixing it
    # into a table with medians without saying so is not, which is why `provenance()` prints it.
    "min": min,
    "mean": statistics.fmean,
}


class BenchRefused(RuntimeError):
    """The guard said no. Raised rather than returned so it cannot be ignored by accident."""


@dataclass(frozen=True)
class Ratio:
    """A ratio that cannot be printed bare, and cannot be printed upside down.

    `value` is always baseline/other, so >= 1 means `other` won. `lo`/`hi` are the paired
    per-burst min and max of that same quantity -- the spread of the *ratio*, not of either
    column, because the ratio is what gets quoted into another session.
    """

    value: float
    lo: float | None = None
    hi: float | None = None
    n: int = 0
    better: str = "faster"
    worse: str = "slower"
    # A deterministic quantity -- bytes on disk, parameter count -- has no burst-to-burst spread
    # and an interval on it would be a fiction. It still goes through this type for the direction
    # rule, which is the half that applies.
    exact: bool = False

    def _text(self) -> str:
        if self.value != self.value or self.value <= 0:
            return "n/a"
        if self.value >= 1.0:
            head = f"{self.value:.2f}x"
            if self.better:
                head += f" {self.better}"
            lo, hi = self.lo, self.hi
        else:
            # Never 0.29x. The direction of a ratio below 1 is the thing readers invert wrongly,
            # so it is stated as the losing factor instead, and the interval inverts with it.
            head = f"{1.0 / self.value:.2f}x {self.worse}"
            lo = 1.0 / self.hi if self.hi else None
            hi = 1.0 / self.lo if self.lo else None
        if self.exact:
            return head
        if self.n > 1 and lo is not None and hi is not None:
            return f"{head} [{lo:.2f}-{hi:.2f}]"
        # Not silence. A single burst is the case the 2026-08-22 measurement showed is biased
        # high, so the absence of an interval is itself the finding and is printed as one.
        return f"{head} (1 burst, no interval)"

    def __str__(self) -> str:
        return self._text()

    def __format__(self, spec: str) -> str:
        text = self._text()
        if not spec:
            return text
        # A numeric spec on a Ratio is the exact mistake this type exists to prevent:
        # `f"{sdpa_ms/sage_ms:.2f}x"` at attn_bench.py:109 is how a direction-blind, bracket-less
        # number reached a JSON dump. Precision and type char are dropped; fill, alignment and
        # width survive, so an existing table column keeps its shape and the bracket is not lost.
        cleaned = spec.rstrip("fgeFGE%")
        if "." in cleaned:
            cleaned = cleaned[: cleaned.index(".")]
        try:
            return format(text, cleaned)
        except ValueError:
            return text


@dataclass
class Result:
    """What `compare()` returns. `ratios` are `Ratio`s, never floats, deliberately."""

    estimator: str
    iters: int
    bursts: int
    kept: int
    series: dict[str, list[float]] = field(default_factory=dict)
    first_burst: dict[str, float] | None = None
    times: dict[str, float] = field(default_factory=dict)
    baseline: str | None = None
    ratios: dict[str, Ratio] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    last: dict[str, object] = field(default_factory=dict)

    def as_json(self) -> dict:
        """Plain types for a JSON dump, with the estimator and the spread carried along.

        `attn_bench.py` used to emit `"sage_speedup": 1.676` -- a bare float, three decimals, one
        burst. A number in that shape gets pasted into another session as if it were measured.
        Every ratio here goes out as an object with its interval, its burst count and the
        estimator that produced it, so it cannot be quoted without its condition.
        """
        return {
            "estimator": self.estimator,
            "iters": self.iters,
            "bursts": self.bursts,
            "kept_bursts": self.kept,
            "first_burst_discarded": (None if self.first_burst is None else
                                      {k: (round(v, 3) if v == v else None)
                                       for k, v in self.first_burst.items()}),
            "ms": {k: (round(v, 3) if v == v else None) for k, v in self.times.items()},
            # `None`, not NaN. `json.dumps` writes a bare `NaN`, which Python reads back but no
            # other JSON parser will, and these dumps get pasted between sessions and tools.
            "series_ms": {k: [round(x, 3) if x == x else None for x in v]
                          for k, v in self.series.items()},
            "baseline": self.baseline,
            "ratios": {
                k: {self.estimator: round(r.value, 3) if r.value == r.value else None,
                    "min": round(r.lo, 3) if r.lo is not None else None,
                    "max": round(r.hi, 3) if r.hi is not None else None,
                    "n_bursts": r.n,
                    "text": str(r)}
                for k, r in self.ratios.items()
            },
            "failed": self.failed,
        }


def provenance(result: Result) -> str:
    """The line every tool built on this prints under its table.

    A column of numbers with no estimator named reads as "the measurement". Three tools in this
    directory reduced with a median, one with `min(passes)`, and two with the mean of a single
    CUDA-event span, and nothing in any of their output said which -- so their rows looked
    comparable and were not.
    """
    head = (f"{result.estimator} of {result.kept} kept burst(s) x {result.iters} iteration(s); "
            f"within a burst the reducer is median")
    if result.first_burst:
        shown = ", ".join(f"{k} {v:.3f}" for k, v in result.first_burst.items() if v == v)
        head += (f"\nburst 1 discarded as warm-up-biased ({shown} ms). MEASURED 2026-08-22 on the "
                 f"3090: the first burst of an\nSDPA-vs-Sage A/B read 1.676x against a "
                 f"1.610-1.625 cluster -- 3% high, in the same direction, every time.")
    else:
        head += ("\nfirst burst NOT discarded, so any warm-up bias is still in these numbers "
                 "(measured at ~3%, directional).")
    return head


# --------------------------------------------------------------------------- timers
# Both take (fn, iters) and return milliseconds. Both do their own warm-up on every burst, which
# is what makes a burst independent of the one before it -- and, per the 2026-08-22 result, still
# not enough to make burst 1 comparable to burst 2.

def wall_ms(fn: Callable[[], object], iters: int, warmup: int = 3) -> float:
    """Median of per-iteration wall time with a sync on both sides. `m_crossover.py:67-78` exactly.

    Kept bit-for-bit so that m_crossover rebuilt on this primitive measures the same quantity it
    measured before. Wall time includes host dispatch, which is the point at small M -- see
    `w4a4_breakdown.py`, where host cost exceeds GPU work entirely at M=1.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000.0


def cuda_event_ms(fn: Callable[[], object], iters: int, warmup: int = 5) -> float:
    """Median of per-iteration CUDA-event spans -- device time, host dispatch excluded.

    Per-iteration events, not one span around the whole loop divided by `iters`. The single-span
    form is what `attn_bench.py:32-39` did, and a mean over one span cannot report a spread at all:
    a single slow iteration and twenty uniformly slow ones produce the identical number. This is
    the timer the 2026-08-22 measurement used, so its result is directly comparable.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(iters)]
    for start, end in pairs:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in pairs)


def seconds_timer(fn: Callable[[], object], iters: int) -> float:
    """For paths whose single call already takes seconds -- a whole sampling run.

    `iters` is honoured but is normally 1: repeating a 40 s render twenty times inside one burst is
    not a measurement, it is a GPU window spent. No `torch.cuda.synchronize` here, because the
    callers that need one (fbcache, nunchaku) already synchronize inside their own sampling
    function and adding a second import for it would make this file need torch on a CPU-only host.
    """
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000.0


# --------------------------------------------------------------------------- the primitive

def _resolve_guard(guard, owner):
    """Return (guard, needs_exit). Raises rather than timing anything unguarded.

    Three cases, and none of them is "carry on without a lock":
      * an explicit guard the caller already entered (a tool whose whole run is one block);
      * the ambient one `BenchGuard.__enter__` registers, so a tool that already wrapped its
        `main()` does not deadlock against its own lock file when it calls `compare()`;
      * nothing held, in which case this takes the lock itself and needs an owner string to name
        the block with. `Assert-GpuLock -Owner 'cortiq:bench'` tells the other side nothing it can
        plan around; `bench:attn_bench` does.
    """
    if guard is not None:
        if not isinstance(guard, BenchGuard):
            raise TypeError(f"guard= must be a BenchGuard, got {type(guard).__name__}")
        if guard.refused:
            raise BenchRefused(guard.refused)
        if not guard.held:
            raise BenchRefused("the BenchGuard passed to compare() is not held; enter it first")
        return guard, False
    ambient = active_guard()
    if ambient is not None:
        return ambient, False
    if not owner:
        raise TypeError(
            "compare() needs owner='<repo>:<what>' so it can take the GPU lock, or an already "
            "entered BenchGuard. There is no unguarded path: two sessions share this 3090 and a "
            "benchmark that ran contended is worse than one that did not run.")
    taken = BenchGuard(owner)
    taken.__enter__()
    if taken.refused:
        taken.__exit__(None, None, None)
        raise BenchRefused(taken.refused)
    return taken, True


def compare(paths: dict[str, Callable[[], object]], *,
            iters: int,
            owner: str | None = None,
            repeats: int = 3,
            baseline: str | None = None,
            estimator: str = "median",
            discard_first_burst: bool = True,
            timer: Callable[[Callable[[], object], int], float] = wall_ms,
            guard: BenchGuard | None = None,
            keep_last: bool = False,
            better: str = "faster",
            worse: str = "slower") -> Result:
    """Time every path against every other one, interleaved, under the GPU lock.

    `repeats` is the number of bursts **kept**. With `discard_first_burst` (the default) one extra
    burst is run and thrown away, so `repeats=3` costs four bursts of `iters` iterations per path.
    That is a real ~33% increase in GPU-window time over the old three-burst form and it is
    deliberate: dropping to two kept bursts instead would have made the interval a min-max of two
    samples, which is not an interval.

    Repeats are **interleaved**, not batched per path: every path is timed once, then every path
    again. Timing one path to completion before starting the next lets any drift over the burst --
    clock boost decaying, another process arriving -- land entirely on whichever path happened to
    be running, which is exactly how a ratio picks up a bias no single median reveals.
    `attn_dtype_ab.py:138-161` timed fp16 to completion before bf16 and its own docstring records
    the resulting artefact: bf16 reported 2.65x slower than fp16, where forcing each backend by
    hand put them within 1%.

    A path that raises is recorded in `failed` and stops being timed -- it does not become a silent
    `nan` column with the verdict still naming a winner (`m_crossover.py:170-173`, same fix).
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if iters < 1:
        raise ValueError("iters must be at least 1")
    if not paths:
        raise ValueError("compare() needs at least one path")
    if baseline is not None and baseline not in paths:
        raise ValueError(f"baseline {baseline!r} is not one of the paths {sorted(paths)}")

    held, needs_exit = _resolve_guard(guard, owner)
    try:
        bursts = repeats + (1 if discard_first_burst else 0)
        series: dict[str, list[float]] = {k: [] for k in paths}
        failed: dict[str, str] = {}
        last: dict[str, object] = {}
        for _ in range(bursts):
            for label, call in paths.items():
                if label in failed:
                    series[label].append(float("nan"))
                    continue
                try:
                    series[label].append(timer(call, iters))
                except Exception as exc:
                    series[label].append(float("nan"))
                    failed[label] = f"{type(exc).__name__}: {str(exc)[:120]}"

        if keep_last:
            # One extra call per path, after every burst is done, never inside one. Capturing the
            # artefact during a burst would add an untimed call to each burst and -- worse -- keep
            # a tensor per path alive for the rest of the comparison. `attn_dtype_ab.py:54-67`
            # records what resident tensors do to this bench: three 32 MiB fp32 tensors left on
            # the card made bf16 read 2.65x slower than fp16, an allocator artefact end to end.
            for label, call in paths.items():
                if label not in failed:
                    last[label] = call()

        first: dict[str, float] | None = None
        if discard_first_burst:
            first = {k: v[0] for k, v in series.items()}
            series = {k: v[1:] for k, v in series.items()}

        reduce = ESTIMATORS[estimator]
        times = {k: (reduce(v) if v and all(x == x for x in v) else float("nan"))
                 for k, v in series.items()}

        ratios: dict[str, Ratio] = {}
        if baseline is not None:
            base_series = series[baseline]
            for label, other in series.items():
                if label == baseline:
                    continue
                pairs = [b / o for b, o in zip(base_series, other)
                         if b == b and o == o and o > 0]
                if not pairs:
                    ratios[label] = Ratio(float("nan"), n=0, better=better, worse=worse)
                    continue
                ratios[label] = Ratio(reduce(pairs), min(pairs), max(pairs), len(pairs),
                                      better=better, worse=worse)
        return Result(estimator=estimator, iters=iters, bursts=bursts, kept=repeats,
                      series=series, first_burst=first, times=times, baseline=baseline,
                      ratios=ratios, failed=failed, last=last)
    finally:
        if needs_exit:
            held.__exit__(None, None, None)


def ratio_of(baseline_value: float, other_value: float, *,
             series: list[float] | None = None,
             better: str = "faster", worse: str = "slower",
             exact: bool = False) -> Ratio:
    """A `Ratio` from two numbers this module did not time itself.

    For quantities that are not burst-structured -- bytes on disk, a peak VRAM figure, the seconds
    of a whole sampling run measured elsewhere. It buys the direction rule and the "no bare
    two-decimal ratio" rule; it cannot invent an interval, and `exact=True` says the quantity has
    none rather than pretending the absence is a measurement failure.

    `series` accepts an already-paired list of ratios (one per pass) when the caller does have
    repeats, which is how `nunchaku_compare.py` gets a real bracket out of its `passes` list.
    """
    if not other_value or other_value != other_value or baseline_value != baseline_value:
        return Ratio(float("nan"), better=better, worse=worse, exact=exact)
    value = baseline_value / other_value
    if series:
        return Ratio(value, min(series), max(series), len(series), better=better, worse=worse)
    return Ratio(value, n=1, better=better, worse=worse, exact=exact)


def summarize(samples: list[float], *, estimator: str = "median") -> tuple[float, float, float]:
    """(reduced, min, max) for a single path's samples, with the estimator named by the caller.

    The one-path counterpart to `compare()`. `nunchaku_compare.py` loads one model per process on
    purpose -- there is no second path in the same interpreter to interleave against -- so it has
    passes but no A/B, and this is the shape that fits it.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}")
    clean = [s for s in samples if s == s]
    if not clean:
        return float("nan"), float("nan"), float("nan")
    return ESTIMATORS[estimator](clean), min(clean), max(clean)
