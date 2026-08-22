"""CPU-only tests for `_timing.py`. No CUDA, no torch, and no touching F:/GPU_BENCH.lock.

    python_embeded\\python.exe -s tools/test_timing.py

pytest is not installed in the embedded interpreter and installing it is the owner's decision, so
this file carries its own runner.

**Two things it deliberately does not test**, because they need the card and faking them would be
worse than admitting the gap:

  * that `m_crossover.py` rebuilt on `compare()` produces the numbers it produced before. That is
    the regression test for ticket 09 and the only honest way to run it is
    `python_embeded\\python.exe -s tools/m_crossover.py --repeats 3` in a GPU window, against the
    table in `W4A4_PROGRESS.md` part 11. A synthetic timing would prove the arithmetic, which is
    not what is in doubt.
  * that `wall_ms` and `cuda_event_ms` measure what they claim. They call into torch; the fake
    timers below do not, and nothing here would notice if `cuda_event_ms` forgot to synchronize.

What it does test is the part that has no GPU in it: the interleaving order, the estimators, the
bracket, the direction rule, the first-burst discard, and that there is no path through `compare()`
that times anything without a held guard.

The guard used below is a real `BenchGuard` object that is **never entered** -- `_held` is set by
hand. `BenchGuard.__init__` builds a `GpuLock` but does not touch the file until `__enter__`, so
this never creates, reads or deletes the shared lock a sibling session may be holding right now.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_guard import BenchGuard  # noqa: E402
from _timing import (BenchRefused, Ratio, compare, provenance,  # noqa: E402
                     ratio_of, summarize)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def fake_guard() -> BenchGuard:
    """A guard that is held without ever having taken the real lock file. See the module docstring."""
    guard = BenchGuard("test:_timing")
    guard._held = True
    return guard


def scripted(values: list[float], log: list[str], label: str):
    """A timer that returns a scripted sequence and records the order it was asked in.

    Deterministic on purpose: the properties under test are ordering and arithmetic, and a real
    `time.sleep` would make the assertions probabilistic on a machine that is doing other things.
    """
    state = {"i": 0}

    def timer(_fn, _iters):
        log.append(label)
        value = values[min(state["i"], len(values) - 1)]
        state["i"] += 1
        return value

    return timer


def dispatching(scripts: dict[str, list[float]], log: list[str]):
    """One timer for all paths; which script it draws from is decided by the callable it is given."""
    timers = {}

    def timer(fn, iters):
        label = fn.__name__
        if label not in timers:
            timers[label] = scripted(scripts[label], log, label)
        return timers[label](fn, iters)

    return timer


# ---------------------------------------------------------------- the guard is not optional
def test_refuses_without_guard() -> None:
    print("\ncompare() cannot be called without a guard")
    try:
        compare({"a": lambda: None}, iters=1, repeats=1, discard_first_burst=False)
        check("no owner, no guard -> raises", False, "it returned instead of raising")
    except TypeError as exc:
        check("no owner, no guard -> TypeError", "GPU lock" in str(exc), str(exc)[:80])
    except Exception as exc:
        check("no owner, no guard -> TypeError", False, f"raised {type(exc).__name__}")

    try:
        compare({"a": lambda: None}, iters=1, repeats=1, guard=object())
        check("a non-BenchGuard guard is rejected", False, "it returned instead of raising")
    except TypeError as exc:
        check("a non-BenchGuard guard is rejected", "BenchGuard" in str(exc), str(exc)[:80])

    unheld = BenchGuard("test:unheld")  # constructed, never entered: no lock file is touched
    try:
        compare({"a": lambda: None}, iters=1, repeats=1, guard=unheld)
        check("an unheld BenchGuard is rejected", False, "it returned instead of raising")
    except BenchRefused as exc:
        check("an unheld BenchGuard is rejected", "not held" in str(exc), str(exc)[:80])

    refused = BenchGuard("test:refused")
    refused._held = True
    refused.refused = "2.90 GiB already resident"
    try:
        compare({"a": lambda: None}, iters=1, repeats=1, guard=refused)
        check("a refused BenchGuard is rejected", False, "it returned instead of raising")
    except BenchRefused as exc:
        check("a refused BenchGuard is rejected", "2.90 GiB" in str(exc), str(exc)[:80])


# ---------------------------------------------------------------- interleaving
def test_interleaved() -> None:
    print("\nrepeats are interleaved across paths, not batched per path")

    def a():
        return None

    def b():
        return None

    log: list[str] = []
    scripts = {"a": [10.0, 10.0, 10.0, 10.0], "b": [5.0, 5.0, 5.0, 5.0]}
    compare({"a": a, "b": b}, iters=1, repeats=3, guard=fake_guard(),
            timer=dispatching(scripts, log))
    check("order is a,b,a,b,...", log == ["a", "b"] * 4, f"got {log}")
    check("4 bursts run for repeats=3 (one discarded)", len(log) == 8, f"got {len(log)} calls")


# ---------------------------------------------------------------- first burst
def test_first_burst_discarded() -> None:
    print("\nthe first burst is treated as suspect and reported separately")

    def sdpa():
        return None

    def sage():
        return None

    # The 2026-08-22 numbers, verbatim: burst 1 is the high one and the tool must not report it.
    scripts = {"sdpa": [3.971, 4.007, 3.972, 3.970, 3.974],
               "sage": [2.370, 2.466, 2.466, 2.446, 2.446]}
    log: list[str] = []
    result = compare({"sdpa": sdpa, "sage": sage}, iters=20, repeats=4, baseline="sdpa",
                     guard=fake_guard(), timer=dispatching(scripts, log))
    check("first burst captured, not dropped",
          result.first_burst == {"sdpa": 3.971, "sage": 2.370}, str(result.first_burst))
    check("first burst excluded from the series",
          result.series["sage"] == [2.466, 2.466, 2.446, 2.446], str(result.series["sage"]))
    ratio = result.ratios["sage"]
    text = str(ratio)
    # attn_bench would have printed 3.971/2.370 = 1.676x. The honest answer in the ticket is
    # 1.63x [1.61-1.68]; with burst 1 gone the cluster is 1.62-1.63.
    check("the 1.676x first-burst ratio is not what is reported", "1.68" not in text, text)
    check("reported ratio is the cluster", text.startswith("1.62x faster"), text)
    # [1.61-1.62], not the ticket's [1.61-1.68]: the ticket's interval is over all five bursts and
    # this one is over the four that survive the discard. Dropping the biased burst narrows the
    # interval as well as moving the point estimate, which is the whole argument for dropping it.
    check("bracket present", "[1.61-1.62]" in text, text)

    kept = compare({"sdpa": sdpa, "sage": sage}, iters=20, repeats=5, baseline="sdpa",
                   guard=fake_guard(), timer=dispatching(scripts, []),
                   discard_first_burst=False)
    check("discard_first_burst=False keeps burst 1",
          kept.first_burst is None and kept.series["sdpa"][0] == 3.971, str(kept.series["sdpa"]))
    check("provenance says which it did",
          "NOT discarded" in provenance(kept) and "discarded as warm-up-biased" in
          provenance(result), "provenance text did not name the treatment")


# ---------------------------------------------------------------- direction rule
def test_direction_and_bracket() -> None:
    print("\na ratio below 1.0 is inverted and the direction is named (razoes-na-direcao-certa)")
    slower = Ratio(0.29, lo=0.25, hi=0.34, n=3)
    text = str(slower)
    check("never emits 0.29x", "0.29" not in text, text)
    check("emits the inverted factor with a direction", text.startswith("3.45x slower"), text)
    check("the interval inverts with it", "[2.94-4.00]" in text, text)

    faster = Ratio(4.89, lo=4.72, hi=5.06, n=3)
    check("above 1.0 keeps its direction", str(faster) == "4.89x faster [4.72-5.06]", str(faster))

    lighter = ratio_of(21.93, 6.43, better="lighter", worse="heavier", exact=True)
    check("a deterministic quantity names its own direction",
          str(lighter) == "3.41x lighter", str(lighter))

    single = Ratio(1.676, lo=1.676, hi=1.676, n=1)
    check("one burst says so instead of faking an interval",
          str(single) == "1.68x faster (1 burst, no interval)", str(single))

    print("\na Ratio cannot be formatted as a bare two-decimal number")
    check("plain f-string keeps the bracket",
          f"{faster}" == "4.89x faster [4.72-5.06]", f"{faster}")
    check("a numeric spec keeps the bracket and drops the precision",
          f"{faster:.2f}" == "4.89x faster [4.72-5.06]", f"{faster:.2f}")
    check("width and alignment survive",
          f"{faster:>30}" == "     4.89x faster [4.72-5.06]".rjust(30), repr(f"{faster:>30}"))
    check("a width+precision spec keeps the width, drops the precision",
          f"{faster:>30.2f}".strip() == "4.89x faster [4.72-5.06]", repr(f"{faster:>30.2f}"))

    empty = Ratio(float("nan"))
    check("a ratio with nothing behind it prints n/a, not a number", str(empty) == "n/a", str(empty))


# ---------------------------------------------------------------- estimators
def test_estimators_named() -> None:
    print("\nthe estimator is explicit and named in the output")

    def a():
        return None

    def b():
        return None

    scripts = {"a": [10.0, 12.0, 14.0, 100.0], "b": [5.0, 5.0, 5.0, 5.0]}
    med = compare({"a": a, "b": b}, iters=1, repeats=3, baseline="a", guard=fake_guard(),
                  timer=dispatching(scripts, []), estimator="median",
                  discard_first_burst=False)
    fastest = compare({"a": a, "b": b}, iters=1, repeats=3, baseline="a", guard=fake_guard(),
                      timer=dispatching(scripts, []), estimator="min",
                      discard_first_burst=False)
    check("median reduces to the middle burst", med.times["a"] == 12.0, str(med.times["a"]))
    check("min reduces to the fastest burst", fastest.times["a"] == 10.0, str(fastest.times["a"]))
    check("median is named", "median of" in provenance(med), provenance(med)[:60])
    check("min is named", "min of" in provenance(fastest), provenance(fastest)[:60])
    check("as_json carries the estimator and the interval",
          med.as_json()["estimator"] == "median"
          and med.as_json()["ratios"]["b"]["min"] is not None,
          str(med.as_json()["ratios"]["b"]))

    try:
        compare({"a": a}, iters=1, repeats=1, guard=fake_guard(), estimator="mode")
        check("an unknown estimator is rejected", False, "it returned instead of raising")
    except ValueError as exc:
        check("an unknown estimator is rejected", "estimator must be" in str(exc), str(exc)[:60])

    reduced, lo, hi = summarize([4.0, 2.0, 6.0], estimator="min")
    check("summarize honours its estimator", (reduced, lo, hi) == (2.0, 2.0, 6.0),
          str((reduced, lo, hi)))


# ---------------------------------------------------------------- failures
def test_failed_path() -> None:
    print("\na path that raises is recorded, and stops being timed")

    def good():
        return None

    def bad():
        return None

    calls: list[str] = []

    def timer(fn, _iters):
        calls.append(fn.__name__)
        if fn.__name__ == "bad":
            raise RuntimeError("kernel refused")
        return 1.0

    result = compare({"good": good, "bad": bad}, iters=1, repeats=2, baseline="good",
                     guard=fake_guard(), timer=timer)
    check("the failure is named", "kernel refused" in result.failed.get("bad", ""),
          str(result.failed))
    check("the failed path is timed once, not every burst",
          calls.count("bad") == 1, f"{calls.count('bad')} attempts")
    check("its time is nan, not a silent zero", result.times["bad"] != result.times["bad"],
          str(result.times["bad"]))
    check("its ratio prints n/a rather than a winner",
          str(result.ratios["bad"]) == "n/a", str(result.ratios["bad"]))


# ---------------------------------------------------------------- which cards the guard watches
def test_visible_devices() -> None:
    """`visible_nvml_indices` is pure -- no NVML, no CUDA -- so it is testable here.

    What it decides is not cosmetic: it is the only way to stop the guard refusing on a card the
    run cannot use, and it is deliberately the *same* switch that stops the run using it. MEASURED
    2026-08-22 on this bench, with the real NVML: 3090 at 0.99 GiB idle, 3080 Ti at 5.12 GiB busy.
    With CUDA_VISIBLE_DEVICES unset the guard refuses (correct: work could land on either card);
    with `CUDA_VISIBLE_DEVICES=0` it passes and still prints the 3080 Ti's 5.12 GiB as BUSY.
    """
    import os

    from _bench_guard import visible_nvml_indices

    print("\nthe guard watches every visible device, and CUDA_VISIBLE_DEVICES is what 'visible' means")
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        indices, how = visible_nvml_indices(2)
        check("unset -> every device", indices == {0, 1} and "unset" in how, f"{indices} {how}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        indices, how = visible_nvml_indices(2)
        check("=0 -> only device 0", indices == {0}, str(indices))
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        indices, how = visible_nvml_indices(2)
        check("=1 -> only device 1", indices == {1}, str(indices))
        os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-deadbeef"
        indices, how = visible_nvml_indices(2)
        check("a UUID form falls back to watching everything",
              indices == {0, 1} and "not a plain index list" in how, f"{indices} {how}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        indices, how = visible_nvml_indices(2)
        check("an out-of-range index falls back to watching everything",
              indices == {0, 1} and "no real device" in how, f"{indices} {how}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        indices, how = visible_nvml_indices(2)
        check("empty falls back to watching everything", indices == {0, 1}, str(indices))
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


def main() -> int:
    test_refuses_without_guard()
    test_visible_devices()
    test_interleaved()
    test_first_burst_discarded()
    test_direction_and_bracket()
    test_estimators_named()
    test_failed_path()
    print()
    print("=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for line in FAILURES:
            print(f"  - {line}")
    else:
        print("all checks passed")
    print("NOT covered here (needs the RTX 3090, and a lock on it):")
    print("  * that m_crossover rebuilt on compare() reproduces its old table -- run")
    print("    `python_embeded\\python.exe -s tools/m_crossover.py --repeats 3` in a GPU window")
    print("  * that wall_ms / cuda_event_ms measure what they claim; the timers here are fakes")
    print("  * that a guard which passes leads to a correct measurement. The occupancy read")
    print("    itself WAS executed against real NVML on 2026-08-22 (3090 0.99 GiB idle, 3080 Ti")
    print("    5.12 GiB busy; refused with CUDA_VISIBLE_DEVICES unset, passed with it =0), but")
    print("    nothing here re-runs that -- these checks only cover the pure index arithmetic.")
    print("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
