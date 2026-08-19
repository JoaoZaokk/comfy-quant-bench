"""Where does the W4A4 time actually go at M=1, and where at M=5856?

tools/m_crossover.py answers *that* small M hurts -- W4A4 is 1.8x slower than a bf16 matmul at
M=1 and 4.6x faster at M=5856. It does not answer *why*, and "the GPU has no compute to give at
M=1" and "the bookkeeping costs more than the multiply" are different diagnoses with different
consequences. The first says nothing can be done; the second says the fixed cost is the target.

So this profiles the CUDA kernels inside one call and names them. No guessing about which stage
dominates: the profiler reports every launch the call makes, with its own duration.

Read it as: if the quantize/pack kernels dwarf the MMA at M=1, the mystery is over -- the
administration exceeds the arithmetic, and it is fixed cost that a larger M amortizes away. That
is a much more useful sentence than "int4 loses in decode".

    python_embeded\\python.exe -s tools/w4a4_breakdown.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import comfy.quant_ops  # noqa: E402,F401
from comfy_kitchen import registry as R  # noqa: E402
from _bench_guard import BenchGuard  # noqa: E402

REPEATS = 30
# How many times the whole wall pass is repeated. `host us` is a difference of two separately
# measured quantities, so it inherits both their errors, and it is the number from this file that
# has already been quoted elsewhere. Three passes, and the spread is printed rather than hidden:
# m_crossover was measured on 2026-08-19 to swing 1.4x between consecutive single-shot runs on an
# idle card, and nothing about that is specific to that file.
WALL_PASSES = 3


def kernel_table(fn, label: str) -> None:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPEATS):
            fn()
        torch.cuda.synchronize()

    totals = defaultdict(float)
    counts = defaultdict(int)
    for event in prof.key_averages():
        # device_time is the sum over all occurrences, in microseconds.
        device_us = getattr(event, "device_time_total", 0.0) or 0.0
        if device_us <= 0:
            continue
        totals[event.key] += device_us
        counts[event.key] += event.count

    grand = sum(totals.values())
    if grand <= 0:
        print(f"  {label}: profiler reported no device time")
        return
    print(f"  {label}: {grand / REPEATS:.1f} us per call across "
          f"{sum(counts.values()) // REPEATS} kernel launches")
    for name in sorted(totals, key=lambda k: -totals[k])[:8]:
        share = totals[name] / grand * 100
        print(f"     {totals[name] / REPEATS:>8.1f} us  {share:>5.1f}%  "
              f"x{counts[name] // REPEATS:<3} {name[:74]}")


def wall_us(fn, iters: int) -> float:
    """Median wall time per call, measured with no profiler attached.

    Two traps, both of which this file fell into and reported wrong numbers from:

    1. A wall figure taken inside `profile()` folds kineto's per-launch cost into the result. So
       the profiler pass and the wall pass are separate.
    2. Worse and less obvious: running the wall pass in a process where the profiler has *already*
       run inflates it too. Measured on this host, w4a4 at M=1 reads 196 us after a profiling pass
       and 112 us in a quiet process -- a 1.7x error, in the direction that made host overhead look
       twice as large as it is. `overhead_table` therefore runs before any profiling.

    The distribution has a long right tail (p10 110 us, p90 ~165 us, max ~240 us at M=1), so 50
    samples is not enough for a stable median; the default here is deliberately larger.
    """
    import statistics
    import time
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6


def device_us(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = sum((getattr(e, "device_time_total", 0.0) or 0.0)
                for e in prof.key_averages()
                if (getattr(e, "device_time_total", 0.0) or 0.0) > 0)
    return total / iters


def overhead_table(paths: dict, batches) -> None:
    """Wall time minus GPU time: what the host spends getting each call to the card.

    This is the column that decides whether a small-M penalty is a kernel problem or a dispatch
    problem, and they have opposite fixes. A kernel problem needs a better kernel. A dispatch
    problem disappears under CUDA graph capture, which replays the launches with no host work.
    """
    # Every wall measurement first, across all M and all paths, before the profiler is attached
    # even once. Interleaving them contaminated the later rows: profiling path A inflated the wall
    # figure for path B in the same process.
    walls = defaultdict(list)
    for _ in range(WALL_PASSES):
        for m, make in batches:
            x = make()
            for label, call in paths.items():
                walls[(m, label)].append(wall_us(call(x), 300 if m <= 512 else 30))
            del x
            torch.cuda.empty_cache()

    print(f"\n{'M':>7}{'path':>10}{'wall us':>10}{'gpu us':>10}{'host us':>10}{'host %':>9}"
          f"{'wall min-max':>18}")
    for m, make in batches:
        x = make()
        for label, call in paths.items():
            passes = walls[(m, label)]
            w = statistics.median(passes)
            d = device_us(call(x), 50 if m <= 512 else 15)
            # The spread shown is the wall pass's, not the host figure's: `gpu us` comes from a
            # single profiled pass, so a host number cannot be given an honest interval here. It
            # is a floor on the uncertainty, not the whole of it.
            span = f"[{min(passes):.0f}-{max(passes):.0f}]" if len(passes) > 1 else ""
            print(f"{m:>7}{label:>10}{w:>10.1f}{d:>10.1f}{w - d:>10.1f}"
                  f"{(w - d) / w * 100:>8.1f}%{span:>18}")
        del x
        torch.cuda.empty_cache()


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    # The occupancy check used to live here, inline, and it failed **open**: NVML unavailable
    # printed a warning and profiled anyway. That is the wrong default on a machine where a
    # sibling project holds most of the card, and it is the reason `_bench_guard` exists. This
    # file kept its private copy after the others were migrated; the guard in `__main__` is now
    # the only one, and it refuses when occupancy cannot be read at all.

    q_w4a4 = R.get_implementation("quantize_convrot_w4a4_weight")
    lin_w4a4 = R.get_implementation("convrot_w4a4_linear")
    q_w4a8 = R.get_implementation("quantize_w4a8_int8_weight")
    lin_w4a8 = R.get_implementation("w4a8_int8_linear")
    for name, impl in (("convrot_w4a4_linear", lin_w4a4), ("w4a8_int8_linear", lin_w4a8)):
        module = getattr(impl, "__module__", "?")
        if "comfy_kitchen.backends.cuda" not in module:
            print(f"{name} resolves to {module}, not CUDA; refusing")
            return 1

    out_features, in_features = 3840, 3840
    torch.manual_seed(1)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
              / in_features ** 0.5).to(torch.bfloat16)
    packed4 = q_w4a4(weight, 256, 64)
    packed8 = q_w4a8(weight, group_size=16, convrot_groupsize=256, symmetric=True,
                     scale_dtype=torch.float8_e4m3fn, codebook=True,
                     codebook_tensor=None, stochastic_rounding=0)

    print(f"{torch.cuda.get_device_name(0)}, weight [{out_features}, {in_features}]")

    # Wall/host accounting FIRST: it must run before the profiler has touched this process.
    paths = {
        "bf16": lambda x: (lambda: F.linear(x, weight)),
        "w4a4": lambda x: (lambda: lin_w4a4(x, packed4[0], packed4[1], None, 256, 64, "int4")),
        "w4a8": lambda x: (lambda: lin_w4a8(x, packed8[0], packed8[1], packed8[2],
                                            codebook=packed8[4], correction=packed8[3],
                                            bias=None, group_size=16, convrot_groupsize=256,
                                            out_dtype=torch.bfloat16)),
    }
    overhead_table(paths, [
        (m, (lambda m=m: torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)))
        for m in (1, 8, 128, 5856)
    ])
    print("\n'host us' is wall time the GPU was not busy: python dispatch, argument checking,")
    print("kernel launch. It is roughly constant per call, so at M=1 it can exceed the GPU work")
    print("entirely -- and unlike a slow kernel, CUDA graph capture removes most of it.")

    for m in (1, 8, 128, 5856):
        x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)
        print(f"\nM = {m}")
        kernel_table(lambda: F.linear(x, weight), "bf16")
        kernel_table(lambda: lin_w4a4(x, packed4[0], packed4[1], None, 256, 64, "int4"), "w4a4")
        kernel_table(lambda: lin_w4a8(x, packed8[0], packed8[1], packed8[2],
                                      codebook=packed8[4], correction=packed8[3], bias=None,
                                      group_size=16, convrot_groupsize=256,
                                      out_dtype=torch.bfloat16), "w4a8")
        del x
        torch.cuda.empty_cache()

    print("\nThe GEMM kernel is the one whose share grows with M. Anything whose per-call time is")
    print("flat from M=1 to M=5856 is fixed cost -- rotation, activation quantization, packing,")
    print("epilogue -- and that is what makes small M expensive, not a shortage of compute.")
    return 0


if __name__ == "__main__":
    with BenchGuard("w4a4_breakdown") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
