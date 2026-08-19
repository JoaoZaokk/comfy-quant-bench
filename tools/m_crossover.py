"""At what batch size does computing in 4 bits start beating dequantizing to 16?

This is the decode-versus-prefill question, isolated to one Linear. An LLM spends most of its life
in decode with M = 1 to 8 tokens, where the matmul is bound by how fast the weight can be read;
the activation is a speck and quantizing it buys nothing while still costing a quantize kernel per
forward. Diffusion never has that regime at all -- Z-Image calls every Linear with M = 5856 on
every one of its 240 attention blocks per 8-step run, so it sits permanently at the end of the
curve where arithmetic dominates.

So the interesting number is not "is W4A4 faster", it is *where the two lines cross*. This sweeps
M over three decades on one weight and times:

    bf16      F.linear on a bf16 weight                          arithmetic in 16 bits
    w4a4      convrot_w4a4_linear(linear_dtype="int4")           int4 weights AND activations
    w4a4/a8   convrot_w4a4_linear(linear_dtype="int8")           int8 activations on the W4A4
                                                                 weight layout
    w4a8      w4a8_int8_linear                                   the separate W4A8 tier, with its
                                                                 own qdata + s_rel + s_channel

and reports error against an fp32 reference at the same time, because a format that wins on speed
and loses the answer has not won.

The third and fourth rows are not the same thing, and conflating them is easy: an activation-dtype
switch on the W4A4 weight layout is not the W4A8 tier, which quantizes the weight differently and
carries a Lloyd-Max codebook. A sibling project measuring quantized Qwen serving hit exactly this
-- its `ACT_DTYPE=int8` knob is the third row, and the fourth was abandoned there for breaking
CUDA-graph capture in vLLM's model runner. A bare kernel sweep has no model runner and no graph
capture, so both are measurable here.

**Read the bf16 column with care.** It reads a full bf16 weight from memory, so at small M it
moves twice the bytes a weight-only-int4 kernel like Marlin would. That makes it pessimistic
exactly where memory traffic decides the race, so the crossover measured here lands *earlier*
than it would against Marlin. The `bf16 bytes` column makes the size of that handicap explicit
rather than leaving it as a footnote. This measures comfy_kitchen on this GPU; it is not a Marlin
benchmark and cannot be quoted as one.

    python_embeded\\python.exe -s tools/m_crossover.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
# The embedded interpreter ships a `python313._pth`, which suppresses the usual script-directory
# entry, so a sibling module in `tools/` is not importable without this. The other three GPU tools
# already carried the line; this one did not, and `from _bench_guard import BenchGuard` died with
# ModuleNotFoundError on the first real execution -- the guard had been written but never run.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.quant_ops  # noqa: E402,F401  (registers the backends)
from comfy_kitchen import registry as R  # noqa: E402

REQUIRED_OPS = ("quantize_convrot_w4a4_weight", "convrot_w4a4_linear",
                "quantize_w4a8_int8_weight", "w4a8_int8_linear")
# 1..8 is the decode regime an LLM lives in; 5856 is what Z-Image hands every Linear.
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5856, 8192)


def timed(fn, iters: int) -> float:
    for _ in range(3):
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


def relative(reference: torch.Tensor, got: torch.Tensor) -> float:
    return float((reference.float() - got.float()).norm()
                 / reference.float().norm().clamp(min=1e-12))


def main(reverse: bool = False, repeats: int = 3) -> int:
    """Sweep the M column. `reverse` walks it from 8192 down to 1 instead of up.

    `repeats` times every path that many times, interleaved, and reports the ratio's own min-max
    alongside the median. It defaults to 3 because 1 was measured to be misleading: three
    single-repeat runs on an idle, locked 3090 disagreed by up to 1.4x at the same M and the same
    shape (M=2048, weight [3840, 3840]: 4.47x, 4.55x, 3.26x), and the crossover for weight
    [10240, 3840] landed between 64 and 128 in two runs and between 128 and 256 in the third.
    A two-decimal number printed from one run is a precision this instrument does not have.

    The sweep is not order-free, and the default order is the one that flatters the right-hand
    end: M climbs monotonically, so every large-M row is measured on a card that has been under
    load for the whole table, while M=1 is measured on a cold one. Clocks, and the allocator's
    fragmentation, both move over that span. If the crossover is a property of the kernels it
    lands in the same place walking down; if it moves, part of what the ascending table called a
    crossover was the machine warming up. Run both and compare -- a single direction cannot tell
    the two apart.
    """
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1

    for name in REQUIRED_OPS:
        impl = R.get_implementation(name)
        module = getattr(impl, "__module__", "?")
        if "comfy_kitchen.backends.cuda" not in module:
            print(f"{name} resolves to {module}, not the CUDA backend; refusing")
            return 1

    q_w4a4 = R.get_implementation("quantize_convrot_w4a4_weight")
    lin_w4a4 = R.get_implementation("convrot_w4a4_linear")
    q_w4a8 = R.get_implementation("quantize_w4a8_int8_weight")
    lin_w4a8 = R.get_implementation("w4a8_int8_linear")

    order = tuple(reversed(BATCHES)) if reverse else BATCHES
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    # Printed, not implied: these two tables are only comparable to each other if the reader can
    # see which direction produced each one, and the file gets pasted without its command line.
    print(f"M order: {'descending 8192 -> 1 (--reverse)' if reverse else 'ascending 1 -> 8192'}")

    for out_features, in_features in ((3840, 3840), (10240, 3840)):
        torch.manual_seed(1)
        weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
                  / in_features ** 0.5).to(torch.bfloat16)
        packed4 = q_w4a4(weight, 256, 64)
        packed8 = q_w4a8(weight, group_size=16, convrot_groupsize=256, symmetric=True,
                         scale_dtype=torch.float8_e4m3fn, codebook=True,
                         codebook_tensor=None, stochastic_rounding=0)

        bf16_bytes = weight.numel() * 2
        print(f"\nweight [{out_features}, {in_features}]  "
              f"bf16 {bf16_bytes / 2**20:.0f} MiB read per call, "
              f"int4 {bf16_bytes / 4 / 2**20:.0f} MiB")
        print(f"{'M':>7}{'bf16':>9}{'w4a4':>9}{'w4a4/a8':>9}{'w4a8':>9}"
              f"{'best vs bf16 [min-max]':>34}{'w4a4 err':>10}{'w4a4/a8':>9}{'w4a8 err':>10}")

        for m in order:
            torch.manual_seed(2)
            x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)
            reference = F.linear(x.float(), weight.float())
            iters = 50 if m <= 512 else 15

            paths = {
                "bf16": lambda: F.linear(x, weight),
                "w4a4": lambda: lin_w4a4(x, packed4[0], packed4[1], None, 256, 64, "int4"),
                "w4a4/a8": lambda: lin_w4a4(x, packed4[0], packed4[1], None, 256, 64, "int8"),
                "w4a8": lambda: lin_w4a8(x, packed8[0], packed8[1], packed8[2],
                                         codebook=packed8[4], correction=packed8[3], bias=None,
                                         group_size=16, convrot_groupsize=256,
                                         out_dtype=torch.bfloat16),
            }
            # Repeats are **interleaved**, not batched per path: all four paths are timed once,
            # then all four again. Timing one path to completion before starting the next lets
            # any drift over the burst -- clock boost decaying, another process arriving -- land
            # entirely on whichever path happened to be running, which is exactly how a ratio
            # picks up a bias that no single median reveals. Interleaved, drift hits every path
            # alike and cancels in the ratio.
            reps, errors, falhas = {k: [] for k in paths}, {}, []
            for label, call in paths.items():
                try:
                    errors[label] = relative(reference, call())
                except Exception as exc:
                    errors[label] = float("nan")
                    falhas.append(label)
                    # Reported at every M, not only the first: an earlier version printed the
                    # exception only for BATCHES[0], so a path that failed from M=128 upward
                    # turned into a silent nan column and the verdict still named a winner.
                    print(f"   M={m} {label} raised: {type(exc).__name__}: {str(exc)[:60]}")
            for _ in range(repeats):
                for label, call in paths.items():
                    if label in falhas:
                        reps[label].append(float("nan"))
                        continue
                    try:
                        reps[label].append(timed(call, iters))
                    except Exception:
                        reps[label].append(float("nan"))
                        falhas.append(label)

            times = {k: (statistics.median(v) if all(s == s for s in v) else float("nan"))
                     for k, v in reps.items()}
            quant = {k: v for k, v in times.items() if k != "bf16" and v == v}
            if not quant:
                verdict = "todos falharam"
            elif times["bf16"] != times["bf16"]:
                # No baseline, so "N times faster than bf16" has nothing to be faster than.
                verdict = f"{min(quant, key=quant.get)} (sem bf16)"
            else:
                best = min(quant, key=quant.get)
                # Paired per repeat, so the spread shown is the spread of the *ratio*, which is
                # the quantity that gets quoted. Three runs of this file on an idle 3090 put the
                # same M at 3.26x and 4.55x, so a bare two-decimal ratio claims a precision the
                # measurement does not have, and it travels out of here as if it did.
                pairs = [b / q for b, q in zip(reps["bf16"], reps[best])
                         if b == b and q == q and q > 0]
                ratio = statistics.median(pairs) if pairs else times["bf16"] / quant[best]
                if ratio >= 1:
                    verdict = f"{best} {ratio:.2f}x"
                    lo, hi = min(pairs), max(pairs)
                else:
                    # Never as 0.29x: the direction of a ratio below 1 is the thing readers
                    # invert wrongly, so it is stated as the slower factor instead.
                    verdict = f"{best} {1 / ratio:.2f}x slower"
                    lo, hi = 1 / max(pairs), 1 / min(pairs)
                if len(pairs) > 1:
                    verdict += f" [{lo:.2f}-{hi:.2f}]"
                # Naming a winner over a field that lost entrants reads as a complete comparison.
                if falhas:
                    verdict += f" (de {len(quant)})"
            print(f"{m:>7}{times['bf16']:>9.3f}{times['w4a4']:>9.3f}"
                  f"{times['w4a4/a8']:>9.3f}{times['w4a8']:>9.3f}{verdict:>34}"
                  f"{errors['w4a4']:>10.4f}{errors['w4a4/a8']:>9.4f}{errors['w4a8']:>10.4f}")
            del x, reference
            torch.cuda.empty_cache()

        del weight, packed4, packed8
        torch.cuda.empty_cache()

    print("\nThe crossover is where the 'vs bf16' columns flip from slower to faster. Below it the")
    print("matmul is bound by reading the weight and 4-bit arithmetic buys nothing while still")
    print("paying to quantize the activation; above it arithmetic dominates and the int4 tensor")
    print("core is the whole point. An LLM in decode sits at M=1..8. Z-Image hands every Linear")
    print("M=5856 on every step and never leaves the right-hand end.")
    print("\nThe bf16 baseline reads a full bf16 weight, twice what a weight-only-int4 kernel")
    print("moves, so it is handicapped at small M and the real crossover against Marlin is")
    print("further right than this table shows.")
    return 0


if __name__ == "__main__":
    # The lock and the occupancy check both live in _bench_guard now. This file previously had a
    # private copy of the occupancy check with the threshold test outside the try (leaking the
    # NVML handle and continuing as if idle) and took no lock at all, despite gpu_lock's own
    # docstring using this file as its example.
    from _bench_guard import BenchGuard

    import argparse

    # argparse rather than an `in sys.argv` test: silently ignoring a misspelled flag would print
    # an ascending table labelled as the descending counterproof, or one repeat labelled as three.
    _parser = argparse.ArgumentParser(description="M sweep: bf16 vs w4a4 vs w4a4/a8 vs w4a8")
    _parser.add_argument("--reverse", action="store_true",
                         help="walk M from 8192 down to 1 (order-effect counterproof)")
    _parser.add_argument("--repeats", type=int, default=3,
                         help="interleaved timing repeats per M; 1 reproduces the old, "
                              "misleadingly precise single-shot table")
    _args = _parser.parse_args()
    if _args.repeats < 1:
        print("--repeats must be at least 1")
        raise SystemExit(2)

    with BenchGuard("m_crossover") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main(reverse=_args.reverse, repeats=_args.repeats))
