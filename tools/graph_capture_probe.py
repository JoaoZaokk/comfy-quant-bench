"""Does CUDA graph capture work on these ops, and how much of the host cost does it remove?

Two separate questions, and this file exists because the answers had been quoted without one.

**Whether they capture** matters to the sibling project (quantized Qwen under vLLM), which
abandoned the W4A8 tier for breaking capture in its model runner. If the bare op captures here,
whatever broke there is the integration, not the kernel -- and that is a different bug to hunt.

**How much it removes** matters here. `w4a4_breakdown.py` measures ~81 us of host time per w4a4
call at M=1 against ~35 us of GPU work: the small-M penalty is dispatch, not arithmetic, and
dispatch is exactly what a replayed graph does not pay. Whether it removes all of it or most of it
decides whether a graph is a fix or a partial one.

That second number was previously stated as "removes 83%", measured in a scratch script that no
longer exists. A number with no instrument cannot be re-checked when the stack moves under it, and
this one had already travelled to another project. This file is that instrument.

**Correctness is checked, not assumed.** A graph that replays stale or wrong results would post
the best possible timing, so every replay is compared against the eager output it is supposed to
reproduce, and a mismatch is a failure rather than a fast number.

Caveats, which belong next to whatever gets quoted from here:

  * This captures **one op** with static shapes and static input buffers. A real model has
    dynamic batch, allocator churn and control flow; capturing an op proves the op is capturable,
    not that a model is.
  * Windows/WDDM. The launch path here goes through the OS scheduler and is expensive -- the
    absolute microseconds removed are larger than they would be on Linux. The *fraction* removed
    should travel better than the microseconds, but neither has been checked on Linux from here.

    python_embeded\\python.exe -s tools/graph_capture_probe.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.quant_ops  # noqa: E402,F401  (registers the backends)
from comfy_kitchen import registry as R  # noqa: E402

REQUIRED_OPS = ("quantize_convrot_w4a4_weight", "convrot_w4a4_linear",
                "quantize_w4a8_int8_weight", "w4a8_int8_linear")
# Graph capture pays off where host cost is a large share of the call, which is small M. M=5856 is
# carried anyway so the table shows the regime where it stops mattering.
BATCHES = (1, 8, 128, 5856)
PASSES = 3
ITERS = 300


def wall_us(fn, iters: int) -> float:
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


def capture(fn):
    """Capture `fn` into a CUDA graph, or return the exception that stopped it.

    The side-stream warm-up is not optional and not a formality: allocator behaviour and any
    lazily-initialised handle inside the op must happen before capture begins, or capture records
    them as part of the graph or fails outright.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fn()
        return graph, static_out, None
    except Exception as exc:
        return None, None, exc


def main(batches=BATCHES, only=None, shape=(3840, 3840)) -> int:
    """`only` restricts which paths run, `batches` which M, `shape` the weight.

    `shape` is what separates "this op refuses to capture above M=5600" from "it refuses above
    some number of bytes". Halving K should double the M threshold if the limit is on a size in
    bytes, and leave it where it is if the limit is on M.

    Both exist to isolate a capture failure. `cudaErrorStreamCaptureInvalidated` reports that
    *something* invalidated the capture, and in a process that has already captured eleven graphs
    it does not say whether the cause is this M, this op, or the ten captures before it. Running
    one path at one M in a fresh process is what separates those.
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

    out_features, in_features = shape
    torch.manual_seed(1)
    weight = (torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
              / in_features ** 0.5).to(torch.bfloat16)
    packed4 = q_w4a4(weight, 256, 64)
    packed8 = q_w4a8(weight, group_size=16, convrot_groupsize=256, symmetric=True,
                     scale_dtype=torch.float8_e4m3fn, codebook=True,
                     codebook_tensor=None, stochastic_rounding=0)

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"weight [{out_features}, {in_features}]")
    print(f"{PASSES} passes of {ITERS} timed iterations each; the bracket is the pass min-max")
    print(f"\n{'M':>7}{'path':>8}{'eager us':>10}{'replay us':>11}{'removed':>10}{'removed %':>11}"
          f"{'match':>8}{'eager min-max':>18}")

    failures = []
    poisoned = False
    for m in batches:
        if poisoned:
            break
        x = torch.randn(m, in_features, device="cuda", dtype=torch.bfloat16)
        paths = {
            "bf16": lambda: F.linear(x, weight),
            "w4a4": lambda: lin_w4a4(x, packed4[0], packed4[1], None, 256, 64, "int4"),
            "w4a8": lambda: lin_w4a8(x, packed8[0], packed8[1], packed8[2],
                                     codebook=packed8[4], correction=packed8[3], bias=None,
                                     group_size=16, convrot_groupsize=256,
                                     out_dtype=torch.bfloat16),
        }
        if only:
            paths = {k: v for k, v in paths.items() if k in only}
        for label, call in paths.items():
            reference = call().clone()
            graph, static_out, exc = capture(call)
            if graph is None:
                # A capture failure is the headline result for that op, not a missing row -- but
                # the eager column is still measurable and still wanted. A capture threshold that
                # coincides with a step in eager time is an algorithm switch; one that does not is
                # something in the capture path alone. Skipping the timing here would throw away
                # the evidence that tells those apart.
                failures.append((m, label, exc))
                try:
                    eager_passes = [wall_us(call, ITERS) for _ in range(PASSES)]
                    eager = statistics.median(eager_passes)
                    span = f"[{min(eager_passes):.0f}-{max(eager_passes):.0f}]"
                    print(f"{m:>7}{label:>8}{eager:>10.1f}{'-':>11}{'-':>10}"
                          f"{'CAPTURE FAILED':>11}{'-':>8}{span:>18}")
                except Exception:
                    print(f"{m:>7}{label:>8}{'-':>10}{'-':>11}{'-':>10}"
                          f"{'CAPTURE FAILED':>11}{'-':>8}{'eager unusable':>18}")
                print(f"         {type(exc).__name__}: {str(exc)[:80]}")
                # A failed capture leaves the CUDA context poisoned: every later call in this
                # process raises the same cascaded error, so any row printed after this one would
                # be an artifact of the first failure rather than a result. Measured -- a run of
                # 5600,5664,5680,5700,5856 died inside the 5700 row. Stop here and say why; to
                # test more M values, run this file once per M.
                print("\nCUDA context is poisoned after a capture failure. Stopping: later rows "
                      "in this process would be cascades, not measurements. Re-run one M per "
                      "process to test further.")
                poisoned = True
                break

            graph.replay()
            torch.cuda.synchronize()
            # Same input buffer, so the replay must reproduce the eager result bit for bit. A
            # graph that replays something else would otherwise report the best timing in the
            # table purely by not doing the work.
            match = torch.equal(static_out, reference)

            eager_passes, replay_passes = [], []
            for _ in range(PASSES):
                eager_passes.append(wall_us(call, ITERS))
                replay_passes.append(wall_us(graph.replay, ITERS))
            eager = statistics.median(eager_passes)
            replay = statistics.median(replay_passes)
            removed = eager - replay
            print(f"{m:>7}{label:>8}{eager:>10.1f}{replay:>11.1f}{removed:>10.1f}"
                  f"{removed / eager * 100:>10.1f}%{'yes' if match else 'NO':>8}"
                  f"{f'[{min(eager_passes):.0f}-{max(eager_passes):.0f}]':>18}")
            if not match:
                failures.append((m, label, "replay output != eager output"))
            del graph, static_out
            torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()

    print("\nWhat this does and does not show:")
    print("  * One op, static shapes, static input buffer. Capturing an op is not capturing a")
    print("    model -- dynamic batch, allocator churn and control flow are all absent here.")
    print("  * Windows/WDDM, where a launch crosses the OS scheduler and costs more than on")
    print("    Linux. The microseconds removed do not transfer; the fraction should transfer")
    print("    better, but that has not been checked on Linux from this machine.")
    print("  * 'removed' is host dispatch, not GPU work. It cannot exceed the host share that")
    print("    w4a4_breakdown.py measures, and at large M there is little of it left to remove.")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for m, label, exc in failures:
            print(f"  M={m} {label}: {exc}")
        return 1
    print("\nNo capture failures and every replay matched its eager output.")
    return 0


if __name__ == "__main__":
    import argparse

    from _bench_guard import BenchGuard

    _parser = argparse.ArgumentParser(description="CUDA graph capture on the quantized Linear ops")
    _parser.add_argument("--batches", default=None,
                         help="comma-separated M values (default 1,8,128,5856)")
    _parser.add_argument("--paths", default=None,
                         help="comma-separated subset of bf16,w4a4,w4a8")
    _parser.add_argument("--shape", default="3840,3840",
                         help="out_features,in_features of the weight (default 3840,3840)")
    _args = _parser.parse_args()
    _shape = tuple(int(v) for v in _args.shape.split(","))
    if len(_shape) != 2:
        print("--shape takes exactly out_features,in_features")
        raise SystemExit(2)
    _batches = (tuple(int(v) for v in _args.batches.split(",")) if _args.batches else BATCHES)
    _only = set(_args.paths.split(",")) if _args.paths else None
    if _only and not _only <= {"bf16", "w4a4", "w4a8"}:
        print(f"unknown path(s): {sorted(_only - {'bf16', 'w4a4', 'w4a8'})}")
        raise SystemExit(2)

    with BenchGuard("graph_capture_probe") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main(batches=_batches, only=_only, shape=_shape))
