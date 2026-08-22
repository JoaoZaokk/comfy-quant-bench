"""Time the installed attention backends on this machine at diffusion-like shapes.

Import success and a numeric check say a backend works; they say nothing about whether it is
worth using. This measures it.

**This file used to report one burst, as a mean, with no lock.** All three were wrong in ways that
were measured rather than argued, on 2026-08-22 on the 3090 with the lock held
(`tools/probe_groupsize_and_spread.py`, five interleaved bursts of SDPA against SageAttention at
B=2 H=24 S=4096 D=64 bf16, median of 20 per-iteration CUDA-event timings per burst):

    sdpa   3.971  4.007  3.972  3.970  3.974     median 3.972   spread 1.01x
    sage   2.370  2.466  2.466  2.446  2.446     median 2.446   spread 1.04x
    ratio  1.676x 1.625x 1.610x 1.623x 1.625x

The spread on the ratio is 1.04x, so the 1.4x run-to-run disagreement `m_crossover.py` records is
a GEMM result and does **not** transfer to attention -- do not quote 1.4x here. What does transfer
is worse than noise: the outlier is the **first** burst, 1.676x against a 1.610-1.625 cluster, and
five warm-up iterations did not settle it. This file ran exactly one burst -- the first one -- so
it did not draw randomly from that distribution, it reported the draw that is systematically ~3%
high. Repeating the tool could not average that away. `_timing.compare()` discards the first burst
and prints what it discarded.

It printed `sage_speedup` to three decimals into a JSON dump, too. The third decimal was noise
under any reading and the second was the bias; the JSON now carries the interval and the burst
count with every ratio, so the number cannot be pasted into another session without its condition.

    python_embeded\\python.exe -s tools/attn_bench.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _timing import compare, cuda_event_ms, provenance  # noqa: E402

DEVICE = "cuda"
ITERS = 20

# (batch, heads, seq, head_dim) — shapes typical of image/video DiT attention
SHAPES = [
    (1, 24, 1024, 64),
    (1, 24, 4096, 64),
    (1, 24, 16384, 64),
    (1, 16, 32768, 128),
]


def main(repeats: int = 3) -> int:
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device      : {name}  sm{capability[0]}{capability[1]}")
    print(f"torch       : {torch.__version__}  cuda {torch.version.cuda}")

    try:
        import sageattention
        from sageattention import sageattn
        print(f"sageattention: {getattr(sageattention, '__version__', 'unknown')}")
    except Exception as error:
        print(f"sageattention: UNAVAILABLE ({error})")
        sageattn = None

    try:
        from flash_attn import flash_attn_func
        import flash_attn
        print(f"flash_attn  : {getattr(flash_attn, '__version__', 'unknown')}")
    except Exception as error:
        print(f"flash_attn  : UNAVAILABLE ({error})")
        flash_attn_func = None

    print()
    header = (f"{'shape (B,H,S,D)':<24} {'SDPA ms':>10} {'Sage ms':>10} "
              f"{'Sage vs SDPA':>28} {'Flash ms':>10} {'Flash vs SDPA':>28}")
    print(header)
    print("-" * len(header))

    results = []
    last_result = None
    for batch, heads, seq, dim in SHAPES:
        torch.manual_seed(0)
        q = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        paths = {"sdpa": lambda: F.scaled_dot_product_attention(q, k, v)}
        if sageattn is not None:
            paths["sage"] = lambda: sageattn(q, k, v)
        qf = kf = vf = None
        if flash_attn_func is not None:
            # Hoisted out of the timed callable deliberately. Charging flash three `.contiguous()`
            # copies per call that SDPA and Sage never pay is not modelling anything real -- a
            # model keeps its tensors in the layout its attention wants -- and it is the same
            # mistake `attn_dtype_ab.py:82-87` records having made and fixed.
            qf, kf, vf = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            paths["flash"] = lambda: flash_attn_func(qf, kf, vf)

        # This file took no lock at all, which was worse for the sibling session than for this
        # one: with no lock file present, their `Assert-GpuLock` would have been *granted*
        # mid-burst. `compare()` refuses to time anything unguarded, so the guard cannot be
        # forgotten again; `owner=` is the fallback for someone calling `main()` directly, and
        # `__main__` below holds one guard across all four shapes rather than taking and
        # releasing the shared lock file once per shape.
        result = compare(paths, iters=ITERS, repeats=repeats, baseline="sdpa",
                         timer=cuda_event_ms, owner="comfy_portable:attn_bench")
        last_result = result
        for label, why in result.failed.items():
            print(f"  {label} failed at {(batch, heads, seq, dim)}: {why}")
        if result.times["sdpa"] != result.times["sdpa"]:
            print(f"{str((batch, heads, seq, dim)):<24} SDPA failed, nothing to compare against")
            del q, k, v, qf, kf, vf
            torch.cuda.empty_cache()
            continue

        row = {"shape": [batch, heads, seq, dim], **result.as_json()}
        results.append(row)
        sage_ms = result.times.get("sage", float("nan"))
        flash_ms = result.times.get("flash", float("nan"))
        print(
            f"{str((batch, heads, seq, dim)):<24} {result.times['sdpa']:>10.3f} "
            f"{(f'{sage_ms:.3f}' if sage_ms == sage_ms else 'n/a'):>10} "
            f"{result.ratios.get('sage', 'n/a'):>28} "
            f"{(f'{flash_ms:.3f}' if flash_ms == flash_ms else 'n/a'):>10} "
            f"{result.ratios.get('flash', 'n/a'):>28}"
        )
        del q, k, v, qf, kf, vf
        torch.cuda.empty_cache()

    print()
    if last_result is not None:
        print(provenance(last_result))
        print("(the discarded burst quoted above is the last shape's; each shape discards its own)")
    print()
    print(json.dumps({"device": name, "sm": f"{capability[0]}{capability[1]}",
                      "results": results}, indent=2))
    print("\nNOT covered: this runs fp16 only, at head_dim 64 for three of the four shapes, and")
    print("the SDPA baseline is `F.scaled_dot_product_attention` with no `sdpa_kernel` pinned --")
    print("torch is free to dispatch it to flash, in which case the Flash column is comparing")
    print("flash against flash. Which shapes that happens on has not been measured here.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved bursts KEPT per shape; one more is run and discarded as "
                             "warm-up-biased. The default costs 4x the GPU time the old "
                             "single-burst version did -- that is the price of the interval")
    args = parser.parse_args()
    if args.repeats < 1:
        print("--repeats must be at least 1")
        raise SystemExit(2)

    from _bench_guard import BenchGuard

    with BenchGuard("comfy_portable:attn_bench") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main(repeats=args.repeats))
