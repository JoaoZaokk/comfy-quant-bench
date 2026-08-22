"""Does fp16 buy anything over bf16 in the attention kernels installed here?

The claim under test: bf16's wider exponent has to be "paid for" somewhere, so attention
implementations support less of it -- fewer paths, fewer features, slower -- than fp16.

The bit layout already answers part of it (bf16 is 1+8+7, fp16 is 1+5+10; the range is paid for
in mantissa, inside the same 16 bits, not in hardware elsewhere). What it cannot answer is whether
a given *kernel* implements both equally, which is a software question and differs per library.
So each backend is run twice, same shapes and same seed, and reported on three axes:

    runs?     an exception here is the strongest possible answer
    time      median over interleaved bursts, both dtypes on the same shapes, first burst dropped
    error     mean |difference| against an fp32 SDPA reference computed from the same inputs

The timing goes through `_timing.compare()`. This file used to run fp16 to completion and only
then bf16 -- the non-interleaved order `m_crossover.py:158-162` argues against at length, where
any drift over the run lands entirely on whichever arm happened to be second. It also reported a
single burst; MEASURED 2026-08-22 on the 3090, the first burst of an attention A/B reads ~3% high
in the same direction every time, so it is now run and discarded rather than reported.

The reference is fp32 for both, so the two error columns are comparable to each other. Note this
measures the kernel, not the format: fp16 starting from fp32 inputs already has 3 more mantissa
bits, so it *should* be more accurate here. What matters is whether the gap is bigger than that
baseline, which the `sdpa` row gives.

    python_embeded\\python.exe -s tools/attn_dtype_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from _timing import compare, provenance, wall_ms  # noqa: E402

DEV = "cuda"
B, H, S, D = 2, 16, 4096, 64   # a real image-model attention shape, not a toy one
ITERS = 20


def reference_inputs():
    """fp32 inputs held on the CPU, plus the fp32 SDPA reference, also on the CPU.

    Both stay off the device on purpose. An earlier version kept the three fp32 tensors (32 MiB
    each at this shape) and the fp32 output resident while timing, and reported bf16 as 2.65x
    slower than fp16 for `sdpa` -- an artefact of allocator pressure, not of the kernel. Forcing
    each SDPA backend by hand showed fp16 and bf16 within 1% of each other on all of them.
    """
    torch.manual_seed(0)
    base = [torch.randn(B, H, S, D, device=DEV, dtype=torch.float32) for _ in range(3)]
    reference = F.scaled_dot_product_attention(*base).cpu()
    base = [t.cpu() for t in base]
    torch.cuda.empty_cache()
    return base, reference


def backends():
    """Each entry returns a callable taking (q, k, v) in (B, H, S, D) and returning the same."""
    out = {}

    out["sdpa"] = lambda q, k, v: F.scaled_dot_product_attention(q, k, v)

    try:
        from sageattention import sageattn
        out["sageattention"] = lambda q, k, v: sageattn(q, k, v, tensor_layout="HND")
    except Exception as exc:
        print(f"sageattention unavailable: {exc!r}"[:160])

    # The layout conversion is hoisted out of the timed callable for the two backends that need
    # it. Leaving it inside charged flash and xformers three `.contiguous()` copies (48 MiB per
    # call at B=2,H=16,S=4096,D=64) that sdpa, sage and sparge never paid -- and since that cost
    # is identical in fp16 and bf16, it *shrank* the relative bf16 penalty this file exists to
    # measure. A real model keeps its tensors in the layout its attention wants, so charging the
    # conversion per call was not modelling anything real either.
    try:
        from flash_attn import flash_attn_func

        def flash(q, k, v, _pre=None):
            qf, kf, vf = _pre
            return flash_attn_func(qf, kf, vf).transpose(1, 2)
        flash.needs_bshd = True
        out["flash_attn"] = flash
    except Exception as exc:
        print(f"flash_attn unavailable: {exc!r}"[:160])

    try:
        import xformers.ops as xops

        def xf(q, k, v, _pre=None):
            qx, kx, vx = _pre
            return xops.memory_efficient_attention(qx, kx, vx).transpose(1, 2)
        xf.needs_bshd = True
        out["xformers"] = xf
    except Exception as exc:
        print(f"xformers unavailable: {exc!r}"[:160])

    try:
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

        def sparge(q, k, v):
            # smooth_k=True: ComfyUI passes False and upstream then reads an unassigned `km`
            # (SpargeAttn#121, patched locally). Not the subject here, so take the working path.
            return spas_sage2_attn_meansim_topk_cuda(
                q, k, v, tensor_layout="HND", is_causal=False, smooth_k=True,
                topk=0.5, output_dtype=q.dtype)
        out["spargeattn"] = sparge
    except Exception as exc:
        print(f"spargeattn unavailable: {exc!r}"[:160])

    return out


def main(repeats: int = 3) -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    print(f"device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)}, "
          f"torch {torch.__version__}")
    print(f"shape B={B} H={H} S={S} D={D}, {ITERS} timed iterations per burst\n")

    impls = backends()
    base, reference = reference_inputs()
    print(f"{'backend':<16}{'fp16 ms':>10}{'bf16 ms':>10}{'bf16 vs fp16':>30}"
          f"{'fp16 err':>11}{'bf16 err':>11}")
    last_result = None
    for name, fn in impls.items():
        # Both dtypes are built and held for the whole comparison, so `compare()` can interleave
        # them. This file used to time fp16 to completion and only then start bf16 -- the exact
        # non-interleaved order `m_crossover.py` argues against, and the order that lets any drift
        # over the run land entirely on bf16.
        #
        # The residency question this raises is the one this file already has a scar from: keeping
        # the three fp32 tensors resident while timing once reported bf16 as 2.65x slower than
        # fp16, an allocator artefact (see `reference_inputs`). Holding both 16-bit copies is 96
        # MiB at this shape, 192 with the pre-transposed pair -- and, unlike the fp32 case, the
        # residency is now *identical* for both arms rather than growing between them, which is
        # the property that makes an interleaved A/B fair. NOT verified on the card: if a backend
        # suddenly reads far off its old figure here, this is the first thing to suspect.
        errors, calls, broken = {}, {}, {}
        held = []
        for label, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            q, k, v = (t.to(device=DEV, dtype=dtype) for t in base)
            pre = ([t.transpose(1, 2).contiguous() for t in (q, k, v)]
                   if getattr(fn, "needs_bshd", False) else None)
            held.append((q, k, v, pre))
            call = ((lambda f=fn, a=q, b=k, c=v, p=pre: f(a, b, c, p)) if pre is not None
                    else (lambda f=fn, a=q, b=k, c=v: f(a, b, c)))
            calls[label] = call
            try:
                got = call().float().cpu()
                if not torch.isfinite(got).all():
                    # An error reported as mean(abs(.)) over a tensor holding NaN prints "nan"
                    # in one column while the timing columns and the verdict look completely
                    # normal. A backend that only breaks in bf16 would slip through.
                    raise RuntimeError("output contains inf or nan")
                errors[label] = float((got - reference).abs().mean())
                del got
                torch.cuda.empty_cache()
            except Exception as exc:
                broken[label] = repr(exc)[:70]
                errors[label] = float("nan")

        if broken:
            which = ", ".join(sorted(broken))
            print(f"{name:<16}{'FAILED on ' + which:>10}  {list(broken.values())[0]}")
            del held, calls
            torch.cuda.empty_cache()
            continue

        result = compare(calls, iters=ITERS, repeats=repeats, baseline="fp16", timer=wall_ms,
                         owner="comfy_portable:attn_dtype_ab")
        last_result = result
        if result.failed:
            print(f"{name:<16}{'FAILED while timing':>10}  {result.failed}")
            del held, calls
            torch.cuda.empty_cache()
            continue
        # The Ratio inverts and names its own direction, so a bf16 that wins reads "1.04x faster"
        # instead of "0.96x" -- the form `razoes-na-direcao-certa` exists to keep out of here.
        print(f"{name:<16}{result.times['fp16']:>10.2f}{result.times['bf16']:>10.2f}"
              f"{result.ratios['bf16']:>30}{errors['fp16']:>11.5f}{errors['bf16']:>11.5f}")
        del held, calls
        torch.cuda.empty_cache()

    if last_result is not None:
        print(f"\n{provenance(last_result)}")
        print("(the discarded burst quoted above is the last backend's; each discards its own)")

    print("\nerror is mean|out - fp32 SDPA| on identical fp32 inputs, so the two columns are")
    print("comparable. The `sdpa` row is the baseline: bf16 carries 3 fewer mantissa bits, so")
    print("some gap is the format and not the kernel. A backend is only worse *at* bf16 if its")
    print("gap exceeds sdpa's.")
    return 0


if __name__ == "__main__":
    import argparse

    from _bench_guard import BenchGuard

    _parser = argparse.ArgumentParser(description="fp16 vs bf16 across the installed attention "
                                                  "backends")
    _parser.add_argument("--repeats", type=int, default=3,
                         help="interleaved bursts KEPT per backend; one more is run and discarded "
                              "as warm-up-biased")
    _args = _parser.parse_args()
    if _args.repeats < 1:
        print("--repeats must be at least 1")
        raise SystemExit(2)

    with BenchGuard("attn_dtype_ab") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main(repeats=_args.repeats))
