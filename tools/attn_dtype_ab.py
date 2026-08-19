"""Does fp16 buy anything over bf16 in the attention kernels installed here?

The claim under test: bf16's wider exponent has to be "paid for" somewhere, so attention
implementations support less of it -- fewer paths, fewer features, slower -- than fp16.

The bit layout already answers part of it (bf16 is 1+8+7, fp16 is 1+5+10; the range is paid for
in mantissa, inside the same 16 bits, not in hardware elsewhere). What it cannot answer is whether
a given *kernel* implements both equally, which is a software question and differs per library.
So each backend is run twice, same shapes and same seed, and reported on three axes:

    runs?     an exception here is the strongest possible answer
    time      median of timed iterations after warm-up, both dtypes on the same shapes
    error     mean |difference| against an fp32 SDPA reference computed from the same inputs

The reference is fp32 for both, so the two error columns are comparable to each other. Note this
measures the kernel, not the format: fp16 starting from fp32 inputs already has 3 more mantissa
bits, so it *should* be more accurate here. What matters is whether the gap is bigger than that
baseline, which the `sdpa` row gives.

    python_embeded\\python.exe -s tools/attn_dtype_ab.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

DEV = "cuda"
B, H, S, D = 2, 16, 4096, 64   # a real image-model attention shape, not a toy one
ITERS = 20


def timed(fn, iters: int = ITERS) -> float:
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


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    print(f"device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)}, "
          f"torch {torch.__version__}")
    print(f"shape B={B} H={H} S={S} D={D}, median of {ITERS} timed iterations\n")

    impls = backends()
    base, reference = reference_inputs()
    print(f"{'backend':<16}{'fp16 ms':>10}{'bf16 ms':>10}{'bf16 vs fp16':>14}"
          f"{'fp16 err':>11}{'bf16 err':>11}")
    for name, fn in impls.items():
        row = {}
        for dtype in (torch.float16, torch.bfloat16):
            q, k, v = (t.to(device=DEV, dtype=dtype) for t in base)
            pre = ([t.transpose(1, 2).contiguous() for t in (q, k, v)]
                   if getattr(fn, "needs_bshd", False) else None)
            call = (lambda: fn(q, k, v, pre)) if pre is not None else (lambda: fn(q, k, v))
            try:
                got = call().float().cpu()
                if not torch.isfinite(got).all():
                    # An error reported as mean(abs(.)) over a tensor holding NaN prints "nan"
                    # in one column while the timing columns and the verdict look completely
                    # normal. A backend that only breaks in bf16 would slip through.
                    raise RuntimeError("output contains inf or nan")
                error = float((got - reference).abs().mean())
                del got
                torch.cuda.empty_cache()
                # Nothing but q, k, v (and the pre-transposed copies) is resident from here to
                # the end of the timing loop.
                row[dtype] = (timed(call), error)
            except Exception as exc:
                row[dtype] = (None, repr(exc)[:70])
            del q, k, v, pre, call
            torch.cuda.empty_cache()

        fp16, bf16 = row[torch.float16], row[torch.bfloat16]
        if fp16[0] is None or bf16[0] is None:
            broken = "fp16" if fp16[0] is None else "bf16"
            print(f"{name:<16}{'FAILED on ' + broken:>10}  {row[torch.float16 if fp16[0] is None else torch.bfloat16][1]}")
            continue
        ratio = bf16[0] / fp16[0]
        verdict = f"{ratio:.2f}x slower" if ratio > 1 else f"{1 / ratio:.2f}x faster"
        print(f"{name:<16}{fp16[0]:>10.2f}{bf16[0]:>10.2f}{verdict:>14}"
              f"{fp16[1]:>11.5f}{bf16[1]:>11.5f}")

    print("\nerror is mean|out - fp32 SDPA| on identical fp32 inputs, so the two columns are")
    print("comparable. The `sdpa` row is the baseline: bf16 carries 3 fewer mantissa bits, so")
    print("some gap is the format and not the kernel. A backend is only worse *at* bf16 if its")
    print("gap exceeds sdpa's.")
    return 0


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("attn_dtype_ab") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
