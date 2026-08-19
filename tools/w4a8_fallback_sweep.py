"""Which shapes make the W4A8 path give up on its kernels and fall back to eager?

`comfy_kitchen/backends/cuda/__init__.py` builds `w4a8_int8_linear` as a chain of attempts, each
returning a host-visible boolean:

    used = _C.w4a8_codebook_linear_chunked(...)     # :2213 region
    if used: return ...
    ...
    used = _C.cutlass_int8_dequant(...)
    if not used:
        return eager_w4a8_int8_linear(...)          # :2261 region

The audit flagged this because the eager tail is exactly what this project forbids -- dequantized
math wearing the name of a quantized op -- and it is reached without a log line. What the audit
could not say is **which shapes reach it**, because that is decided inside the compiled extension.

So this sweeps shapes and reads the booleans. It does not guess: every `_C` entry point in the
chain is wrapped and its return value recorded, so a row saying "eager" is a row where the C++
said no, not a row where this file inferred it.

Constraints on the grid, from the quantizer rather than from taste: K must be divisible by the
convrot group size (256), and the W4A8 tier wants K divisible by its group size (16) as well.

    python_embeded\\python.exe -s tools/w4a8_fallback_sweep.py
    python_embeded\\python.exe -s tools/w4a8_fallback_sweep.py --batches 1,5856 --odd

Caveat for anything quoted from here: one GPU (sm86), one comfy-kitchen build. A shape that keeps
the kernel here can lose it on another card, because the refusal is the extension's own
capability test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import comfy.quant_ops  # noqa: E402,F401
from comfy_kitchen import registry as R  # noqa: E402
from comfy_kitchen.backends.cuda import _C  # noqa: E402

WATCH = ("w4a8_codebook_linear_chunked", "w4a8_codebook_gemm_chunked",
         "cutlass_int8_dequant", "int4_weight_int8_act_gemm_dequant_chunked")

CALLS: list[tuple[str, object]] = []


def spy_on(names) -> list[str]:
    wrapped = []
    for name in names:
        original = getattr(_C, name, None)
        if original is None:
            continue

        def make(fn, label):
            def spy(*a, **kw):
                result = fn(*a, **kw)
                CALLS.append((label, result))
                return result
            return spy

        setattr(_C, name, make(original, name))
        wrapped.append(name)
    return wrapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8,256,5856")
    parser.add_argument("--out-features", default="512,1024,3840,10240,11520")
    parser.add_argument("--in-features", default="256,512,1024,2560,3840")
    parser.add_argument("--odd", action="store_true",
                        help="also try N values that are not multiples of 128, where a tiled "
                             "kernel is likeliest to refuse")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    for name in ("quantize_w4a8_int8_weight", "w4a8_int8_linear"):
        module = getattr(R.get_implementation(name), "__module__", "?")
        if "comfy_kitchen.backends.cuda" not in module:
            print(f"{name} resolves to {module}, not the CUDA backend; refusing")
            return 1

    wrapped = spy_on(WATCH)
    print(f"{torch.cuda.get_device_name(0)}, watching {len(wrapped)} entry point(s): "
          f"{', '.join(wrapped)}")
    missing = [n for n in WATCH if n not in wrapped]
    if missing:
        # An absent entry point is not a neutral fact: this sweep would report "kernel" for a
        # path it never watched.
        print(f"NOT PRESENT in this build, therefore not watched: {', '.join(missing)}")

    q = R.get_implementation("quantize_w4a8_int8_weight")
    lin = R.get_implementation("w4a8_int8_linear")

    outs = [int(v) for v in args.out_features.split(",")]
    if args.odd:
        outs += [513, 1000, 3841]
    ins = [int(v) for v in args.in_features.split(",")]
    batches = [int(v) for v in args.batches.split(",")]

    # Relative error against a float32 reference, on every row. Knowing a shape fell to eager is
    # half the finding; the other half is whether the fallback still computes the right answer.
    # A slow-but-correct fallback and a slow-and-wrong one need different responses, and "it
    # returned finite numbers" does not tell them apart.
    print(f"\n{'N (out)':>9}{'K (in)':>9}{'M':>8}  {'verdict':<12}{'rel err':>9}  "
          f"{'entry points called'}")
    eager_rows = 0
    skipped = []
    for n in outs:
        for k in ins:
            if k % 256 or k % 16:
                skipped.append((n, k, "K not divisible by 256/16"))
                continue
            torch.manual_seed(1)
            weight = (torch.randn(n, k, device="cuda", dtype=torch.float32)
                      / k ** 0.5).to(torch.bfloat16)
            try:
                packed = q(weight, group_size=16, convrot_groupsize=256, symmetric=True,
                           scale_dtype=torch.float8_e4m3fn, codebook=True, codebook_tensor=None,
                           stochastic_rounding=0)
            except Exception as exc:
                skipped.append((n, k, f"quantizer refused: {type(exc).__name__}: {exc}"))
                del weight
                torch.cuda.empty_cache()
                continue
            for m in batches:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                CALLS.clear()
                try:
                    out = lin(x, packed[0], packed[1], packed[2], codebook=packed[4],
                              correction=packed[3], bias=None, group_size=16,
                              convrot_groupsize=256, out_dtype=torch.bfloat16)
                    torch.cuda.synchronize()
                    finite = bool(torch.isfinite(out).all())
                except Exception as exc:
                    print(f"{n:>9}{k:>9}{m:>8}  {'RAISED':<12}"
                          f"{type(exc).__name__}: {str(exc)[:40]}")
                    del x
                    torch.cuda.empty_cache()
                    continue
                # "eager" is read off the booleans, not inferred: the chain reached its tail only
                # if every entry point it called said no.
                took_kernel = any(bool(r) for _, r in CALLS)
                verdict = "kernel" if took_kernel else "EAGER"
                if not took_kernel:
                    eager_rows += 1
                if not finite:
                    verdict += "/nonfinite"
                reference = torch.nn.functional.linear(x.float(), weight.float())
                rel = float((reference - out.float()).norm() / reference.norm())
                trace = ", ".join(f"{name}={result}" for name, result in CALLS) or "(none)"
                print(f"{n:>9}{k:>9}{m:>8}  {verdict:<12}{rel:>9.4f}  {trace}")
                del reference
                del x, out
                torch.cuda.empty_cache()
            del weight, packed
            torch.cuda.empty_cache()

    if skipped:
        print(f"\n{len(skipped)} shape(s) never reached the linear op:")
        for n, k, why in skipped:
            print(f"  N={n} K={k}: {why}")

    print(f"\n{eager_rows} shape(s) fell through to the eager tail.")
    if not eager_rows:
        print("No shape in this grid reached it. That is not proof it is unreachable -- the grid "
              "is small, the refusal is decided inside the compiled extension, and this is one "
              "GPU. It does mean the audit's eager-fallback concern has no repro yet.")
    return 0


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("w4a8_fallback_sweep") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
