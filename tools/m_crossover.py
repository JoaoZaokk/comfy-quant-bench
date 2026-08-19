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


def main() -> int:
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

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")

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
              f"{'best vs bf16':>22}{'w4a4 err':>10}{'w4a4/a8':>9}{'w4a8 err':>10}")

        for m in BATCHES:
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
            times, errors, falhas = {}, {}, []
            for label, call in paths.items():
                try:
                    errors[label] = relative(reference, call())
                    times[label] = timed(call, iters)
                except Exception as exc:
                    errors[label] = float("nan")
                    times[label] = float("nan")
                    # Reported at every M, not only the first: an earlier version printed the
                    # exception only for BATCHES[0], so a path that failed from M=128 upward
                    # turned into a silent nan column and the verdict still named a winner.
                    falhas.append(label)
                    print(f"   M={m} {label} raised: {type(exc).__name__}: {str(exc)[:60]}")

            quant = {k: v for k, v in times.items() if k != "bf16" and v == v}
            if not quant:
                verdict = "todos falharam"
            elif times["bf16"] != times["bf16"]:
                # No baseline, so "N times faster than bf16" has nothing to be faster than.
                verdict = f"{min(quant, key=quant.get)} (sem bf16)"
            else:
                best = min(quant, key=quant.get)
                ratio = times["bf16"] / quant[best]
                verdict = (f"{best} {ratio:.2f}x" if ratio >= 1
                           else f"{best} {1 / ratio:.2f}x slower")
                # Naming a winner over a field that lost entrants reads as a complete comparison.
                if falhas:
                    verdict += f" (de {len(quant)})"
            print(f"{m:>7}{times['bf16']:>9.3f}{times['w4a4']:>9.3f}"
                  f"{times['w4a4/a8']:>9.3f}{times['w4a8']:>9.3f}{verdict:>22}"
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

    with BenchGuard("m_crossover") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
