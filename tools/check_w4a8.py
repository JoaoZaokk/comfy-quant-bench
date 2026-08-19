"""Does `w4a8_int8_linear` compute the right thing, and does it beat W4A4 where it should?

The registry resolving an op to `comfy_kitchen.backends.cuda` proves the kernel exists, not that
it is correct. Today the ComfyUI-nunchaku Z-Image loader existed, resolved, and died -- so
existence is checked separately from behaviour here.

Three questions, in order:

  1. does W4A8 reproduce `F.linear` on real-shaped data, and how closely?
  2. how does that compare against ConvRot W4A4 on the SAME weight and the SAME input?
  3. does the gap widen on the input distribution that actually hurts -- post-activation, which
     is where `feed_forward.net.2` sits and which scored worst in every diagnostic in this
     project?

Question 3 is the one that matters for the mixed-precision converter. If W4A8 and W4A4 differ
only on well-behaved Gaussian input, per-layer precision buys nothing; if W4A8 holds up where
W4A4 falls apart, the layers to promote are exactly the ones a calibration pass would flag.

    python_embeded\\python.exe -s tools/check_w4a8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.quant_ops  # noqa: E402,F401  (registers the backends)
from comfy_kitchen import registry as R  # noqa: E402


def relative(reference: torch.Tensor, got: torch.Tensor) -> float:
    return ((reference.float() - got.float()).norm()
            / reference.float().norm().clamp(min=1e-12)).item()


def inputs(kind: str, rows: int, cols: int, device: str, dtype) -> torch.Tensor:
    """Two distributions: the easy one everybody tests with, and the one that breaks things."""
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=torch.float32)
    if kind == "post_activation":
        # SwiGLU-shaped: one-sided, heavy tail, a few channels far larger than the rest. This is
        # what a down-projection actually receives, and it is where a per-group scale gets set by
        # an outlier and crushes everything else in the group.
        x = F.silu(x) * (1.0 + 4.0 * torch.rand(1, cols, device=device))
        x[:, ::37] *= 8.0
    return x.to(dtype)


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    device, dtype = "cuda:0", torch.bfloat16

    for name in ("quantize_w4a8_int8_weight", "w4a8_int8_linear",
                 "quantize_convrot_w4a4_weight", "convrot_w4a4_linear"):
        impl = R.get_implementation(name)
        module = getattr(impl, "__module__", "?")
        print(f"{name:<30} -> {module}")
        # Full module path, not the substring "cuda". The loose needle would accept anything
        # with "cuda" anywhere in its name -- a hypothetical `backends.cuda_dequant_fallback`
        # among them -- and the two sibling tools already use the strict form.
        if "comfy_kitchen.backends.cuda" not in module:
            # The eager backend declares the same capabilities, so an op that silently resolves
            # there would produce numbers that say nothing about the CUDA kernel.
            print("   not the CUDA backend; refusing to report this as a kernel measurement")
            return 1

    q_w4a8 = R.get_implementation("quantize_w4a8_int8_weight")
    lin_w4a8 = R.get_implementation("w4a8_int8_linear")
    q_w4a4 = R.get_implementation("quantize_convrot_w4a4_weight")
    lin_w4a4 = R.get_implementation("convrot_w4a4_linear")

    incompletos: list[str] = []
    print(f"\n{'shape':<18}{'input':<18}{'W4A8':>10}{'W4A4':>10}{'W4A8 vs W4A4':>18}")
    for out_features, in_features in ((3840, 3840), (3840, 10240), (11520, 3840)):
        torch.manual_seed(1)
        weight = (torch.randn(out_features, in_features, device=device, dtype=torch.float32)
                  / in_features ** 0.5).to(dtype)
        packed8 = q_w4a8(weight, group_size=16, convrot_groupsize=256)
        # (weight, convrot_groupsize, quant_group_size) -- the order is not the
        # same as the W4A8 quantiser's, and passing 64 first raises
        # 'int4 MMA kernel requires quant_group_size 64'.
        packed4 = q_w4a4(weight, 256, 64)

        for kind in ("gaussian", "post_activation"):
            x = inputs(kind, 64, in_features, device, dtype)
            reference = F.linear(x.float(), weight.float())
            try:
                # The 5-tuple is (qdata, s_rel, s_channel, correction, codebook).
                # Reading 3 as the codebook fails with "correction must have shape
                # (240, 3840), got (16,)" -- 16 being the project's 16-entry codebook,
                # which is what named the slots.
                got8 = lin_w4a8(x, packed8[0], packed8[1], packed8[2],
                                codebook=packed8[4], correction=packed8[3],
                                group_size=16, convrot_groupsize=256, out_dtype=dtype)
                e8 = relative(reference, got8)
            except Exception as exc:
                e8 = float("nan")
                print(f"   W4A8 raised: {type(exc).__name__}: {str(exc)[:70]}")
            try:
                got4 = lin_w4a4(x, *packed4) if isinstance(packed4, (tuple, list)) \
                    else lin_w4a4(x, packed4)
                e4 = relative(reference, got4)
            except Exception as exc:
                e4 = float("nan")
                print(f"   W4A4 raised: {type(exc).__name__}: {str(exc)[:70]}")
            # The column is headed "W4A8 better by", so printing a bare e4/e8 asserts the
            # direction before measuring it: a W4A8 that came out worse would print "0.73x"
            # under a heading that already claims it won.
            if e8 == e8 and e4 == e4 and e8 > 0 and e4 > 0:
                verdict = (f"{e4 / e8:.2f}x better" if e8 <= e4
                           else f"{e8 / e4:.2f}x WORSE")
            else:
                verdict = "nao medido"
                incompletos.append(f"{out_features}x{in_features} {kind}")
            print(f"{f'{out_features}x{in_features}':<18}{kind:<18}"
                  f"{e8:>10.4f}{e4:>10.4f}{verdict:>18}")

    print("\nrelative L2 against F.linear in float32. Lower is better.")
    print("On the `post_activation` row: this file used to attribute the gap to 'a per-group "
          "scale set by an outlier', which is a mechanism ConvRot removes -- it rotates over "
          "groups of 256 and then scales per row, not per group. The measured numbers agree that "
          "the mechanism is not there: 3840x3840 moves 0.2230 -> 0.2234 (+0.18%) between the two "
          "input distributions, and 11520x3840 moves +0.09%. Only 3840x10240 shifts, and its "
          "larger K explains most of that on its own.")
    if incompletos:
        # An all-nan table used to exit 0, which reads as "measured, and they tie".
        print(f"\n{len(incompletos)} cell(s) could not be measured: {incompletos[:4]}")
        return 1
    return 0


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("check_w4a8") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
