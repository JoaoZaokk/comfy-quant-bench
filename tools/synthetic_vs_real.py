"""Does measuring on random input pick the same layers as measuring on real activations?

This is the version of "skip the calibration pass" that has a chance of working. Weight-only
statistics do not predict `err_w4a4` -- measured in `tools/predict_promotion.py`, best feature
Spearman +0.225 and a layer selection barely above chance. But the expensive half of calibration
is not the measurement, it is *getting the activations*: loading the high-precision model, running
a real generation with hooks on 170 Linears, reservoir-sampling rows, writing a 226 MiB file.

The measurement itself is cheap. So the question is narrower and better: if the same kernels are
run on **gaussian noise** of the right width instead of the rows the model actually produced, does
the per-layer error rank the layers the same way?

If yes, converting a new model needs no generation pass at all -- just its weights.
If no, that is worth knowing too, and it says the real activations carry information the weight
and the shape do not.

Scored two ways, because they answer different questions:

    spearman   do the two measurements rank the layers the same
    overlap    at a given threshold, do they promote the same layers -- against the chance
               baseline, since picking k of n at random already hits k^2/n

    python_embeded\\python.exe -s tools/synthetic_vs_real.py ^
        --weights ComfyUI/models/diffusion_models/beyond-reality-zimage-v2_native.safetensors ^
        --analysis calib/zimage_v2_native.analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.quant_ops  # noqa: E402,F401
from comfy_kitchen import registry as R  # noqa: E402
from predict_promotion import spearman  # noqa: E402

CONVROT_GROUP = 256
QUANT_GROUP = 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=128,
                        help="synthetic rows per layer; matches the calibration sample size")
    parser.add_argument("--promote-error", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distribution", default="gaussian",
                        choices=["gaussian", "heavy"],
                        help="'heavy' is gaussian times a per-channel lognormal, a cheap stand-in "
                             "for the channel-magnitude spread real activations have")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA: this runs the real kernels")
    for name in ("quantize_convrot_w4a4_weight", "convrot_w4a4_linear"):
        module = getattr(R.get_implementation(name), "__module__", "?")
        if "comfy_kitchen.backends.cuda" not in module:
            print(f"{name} resolves to {module}, not the CUDA backend; refusing")
            return 1
    q_w4a4 = R.get_implementation("quantize_convrot_w4a4_weight")
    lin_w4a4 = R.get_implementation("convrot_w4a4_linear")

    from safetensors.torch import safe_open

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = analysis["layers"] if isinstance(analysis, dict) and "layers" in analysis else analysis
    if isinstance(rows, dict):
        rows = list(rows.values())
    measured = {r["layer"]: r["err_w4a4"] for r in rows if r.get("err_w4a4") is not None}

    real, synthetic, names = [], [], []
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with safe_open(str(args.weights), framework="pt") as f:
        available = set(f.keys())
        for layer, err_real in measured.items():
            key = f"{layer}.weight"
            if key not in available:
                continue
            weight = f.get_tensor(key).cuda().to(torch.bfloat16)
            if weight.shape[1] % CONVROT_GROUP:
                continue
            x = torch.randn(args.rows, weight.shape[1], device="cuda", dtype=torch.float32,
                            generator=generator)
            if args.distribution == "heavy":
                # Real Z-Image activations differ across channels by orders of magnitude; plain
                # gaussian has none of that, and the ConvRot group scale is set per group of
                # channels, so a flat input cannot exercise what the format is bad at.
                scale = torch.exp(torch.randn(weight.shape[1], device="cuda",
                                              generator=generator) * 1.5)
                x = x * scale
            x = x.to(torch.bfloat16)
            packed = q_w4a4(weight, CONVROT_GROUP, QUANT_GROUP)
            reference = F.linear(x.float(), weight.float())
            got = lin_w4a4(x, packed[0], packed[1], None, CONVROT_GROUP, QUANT_GROUP, "int4")
            err = float((reference - got.float()).norm() / reference.norm())
            real.append(err_real)
            synthetic.append(err)
            names.append(layer)
            del weight, x, packed, reference, got
            torch.cuda.empty_cache()

    print(f"{len(names)} layer(s) measured both ways ({args.distribution} input, "
          f"{args.rows} rows)\n")
    rho = spearman(synthetic, real)
    print(f"spearman(synthetic, real) = {rho:+.3f}")

    promoted = {n for n, e in zip(names, real) if e > args.promote_error}
    chance = len(promoted) ** 2 / len(names)
    ranked = sorted(range(len(names)), key=lambda i: synthetic[i], reverse=True)
    picked = {names[i] for i in ranked[:len(promoted)]}
    hit = len(picked & promoted)
    print(f"\nat --promote-error {args.promote_error}: real promotes {len(promoted)} of "
          f"{len(names)}; chance overlap {chance:.1f}")
    print(f"synthetic picks the same {hit} of them ({hit - chance:+.1f} vs chance)")

    mean_real = sum(real) / len(real)
    mean_syn = sum(synthetic) / len(synthetic)
    print(f"\nmean err_w4a4: real {mean_real:.4f}, synthetic {mean_syn:.4f}")
    if abs(rho) < 0.4 or hit - chance < 0.15 * len(promoted):
        print("\nSynthetic input does NOT reproduce the real ranking. The activations carry "
              "information the weight and the shape do not, so the calibration pass cannot be "
              "replaced by noise -- at least not by this noise, on this model.")
    else:
        print("\nSynthetic input tracks the real ranking on this model. That is one model. "
              "Before dropping the calibration pass it needs to hold on a second architecture, "
              "because a result that holds once is a coincidence you have not ruled out.")
    return 0


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("synthetic_vs_real") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
