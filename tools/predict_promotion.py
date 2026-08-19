"""Can the promotion decision be made from the weight alone, with no calibration pass?

`quant_mixed.py` decides 4-bit or 8-bit per layer by measuring the real kernels on real
activations, which costs a full calibration run: load the high-precision model, sample with hooks
on 170 Linears, reservoir-sample rows, then measure three formats per layer. Useful, but it means
every new model needs a generation pass before it can be converted.

If something computable from the weight alone predicts `err_w4a4`, that pass goes away.

The obvious candidate is already dead. Crest factor of the *activation* was measured against
`err_w4a4` over these same 170 layers and scored Spearman +0.10 (part 9) -- no relationship. That
was a statistic of the input. Statistics of the **weight** have never been tried here, and they are
free: no sampling, no hooks, no GPU time beyond reading the tensor.

So this scores candidate weight features against the `err_w4a4` that was already measured, and
reports Spearman for each. It is a screening tool, not a decision: a feature that correlates here
has been shown to correlate on one model's 170 layers, which is a reason to test it on a second
model, not a reason to ship it.

**A high correlation is not enough on its own.** What matters is whether thresholding on the
feature selects the same layers the measurement selects, so that is reported too -- as the overlap
between the two selections at the same promotion count, which is the quantity the converter would
actually depend on.

    python_embeded\\python.exe -s tools/predict_promotion.py ^
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

CONVROT_GROUP = 256
QUANT_GROUP = 64


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, computed here rather than pulled in, because scipy is not installed."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den = (sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def features(weight: torch.Tensor) -> dict:
    """Weight-only statistics, each with a reason to be here rather than a swept net.

    The quantizer is per-output-row with a symmetric group scale, so what should hurt is a row
    whose magnitude is set by a few entries -- the group scale is pinned by the largest element in
    the group and everything smaller loses resolution. Every feature below is a different way of
    asking "how much of this row is decided by its outliers".
    """
    w = weight.float()
    rows, cols = w.shape
    absw = w.abs()

    row_max = absw.amax(dim=1)
    row_std = w.std(dim=1).clamp(min=1e-12)
    row_crest = row_max / row_std                       # per-row crest, the weight-side analogue

    groups = w.reshape(rows, cols // QUANT_GROUP, QUANT_GROUP)
    group_max = groups.abs().amax(dim=2)
    group_mean = groups.abs().mean(dim=2).clamp(min=1e-12)
    # How much of each group's representable range is wasted on its own largest element.
    group_waste = (group_max / group_mean)

    kurt = ((w - w.mean()) ** 4).mean() / (w.var().clamp(min=1e-12) ** 2)

    # Fraction of mass in the top 0.1% of entries: a blunt outlier measure that does not assume
    # a distribution shape the way kurtosis does.
    flat = absw.reshape(-1)
    k = max(1, flat.numel() // 1000)
    top = torch.topk(flat, k).values
    top_mass = float(top.sum() / flat.sum().clamp(min=1e-12))

    return {
        "row_crest_p99": float(torch.quantile(row_crest, 0.99)),
        "row_crest_mean": float(row_crest.mean()),
        "group_waste_p99": float(torch.quantile(group_waste.reshape(-1).float(), 0.99)),
        "group_waste_mean": float(group_waste.mean()),
        "kurtosis": float(kurt),
        "top0.1pct_mass": top_mass,
        "std": float(w.std()),
        "cols": float(cols),
        "rows": float(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--target", default="err_w4a4",
                        choices=["err_w4a4", "err_w4a8", "err_bf16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--promote-error", type=float, default=0.10,
                        help="the threshold whose selection the features are asked to reproduce")
    args = parser.parse_args()

    from safetensors.torch import safe_open

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = analysis["layers"] if isinstance(analysis, dict) and "layers" in analysis else analysis
    if isinstance(rows, dict):
        rows = list(rows.values())
    measured = {r["layer"]: r for r in rows if r.get(args.target) is not None}
    print(f"{len(measured)} layer(s) with a measured {args.target}")

    table = []
    with safe_open(str(args.weights), framework="pt") as f:
        available = set(f.keys())
        for name, row in measured.items():
            key = f"{name}.weight"
            if key not in available:
                continue
            w = f.get_tensor(key).to(args.device)
            entry = features(w)
            entry["layer"] = name
            entry["target"] = row[args.target]
            entry["crest_p99_activation"] = row.get("crest_p99")
            table.append(entry)
            del w
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print(f"{len(table)} layer(s) matched a weight tensor\n")

    names = [k for k in table[0] if k not in ("layer", "target", "crest_p99_activation")]
    target = [e["target"] for e in table]

    print(f"{'feature':<24}{'spearman':>10}   {'reads as'}")
    scored = []
    for feature in names:
        rho = spearman([e[feature] for e in table], target)
        scored.append((abs(rho), rho, feature))
    # The activation-side statistic is scored alongside, as the control: it is already known to
    # fail, so a weight feature that does not beat it has found nothing.
    if table[0].get("crest_p99_activation") is not None:
        rho = spearman([e["crest_p99_activation"] for e in table], target)
        scored.append((abs(rho), rho, "crest_p99_activation*"))
    for _, rho, feature in sorted(scored, reverse=True):
        strength = ("strong" if abs(rho) >= 0.7 else
                    "moderate" if abs(rho) >= 0.4 else
                    "weak" if abs(rho) >= 0.2 else "none")
        print(f"{feature:<24}{rho:>+10.3f}   {strength}")
    print("\n* the activation-side crest factor, scored as the control. It was measured at +0.10 "
          "against this target in part 9; a weight feature that does not clearly beat it has "
          "found nothing.")

    # Correlation is not the deliverable. The converter would threshold on a feature and promote a
    # set of layers; what matters is whether that set is the set the measurement chose.
    promoted = {e["layer"] for e in table if e["target"] > args.promote_error}
    # Without this line the overlap column reads as a hit rate. Picking k layers at random out of
    # n, when the true set is also size k, already hits k^2/n of them -- at 119 of 170 that is 83
    # before any feature has done anything.
    chance = len(promoted) ** 2 / len(table)
    print(f"\nAt --promote-error {args.promote_error}, the measurement promotes {len(promoted)} "
          f"of {len(table)} layers. A random pick of the same size scores {chance:.1f} by chance.")
    print(f"{'feature':<24}{'overlap':>10}{'missed':>9}{'wrong':>8}{'vs chance':>11}")
    for _, rho, feature in sorted(scored, reverse=True):
        if feature.endswith("*"):
            key = "crest_p99_activation"
        else:
            key = feature
        # Take the same number of layers the measurement took, ranked by this feature, so the two
        # selections are the same size and the overlap is the whole story.
        ranked = sorted(table, key=lambda e: e[key], reverse=rho > 0)
        picked = {e["layer"] for e in ranked[:len(promoted)]}
        hit = len(picked & promoted)
        print(f"{feature:<24}{hit:>7}/{len(promoted):<3}{len(promoted) - hit:>9}"
              f"{len(picked - promoted):>8}{hit - chance:>+11.1f}")

    print("\nRead the overlap column, not the correlation column. A feature can rank-correlate "
          "and still pick the wrong layers, and it is the picked set the converter would ship.")
    print("Whatever wins here has been shown to work on ONE model's layers. That is a reason to "
          "test it on a second model, not a reason to drop the calibration pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
