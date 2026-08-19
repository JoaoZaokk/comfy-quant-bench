"""Does a layer profile measured on one checkpoint describe another?

The question this project cares about is whether the calibration pass can be paid once per
architecture instead of once per checkpoint. `tools/quant_mixed.py --foreign-analysis` allows that;
this is what decides whether allowing it was right for a given pair.

Two numbers, because they answer different things:

    spearman   do the two measurements rank the layers the same way
    overlap    at a promotion threshold, do they select the same layers -- reported against the
               chance baseline, because picking k of n at random already hits k^2/n and an
               overlap quoted without it reads as a hit rate

Both are computed only over layers the two analyses share, and the count of shared layers is
printed: two files that overlap in 3 layers can score +1.0 and mean nothing.

    python_embeded\\python.exe -s tools/profile_transfer.py calib/a.analysis.json calib/b.json ...

Any number of analyses; every pair is compared. Mixing architectures is allowed and expected to
score badly -- that is the control, not a mistake.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_promotion import spearman  # noqa: E402


def load(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["layers"] if isinstance(payload, dict) and "layers" in payload else payload
    if isinstance(rows, dict):
        rows = list(rows.values())
    source = Path(payload.get("source", path.stem)).name if isinstance(payload, dict) else path.stem
    return source, {r["layer"]: r for r in rows if r.get("err_w4a4") is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyses", nargs="+", type=Path)
    parser.add_argument("--promote-error", type=float, default=0.10)
    parser.add_argument("--top-fraction", type=float, default=0.5,
                        help="compare the worst F of each analysis by rank instead of an "
                             "absolute threshold; 0 restores threshold mode")
    parser.add_argument("--label-width", type=int, default=34)
    args = parser.parse_args()

    loaded = {}
    for path in args.analyses:
        source, rows = load(path)
        loaded[path.stem] = (source, rows)
        print(f"{path.stem:<28} {len(rows):>4} layers   from {source}")

    width = args.label_width
    print(f"\n{'pair':<{width}}{'shared':>8}{'spearman':>10}{'overlap':>12}{'chance':>8}"
          f"{'vs chance':>11}")
    for a, b in itertools.combinations(loaded, 2):
        (_, ra), (_, rb) = loaded[a], loaded[b]
        shared = sorted(set(ra) & set(rb))
        if not shared:
            print(f"{a + ' vs ' + b:<{width}}{0:>8}   no layer in common -- different "
                  f"architectures, or a naming mismatch")
            continue
        rho = spearman([ra[k]["err_w4a4"] for k in shared], [rb[k]["err_w4a4"] for k in shared])
        if args.top_fraction:
            # Rank-based selection, because an absolute threshold degenerates. Measured: at
            # --promote-error 0.10 the two HunyuanVideo checkpoints each promote 424 of 432, so
            # chance overlap is 417 and a perfect 424/424 scores +6.9 -- the statistic says almost
            # nothing while looking like a triumph. Taking the worst half of each is the same
            # question asked where it can still be answered.
            take = max(1, int(len(shared) * args.top_fraction))
            pa = set(sorted(shared, key=lambda k: ra[k]["err_w4a4"], reverse=True)[:take])
            pb = set(sorted(shared, key=lambda k: rb[k]["err_w4a4"], reverse=True)[:take])
        else:
            pa = {k for k in shared if ra[k]["err_w4a4"] > args.promote_error}
            pb = {k for k in shared if rb[k]["err_w4a4"] > args.promote_error}
        chance = len(pa) * len(pb) / len(shared) if shared else 0.0
        hit = len(pa & pb)
        print(f"{a + ' vs ' + b:<{width}}{len(shared):>8}{rho:>+10.3f}"
              f"{f'{hit}/{len(pa)}':>12}{chance:>8.1f}{hit - chance:>+11.1f}")

        # The decision-relevant number, and it is not the correlation. If `b`'s profile is reused
        # to convert `a`, these are the layers that come out assigned differently than measuring
        # `a` would have assigned them, split by which way the mistake goes: a layer promoted
        # without needing it costs size and speed, a layer left at 4 bits that needed 8 costs
        # accuracy. Reported at the absolute threshold, because that is what the converter uses.
        ta = {k for k in shared if ra[k]["err_w4a4"] > args.promote_error}
        tb = {k for k in shared if rb[k]["err_w4a4"] > args.promote_error}
        wasted, exposed = tb - ta, ta - tb
        worst = max((ra[k]["err_w4a4"] for k in exposed), default=None)
        detail = f", worst left at 4 bits {worst:.4f}" if worst is not None else ""
        # The direction is named because the two are not the same number: reusing b's profile on a
        # exposes the layers a needs and b does not, and swapping the pair swaps which those are.
        print(f"{'':<{width}}reusing {b} on {a} at {args.promote_error}: "
              f"{len(wasted | exposed)} of {len(shared)} layers differ "
              f"({len(wasted)} promoted needlessly, {len(exposed)} left exposed{detail})")

    print(f"\n{'analysis':<28}{'median err_w4a4':>18}{'promoted':>12}{'of':>6}")
    for name, (_, rows) in loaded.items():
        errs = [r["err_w4a4"] for r in rows.values()]
        promoted = sum(1 for e in errs if e > args.promote_error)
        print(f"{name:<28}{statistics.median(errs):>18.4f}{promoted:>12}{len(errs):>6}")
    print(f"\nPromotion counts are at --promote-error {args.promote_error}. They are not "
          "comparable across architectures: the same threshold means a different fraction of the "
          "model in each, which is the reason a default cannot be shared between families.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
