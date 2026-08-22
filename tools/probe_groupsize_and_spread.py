"""Two more, both decisive for a ticket the read-only pass could only argue about.

  T06  verify_w4a4.py:223-226 hardcodes convrot_groupsize=256 and never reads the metadata
       it just loaded. A file converted at 64 -- a documented flag -- is smoked at 256. The
       RMSE ceiling is 0.9, deliberately loose. Two outcomes, both bad: a correct file fails,
       or the mismatch lands under 0.9 and PASS is printed for a comparison never performed.
       Which one is it? Measurable in one shot.

  T09  attn_bench.py takes ONE burst and prints a ratio to three decimals. m_crossover's
       docstring records three consecutive runs disagreeing by up to 1.4x. Is that specific
       to m_crossover's shapes, or does one burst lie here too? Interleave and look.
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(ROOT / "ComfyUI"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.quant_ops  # noqa: E402,F401
import comfy_kitchen as ck  # noqa: E402

DEV = torch.cuda.get_device_name(0)
print(f"device: {DEV}   cc {torch.cuda.get_device_capability(0)}")


def rule(t):
    print()
    print("-" * 78)
    print(t)
    print("-" * 78)


# ================================================================= T06
rule("T06  a file quantized at cg=64, run through the kernel at cg=256")
print("verify_w4a4's smoke ceiling is relative RMSE 0.9. Anything under it prints PASS.")
print()
torch.manual_seed(1234)
w = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
ref = F.linear(x.float(), w.float())


def rrmse(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


rows = []
for made_at in (64, 256):
    q, s = ck.quantize_convrot_w4a4_weight(w, made_at, 64)
    for run_at in (64, 256):
        try:
            out = ck.convrot_w4a4_linear(x, q, s, None, run_at, 64, "int4").float()
            e = rrmse(out, ref)
            rows.append((made_at, run_at, f"{e:.4f}", "PASS" if e < 0.9 else "FAIL",
                         "finite" if torch.isfinite(out).all() else "NON-FINITE"))
        except Exception as exc:
            rows.append((made_at, run_at, "-", "RAISED", f"{type(exc).__name__}: {str(exc)[:60]}"))

print(f"{'made at':>8}  {'run at':>7}  {'rel RMSE':>9}  {'verdict':>8}  note")
for m, r, e, v, n in rows:
    mark = "   <-- MATCHED" if m == r else "   <-- MISMATCH"
    print(f"{m:>8}  {r:>7}  {e:>9}  {v:>8}  {n}{mark}")

mism = [r for r in rows if r[0] != r[1]]
passing_mismatch = [r for r in mism if r[3] == "PASS"]
print()
if passing_mismatch:
    print("VERDICT: a groupsize MISMATCH lands UNDER the 0.9 ceiling and prints PASS.")
    print("         verify_w4a4 --kernel-smoke on a cg=64 checkpoint reports a pass for a")
    print("         comparison it did not perform. The check is not merely untidy: its")
    print("         output is affirmatively wrong for one of the two supported groupsizes.")
elif mism and all(r[3] in ("FAIL", "RAISED") for r in mism):
    print("VERDICT: a mismatch FAILS or RAISES loudly. The bug costs a false alarm on a")
    print("         correct file, not a false PASS on a broken one. Lower severity, still")
    print("         worth fixing -- a false FAIL on a supported flag is a trap too.")
else:
    print("VERDICT: mixed -- read the table.")

# ================================================================= T09
rule("T09  does one timing burst lie? interleaved repeats of an attention A/B")
B, H, S, D = 2, 24, 4096, 64
print(f"shape B={B} H={H} S={S} D={D} bf16 -- DiT-like, the shape attn_bench uses")

q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

paths = {}
paths["sdpa"] = lambda: F.scaled_dot_product_attention(q, k, v)
try:
    from sageattention import sageattn
    paths["sage"] = lambda: sageattn(q, k, v, tensor_layout="HND")
except Exception as exc:
    print(f"(sage unavailable: {type(exc).__name__})")


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


REPEATS = 5
series = {name: [] for name in paths}
for rep in range(REPEATS):
    for name, fn in paths.items():          # interleaved, not one path to completion
        series[name].append(timed(fn))

print()
print(f"{'path':<8} {'per-burst medians (ms)':<44} {'median':>8} {'spread':>8}")
for name, xs in series.items():
    lo, hi = min(xs), max(xs)
    print(f"{name:<8} {' '.join(f'{v:7.3f}' for v in xs):<44} "
          f"{statistics.median(xs):8.3f} {hi/lo:7.2f}x")

if "sage" in series:
    ratios = [s / g for s, g in zip(series["sdpa"], series["sage"])]
    print()
    print("The number attn_bench.py:101 prints, computed once per burst:")
    print("   " + "  ".join(f"{r:.3f}x" for r in ratios))
    print()
    print(f"   one burst would report : {ratios[0]:.3f}x  (whichever burst it happened to get)")
    print(f"   median of {REPEATS}          : {statistics.median(ratios):.3f}x  "
          f"[{min(ratios):.3f}-{max(ratios):.3f}]")
    swing = max(ratios) / min(ratios)
    print()
    print(f"VERDICT: the ratio's own min-max spans {swing:.2f}x across {REPEATS} interleaved bursts.")
    if swing > 1.05:
        print("         attn_bench prints THREE DECIMALS off one burst. The third decimal is")
        print("         noise, and so is the second. A number in that shape gets pasted into")
        print("         another session as if it were measured.")
    else:
        print("         Tight here. attn_bench's single burst is not lying on THIS shape --")
        print("         which is not the same as it being safe to print three decimals, and")
        print("         says nothing about the shapes where m_crossover saw 1.4x.")

print()
print("=" * 78)
print(f"RAN ON: {DEV}")
print("NOT covered: T06 uses a synthetic weight, not a real checkpoint -- it measures the")
print("kernel's behaviour under a groupsize mismatch, not verify_w4a4.py's full path. T09 is")
print("one shape on one card with the machine in whatever state it is in; it shows that one")
print("burst can lie, not how much it lies in general.")
print("=" * 78)
