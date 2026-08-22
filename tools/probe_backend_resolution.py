"""Settle, on the card, the questions the read-only pass could not.

Four of them, in order of how much they change what the bench believes:

  Q1  Does `_native_probe.native_backend_ready()` -- the shared probe written to end the
      duplication, and whose own docstring says it was never executed -- actually work?

  Q2  Does the resolved implementation depend on `convrot_groupsize`? `quant_w4a4.py:98-99`
      preflights at 64/64 and converts at whatever `--convrot-groupsize` says (default 256).
      If the answer is "no difference", that preflight is merely untidy. If "yes", it is
      answering a question about a configuration it is not about to run.

  Q3  Does dummy `torch.empty` vs real tensors change the answer? `_native_probe.py:9-15`
      asserts it can, citing `comfy_kitchen/registry.py:246` -- empty/None kwargs skip
      constraint validation. Never tested.

  Q4  Does `.backends.cuda` in `__module__` prove the native INT4 path ran? The whole
      preflight rests on that string match. W4A4 means the ACTIVATION is 4-bit too, so a
      true native call and a dequantized `F.linear(x, W_deq)` must differ numerically. If
      they agree to the last bit, the "native" resolution is doing BF16 GEMM.

Run with CUDA_VISIBLE_DEVICES=0. Every result prints the device it ran on, because a lock on
the 3090 is not evidence the work went there.
"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ComfyUI"))

import torch  # noqa: E402
import comfy.quant_ops  # noqa: E402,F401  -- registers the ops
import comfy_kitchen as ck  # noqa: E402
from comfy_kitchen import registry as R  # noqa: E402

DEV = torch.cuda.get_device_name(0)
print("=" * 78)
print(f"device       {DEV}   cc {torch.cuda.get_device_capability(0)}")
print(f"torch        {torch.__version__}")
try:
    import importlib.metadata as md
    print(f"comfy-kitchen {md.version('comfy-kitchen')}")
except Exception as exc:
    print(f"comfy-kitchen version unreadable: {exc}")
print(f"backends     {ck.list_backends()}")
print("=" * 78)


def rule(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ----------------------------------------------------------------- Q1
rule("Q1  _native_probe.native_backend_ready() -- first execution ever")
try:
    import _native_probe
    payload = _native_probe.native_backend_ready(ROOT)
    print(json.dumps(payload, indent=2))
    print(f"VERDICT: it runs. native_ready = {payload['native_ready']}")
except Exception:
    print("VERDICT: it does NOT run. Traceback:")
    traceback.print_exc()

# ----------------------------------------------------------------- Q2 + Q3
rule("Q2/Q3  does groupsize, or dummy-vs-real tensors, change which impl resolves?")

results = {}
for cg in (64, 256):
    for kind in ("real", "dummy"):
        key = f"cg={cg:<3} {kind}"
        try:
            if kind == "real":
                w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
                x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
                q4, s4 = ck.quantize_convrot_w4a4_weight(w, cg, 64)
            else:
                # exactly what quant_w4a4.py:93-99 allocates: uninitialized, fp16, 64x64
                w = torch.empty((64, 64), device="cuda", dtype=torch.float16)
                q4 = torch.empty((64, 32), device="cuda", dtype=torch.int8)
                s4 = torch.empty((64,), device="cuda", dtype=torch.float32)
                x = torch.empty((2, 64), device="cuda", dtype=torch.float16)

            qi = R.get_implementation("quantize_convrot_w4a4_weight", kwargs={
                "weight": w, "convrot_groupsize": cg, "quant_group_size": 64,
                "stochastic_rounding": 0})
            li = R.get_implementation("convrot_w4a4_linear", kwargs={
                "x": x, "qweight": q4, "wscales": s4, "bias": None,
                "convrot_groupsize": cg, "quant_group_size": 64, "linear_dtype": "int4"})
            results[key] = (qi.__module__, li.__module__)
        except Exception as exc:
            results[key] = ("EXC: " + type(exc).__name__ + ": " + str(exc)[:90], "-")

w_ = max(len(k) for k in results)
print(f"{'config':<{w_}}  {'quantize impl':<44}  linear impl")
for k, (q, l) in results.items():
    print(f"{k:<{w_}}  {q:<44}  {l}")

qmods = {v[0] for v in results.values()}
lmods = {v[1] for v in results.values()}
print()
print(f"distinct quantize impls across the 4 configs: {len(qmods)}")
print(f"distinct linear   impls across the 4 configs: {len(lmods)}")
print("VERDICT: " + ("groupsize/tensor-kind DOES change resolution -- the 64/64 preflight is "
                     "answering about a config the conversion does not use"
                     if len(qmods) > 1 or len(lmods) > 1 else
                     "resolution is INVARIANT to groupsize and to dummy-vs-real here. The "
                     "quant_w4a4 preflight is untidy, not wrong. NOTE: this tests resolution "
                     "only -- it does not test whether the real CALL succeeds at both."))

# ----------------------------------------------------------------- Q3b
rule("Q3b  does the real CALL succeed where resolution said yes?")
for cg in (64, 256):
    try:
        w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
        q4, s4 = ck.quantize_convrot_w4a4_weight(w, cg, 64)
        out = ck.convrot_w4a4_linear(x, q4, s4, None, cg, 64, "int4")
        print(f"cg={cg:<4} qweight {tuple(q4.shape)} {q4.dtype}  scales {tuple(s4.shape)} "
              f"{s4.dtype}  out {tuple(out.shape)} {out.dtype}  finite={torch.isfinite(out).all().item()}")
    except Exception as exc:
        print(f"cg={cg:<4} CALL FAILED: {type(exc).__name__}: {str(exc)[:160]}")

# ----------------------------------------------------------------- Q4
rule("Q4  does '.backends.cuda' prove the native INT4 path ran?")
print("W4A4 quantizes the ACTIVATION to 4 bits as well. A dequantized fallback")
print("(F.linear(x, W_deq)) keeps x in bf16. So the two must differ measurably.")
print()
try:
    torch.manual_seed(0)
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    q4, s4 = ck.quantize_convrot_w4a4_weight(w, 256, 64)

    native = ck.convrot_w4a4_linear(x, q4, s4, None, 256, 64, "int4")
    li = R.get_implementation("convrot_w4a4_linear", kwargs={
        "x": x, "qweight": q4, "wscales": s4, "bias": None,
        "convrot_groupsize": 256, "quant_group_size": 64, "linear_dtype": "int4"})
    print(f"resolved impl : {li.__module__}")

    bf16_ref = torch.nn.functional.linear(x.float(), w.float())

    # the dequantized-weight comparison: what the eager fallback at
    # comfy_kitchen/tensor/convrot_w4a4.py:237 would compute
    deq = None
    for name in ("dequantize_convrot_w4a4_weight", "dequantize_convrot_w4a4",
                 "convrot_w4a4_dequantize"):
        if hasattr(ck, name):
            deq = getattr(ck, name)
            print(f"dequantizer   : ck.{name}")
            break
    if deq is None:
        print("dequantizer   : none exported by ck -- "
              + str([n for n in dir(ck) if "dequant" in n.lower()]))

    n = native.float()
    r = bf16_ref
    rel = ((n - r).norm() / r.norm()).item()
    print()
    print(f"native  vs fp32 reference : relative L2 = {rel:.6f}")

    if deq is not None:
        wdeq = deq(q4, s4, 256, 64) if deq.__code__.co_argcount >= 4 else deq(q4, s4)
        w4_only = torch.nn.functional.linear(x.float(), wdeq.float())
        rel_w4 = ((w4_only - r).norm() / r.norm()).item()
        gap = ((n - w4_only).norm() / w4_only.norm()).item()
        print(f"W4-only vs fp32 reference : relative L2 = {rel_w4:.6f}")
        print(f"native  vs W4-only        : relative L2 = {gap:.6e}")
        print()
        if gap < 1e-6:
            print("VERDICT: native output is INDISTINGUISHABLE from dequantized-weight BF16 GEMM.")
            print("         The activation is not being quantized. '.backends.cuda' resolved,")
            print("         but W4A4 is not what ran.")
        else:
            print("VERDICT: native differs from dequantized-weight GEMM -- consistent with the")
            print("         activation also being quantized. The A4 half is real.")
    else:
        print()
        print("Cannot run the decisive comparison without a dequantizer. What IS measurable:")
        print(f"  native relative L2 vs fp32 = {rel:.6f}")
        print("  A weight-only-INT4 path lands near 0.02-0.06 on random data; a W4A4 path is")
        print("  markedly worse because the activation is quantized too. NOT CONCLUSIVE alone.")
except Exception:
    print("Q4 failed:")
    traceback.print_exc()

# ----------------------------------------------------------------- Q5
rule("Q5  what dtypes does quantize_w4a8_int8_weight actually return?")
print("quant_w4a8.py:177-208 grew header_dtype()/as_bytes() for bfloat16 and fp8_e5m2.")
print("quant_mixed.py:618/645 handles only float8_e4m3fn and calls .numpy() directly --")
print("so a bfloat16 codebook would raise AFTER every layer has been quantized.")
print()
try:
    w = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    ret = ck.quantize_w4a8_int8_weight(w, group_size=16, convrot_groupsize=256,
                                       symmetric=True, scale_dtype=torch.float8_e4m3fn,
                                       codebook=True, codebook_tensor=None,
                                       stochastic_rounding=0)
    for i, t in enumerate(ret if isinstance(ret, (tuple, list)) else [ret]):
        print(f"  [{i}] {getattr(t, 'dtype', type(t).__name__)}  "
              f"shape={tuple(t.shape) if hasattr(t, 'shape') else '-'}")
    dtypes = [getattr(t, "dtype", None) for t in (ret if isinstance(ret, (tuple, list)) else [ret])]
    risky = [d for d in dtypes if d in (torch.bfloat16, torch.float8_e5m2, torch.float16)]
    print()
    print("VERDICT: " + (f"quant_mixed.py WOULD trip on {risky} -- .numpy() raises on these."
                         if risky else
                         "no returned dtype is one quant_mixed.py mishandles, on this build, "
                         "with these arguments. Latent, not live."))
except Exception as exc:
    print(f"could not call with these arguments: {type(exc).__name__}: {str(exc)[:300]}")
    print("(the signature may differ on this build -- that itself is worth recording)")

print()
print("=" * 78)
print(f"ALL OF THE ABOVE RAN ON: {DEV}")
print("NOT covered: no checkpoint was converted, no end-to-end generation was run, and no")
print("timing was taken -- these are resolution and correctness probes only. Nothing here")
print("says anything about speed.")
print("=" * 78)
