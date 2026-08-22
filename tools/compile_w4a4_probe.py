# SUPERSEDED -- 2026-08-18. Do not trust this file's verdict.
#
# The question it asks (does torch.compile survive a ConvRot QuantizedTensor?) is answered, and
# the answer is in W4A4_PROGRESS.md: "torch.compile works, max_abs_diff 0.000000", re-verified
# under torch 2.13. Running probe2.py today prints [FAIL] for the backend and contradicts that
# measured result, so it produces a wrong answer rather than no answer.
#
# Kept rather than deleted because deleting is not reversible and nothing depends on the call.
# The working reproduction lives in compile_w4a4_fix_poc.py, whose docstring carries the fix.
# These two also duplicate read_header/load_tensor/build_module/DTYPES from that file almost
# byte for byte.

"""Does torch.compile survive a ConvRot W4A4 QuantizedTensor weight?

QuantizedTensor is a __torch_dispatch__ subclass, so torch.compile may graph-break, fall back to
eager, silently dequantize, or fail outright. This builds a real MixedPrecisionOps.Linear from a
converted checkpoint, runs it eager and compiled, and reports correctness, native dispatch and
speed for both.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
# The embedded interpreter's `python313._pth` suppresses the script-directory entry, so the two
# sibling imports below need this (`m_crossover.py:49-52`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _timing import compare, cuda_event_ms, provenance  # noqa: E402

DTYPES = {"I8": torch.int8, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20,
                        help="timed iterations inside one burst")
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved bursts KEPT; one more is run and discarded as "
                             "warm-up-biased")
    return parser.parse_args()


def read_header(path: Path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    return raw, metadata


def load_tensor(path: Path, info: dict) -> torch.Tensor:
    start, end = info["data_offsets"]
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_size + start)
        raw = bytearray(end - start)
        view = memoryview(raw)
        position = 0
        while position < len(raw):
            read = handle.readinto(view[position:])
            if not read:
                raise EOFError("short read")
            position += read
    return torch.frombuffer(raw, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).cuda()


def instrument():
    import comfy_kitchen.tensor.convrot_w4a4 as convrot
    from comfy_kitchen.registry import registry

    counters = {"linear": 0, "dequant": 0, "impls": set()}
    original_linear = convrot.convrot_w4a4_linear
    original_dequant = convrot.TensorCoreConvRotW4A4Layout.dequantize.__func__

    def counting_linear(x, qweight, wscales, bias=None, **kwargs):
        counters["linear"] += 1
        impl = registry.get_implementation(
            "convrot_w4a4_linear",
            kwargs={"x": x, "qweight": qweight, "wscales": wscales, "bias": bias, **kwargs},
        )
        counters["impls"].add(f"{impl.__module__}.{impl.__name__}")
        return original_linear(x, qweight, wscales, bias=bias, **kwargs)

    def counting_dequant(cls, qdata, params):
        counters["dequant"] += 1
        return original_dequant(cls, qdata, params)

    convrot.convrot_w4a4_linear = counting_linear
    convrot.TensorCoreConvRotW4A4Layout.dequantize = classmethod(counting_dequant)
    return counters


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    header, metadata = read_header(model)
    layers = json.loads(metadata["_quantization_metadata"])["layers"]
    layer = next(iter(layers))

    import comfy.ops

    counters = instrument()
    weight_info = header[f"{layer}.weight"]
    scale_info = header[f"{layer}.weight_scale"]
    rows, packed_columns = weight_info["shape"]
    in_features = packed_columns * 2

    operations = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=False)
    module = operations.Linear(in_features, rows, bias=False, device="cuda")
    layer_conf = json.dumps({"format": "convrot_w4a4", "convrot_groupsize": 256}).encode("utf-8")
    state_dict = {
        "weight": load_tensor(model, weight_info),
        "weight_scale": load_tensor(model, scale_info),
        "comfy_quant": torch.tensor(list(layer_conf), dtype=torch.uint8),
    }
    module._load_from_state_dict(state_dict, "", {}, False, [], [], [])

    x = torch.randn((1, args.tokens, in_features), device="cuda", dtype=torch.bfloat16)

    report = {"layer": layer, "shape": [1, args.tokens, in_features], "out_features": rows}

    # The dispatch counters have to be read from a run of their own, before any timing: they count
    # per call, and a burst of 20 would make "native_calls" a burst size rather than an answer.
    with torch.no_grad():
        counters["linear"] = counters["dequant"] = 0
        counters["impls"] = set()
        eager_out = module(x)
        torch.cuda.synchronize()
        report["eager"] = {
            "native_calls": counters["linear"],
            "dequant_calls": counters["dequant"],
            "impls": sorted(counters["impls"]),
        }

    compiled = None
    try:
        compiled = torch.compile(module)
        with torch.no_grad():
            counters["linear"] = counters["dequant"] = 0
            counters["impls"] = set()
            compiled_out = compiled(x)
            torch.cuda.synchronize()
        difference = (compiled_out.float() - eager_out.float())
        report["compiled"] = {
            "status": "ok",
            "native_calls_first_run": counters["linear"],
            "dequant_calls_first_run": counters["dequant"],
            "impls": sorted(counters["impls"]),
            "max_abs_diff_vs_eager": difference.abs().max().item(),
            "still_native": counters["linear"] > 0 and counters["dequant"] == 0,
        }
    except Exception as error:
        report["compiled"] = {"status": "FAILED",
                              "error": f"{type(error).__name__}: {str(error)[:400]}"}

    # Both arms exist by now, so they can be interleaved. This file used to time eager to
    # completion, then compile, then time compiled -- with a whole Inductor compilation sitting
    # between the two measurements, which is the single largest thing that can move a clock on
    # this machine. It also took the mean of one CUDA-event span over 20 iterations: a mean over
    # one span cannot express a spread at all, and MEASURED 2026-08-22 on the 3090 the first burst
    # of an A/B here reads ~3% high in a fixed direction. `compare()` discards that burst.
    with torch.no_grad():
        paths = {"eager": lambda: module(x)}
        if compiled is not None and report["compiled"]["status"] == "ok":
            paths["compiled"] = lambda: compiled(x)
        result = compare(paths, iters=args.iters, repeats=args.repeats,
                         baseline="eager" if len(paths) > 1 else None,
                         timer=cuda_event_ms, owner="comfy_portable:compile_w4a4_probe")
    report["timing"] = result.as_json()
    report["eager"]["ms"] = round(result.times["eager"], 3)
    if "compiled" in result.times:
        report["compiled"]["ms"] = round(result.times["compiled"], 3)
        # Not a bare float any more. `"speedup_vs_eager": 1.676` is a number in exactly the shape
        # that gets pasted into another session as if it had been measured three times.
        report["compiled"]["speedup_vs_eager"] = str(result.ratios["compiled"])

    print(json.dumps(report, indent=2))
    print(f"\n{provenance(result)}")
    return 0


if __name__ == "__main__":
    # `compare()` would take the lock on its own, but taking it here covers the load, the eager
    # forward and the Inductor compilation too -- all of which are on the card, and the last of
    # which is the longest single thing this file does.
    from _bench_guard import BenchGuard

    with BenchGuard("comfy_portable:compile_w4a4_probe") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
