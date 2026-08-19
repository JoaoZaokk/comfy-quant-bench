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

"""Full-traceback torch.compile investigation for ConvRot W4A4.

comfy_kitchen's QuantizedTensor implements __tensor_flatten__/__tensor_unflatten__, so it was
built to be traceable. This finds out what actually fails, and whether any backend/mode works.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import traceback
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

DTYPES = {"I8": torch.int8, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def read_header(path: Path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    return raw, dict(raw.pop("__metadata__", {}) or {})


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


def build_module(model: Path):
    import comfy.ops

    header, metadata = read_header(model)
    layers = json.loads(metadata["_quantization_metadata"])["layers"]
    layer = next(iter(layers))
    weight_info = header[f"{layer}.weight"]
    scale_info = header[f"{layer}.weight_scale"]
    rows, packed_columns = weight_info["shape"]

    operations = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=False)
    module = operations.Linear(packed_columns * 2, rows, bias=False, device="cuda")
    conf = json.dumps({"format": "convrot_w4a4", "convrot_groupsize": 256}).encode("utf-8")
    module._load_from_state_dict(
        {
            "weight": load_tensor(model, weight_info),
            "weight_scale": load_tensor(model, scale_info),
            "comfy_quant": torch.tensor(list(conf), dtype=torch.uint8),
        },
        "", {}, False, [], [], [],
    )
    return module, layer, packed_columns * 2


def attempt(label, build_callable, x, reference):
    torch._dynamo.reset()
    try:
        fn = build_callable()
        with torch.no_grad():
            out = fn(x)
            torch.cuda.synchronize()
        diff = (out.float() - reference.float()).abs().max().item()
        print(f"[ OK ] {label}: max_abs_diff_vs_eager={diff:.6f}")
        return True
    except Exception as error:
        print(f"[FAIL] {label}: {type(error).__name__}")
        text = "".join(traceback.format_exception_only(type(error), error))
        for line in text.splitlines()[:14]:
            print(f"       {line}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()

    module, layer, in_features = build_module(args.model.resolve())
    x = torch.randn((1, args.tokens, in_features), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference = module(x)
    print(f"layer {layer}  in_features {in_features}  eager ok, out {tuple(reference.shape)} {reference.dtype}")
    print(f"torch {torch.__version__}")
    print()

    attempt("backend=eager", lambda: torch.compile(module, backend="eager"), x, reference)
    attempt("backend=aot_eager", lambda: torch.compile(module, backend="aot_eager"), x, reference)
    attempt("backend=inductor (default)", lambda: torch.compile(module), x, reference)
    attempt("inductor, dynamic=False", lambda: torch.compile(module, dynamic=False), x, reference)

    def with_subclass_registered():
        import comfy_kitchen.tensor.base as base
        torch._dynamo.config.traceable_tensor_subclasses.add(base.QuantizedTensor)
        return torch.compile(module)

    if hasattr(torch._dynamo.config, "traceable_tensor_subclasses"):
        attempt("inductor + traceable_tensor_subclasses", with_subclass_registered, x, reference)
    else:
        print("[skip] torch._dynamo.config.traceable_tensor_subclasses not present in this torch")

    print()
    print("=== ComfyUI's own wrapper path ===")
    torch._dynamo.reset()
    try:
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        print("set_torch_compile_wrapper importable:", set_torch_compile_wrapper is not None)
    except Exception as error:
        print(f"could not import ComfyUI wrapper: {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
