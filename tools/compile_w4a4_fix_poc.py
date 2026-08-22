"""Proof of concept: wrap convrot_w4a4_linear as an opaque custom op so torch.compile works.

The failure is not the QuantizedTensor subclass -- comfy_kitchen already implements the
flatten/unflatten protocol. It is that the CUDA kernel is not registered as an opaque custom op,
so Dynamo traces into it and hits a FakeTensor data pointer. PyTorch's documented fix is
torch.library.custom_op + register_fake. This patches that in at runtime and re-tests.

Runtime only. No comfy_kitchen file is modified.
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
# The embedded interpreter's `python313._pth` suppresses the script-directory entry, so the two
# sibling imports below need this (`m_crossover.py:49-52`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _timing import compare, cuda_event_ms, provenance  # noqa: E402

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


def install_custom_op():
    """Register convrot_w4a4_linear as an opaque custom op with a meta (fake) implementation."""
    import comfy_kitchen.tensor.convrot_w4a4 as convrot

    original = convrot.convrot_w4a4_linear

    @torch.library.custom_op("convrot_poc::w4a4_linear", mutates_args=())
    def w4a4_linear(
        x: torch.Tensor,
        qweight: torch.Tensor,
        wscales: torch.Tensor,
        bias: torch.Tensor | None,
        convrot_groupsize: int,
        quant_group_size: int,
        linear_dtype: str,
    ) -> torch.Tensor:
        return original(
            x, qweight, wscales, bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )

    @w4a4_linear.register_fake
    def _(x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype):
        # packed int8 holds two int4 per byte, so out_features is qweight.shape[0]
        out_features = qweight.shape[0]
        return x.new_empty((*x.shape[:-1], out_features))

    def dispatching(x, qweight, wscales, bias=None, convrot_groupsize=256, quant_group_size=64, linear_dtype="int4"):
        return torch.ops.convrot_poc.w4a4_linear(
            x, qweight, wscales, bias, int(convrot_groupsize), int(quant_group_size), str(linear_dtype)
        )

    convrot.convrot_w4a4_linear = dispatching
    return original


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=30,
                        help="timed iterations inside one burst")
    parser.add_argument("--repeats", type=int, default=3,
                        help="interleaved bursts KEPT; one more is run and discarded as "
                             "warm-up-biased")
    args = parser.parse_args()

    install_custom_op()
    print("registered opaque custom op convrot_poc::w4a4_linear with a fake impl")

    module, layer, in_features = build_module(args.model.resolve())
    x = torch.randn((1, args.tokens, in_features), device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        eager_out = module(x)
        torch.cuda.synchronize()
    print(f"layer {layer}  shape {tuple(x.shape)}")
    print(f"eager     : out {tuple(eager_out.shape)} {eager_out.dtype}")

    torch._dynamo.reset()
    compiled = None
    try:
        compiled = torch.compile(module)
        with torch.no_grad():
            compiled_out = compiled(x)
            torch.cuda.synchronize()
        diff = (compiled_out.float() - eager_out.float()).abs().max().item()
    except Exception as error:
        print(f"compiled  : STILL FAILS -> {type(error).__name__}")
        print("".join(traceback.format_exception_only(type(error), error))[:1200])
        print("RESULT: the opaque custom op alone is not sufficient")
        return 0

    # Interleaved, and only after the compilation is finished. Timing eager, then compiling, then
    # timing compiled puts an Inductor build -- minutes of CPU and a busy card -- between the two
    # halves of the comparison, so any clock or allocator drift over it lands entirely on the
    # compiled arm and reads as a speedup. The old form also took the mean of a single CUDA-event
    # span, which cannot express a spread at all.
    with torch.no_grad():
        result = compare({"eager": lambda: module(x), "compiled": lambda: compiled(x)},
                         iters=args.iters, repeats=args.repeats, baseline="eager",
                         timer=cuda_event_ms, owner="comfy_portable:compile_w4a4_fix_poc")
    print(f"eager     : {result.times['eager']:.3f} ms")
    print(f"compiled  : {result.times['compiled']:.3f} ms  "
          f"speedup {result.ratios['compiled']}  max_abs_diff {diff:.6f}")
    print("RESULT: torch.compile WORKS on ConvRot W4A4 once the kernel is an opaque custom op")
    print(f"\n{provenance(result)}")
    return 0


if __name__ == "__main__":
    # Held across the load and the Inductor compilation as well as the timing; `compare()` would
    # take it anyway, but only around the bursts, and the compilation is the longer occupation.
    from _bench_guard import BenchGuard

    with BenchGuard("comfy_portable:compile_w4a4_fix_poc") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
