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


def timed(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()

    install_custom_op()
    print("registered opaque custom op convrot_poc::w4a4_linear with a fake impl")

    module, layer, in_features = build_module(args.model.resolve())
    x = torch.randn((1, args.tokens, in_features), device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        eager_out = module(x)
        torch.cuda.synchronize()
        eager_ms = timed(lambda: module(x))
    print(f"layer {layer}  shape {tuple(x.shape)}")
    print(f"eager     : {eager_ms:.3f} ms  out {tuple(eager_out.shape)} {eager_out.dtype}")

    torch._dynamo.reset()
    try:
        compiled = torch.compile(module)
        with torch.no_grad():
            compiled_out = compiled(x)
            torch.cuda.synchronize()
            compiled_ms = timed(lambda: compiled(x))
        diff = (compiled_out.float() - eager_out.float()).abs().max().item()
        print(f"compiled  : {compiled_ms:.3f} ms  speedup {eager_ms / compiled_ms:.2f}x  max_abs_diff {diff:.6f}")
        print("RESULT: torch.compile WORKS on ConvRot W4A4 once the kernel is an opaque custom op")
    except Exception as error:
        print(f"compiled  : STILL FAILS -> {type(error).__name__}")
        print("".join(traceback.format_exception_only(type(error), error))[:1200])
        print("RESULT: the opaque custom op alone is not sufficient")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
