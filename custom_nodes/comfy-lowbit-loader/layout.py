"""The `lowbit_affine` quantization format: a comfy-kitchen layout plus its ComfyUI registration.

Weights stay packed (1, 2 or 4 bits) in VRAM and in the offload copy; each linear dequantizes to the
compute dtype right before a regular matmul. Nothing here edits a ComfyUI file: the layout is
registered through `comfy.quant_ops`, and the one closed `if/elif` in the loader
(`comfy.ops._load_quantized_module`) is wrapped so that only `lowbit_affine` layers take the new
branch. Saving needs no hook: bits and group size are recovered from the tensor shapes.
"""

import json
import logging
from dataclasses import dataclass

import torch

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor
from comfy_kitchen.tensor.base import BaseLayoutParams, QuantizedLayout

from . import kernel

FORMAT = "lowbit_affine"
LAYOUT = "LowBitAffineLayout"


class LowBitAffineLayout(QuantizedLayout):
    """Packed unsigned codes with a per-group affine map along K: W = code * scale + zero."""

    MIN_SM_VERSION = None
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        zero: torch.Tensor
        bits: int = 2
        group_size: int = 128

        def _tensor_fields(self):
            return ["scale", "zero"]

        def _validate_tensor_fields(self):
            return

    @classmethod
    def quantize(cls, tensor, **kwargs):
        raise NotImplementedError("lowbit_affine weights are loaded pre-quantized; requantizing a float tensor would change the model.")

    @classmethod
    def dequantize(cls, qdata, params):
        return kernel.dequantize(qdata, params.scale, params.zero, params.bits, params.group_size, params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale, qtensor._params.zero

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        return {"": qdata, "_scale": params.scale, "_zeros": params.zero}


def shape_params(qdata, scale, k):
    """(bits, group_size) implied by the packed width and the number of groups for an input size K."""
    bits = qdata.shape[1] * 8 // k
    if bits not in kernel.SUPPORTED_BITS or qdata.shape[1] * 8 != bits * k or k % scale.shape[1]:
        raise ValueError(f"lowbit_affine: packed width {qdata.shape[1]} and {scale.shape[1]} groups do not fit K={k}")
    return bits, k // scale.shape[1]


_original_load = comfy.ops._load_quantized_module


def _load_quantized_module(module, super_load, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    conf = state_dict.get(f"{prefix}comfy_quant")
    if conf is None or json.loads(conf.numpy().tobytes()).get("format") != FORMAT:
        return _original_load(module, super_load, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, load_extra_params=load_extra_params)

    device = module.factory_kwargs["device"]
    keys = [f"{prefix}{name}" for name in ("comfy_quant", "weight", "weight_scale", "weight_zeros")]
    _, qdata, scale, zero = (state_dict.pop(key) for key in keys)
    bits, group_size = shape_params(qdata, scale, module._orig_shape[1])
    params = LowBitAffineLayout.Params(
        scale=scale.to(device=device), zero=zero.to(device=device), bits=bits, group_size=group_size,
        orig_dtype=module.factory_kwargs["dtype"], orig_shape=module._orig_shape,
    )
    module.quant_format = FORMAT
    module.layout_type = LAYOUT
    module._full_precision_mm_config = False
    module.weight = torch.nn.Parameter(QuantizedTensor(qdata.to(device=device, dtype=torch.uint8), LAYOUT, params), requires_grad=False)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in keys:
        if key in missing_keys:
            missing_keys.remove(key)


def register():
    comfy.quant_ops.register_layout_class(LAYOUT, LowBitAffineLayout)
    QUANT_ALGOS[FORMAT] = {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_zeros"},
        "comfy_tensor_layout": LAYOUT,
        "quantize_input": False,
    }
    if comfy.ops._load_quantized_module is not _load_quantized_module:
        comfy.ops._load_quantized_module = _load_quantized_module
    logging.info("lowbit_affine registered (%s kernel on CUDA)", kernel.kernel_for("cuda") if torch.cuda.is_available() else "torch")
