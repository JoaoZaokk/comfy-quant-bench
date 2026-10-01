"""The `lowbit_affine` quantization format: a comfy-kitchen layout plus its ComfyUI registration.

Weights stay packed (1, 2 or 4 bits) in VRAM and in the offload copy; each linear dequantizes to the
compute dtype right before a regular matmul. The layout is registered through `comfy.quant_ops`, and the
loader reads a layer's scales through `QUANT_ALGOS[format]["params_from_state_dict"]`, the reader hook
`comfy.ops._load_quantized_module` calls (local ComfyUI patch `patches/comfyui_awq_w4a16_format.patch`).
Saving needs no hook: bits and group size are recovered from the tensor shapes.
"""

import inspect
import logging
from dataclasses import dataclass

import torch

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QUANT_ALGOS
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


def params_from_state_dict(layer_name, module, weight, pop_scale, layer_conf):
    """Layout params for one layer: scale and zero from the file, bits and group size from the shapes."""
    scale, zero = pop_scale("weight_scale"), pop_scale("weight_zeros")
    if scale is None or zero is None:
        raise ValueError(f"lowbit_affine: missing weight_scale/weight_zeros for layer {layer_name}")
    bits, group_size = shape_params(weight, scale, module._orig_shape[1])
    return {"scale": scale, "zero": zero, "bits": bits, "group_size": group_size}


def register():
    comfy.quant_ops.register_layout_class(LAYOUT, LowBitAffineLayout)
    QUANT_ALGOS[FORMAT] = {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_zeros"},
        "comfy_tensor_layout": LAYOUT,
        "quantize_input": False,
        "params_from_state_dict": params_from_state_dict,
    }
    if "params_from_state_dict" not in inspect.getsource(comfy.ops._load_quantized_module):
        logging.error("lowbit_affine: this ComfyUI has no params_from_state_dict reader hook "
                      "(patches/comfyui_awq_w4a16_format.patch is not applied); lowbit files will fail to load")
    logging.info("lowbit_affine registered (%s kernel on CUDA)", kernel.kernel_for("cuda") if torch.cuda.is_available() else "torch")
