"""Tests for the 2026-10-03 local patches (fase 4) against the live ComfyUI checkout and the installed comfy-kitchen:

- comfy/ops.py `_convrot_w4a4_input_act` (patches/comfyui_w4a4_swiglu_input_act.patch): linear_input_act hands
  W4A4 ConvRot layers the [gate | up] pair with input_act="swiglu" and comfy-kitchen folds the activation into its
  quantizer (patches/comfy_kitchen_swiglu_w4a4_fused.patch + tools/ck_swiglu);
- comfy/ldm/modules/attention.py + comfy/model_base.py (patches/comfyui_sage_model_policy.patch): a model's
  `sage_attention` settings reach attention_sage through transformer_options and override the env globals.

CPU by default (run with CUDA hidden):

    CUDA_VISIBLE_DEVICES=-1 python_embeded/python.exe -s -m pytest patches/tests/test_comfyui_fase4_dispatch.py

The fused-kernel test needs a GPU and runs only when asked, on the card named explicitly:

    KITCHEN_TEST_CUDA=1 KITCHEN_TEST_DEVICE=cuda:0 python_embeded/python.exe -s -m pytest patches/tests/test_comfyui_fase4_dispatch.py
"""

import json
import os
import sys
import unittest.mock

import pytest
import torch

sys.path.insert(0, os.environ.get("COMFYUI_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI")))

from comfy.cli_args import args  # noqa: E402

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_base  # noqa: E402
import comfy.ldm.modules.attention as attention  # noqa: E402
import comfy.utils  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor  # noqa: E402

CUDA = os.environ.get("KITCHEN_TEST_CUDA") == "1"
DEVICE = os.environ.get("KITCHEN_TEST_DEVICE", "cuda:0")


def convrot_layer(k, n, linear_dtype, device="cpu", seed=456):
    torch.manual_seed(seed)
    weight = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n, dtype=torch.bfloat16) * 0.01
    q_weight = QuantizedTensor.from_float(weight.to(device), "TensorCoreConvRotW4A4Layout", convrot_groupsize=256, quant_group_size=64)
    state_dict = {"layer.weight": q_weight._qdata, "layer.bias": bias.to(device), "layer.weight_scale": q_weight._params.scale}
    layers = {"layer": {"format": "convrot_w4a4", "convrot_groupsize": 256, "linear_dtype": linear_dtype}}
    state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layers})})
    model = torch.nn.Module()
    model.layer = ops.mixed_precision_ops({}).Linear(k, n, device=device, dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=False)
    return model.layer


needs_w4a4 = pytest.mark.skipif("convrot_w4a4" not in QUANT_ALGOS, reason="comfy_kitchen does not provide ConvRot W4A4")


@needs_w4a4
def test_w4a4_swiglu_routes_to_kitchen_and_matches_forward():
    layer = convrot_layer(256, 16, "int8")
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    orig = ops.quant_ops.ck.convrot_w4a4_linear
    spy = unittest.mock.Mock(side_effect=orig)
    with unittest.mock.patch.object(ops.quant_ops.ck, "convrot_w4a4_linear", spy):
        out = ops.linear_input_act(layer, x, "swiglu")
    assert spy.call_args.kwargs.get("input_act") == "swiglu"
    assert spy.call_args.args[0] is x                                   # the [gate | up] pair, not the activation
    expected = layer(ops.INPUT_ACT_EAGER["swiglu"](x))                  # the unpatched path: eager act + Linear.forward
    assert torch.equal(out, expected)


@needs_w4a4
def test_w4a4_swiglu_keeps_eager_path_when_disabled_or_patched():
    layer = convrot_layer(256, 16, "int8")
    x = torch.randn(4, 512, dtype=torch.bfloat16)
    with unittest.mock.patch.object(ops, "_W4A4_INPUT_ACT", False):
        assert ops._convrot_w4a4_input_act(layer, x, "swiglu") is None
    assert ops._convrot_w4a4_input_act(layer, x, "gelu_tanh") is None
    layer.weight_function.append(lambda w: w)                           # a LoRA patch: Linear.forward dequantizes
    try:
        assert ops._convrot_w4a4_input_act(layer, x, "swiglu") is None
    finally:
        layer.weight_function.clear()
    assert ops._convrot_w4a4_input_act(torch.nn.Linear(256, 16), x, "swiglu") is None


@needs_w4a4
@pytest.mark.skipif(not CUDA, reason="set KITCHEN_TEST_CUDA=1 to run the fused-kernel test")
def test_w4a4_swiglu_fused_kernel_cuda():
    import comfy_kitchen.backends.cuda as ckc
    assert ckc._swiglu_quant is not None, "tools/ck_swiglu/build.bat --install"
    layer = convrot_layer(1024, 256, "int4", device=DEVICE)
    x = torch.randn(64, 2048, dtype=torch.bfloat16, device=DEVICE)
    out = ops.linear_input_act(layer, x, "swiglu")
    eager = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
    ref = torch.nn.functional.linear(ops.INPUT_ACT_EAGER["swiglu"](x.float()), layer.weight.dequantize().float(), layer.bias.float())
    rel = lambda o: float((o.float() - ref).norm() / ref.norm())  # noqa: E731
    assert torch.isfinite(out).all()
    assert rel(out) <= rel(eager) * 1.05                              # one rounding of the intermediate less, never worse
    # The fused quantizer is the installed one fed silu(gate) * up rounded once: identical codes and scales. The kernel
    # computes silu with __expf (--use_fast_math), so against torch's precise silu a bf16 intermediate rounds differently
    # in ~1 of 1e8 elements (fp16: ~6e-5), which none of these 65k elements hits. (The two
    # layer outputs differ by ~5 %: ~1 % of the int4 codes flip with the intermediate's rounding, as any two valid
    # quantizations do -- so the outputs are compared to the reference above, not to each other.)
    k = x.shape[-1] // 2
    q_f, s_f = ckc._swiglu_quant.swiglu_quantize_int4_rowwise_convrot64(x)
    q_c, s_c = ckc.quantize_int4_rowwise_convrot64((torch.nn.functional.silu(x[:, :k].float()) * x[:, k:].float()).to(x.dtype), 256)
    assert torch.equal(q_f, q_c) and torch.equal(s_f.reshape(-1), s_c.reshape(-1))


def _sage_calls(monkeypatch, transformer_options):
    calls = []

    def fake_kernel(q, k, v, **kw):
        calls.append(kw)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "_sageattn_pv_fp16_cuda", fake_kernel)
    monkeypatch.setattr(attention, "_sage_sm8x", lambda device: True)
    monkeypatch.setattr(attention, "SAGE_PV_ACCUM", "fp16sv")
    monkeypatch.setattr(attention, "SAGE_SMOOTH_K", False)
    monkeypatch.setattr(attention, "SAGE_QK_GRAN", "per_thread")
    q = torch.randn(1, 4, 8, 64)
    k = torch.randn(1, 4, 8, 64)
    v = torch.randn(1, 4, 8, 64)
    attention.attention_sage(q, k, v, 4, skip_reshape=True, transformer_options=transformer_options)
    return calls[-1]


def test_sage_model_policy_overrides_env_globals(monkeypatch):
    kw = _sage_calls(monkeypatch, {"sage_attention": comfy.model_base.Krea2.sage_attention})
    assert kw["smooth_k"] is True and kw["qk_quant_gran"] == "per_warp"


def test_sage_without_policy_keeps_env_globals(monkeypatch):
    kw = _sage_calls(monkeypatch, {})
    assert kw["smooth_k"] is False and kw["qk_quant_gran"] == "per_thread"
    monkeypatch.setattr(attention, "SAGE_MODEL_POLICY", False)
    kw = _sage_calls(monkeypatch, {"sage_attention": {"qk_quant_gran": "per_warp", "smooth_k": True}})
    assert kw["smooth_k"] is False and kw["qk_quant_gran"] == "per_thread"


def test_only_krea2_declares_a_policy():
    assert comfy.model_base.BaseModel.sage_attention is None
    assert comfy.model_base.Krea2.sage_attention == {"qk_quant_gran": "per_warp", "smooth_k": True}
    declared = sorted(name for name, cls in vars(comfy.model_base).items()
                      if isinstance(cls, type) and issubclass(cls, comfy.model_base.BaseModel)
                      and "sage_attention" in vars(cls))
    assert declared == ["BaseModel", "Krea2"]
