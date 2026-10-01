"""Tests for patches/comfy_kitchen_awq_w4a16_triton.patch, against the installed comfy-kitchen.

CPU by default (run with CUDA hidden):

    CUDA_VISIBLE_DEVICES=-1 python_embeded/python.exe -s -m pytest patches/tests/test_comfy_kitchen_awq.py

The Triton parity test needs a GPU and runs only when asked, on the card named explicitly:

    KITCHEN_TEST_CUDA=1 KITCHEN_TEST_DEVICE=cuda:1 python_embeded/python.exe -s -m pytest patches/tests/test_comfy_kitchen_awq.py
"""

import os

import pytest
import torch

from comfy_kitchen.backends.eager import awq as eager_awq
from comfy_kitchen.registry import registry


def awq_inputs(n=48, k=256, group_size=64, dtype=torch.bfloat16, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.int32)
    qweight = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    wscales = (torch.rand(k // group_size, n, generator=gen) * 0.01 + 0.001).to(dtype)
    wzeros = (torch.randn(k // group_size, n, generator=gen) * 0.01).to(dtype)
    return codes, qweight.to(device), wscales.to(device), wzeros.to(device)


def old_cuda_torch_chain(qweight, wscales, wzeros, group_size):
    """The unpack/stack/scale chain comfy-kitchen 0.2.35 ran in _awq_w4a16_dequant_then_matmul."""
    n, k_half = qweight.shape
    k = k_half * 2
    g = group_size
    x32 = qweight.to(torch.int32)
    lo = (x32 & 0xF).to(torch.int8)
    hi = ((x32 >> 4) & 0xF).to(torch.int8)
    nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).to(wscales.dtype)
    return ((nibbles.view(n, k // g, g) - 8.0) * wscales.t().unsqueeze(-1) + wzeros.t().unsqueeze(-1)).view(n, k)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_eager_dequant_matches_the_formula(dtype):
    codes, qweight, wscales, wzeros = awq_inputs(dtype=dtype)
    n, k = codes.shape
    expected = ((codes.view(n, k // 64, 64).to(dtype) - 8.0) * wscales.t().unsqueeze(-1)
                + wzeros.t().unsqueeze(-1)).view(n, k)
    assert torch.equal(eager_awq.dequantize_awq_w4a16(qweight, wscales, wzeros, 64), expected)


def test_eager_dequant_is_bit_identical_to_the_old_cuda_torch_chain():
    _, qweight, wscales, wzeros = awq_inputs()
    assert torch.equal(eager_awq.dequantize_awq_w4a16(qweight, wscales, wzeros, 64),
                       old_cuda_torch_chain(qweight, wscales, wzeros, 64))


def test_eager_gemv_is_unchanged_by_the_refactor():
    _, qweight, wscales, wzeros = awq_inputs()
    x = torch.randn(5, 256, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1))
    bias = torch.randn(48, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(2))
    expected = x @ old_cuda_torch_chain(qweight, wscales, wzeros, 64).t() + bias
    assert torch.equal(eager_awq.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, 64), expected)


def test_large_m_fallback_dispatches_through_the_registry_to_eager_on_cpu():
    cuda_backend = pytest.importorskip("comfy_kitchen.backends.cuda")
    _, qweight, wscales, wzeros = awq_inputs()
    kwargs = {"qweight": qweight, "wscales": wscales, "wzeros": wzeros, "group_size": 64}
    assert registry.get_capable_backend("dequantize_awq_w4a16", kwargs) == "eager"
    x = torch.randn(300, 256, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(3))
    out = cuda_backend._awq_w4a16_dequant_then_matmul(x, qweight, wscales, wzeros, 64)
    assert torch.equal(out, x.matmul(old_cuda_torch_chain(qweight, wscales, wzeros, 64).t()))


def _test_device():
    if os.environ.get("KITCHEN_TEST_CUDA") != "1":
        pytest.skip("GPU test: set KITCHEN_TEST_CUDA=1 and KITCHEN_TEST_DEVICE=cuda:N")
    device = os.environ.get("KITCHEN_TEST_DEVICE", "")
    if not device.startswith("cuda:"):
        pytest.fail("KITCHEN_TEST_DEVICE must name the card explicitly, e.g. cuda:1")
    return torch.device(device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_dequant_matches_a_single_rounding_of_the_fp32_reference(dtype):
    device = _test_device()
    from comfy_kitchen.backends.triton import awq as triton_awq

    _, qweight, wscales, wzeros = awq_inputs(n=4096, k=4096, group_size=32, dtype=dtype, device=device)
    with torch.cuda.device(device):
        got = triton_awq.dequantize_awq_w4a16(qweight, wscales, wzeros, 32)
    reference = eager_awq.dequantize_awq_w4a16(qweight, wscales.float(), wzeros.float(), 32)
    # Triton computes in fp32 and rounds once; FMA contraction may move the fp32 value by an ulp,
    # which can flip the final rounding of a few elements by one ulp of the output dtype.
    assert got.dtype == dtype
    assert torch.allclose(got.float(), reference, rtol=2 ** -7, atol=1e-6)
    assert (got != reference.to(dtype)).float().mean().item() < 1e-3
