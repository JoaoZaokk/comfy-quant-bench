"""Dequantization of the low-bit affine layout.

    W[n, k] = code[n, k] * scale[n, k // G] + zero[n, k // G]

`code` is an unsigned `bits`-wide integer packed LSB-first along K into uint8, 8 // bits codes per
byte. That is the order MLX (`mx.quantize`, uint32 words read as little-endian bytes) and gemlite
(after transposing its (K / r, N) storage) both use, so neither needs repacking at the bit level.

The arithmetic is fp32 and the result is rounded once to the output dtype. For ternary and binary
codes (0..2) every product code * scale is exact, so the Triton kernel and the torch reference agree
bit for bit regardless of FMA contraction.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs, or a platform without Triton
    triton = None

SUPPORTED_BITS = (1, 2, 4)


def dequantize_torch(qdata, scale, zero, bits, group_size, out_dtype):
    n = qdata.shape[0]
    k = qdata.shape[1] * (8 // bits)
    shifts = torch.arange(0, 8, bits, dtype=torch.uint8, device=qdata.device)
    codes = (qdata.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    codes = codes.view(n, k // group_size, group_size).to(torch.float32)
    w = codes * scale.to(torch.float32).unsqueeze(-1) + zero.to(torch.float32).unsqueeze(-1)
    return w.view(n, k).to(out_dtype)


if triton is not None:
    @triton.jit
    def _dequantize_kernel(q_ptr, s_ptr, z_ptr, out_ptr, N, K,
                           BITS: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        PER_BYTE: tl.constexpr = 8 // BITS
        rn = tl.program_id(0) * BN + tl.arange(0, BN)
        rk = tl.program_id(1) * BK + tl.arange(0, BK)
        mask = (rn[:, None] < N) & (rk[None, :] < K)
        byte = tl.load(q_ptr + rn[:, None] * (K // PER_BYTE) + rk[None, :] // PER_BYTE, mask=mask, other=0).to(tl.int32)
        code = ((byte >> ((rk[None, :] % PER_BYTE) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
        sz = rn[:, None] * (K // G) + rk[None, :] // G
        s = tl.load(s_ptr + sz, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + sz, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + rn[:, None] * K + rk[None, :], (code * s + z).to(out_ptr.dtype.element_ty), mask=mask)


def dequantize_triton(qdata, scale, zero, bits, group_size, out_dtype):
    n = qdata.shape[0]
    k = qdata.shape[1] * (8 // bits)
    out = torch.empty(n, k, dtype=out_dtype, device=qdata.device)
    grid = (triton.cdiv(n, 32), triton.cdiv(k, 128))
    _dequantize_kernel[grid](qdata.contiguous(), scale.contiguous(), zero.contiguous(), out, n, k, bits, group_size, 32, 128)
    return out


def kernel_for(device):
    """'triton' on CUDA when Triton imports, otherwise 'torch'. LOWBIT_KERNEL=torch forces the reference."""
    if os.environ.get("LOWBIT_KERNEL", "").lower() == "torch":
        return "torch"
    return "triton" if triton is not None and torch.device(device).type == "cuda" else "torch"


def dequantize(qdata, scale, zero, bits, group_size, out_dtype):
    if kernel_for(qdata.device) == "triton":
        return dequantize_triton(qdata, scale, zero, bits, group_size, out_dtype)
    return dequantize_torch(qdata, scale, zero, bits, group_size, out_dtype)


def pack_codes(codes, bits):
    """(N, K) integer codes -> (N, K * bits / 8) uint8, LSB-first along K."""
    n, k = codes.shape
    per_byte = 8 // bits
    shifts = torch.arange(0, 8, bits, dtype=torch.int32, device=codes.device)
    grouped = codes.to(torch.int32).view(n, k // per_byte, per_byte) << shifts
    return grouped.sum(-1, dtype=torch.int32).to(torch.uint8)
