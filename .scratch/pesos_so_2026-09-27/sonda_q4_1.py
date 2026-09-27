"""Q4_1 -> layout AWQ W4A16 do comfy-kitchen: confere o reempacotamento contra o gguf-py e mede desquantizacao
(torch do comfy-kitchen x Triton fundido) contra a matmul BF16. So leitura do BF16 original.

    python_embeded\\python.exe -s sonda_q4_1.py
"""
import json
import struct
import time

import gguf
import numpy as np
import torch
import triton
import triton.language as tl

FONTE = "P:/ComfyBench/diffusion_models/qwen_image_2.1_bf16.safetensors"
CAMADA = "transformer_blocks.10.img_mlp.gate_up.weight"
G = 32


def le_camada():
    with open(FONTE, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        i = h[CAMADA]
        a, b = i["data_offsets"]
        f.seek(8 + n + a)
        return torch.frombuffer(bytearray(f.read(b - a)), dtype=torch.bfloat16).reshape(i["shape"])


def q4_1_para_awq(w):
    """Codigos Q4_1 do gguf-py (d, m fp16 por bloco de 32) no layout AWQ: (q-8)*s + z, z = m + 8d."""
    n, k = w.shape
    blocos = gguf.quants.quantize(w.float().numpy(), gguf.GGMLQuantizationType.Q4_1).reshape(n, k // G, 20)
    d = blocos[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    m = blocos[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    qs = blocos[..., 4:20]
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(n, k)  # ggml: 16 baixos = j, 16 altos = j+16
    packed = (vals[:, 0::2] | (vals[:, 1::2] << 4)).astype(np.uint8)    # kitchen: baixo = coluna par
    ref = gguf.quants.dequantize(blocos.reshape(n, -1), gguf.GGMLQuantizationType.Q4_1).reshape(n, k)
    return (torch.from_numpy(packed.view(np.int8)), torch.from_numpy(d.T.copy()).to(torch.bfloat16),
            torch.from_numpy((m + 8 * d).T.copy()).to(torch.bfloat16), torch.from_numpy(ref))


@triton.jit
def _awq_dequant(q_ptr, s_ptr, z_ptr, out_ptr, N, K, G: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (rn[:, None] < N) & (rk[None, :] < K)
    byte = tl.load(q_ptr + rn[:, None] * (K // 2) + rk[None, :] // 2, mask=mask, other=0).to(tl.int32) & 0xFF
    nib = tl.where(rk[None, :] % 2 == 0, byte & 0xF, byte >> 4).to(tl.float32)
    g = rk[None, :] // G
    s = tl.load(s_ptr + g * N + rn[:, None], mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + g * N + rn[:, None], mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + rn[:, None] * K + rk[None, :], ((nib - 8.0) * s + z).to(out_ptr.dtype.element_ty), mask=mask)


def dequant_triton(q, s, z, group_size):
    n, k = q.shape[0], q.shape[1] * 2
    out = torch.empty(n, k, dtype=s.dtype, device=q.device)
    _awq_dequant[(triton.cdiv(n, 32), triton.cdiv(k, 128))](q, s, z, out, n, k, group_size, 32, 128)
    return out


def cronometra(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


w = le_camada()
q, s, z, ref = q4_1_para_awq(w)
from comfy_kitchen.tensor.awq_w4a16 import TensorCoreAWQW4A16Layout as L  # noqa: E402

params = L.Params(scale=s, zeros=z, group_size=G, orig_dtype=torch.bfloat16, orig_shape=tuple(w.shape))
cpu = L.dequantize(q, params).float()
print("camada", CAMADA, tuple(w.shape))
print("AWQ(eager, cpu) x dequant do gguf-py: rel", float((cpu - ref).norm() / ref.norm()),
      "| erro do Q4_1 vs BF16: rel", float((ref - w.float()).norm() / w.float().norm()))
qc, sc, zc = q.cuda(), s.cuda(), z.cuda()
tri = dequant_triton(qc, sc, zc, G)
print("Triton x eager: max abs", float((tri.float().cpu() - cpu).abs().max()))
x = torch.randn(4096 + 256, w.shape[1], device="cuda", dtype=torch.bfloat16)
wb = w.cuda()
print(f"dequant torch (kitchen eager)  {cronometra(lambda: L.dequantize(qc, params._replace() if hasattr(params, '_replace') else L.Params(scale=sc, zeros=zc, group_size=G, orig_dtype=torch.bfloat16, orig_shape=tuple(w.shape)))):7.3f} ms")
print(f"dequant Triton                 {cronometra(lambda: dequant_triton(qc, sc, zc, G)):7.3f} ms")
print(f"Triton dequant + linear        {cronometra(lambda: torch.nn.functional.linear(x, dequant_triton(qc, sc, zc, G))):7.3f} ms")
print(f"bf16 linear                    {cronometra(lambda: torch.nn.functional.linear(x, wb)):7.3f} ms")
