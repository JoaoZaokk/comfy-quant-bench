"""apply_rope_split_half em fp32: triton × eager × referência fp64. Rodar sob lock, só a 3090 visível."""
import inspect

import torch
from comfy_kitchen.registry import registry

registry.enable("triton")
torch.manual_seed(0)
eager = getattr(registry._backends["eager"], "apply_rope_split_half")
triton = getattr(registry._backends["triton"], "apply_rope_split_half")
print("assinatura:", inspect.signature(eager))


def casos():
    # formas do text encoder (B, heads, T, D) e freqs como o ComfyUI passa; a forma exata vem da assinatura
    for heads, t, d in ((16, 77, 128), (32, 512, 128), (8, 300, 64)):
        xq = torch.randn(1, heads, t, d, device="cuda", dtype=torch.float32)
        xk = torch.randn(1, heads, t, d, device="cuda", dtype=torch.float32)
        ang = torch.rand(1, 1, t, d // 2, device="cuda", dtype=torch.float64) * 6.28
        yield (heads, t, d), xq, xk, ang


for forma, xq, xk, ang in casos():
    cos, sin = torch.cos(ang), torch.sin(ang)
    # freqs_cis do ComfyUI: matriz de rotação [[cos, -sin], [sin, cos]] em (..., D/2, 2, 2)
    freqs = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2).float()
    try:
        e = eager(xq, xk, freqs)
    except Exception as ex:  # noqa: BLE001
        print("eager recusou a forma", forma, ex)
        continue
    t = triton(xq.clone(), xk.clone(), freqs)
    ref = eager(xq.double(), xk.double(), freqs.double())
    for nome, a, b, r in (("q", e[0], t[0], ref[0]), ("k", e[1], t[1], ref[1])):
        print(forma, nome, f"|triton-eager|max={float((a - b).abs().max()):.3e}",
              f"|eager-fp64|max={float((a.double() - r).abs().max()):.3e}",
              f"|triton-fp64|max={float((b.double() - r).abs().max()):.3e}")
