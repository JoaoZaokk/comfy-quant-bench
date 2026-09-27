"""Qual implementacao o comfy-kitchen escolhe para gemv_awq_w4a16 no uso real do ComfyUI, e quanto custa.

    python_embeded\\python.exe -s sonda_awq_caminho.py
"""
import hashlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "F:/COMFY_PORTABLE/ComfyUI")
import comfy.quant_ops  # noqa: E402,F401
import comfy_kitchen as ck  # noqa: E402
from comfy_kitchen.backends.cuda import _triton_dequantize_awq_w4a16  # noqa: E402

n, k, g = 24576, 4096, 32
q = torch.randint(-128, 127, (n, k // 2), dtype=torch.int8, device="cuda")
s = (torch.rand(k // g, n, device="cuda") * 0.01).bfloat16()
z = (torch.randn(k // g, n, device="cuda") * 0.01).bfloat16()
x = torch.randn(4352, k, device="cuda", dtype=torch.bfloat16)
kw = {"x": x, "qweight": q, "wscales": s, "wzeros": z, "bias": None, "group_size": g}
impl = ck.registry.get_implementation("gemv_awq_w4a16", kwargs=kw)
print("backends:", ck.list_backends().keys(), "| escolhido:", impl.__module__, impl.__name__)
print("triton dequant importado no backend cuda:", _triton_dequantize_awq_w4a16 is not None)


def cronometra(fn, n_=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n_ * 1000


print(f"ck.gemv_awq_w4a16 (M=4352): {cronometra(lambda: ck.gemv_awq_w4a16(x, q, s, z, group_size=g)):.3f} ms")
wb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
print(f"bf16 linear:                {cronometra(lambda: torch.nn.functional.linear(x, wb)):.3f} ms")

R = "F:/COMFY_PORTABLE/ComfyUI/output/qwen21_bateria/"
h = lambda p: hashlib.sha256(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest()  # noqa: E731
iguais = sum(h(f"{R}w4a16_q4_1_nativo/p{p}_s{s_}_00001_.png") == h(f"{R}w4a16_q4_1_nativo_ckstock/p{p}_s{s_}_00001_.png")
             for p in range(6) for s_ in (42, 7))
print("imagens patch Triton x kitchen original identicas:", iguais, "/ 12")
