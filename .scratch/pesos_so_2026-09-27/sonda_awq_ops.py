"""Uma Linear awq_w4a16 carregada pelo comfy/ops.py (como no modelo), na GPU: qual caminho executa e quanto custa.

    python_embeded\\python.exe -s sonda_awq_ops.py
"""
import json
import sys
import time

import torch

sys.path.insert(0, "F:/COMFY_PORTABLE/ComfyUI")
from comfy import ops  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402
try:
    from comfy_kitchen.backends.triton.awq import dequantize_awq_w4a16
except ImportError:  # kitchen sem o patch
    dequantize_awq_w4a16 = None

n, k, g = 24576, 4096, 32
torch.manual_seed(0)
q = torch.randint(0, 16, (n, k), dtype=torch.int32)
qweight = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
s = (torch.rand(k // g, n) * 0.01).bfloat16()
z = (torch.randn(k // g, n) * 0.01).bfloat16()
sd = {"layer.weight": qweight, "layer.weight_scale": s, "layer.weight_zeros": z}
sd, _ = comfy.utils.convert_old_quants(sd, metadata={"_quantization_metadata": json.dumps(
    {"layers": {"layer": {"format": "awq_w4a16", "group_size": g}}})})
m = torch.nn.Module()
m.layer = ops.mixed_precision_ops({}).Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
m.load_state_dict(sd, strict=False)
w = m.layer.weight
print("peso:", type(w).__name__, w._layout_cls if isinstance(w, QuantizedTensor) else w.dtype, "device", w.device)
x = torch.randn(1, 4352, k, device="cuda", dtype=torch.bfloat16)

out = m.layer(x)
qc, sc, zc = qweight.cuda(), s.cuda(), z.cuda()
tri = torch.nn.functional.linear(x, dequantize_awq_w4a16(qc, sc, zc, g)) if dequantize_awq_w4a16 else out * float("nan")
qn = q.cuda().view(n, k // g, g).to(torch.bfloat16)
stock = torch.nn.functional.linear(x, ((qn - 8.0) * sc.t().unsqueeze(-1) + zc.t().unsqueeze(-1)).view(n, k))
print("saida ComfyUI == Triton:", torch.equal(out, tri), "| == formula torch bf16:", torch.equal(out, stock))

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    m.layer(x)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))


def cronometra(fn, n_=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n_ * 1000


wb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
print(f"Linear awq_w4a16 do ComfyUI: {cronometra(lambda: m.layer(x)):.3f} ms | bf16: {cronometra(lambda: torch.nn.functional.linear(x, wb)):.3f} ms")
