"""SDPA na XPU: sequências de t2i (~4,2k) e edição (~6,4k/8,3k/12k tokens), 32 cabeças x 128, bf16."""
import sys, time, torch
import torch.nn.functional as F
for L in [int(x) for x in sys.argv[1:]]:
    q = torch.randn(1, 32, L, 128, device="xpu", dtype=torch.bfloat16)
    try:
        torch.xpu.reset_peak_memory_stats()
        F.scaled_dot_product_attention(q, q, q); torch.xpu.synchronize()
        a = time.time(); F.scaled_dot_product_attention(q, q, q); torch.xpu.synchronize()
        print(L, "ok", round((time.time() - a) * 1000), "ms pico", round(torch.xpu.max_memory_allocated() / 2**30, 2), "GiB", flush=True)
    except Exception as e:
        print(L, "ERRO", str(e)[:120], flush=True)
    del q; torch.xpu.empty_cache()
