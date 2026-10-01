"""M1: cópia host->XPU no torch. python bench_copia.py <page|pin> <MB>  (3 repetições, imprime MB/s)."""
import sys, time, torch
modo, mb = sys.argv[1], int(sys.argv[2])
r = []
for _ in range(3):
    x = torch.ones(mb * 2**20, dtype=torch.uint8)
    if modo == "pin":
        x = x.pin_memory()
    torch.xpu.synchronize(); a = time.time(); y = x.to("xpu"); torch.xpu.synchronize()
    r.append(round(mb / (time.time() - a)))
    del y
print(f"M1 {modo} {mb}MB MB/s {r}", flush=True)
