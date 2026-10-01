import sys, time, json, struct
from safetensors import safe_open
import torch
for p in sys.argv[1:]:
    t0 = time.time()
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
    keys = [k for k in h if k != '__metadata__']
    print(f"## {p}: header ok, {len(keys)} tensors, header {n} B", flush=True)
    try:
        with safe_open(p, framework='pt', device='cpu') as f:
            ks = list(f.keys())
            print(f"   safe_open ok, {len(ks)} keys", flush=True)
            tot = 0
            for i, k in enumerate(ks):
                t = f.get_tensor(k); tot += t.numel() * t.element_size(); del t
                if i % 500 == 0: print(f"   [{i}/{len(ks)}] {tot/2**30:.1f} GiB paged through, {time.time()-t0:.0f} s", flush=True)
            print(f"   FULL PAGE-THROUGH OK: {tot/2**30:.2f} GiB in {time.time()-t0:.0f} s", flush=True)
    except Exception as e:
        print("   FAILED:", type(e).__name__, str(e)[:200], flush=True)
