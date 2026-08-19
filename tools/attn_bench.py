"""Time the installed attention backends on this machine at diffusion-like shapes.

Import success and a numeric check say a backend works; they say nothing about whether it is
worth using. This measures it.
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

DEVICE = "cuda"
WARMUP = 5
ITERS = 20

# (batch, heads, seq, head_dim) — shapes typical of image/video DiT attention
SHAPES = [
    (1, 24, 1024, 64),
    (1, 24, 4096, 64),
    (1, 24, 16384, 64),
    (1, 16, 32768, 128),
]


def timed(fn, *args, **kwargs):
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main() -> int:
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device      : {name}  sm{capability[0]}{capability[1]}")
    print(f"torch       : {torch.__version__}  cuda {torch.version.cuda}")

    try:
        import sageattention
        from sageattention import sageattn
        print(f"sageattention: {getattr(sageattention, '__version__', 'unknown')}")
    except Exception as error:
        print(f"sageattention: UNAVAILABLE ({error})")
        sageattn = None

    try:
        from flash_attn import flash_attn_func
        import flash_attn
        print(f"flash_attn  : {getattr(flash_attn, '__version__', 'unknown')}")
    except Exception as error:
        print(f"flash_attn  : UNAVAILABLE ({error})")
        flash_attn_func = None

    print()
    header = f"{'shape (B,H,S,D)':<24} {'SDPA ms':>10} {'Sage ms':>10} {'Sage x':>8} {'Flash ms':>10} {'Flash x':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for batch, heads, seq, dim in SHAPES:
        torch.manual_seed(0)
        q = torch.randn(batch, heads, seq, dim, device=DEVICE, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        try:
            sdpa_ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        except Exception as error:
            print(f"{str((batch, heads, seq, dim)):<24} SDPA failed: {str(error)[:60]}")
            continue

        sage_ms = None
        if sageattn is not None:
            try:
                sage_ms = timed(lambda: sageattn(q, k, v))
            except Exception as error:
                print(f"  sage failed at {(batch, heads, seq, dim)}: {str(error)[:80]}")

        flash_ms = None
        if flash_attn_func is not None:
            qf, kf, vf = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            try:
                flash_ms = timed(lambda: flash_attn_func(qf, kf, vf))
            except Exception as error:
                print(f"  flash failed at {(batch, heads, seq, dim)}: {str(error)[:80]}")

        row = {
            "shape": [batch, heads, seq, dim],
            "sdpa_ms": round(sdpa_ms, 3),
            "sage_ms": round(sage_ms, 3) if sage_ms else None,
            "sage_speedup": round(sdpa_ms / sage_ms, 3) if sage_ms else None,
            "flash_ms": round(flash_ms, 3) if flash_ms else None,
            "flash_speedup": round(sdpa_ms / flash_ms, 3) if flash_ms else None,
        }
        results.append(row)
        print(
            f"{str((batch, heads, seq, dim)):<24} {sdpa_ms:>10.3f} "
            f"{(f'{sage_ms:.3f}' if sage_ms else 'n/a'):>10} "
            f"{(f'{sdpa_ms/sage_ms:.2f}x' if sage_ms else 'n/a'):>8} "
            f"{(f'{flash_ms:.3f}' if flash_ms else 'n/a'):>10} "
            f"{(f'{sdpa_ms/flash_ms:.2f}x' if flash_ms else 'n/a'):>9}"
        )
        del q, k, v
        torch.cuda.empty_cache()

    print()
    print(json.dumps({"device": name, "sm": f"{capability[0]}{capability[1]}", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
