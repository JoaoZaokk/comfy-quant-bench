r"""
_check_accel.py - verificacao funcional REAL dos aceleradores de atencao.
Import verde nao conta: cada backend faz forward + checagem numerica vs SDPA.
Rodar:  F:\COMFY_PORTABLE\python_embeded\python.exe -s F:\COMFY_PORTABLE\_check_accel.py
"""
import sys
import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.float16
results = {}


def report(name, ok, detail=""):
    results[name] = ok
    tag = "OK  " if ok is True else ("SKIP" if ok is None else "FAIL")
    print(f"[{tag}] {name:14s} {detail}")


def sdpa(q, k, v, causal=False):  # q,k,v: (B,H,S,D)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


print("=== environment ===")
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__, "| cuda_build", torch.version.cuda)
print("device     ", torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0))
print("n_gpu      ", torch.cuda.device_count())
print()
print("=== functional tests (forward + numeric vs SDPA) ===")

# ---------------- Triton ----------------
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _mul2(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * 2, mask=m)

    x = torch.arange(1024, device=DEV, dtype=torch.float32)
    y = torch.empty_like(x)
    _mul2[(1,)](x, y, 1024, BLOCK=1024)
    report("triton", bool(torch.allclose(y, x * 2)), f"v{triton.__version__} kernel compiled+ran")
except Exception as e:
    report("triton", False, repr(e)[:160])

# ---------------- reference tensors ----------------
B, H, S, D = 2, 8, 512, 64
torch.manual_seed(0)
qh = torch.randn(B, H, S, D, device=DEV, dtype=DT)
kh = torch.randn(B, H, S, D, device=DEV, dtype=DT)
vh = torch.randn(B, H, S, D, device=DEV, dtype=DT)
ref = sdpa(qh, kh, vh, causal=False).float()
ref_c = sdpa(qh, kh, vh, causal=True).float()

# ---------------- SageAttention ----------------
try:
    from sageattention import sageattn
    o = sageattn(qh, kh, vh).float()  # v1 layout HND = (B,H,S,D)
    d = (o - ref).abs().mean().item()
    report("sageattention", o.shape == ref.shape and d < 0.06, f"mean|d|={d:.4f} vs SDPA (INT8 approx)")
except Exception as e:
    report("sageattention", False, repr(e)[:160])

# ---------------- FlashAttention ----------------
try:
    from flash_attn import flash_attn_func
    qf, kf, vf = (t.transpose(1, 2).contiguous() for t in (qh, kh, vh))  # (B,S,H,D)
    o1 = flash_attn_func(qf, kf, vf).transpose(1, 2).float()
    o2 = flash_attn_func(qf, kf, vf, causal=True).transpose(1, 2).float()
    d1 = (o1 - ref).abs().mean().item()
    d2 = (o2 - ref_c).abs().mean().item()
    report("flash_attn", d1 < 0.02 and d2 < 0.02, f"mean|d|={d1:.4f} (causal {d2:.4f}) vs SDPA")
except Exception as e:
    report("flash_attn", False, repr(e)[:160])

# ---------------- xformers ----------------
try:
    import xformers
    import xformers.ops as xops
    qx, kx, vx = (t.transpose(1, 2).contiguous() for t in (qh, kh, vh))  # (B,S,H,D)
    ox = xops.memory_efficient_attention(qx, kx, vx).transpose(1, 2).float()
    dx = (ox - ref).abs().mean().item()
    report("xformers", dx < 0.02, f"v{xformers.__version__} mean|d|={dx:.4f} vs SDPA")
except ModuleNotFoundError:
    report("xformers", None, "not installed (bonus)")
except Exception as e:
    report("xformers", False, repr(e)[:160])

print()
print("=== SUMMARY ===")
for k, v in results.items():
    print(f"  {k:16s} {'OK' if v is True else ('SKIP' if v is None else 'FAIL')}")
fails = [k for k, v in results.items() if v is False]
print()
print("RESULT:", "ALL GOOD" if not fails else f"FAILURES -> {fails}")
sys.exit(1 if fails else 0)
