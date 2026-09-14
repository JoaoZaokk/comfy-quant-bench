"""Rastreia o que `safetensors.safe_open(framework='pt').get_tensor` chama no torch, porque a
mesma sequencia feita a mao (`UntypedStorage.from_file(shared=False)` + fatia + asarray) sobrevive
em W: (SMB) em todos os offsets e o safe_open morre com access violation em `storage.py:471`.
Monkeypatch em from_file / frombuffer / asarray / __getitem__ com log ANTES da chamada, para o
ultimo log dizer que chamada estava em curso quando o processo morreu.
"""
import faulthandler
import os
import sys

faulthandler.enable()
PATH = sys.argv[1]
import torch  # noqa: E402

size = os.path.getsize(PATH)
print(f"arquivo {PATH} {size} B", flush=True)

_orig_from_file = torch.UntypedStorage.from_file


import ctypes  # noqa: E402
import ctypes.wintypes as W  # noqa: E402
import time  # noqa: E402


class PI(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", W.DWORD),
                ("ProcessCount", W.DWORD), ("ThreadCount", W.DWORD)]


def commit(label):
    p = PI(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    print(f"    [commit] {label:<34} total {p.CommitTotal * p.PageSize / 2**30:6.1f}  limite {p.CommitLimit * p.PageSize / 2**30:6.1f}", flush=True)


def from_file(*a, **k):
    print(f"  from_file args={[str(x)[:60] for x in a]} kwargs={ {kk: str(v)[:60] for kk, v in k.items()} }", flush=True)
    commit("antes do from_file")
    st = _orig_from_file(*a, **k)
    commit("depois do from_file")
    time.sleep(1.0)
    commit("1 s depois")
    print(f"  from_file -> nbytes={st.nbytes()} ", flush=True)
    return st


torch.UntypedStorage.from_file = staticmethod(from_file)

_orig_getitem = torch.UntypedStorage.__getitem__


def getitem(self, idx):
    print(f"  storage[{idx}]  (nbytes {self.nbytes()})", flush=True)
    r = _orig_getitem(self, idx)
    print(f"  storage[...] -> {type(r).__name__} nbytes {r.nbytes() if hasattr(r, 'nbytes') else '?'}", flush=True)
    return r


torch.UntypedStorage.__getitem__ = getitem

_orig_frombuffer = torch.frombuffer


def frombuffer(*a, **k):
    print(f"  frombuffer kwargs={ {kk: str(v)[:40] for kk, v in k.items()} }", flush=True)
    return _orig_frombuffer(*a, **k)


torch.frombuffer = frombuffer
_orig_asarray = torch.asarray


def asarray(*a, **k):
    print(f"  asarray arg0={type(a[0]).__name__} kwargs={ {kk: str(v)[:40] for kk, v in k.items()} }", flush=True)
    return _orig_asarray(*a, **k)


torch.asarray = asarray

from safetensors import safe_open  # noqa: E402
f = safe_open(PATH, framework="pt", device="cpu")
print("safe_open ok", flush=True)
keys = list(f.keys())
small = min(keys, key=lambda k: len(k))
print(f"get_tensor({small}) ...", flush=True)
t = f.get_tensor(small)
print(f"get_tensor ok {tuple(t.shape)} {t.dtype}", flush=True)
big = max(keys, key=lambda k: f.get_slice(k).get_shape()[0])
print(f"get_tensor({big}) ...", flush=True)
t2 = f.get_tensor(big)
print(f"get_tensor ok {tuple(t2.shape)} {t2.dtype}  soma {float(t2.reshape(-1)[:64].float().sum()):.3f}", flush=True)
print("FIM OK", flush=True)
