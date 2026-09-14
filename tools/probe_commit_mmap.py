"""Mede o que cada caminho de leitura de um safetensors de 39 GiB cobra de COMMIT no Windows,
UMA ETAPA POR PROCESSO (argv[2]), porque a primeira versao morreu em silencio logo depois do
safe_open -- o que ja e um achado: o reprodutor da morte do servidor cabe num processo nu.

Hipotese (escrita antes de rodar): `torch.UntypedStorage.from_file(shared=False)` -- que e o que
`safetensors.safe_open(framework="pt")` usa por baixo (a pilha das mortes mostra
`torch/storage.py __getitem__` sob `f.get_tensor`) -- mapeia o arquivo como copy-on-write
(FILE_MAP_COPY), e o Windows cobra commit pelo TAMANHO INTEIRO da view no momento do mapeamento.
Refutacao: commit sobe menos de 5 GiB ao mapear. Controles: mmap ACCESS_READ deve cobrar ~0;
torch.empty de 39 GiB deve cobrar 39; estourar o teto deve dar excecao limpa, nao AV.

Etapas: A safe_open+get_tensor | B from_file+fatia+toque | C mmap leitura | D torch.empty |
E from_file + torch.empty juntos. Sempre com -X faulthandler.
"""
import ctypes
import ctypes.wintypes as W
import mmap
import os
import sys
import time

PATH = sys.argv[1]
ETAPA = sys.argv[2]
GIB = float(1 << 30)


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", W.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
        ("HandleCount", W.DWORD), ("ProcessCount", W.DWORD), ("ThreadCount", W.DWORD),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", W.DWORD), ("PageFaultCount", W.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t),
    ]


psapi = ctypes.windll.psapi
k32 = ctypes.windll.kernel32


def commit():
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        raise OSError("GetPerformanceInfo failed")
    ps = pi.PageSize
    return pi.CommitTotal * ps / GIB, pi.CommitLimit * ps / GIB, pi.PhysicalAvailable * ps / GIB


def private():
    pm = PROCESS_MEMORY_COUNTERS_EX()
    pm.cb = ctypes.sizeof(pm)
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    if not ok:
        return float("nan"), float("nan")
    return pm.PrivateUsage / GIB, pm.WorkingSetSize / GIB


base_c = None


def mark(label):
    global base_c
    c, lim, avail = commit()
    p, ws = private()
    if base_c is None:
        base_c = c
    print(f"[{ETAPA}] {label:<48} commit {c:6.1f} (d {c - base_c:+5.1f})  lim {lim:6.1f}  fis.livre {avail:5.1f}  priv {p:6.2f}  ws {ws:6.2f}", flush=True)


size = os.path.getsize(PATH)
mark(f"inicio  arquivo {size / GIB:.2f} GiB")
import torch  # noqa: E402
mark("apos import torch")
n = size // 2

if ETAPA == "A":
    from safetensors import safe_open
    t0 = time.time()
    f = safe_open(PATH, framework="pt", device="cpu")
    mark(f"safe_open ({time.time() - t0:.2f}s)")
    keys = list(f.keys())
    mark(f"keys() -> {len(keys)}")
    small = min(keys, key=lambda k: len(k))
    x = f.get_tensor(small)
    mark(f"get_tensor pequeno {tuple(x.shape)}")
    shapes = {k: f.get_slice(k).get_shape() for k in keys}
    big = max(keys, key=lambda k: shapes[k][0] * (shapes[k][1] if len(shapes[k]) > 1 else 1))
    t0 = time.time()
    t = f.get_tensor(big)
    mark(f"get_tensor grande {tuple(t.shape)} {t.numel() * t.element_size() / GIB:.2f} GiB ({time.time() - t0:.2f}s)")
    s = float(t[0, :16].float().sum())
    mark("tocou o grande")
    lidos = 0
    for k in keys:
        if lidos > 2 * GIB:
            break
        y = f.get_tensor(k)
        lidos += y.numel() * y.element_size()
        float(y.reshape(-1)[:1].float().sum())
    mark(f"apos ler ~{lidos / GIB:.1f} GiB em {len([1 for _ in ()])} ")
    del t, x, y
    del f
    mark("apos soltar tudo")

elif ETAPA == "B":
    st = torch.UntypedStorage.from_file(PATH, shared=False, nbytes=size)
    mark("from_file(shared=False)")
    sl = st[1 << 30:(1 << 30) + (2 << 30)]
    mark("fatia de 2 GiB")
    tt = torch.empty(0, dtype=torch.uint8).set_(sl)
    s = int(tt[::4096].sum())
    mark("tocou 2 GiB (uma leitura por pagina)")
    del tt, sl, st
    mark("apos soltar")

elif ETAPA == "B2":
    st = torch.UntypedStorage.from_file(PATH, shared=True, nbytes=size)
    mark("from_file(shared=True)")
    sl = st[1 << 30:(1 << 30) + (2 << 30)]
    tt = torch.empty(0, dtype=torch.uint8).set_(sl)
    s = int(tt[::4096].sum())
    mark("tocou 2 GiB")
    del tt, sl, st
    mark("apos soltar")

elif ETAPA == "C":
    fh = open(PATH, "rb")
    m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    mark("mmap ACCESS_READ")
    acc = 0
    for off in range(1 << 30, (1 << 30) + (2 << 30), 4096):
        acc += m[off]
    mark("tocou 2 GiB")
    m.close(); fh.close()
    mark("apos fechar")

elif ETAPA == "D":
    try:
        e = torch.empty(n, dtype=torch.bfloat16)
        mark(f"torch.empty {n * 2 / GIB:.1f} GiB bf16")
        e[::1 << 20] = 1
        mark("tocou o empty (1 escrita por MiB)")
        del e
        mark("apos soltar")
    except Exception as ex:  # noqa: BLE001
        mark(f"torch.empty FALHOU limpo: {type(ex).__name__}: {str(ex)[:80]}")

elif ETAPA == "E":
    st = None
    try:
        st = torch.UntypedStorage.from_file(PATH, shared=False, nbytes=size)
        mark("from_file(shared=False)")
        e = torch.empty(n, dtype=torch.bfloat16)
        mark(f"+ torch.empty {n * 2 / GIB:.1f} GiB")
        del e
    except Exception as ex:  # noqa: BLE001
        mark(f"FALHOU limpo: {type(ex).__name__}: {str(ex)[:80]}")
    del st
    mark("apos soltar tudo")

print(f"[{ETAPA}] FIM OK", flush=True)
