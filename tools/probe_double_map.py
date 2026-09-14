"""Reprodutor da morte `storage.__getitem__` em safetensors sobre SMB, e o teste da condicao
composta.

Medido ate aqui: (1) `safe_open` charge 39 GiB de commit ANTES de chamar `from_file` (a view do
memmap2 e copy-on-write), e `from_file(shared=False)` cobra outros 39 -- duas views COW do mesmo
arquivo; (2) em C: as duas sao cobradas (total 150,1 com o limite expandido a 150,2) e a leitura
funciona; (3) em W: (SMB) a segunda view NAO e cobrada (total fica em 111, limite expande e volta)
e a primeira leitura pela view do torch da access violation; (4) uma view somente-leitura +
uma COW sobrevivem em W:.

Hipotese (escrita antes de rodar): a morte exige TRES coisas juntas -- arquivo em SMB, segunda
view copy-on-write, e a cobranca da segunda view precisar EXPANDIR o pagefile (isto e, as duas
views somadas passam do commit livre). Previsoes: T5 em W: (39 GiB, duas COW) morre; T5 em C:
sobrevive; T8 em W: (arquivo de 15,5 GiB, duas COW = 31 < ~53 livres) sobrevive; T9 em W: (mesmo
arquivo de 15,5 GiB, mas com 35 GiB privados presos antes, forcando expansao) morre.
Refutacao: T8 morre (tamanho nao importa) ou T9 sobrevive (expansao nao e o gatilho).

Modos: T1 RO+COW | T2 COW+RO | T3 so COW | T5 COW(mmap ACCESS_COPY)+COW(from_file) |
T8 = T5 | T9 = T5 com torch.empty(argv[3] GiB) preso antes.
"""
import ctypes
import ctypes.wintypes as W
import json
import mmap
import os
import struct
import sys

PATH, MODO = sys.argv[1], sys.argv[2]
HOLD_GIB = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
GIB = float(1 << 30)


class PI(ctypes.Structure):
    _fields_ = [("cb", W.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", W.DWORD),
                ("ProcessCount", W.DWORD), ("ThreadCount", W.DWORD)]


def commit(label):
    p = PI(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    print(f"[{MODO}] {label:<44} commit {p.CommitTotal * p.PageSize / GIB:6.1f}  limite {p.CommitLimit * p.PageSize / GIB:6.1f}", flush=True)


import faulthandler  # noqa: E402
faulthandler.enable()
import torch  # noqa: E402
size = os.path.getsize(PATH)
with open(PATH, "rb") as fh:
    hl = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(hl))
k = min((k for k in hdr if k != "__metadata__"), key=lambda k: hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0])
OFF = 8 + hl + hdr[k]["data_offsets"][0]
N = hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0]
print(f"[{MODO}] arquivo {size / GIB:.2f} GiB; menor tensor {k} em {OFF} ({N} B)", flush=True)
commit("inicio")
held = None
if HOLD_GIB > 0:
    held = torch.empty(int(HOLD_GIB * GIB), dtype=torch.uint8)
    held[::1 << 20] = 1
    commit(f"torch.empty {HOLD_GIB:.0f} GiB preso e tocado")
ro = cow = None
if MODO == "T1":
    fh = open(PATH, "rb"); ro = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    commit("mmap ACCESS_READ aberto")
if MODO in ("T5", "T8", "T9", "T10"):
    fh = open(PATH, "rb"); cow = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    commit("mmap ACCESS_COPY (1a view COW) aberto")
if MODO == "T11":
    fh = open(PATH, "rb"); ro = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    commit("mmap ACCESS_READ aberto")
if MODO in ("T10", "T11"):
    # como o safetensors: le o cabecalho pela 1a view antes de chamar from_file
    v = cow if cow is not None else ro
    hb = sum(v[i] for i in range(0, 8 + hl, 4096))
    commit(f"cabecalho ({(8 + hl) / 2**20:.1f} MiB) tocado pela 1a view")
st = torch.UntypedStorage.from_file(PATH, shared=False, nbytes=size)
commit("from_file(shared=False) (view COW do torch)")
if MODO == "T2":
    fh = open(PATH, "rb"); ro = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    commit("mmap ACCESS_READ aberto depois")
sl = st[OFF:OFF + N]
print(f"[{MODO}] fatiado; lendo storage[0] pela view do torch ...", flush=True)
b = sl[0]
print(f"[{MODO}] lido {b}; asarray ...", flush=True)
t = torch.asarray(sl, dtype=torch.uint8)
print(f"[{MODO}] asarray ok soma {int(t.sum())}", flush=True)
if cow is not None:
    print(f"[{MODO}] byte pela 1a view COW: {cow[OFF]}", flush=True)
if ro is not None:
    print(f"[{MODO}] byte pela view RO: {ro[OFF]}", flush=True)
commit("fim")
print(f"[{MODO}] FIM OK", flush=True)
