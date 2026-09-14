"""Reprodutor minimo da morte `torch/storage.py __getitem__` sobre um safetensors em SMB.

`safetensors.safe_open` + primeiro `get_tensor` morre com access violation no arquivo em W: (SMB)
e sobrevive no mesmo arquivo em C:. `UntypedStorage.from_file(shared=False)` + fatia em 1..3 GiB
sobreviveu nos dois. Entao o que difere e ONDE (offset) ou COMO (asarray/dtype) se le. Este
script mapeia como o safetensors mapeia e le uma fatia num offset dado (argv[2], em bytes), do
jeito que o safetensors le (storage[a:b] -> torch.asarray(dtype) -> .view). Um processo por
offset; o codigo de saida 139 (AV) marca onde morre.
"""
import json
import os
import struct
import sys

PATH = sys.argv[1]
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else -1
N = 1 << 20

import torch  # noqa: E402

size = os.path.getsize(PATH)
with open(PATH, "rb") as fh:
    hl = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(hl))
base = 8 + hl
if OFF < 0:
    # offset real do menor tensor, como o safetensors o pediria
    k = min((k for k in hdr if k != "__metadata__"), key=lambda k: hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0])
    a, b = hdr[k]["data_offsets"]
    OFF, N = base + a, b - a
    print(f"menor tensor {k} shape {hdr[k]['shape']} dtype {hdr[k]['dtype']} offset {OFF} ({OFF / 2**30:.2f} GiB) n {N}", flush=True)
print(f"mapeando {PATH} ({size / 2**30:.2f} GiB) shared=False; fatia [{OFF}, {OFF + N}) = {OFF / 2**30:.2f} GiB", flush=True)
st = torch.UntypedStorage.from_file(PATH, shared=False, nbytes=size)
print("mapeado", flush=True)
sl = st[OFF:OFF + N]
print("fatiado", flush=True)
t = torch.asarray(sl, dtype=torch.uint8)
print("asarray ok", flush=True)
v = int(t[:4096:512].sum())
print(f"lido ok soma {v}", flush=True)
print("FIM OK", flush=True)
