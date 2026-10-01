"""Roda UM conversor (main()) de um diretorio de tools dado, em CPU, com a GPU desviada.

    roda_um.py <tools_dir> <modulo> [args do conversor...]

Desvios (so neste processo): torch.cuda.is_available -> True, `.to("cuda")`/`.cuda()` -> CPU,
`native_backend_ready` -> "comfy_kitchen.backends.cuda.FAKE" para todo op. O quantizador e o
REAL do comfy-kitchen, que em CPU resolve para o backend eager. Isto prova layout e bytes do
caminho de escrita, nunca o backend nativo.

Variaveis: FAKE_CALIB=1 (smooth: calibracao deterministica), WRAP_BF16=1 (smooth: quantizador recebe
BF16, como o codigo antigo), FAKE_NUNCHAKU=1 (svdq: nunchaku falso do test_svdq_verify do mesmo dir).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

tools_dir = Path(sys.argv[1]).resolve()
modulo = sys.argv[2]
argv = sys.argv[3:]
sys.path.insert(0, str(tools_dir.parent / "ComfyUI") if (tools_dir.parent / "ComfyUI").is_dir()
                else str(Path("F:/COMFY_PORTABLE/ComfyUI")))
sys.path.insert(0, str(tools_dir))

import torch  # noqa: E402

assert not torch.cuda.is_available(), "a GPU esta visivel: rode com CUDA_VISIBLE_DEVICES=-1"
torch.cuda.is_available = lambda: True
torch.cuda.empty_cache = lambda: None
torch.cuda.get_device_name = lambda *a, **k: "CPU-FAKE"


def _cpu(x):
    if isinstance(x, str) and x.startswith("cuda"):
        return "cpu"
    if isinstance(x, torch.device) and x.type == "cuda":
        return torch.device("cpu")
    return x


_orig_to = torch.Tensor.to


def _to(self, *a, **k):
    a = tuple(_cpu(x) for x in a)
    if "device" in k:
        k["device"] = _cpu(k["device"])
    return _orig_to(self, *a, **k)


torch.Tensor.to = _to
torch.Tensor.cuda = lambda self, *a, **k: self


class Resolvido(dict):
    def __missing__(self, key):
        return "comfy_kitchen.backends.cuda.FAKE"


def fake_backend(root, ops):
    return {"native_ready": True, "resolved": Resolvido({op: "comfy_kitchen.backends.cuda.FAKE" for op in ops})}


try:
    import _native_probe
    _native_probe.native_backend_ready = fake_backend
except ImportError:
    pass

if os.environ.get("FAKE_NUNCHAKU"):
    import test_svdq_verify as T

    class _Fake(T._FakeSVDQW4A4Linear):
        # proj_up e [out, rank] no checkpoint real (svdq_to_bf16._in_out); o fake do teste guarda
        # (rank, out). So a forma do parametro muda; a matematica e a do fake do teste.
        def __init__(self, in_features, out_features, rank=1, **k):
            super().__init__(in_features, out_features, rank=rank, **k)
            self.proj_up = T._zeros_param(out_features, rank, dtype=self.proj_up.dtype,
                                          device=self.proj_up.device)

    T._fake_nunchaku().__enter__()
    sys.modules["nunchaku.models.linear"].SVDQW4A4Linear = _Fake

import importlib  # noqa: E402

m = importlib.import_module(modulo)
if hasattr(m, "native_backend_ready"):
    m.native_backend_ready = fake_backend

if os.environ.get("FAKE_CALIB"):
    import json
    import struct

    def fake_calibrate(args):
        src = Path(args.source)
        with src.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        stats = {}
        for i, k in enumerate(sorted(k for k in h if k.endswith(("input_layernorm.weight",
                                                                 "pre_feedforward_layernorm.weight")))):
            g = torch.Generator().manual_seed(100 + i)
            stats[k] = torch.rand(h[k]["shape"][0], generator=g) * 20 + 0.05
        return stats

    m.calibrate = fake_calibrate

if os.environ.get("WRAP_BF16"):
    import comfy_kitchen as ck
    _orig_q = ck.quantize_convrot_w4a4_weight

    def _q(weight, *a, **k):
        return _orig_q(weight.to(torch.bfloat16), *a, **k)

    ck.quantize_convrot_w4a4_weight = _q

sys.argv = [f"{modulo}.py", *argv]
try:
    rc = m.main()
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if not isinstance(e.code, int) and e.code is not None:
        print(f"SystemExit: {e.code}")
sys.exit(int(rc or 0))
