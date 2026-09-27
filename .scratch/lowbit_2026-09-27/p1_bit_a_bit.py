"""P1 do criterio: os leitores do loader, nas 100 camadas de cada arquivo, contra o unpacked do mesmo modelo.

CPU apenas. Referencia lida tensor a tensor do safetensors (sem carregar o arquivo inteiro).
Controle: o pack ternario contra o unpacked BINARIO tem de falhar.

    python_embeded\\python.exe -s .scratch\\lowbit_2026-09-27\\p1_bit_a_bit.py
"""
import importlib.util
import json
import pathlib
import struct
import sys
import time
import types

import torch

ROOT = pathlib.Path("F:/COMFY_PORTABLE")
sys.path.insert(0, str(ROOT / "ComfyUI"))
from comfy.cli_args import args  # noqa: E402

args.cpu = True
PKG = ROOT / "custom_nodes" / "comfy-lowbit-loader"
pkg = types.ModuleType("lowbit_pkg")
pkg.__path__ = [str(PKG)]
sys.modules["lowbit_pkg"] = pkg
for name in ("kernel", "layout", "formats"):
    spec = importlib.util.spec_from_file_location(f"lowbit_pkg.{name}", PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
kernel, formats = sys.modules["lowbit_pkg.kernel"], sys.modules["lowbit_pkg.formats"]

B = pathlib.Path("F:/bonsai-re")
UNP = {m: B / f"bonsai-image-{m}-4B-unpacked/transformer/diffusion_pytorch_model.safetensors" for m in ("ternary", "binary")}
PACKS = {
    ("ternary", "gemlite"): B / "bonsai-image-ternary-4B-gemlite-2bit/transformer-gemlite-int2/state_dict.pt",
    ("ternary", "mlx"): B / "bonsai-image-ternary-4B-mlx-2bit/transformer-packed-mflux/diffusion_pytorch_model.safetensors",
    ("binary", "gemlite"): B / "bonsai-image-binary-4B-gemlite-1bit/transformer-gemlite-int1/state_dict.pt",
    ("binary", "mlx"): B / "bonsai-image-binary-4B-mlx-1bit/transformer-packed-mflux/diffusion_pytorch_model.safetensors",
}


class Ref:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h, self.base = json.loads(self.f.read(n)), 8 + n

    def __getitem__(self, name):
        e = self.h[name]
        a, b = e["data_offsets"]
        self.f.seek(self.base + a)
        return torch.frombuffer(bytearray(self.f.read(b - a)), dtype=torch.bfloat16).view(e["shape"])


def compara(lowbit, ref):
    identicas, piores = 0, []
    for name, layer in lowbit.items():
        w = kernel.dequantize_torch(layer.qdata, layer.scale, layer.zero, layer.bits, layer.group_size, torch.bfloat16)
        r = ref[f"{name}.weight"]
        if torch.equal(w, r):
            identicas += 1
        else:
            piores.append((name, (w == r).float().mean().item()))
    return identicas, piores


resultado = {}
for (modelo, pack), path in PACKS.items():
    t = time.time()
    _, lowbit = formats.READERS[pack](str(path))
    bits = sorted({l.bits for l in lowbit.values()})
    ok, piores = compara(lowbit, Ref(UNP[modelo]))
    resultado[f"{modelo}/{pack}"] = {"camadas": len(lowbit), "identicas": ok, "bits": bits, "falhas": piores[:3]}
    print(f"{modelo:8s} {pack:8s} bits {bits} camadas {len(lowbit)} identicas bit a bit {ok}  ({time.time() - t:.0f} s)", flush=True)
    if modelo == "ternary" and pack == "mlx":
        ok_c, piores_c = compara(lowbit, Ref(UNP["binary"]))
        frac = sum(p for _, p in piores_c) / max(len(piores_c), 1)
        resultado["controle ternario-vs-unpacked-binario"] = {"identicas": ok_c, "fracao_media_identica": frac}
        print(f"CONTROLE ternario mlx vs unpacked BINARIO: identicas {ok_c}/{len(lowbit)}, fracao media identica {frac:.4f}", flush=True)

for modelo, path in UNP.items():
    t = time.time()
    dense, lowbit = formats.read_dense(str(path))
    bits = sorted({l.bits for l in lowbit.values()})
    ok, piores = compara(lowbit, Ref(path))
    resultado[f"{modelo}/unpacked"] = {"camadas": len(lowbit), "identicas": ok, "bits": bits, "densos": len(dense), "falhas": piores[:3]}
    print(f"{modelo:8s} unpacked bits {bits} camadas empacotadas {len(lowbit)} densos {len(dense)} identicas {ok}  ({time.time() - t:.0f} s)", flush=True)

(pathlib.Path(__file__).parent / "p1_resultado.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
