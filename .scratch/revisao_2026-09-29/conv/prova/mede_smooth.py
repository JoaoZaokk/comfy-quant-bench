"""Mede a diferenca do smooth FP32 (depois) contra o BF16 (antes) nas fontes sinteticas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
import _conversion as C  # noqa: E402

AQUI = Path(__file__).resolve().parent
a = AQUI / "antes" / "smooth_gemma" / "saida.safetensors"
d = AQUI / "depois" / "smooth_gemma" / "saida.safetensors"

ha, ma = C.read_header(a)
hd, md = C.read_header(d)
print("header (nomes/dtypes/formas/offsets) identico:", ha == hd, "| metadata identico:", ma == md)
ta, td = C.LazyTensors(a), C.LazyTensors(d)
layers = json.loads(ma["_quantization_metadata"])["layers"]
difs_cod = tot_cod = 0
rel_esc = []
preservados_iguais = True
for k in ha:
    x, y = ta[k], td[k]
    base = k.removesuffix(".weight").removesuffix(".weight_scale").removesuffix("_scale")
    if k.endswith(".weight") and k.removesuffix(".weight") in layers:
        # int4 empacotado: dois codigos por byte
        xa, ya = x.view(torch.uint8), y.view(torch.uint8)
        lo = (xa & 0x0F) != (ya & 0x0F)
        hi = (xa >> 4) != (ya >> 4)
        difs_cod += int(lo.sum() + hi.sum())
        tot_cod += 2 * xa.numel()
    elif k.endswith(".weight_scale"):
        rel_esc.append(float((x.float() - y.float()).norm() / x.float().norm()))
    elif not torch.equal(x.view(torch.uint8) if x.dtype == torch.bfloat16 else x,
                         y.view(torch.uint8) if y.dtype == torch.bfloat16 else y):
        if "layernorm" in k:
            continue  # normas reescritas: dependem so de lambda, conferidas abaixo
        preservados_iguais = False
        print("preservado diferente:", k)
normas_iguais = all(torch.equal(ta[k].view(torch.int16), td[k].view(torch.int16))
                    for k in ha if k.endswith(("input_layernorm.weight", "pre_feedforward_layernorm.weight")))
print(f"codigos int4 diferentes: {difs_cod}/{tot_cod} = {100 * difs_cod / tot_cod:.3f}%")
print(f"escalas: erro relativo medio {sum(rel_esc) / len(rel_esc):.2e}, max {max(rel_esc):.2e} "
      f"({len(rel_esc)} tensores)")
print("tensores preservados byte a byte iguais:", preservados_iguais)
print("normas reescritas (lambda) byte a byte iguais:", normas_iguais)
sa = json.loads((AQUI / "antes" / "smooth_gemma_bf16" / "saida.quant.json").read_text())
sd = json.loads((AQUI / "depois" / "smooth_gemma_bf16" / "saida.quant.json").read_text())
chaves = sorted(set(sa) ^ set(sd)) + sorted(k for k in set(sa) & set(sd)
                                             if sa[k] != sd[k] and k not in ("output", "conversion_seconds"))
print("sidecar smooth_gemma_bf16, chaves diferentes (fora output/seconds):", chaves)
