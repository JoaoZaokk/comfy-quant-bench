"""Tempo por camada: desquantizar (kernel do layout) + F.linear BF16, contra F.linear BF16 puro. So leitura dos pesos.

    python_embeded\\python.exe -s sonda_dequant.py
"""
import json
import struct
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "F:/COMFY_PORTABLE/ComfyUI")
sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
import comfy.quant_ops  # noqa: E402,F401  (registra os layouts como o ComfyUI)
import comfy_kitchen as ck  # noqa: E402
from comfy_kitchen.tensor import QuantizedTensor, get_layout_class  # noqa: E402
DT = {"I8": torch.int8, "U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn}


def read_tensor(f, start, size, dtype, shape):
    f.seek(start)
    return torch.frombuffer(bytearray(f.read(size)), dtype=DT[dtype]).reshape(shape)

D = Path("P:/ComfyBench/diffusion_models")
CAMADA = "transformer_blocks.10.img_mlp.gate_up"  # a maior linear (K=3072, N=2*9216?)


def carrega(arquivo, nomes):
    with open(D / arquivo, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        meta = h.pop("__metadata__", {})
        out = {}
        for nome in nomes:
            if nome in h:
                i = h[nome]
                a, b = i["data_offsets"]
                out[nome] = read_tensor(f, 8 + n + a, b - a, i["dtype"], i["shape"]).cuda()
    return out, json.loads(meta["_quantization_metadata"])["layers"][CAMADA]


def cronometra(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


x = torch.randn(4096 + 256, 4096, device="cuda", dtype=torch.bfloat16)

w8, conf8 = carrega("qwen_image_2.1_bf16_int8_convrot_f32.safetensors", [f"{CAMADA}.weight", f"{CAMADA}.weight_scale"])
L8 = get_layout_class("TensorWiseINT8Layout")
q8 = QuantizedTensor(w8[f"{CAMADA}.weight"], "TensorWiseINT8Layout",
                     L8.Params(scale=w8[f"{CAMADA}.weight_scale"], orig_dtype=torch.bfloat16,
                               orig_shape=tuple(w8[f"{CAMADA}.weight"].shape), convrot=True, convrot_groupsize=256))
w4, conf4 = carrega("qwen_image_2.1_bf16_w4a8_f32.safetensors",
                    [f"{CAMADA}.weight", f"{CAMADA}.weight_s_rel", f"{CAMADA}.weight_s_channel", f"{CAMADA}.weight_codebook"])
L4 = get_layout_class("AsymW4A8Int8Layout")
s_rel = w4[f"{CAMADA}.weight_s_rel"]
if s_rel.dtype == torch.uint8:
    s_rel = s_rel.view(torch.float8_e4m3fn)
N = w4[f"{CAMADA}.weight"].shape[0]
q4 = QuantizedTensor(w4[f"{CAMADA}.weight"], "AsymW4A8Int8Layout",
                     L4.Params(scale=s_rel, s_channel=w4[f"{CAMADA}.weight_s_channel"],
                               codebook=w4.get(f"{CAMADA}.weight_codebook"), group_size=conf4.get("group_size", 16),
                               convrot_groupsize=conf4.get("convrot_groupsize", 256), orig_dtype=torch.bfloat16,
                               orig_shape=(N, x.shape[1])))
wb = q8.dequantize()
print("camada", CAMADA, "peso", tuple(wb.shape), "x", tuple(x.shape))
print("int8 dequant rel vs w4a8 dequant rel:", float((q4.dequantize().float() - wb.float()).norm() / wb.float().norm()))
for nome, qt in (("int8_convrot", q8), ("w4a8", q4)):
    impl = {"int8_convrot": "dequantize_int8_convrot_weight", "w4a8": "dequantize_w4a8_int8_weight"}[nome]
    try:
        print(nome, "backends registrados:", [b for b in ck.list_backends()], "op:", impl)
    except Exception:
        pass
    print(f"{nome:13s} dequant {cronometra(qt.dequantize):7.3f} ms   dequant+linear "
          f"{cronometra(lambda: torch.nn.functional.linear(x, qt.dequantize())):7.3f} ms")
print(f"{'bf16':13s} linear  {cronometra(lambda: torch.nn.functional.linear(x, wb)):7.3f} ms")

# referencia fp32 pelo backend eager (mesma matematica, sem arredondar para BF16 no meio)
from comfy_kitchen.backends.eager import w4a8_int8 as _eager  # noqa: E402
qd, sr, sc, corr, cb = get_layout_class("AsymW4A8Int8Layout").get_plain_tensors(q4)
ref = _eager.dequantize_w4a8_int8_weight(qd, sr, sc, cb, corr, q4._params.group_size, q4._params.convrot_groupsize, torch.float32)
got = q4.dequantize().float()
print("w4a8 dequant (backend atual) x referencia fp32: rel", float((got - ref).norm() / ref.norm()),
      "| BF16 da propria referencia: rel", float((ref.bfloat16().float() - ref).norm() / ref.norm()))
