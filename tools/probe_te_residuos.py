"""Confere, em CPU, que os residuos quantizados por `quant_te_residuos.py` carregam e rodam pelo
MESMO codigo que o ComfyUI usa: `mixed_precision_ops` (Embedding e Linear) + a embedding escalada
do Gemma (`llama._make_scaled_embedding`). Compara a saida contra a fonte BF16.

    python_embeded\\python.exe -s tools/probe_te_residuos.py --fonte X_bf16.safetensors --quant X_int8.safetensors

NAO COBERTO: a entrada da projecao aqui e ruido gaussiano normalizado, nao os hidden states reais
do Gemma (exigiria rodar o encoder inteiro); o texto do erro e da projecao sobre esse ruido. O
encode real e o render continuam sendo a aceitacao.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "ComfyUI"))

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "I8": torch.int8, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn}


def cabecalho(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, h.pop("__metadata__", {}), 8 + n


def carrega(path: Path, nomes: list[str]) -> dict[str, torch.Tensor]:
    h, _, base = cabecalho(path)
    out = {}
    with path.open("rb") as f:
        for n in nomes:
            if n in h:
                a, b = h[n]["data_offsets"]
                f.seek(base + a)
                raw = bytearray(f.read(b - a))
                out[n] = torch.frombuffer(raw, dtype=DTYPES[h[n]["dtype"]]).reshape(h[n]["shape"])
    return out


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", type=Path, required=True)
    p.add_argument("--quant", type=Path, required=True)
    args = p.parse_args()

    import comfy.ops
    from comfy.text_encoders.llama import _make_scaled_embedding

    _, meta, _ = cabecalho(args.quant)
    camadas = json.loads(meta["_quantization_metadata"])["layers"]
    novas = [k for k in camadas if k.endswith("embed_tokens") or "aggregate_embed" in k
             or k.endswith("text_embedding_projection")]
    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
    torch.manual_seed(0)
    resultado = {}
    for camada in novas:
        chaves = [f"{camada}.{s}" for s in ("weight", "weight_scale", "bias")]
        ref = carrega(args.fonte, chaves)
        q = carrega(args.quant, chaves)
        sd = {k.removeprefix(camada + "."): v for k, v in q.items()}
        sd["comfy_quant"] = torch.tensor(list(json.dumps(camadas[camada]).encode()), dtype=torch.uint8)
        w = ref[f"{camada}.weight"]
        if camada.endswith("embed_tokens"):
            mod = _make_scaled_embedding(ops, w.shape[0], w.shape[1], w.shape[1] ** 0.5, None, torch.bfloat16)
            mod.load_state_dict(sd, strict=True)
            ids = torch.randint(0, w.shape[0], (1, 512))
            y = mod(ids, out_dtype=torch.bfloat16)
            y_ref = torch.nn.functional.embedding(ids, w) * (w.shape[1] ** 0.5)
            tipo = type(mod.weight).__name__
        else:
            mod = ops.Linear(w.shape[1], w.shape[0], bias=f"{camada}.bias" in ref, dtype=torch.bfloat16)
            mod.load_state_dict(sd, strict=True)
            x = torch.randn(1, 64, w.shape[1])
            x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))).to(torch.bfloat16)
            y = mod(x)
            y_ref = torch.nn.functional.linear(x, w, ref.get(f"{camada}.bias"))
            tipo = type(mod.weight).__name__
        resultado[camada] = {"peso_carregado": tipo, "formato": getattr(mod, "quant_format", None),
                             "rel_saida": rel(y, y_ref), "shape_saida": list(y.shape)}
        print(camada, resultado[camada], flush=True)
        del ref, q, sd, mod, w
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
