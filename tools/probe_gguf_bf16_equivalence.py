"""Prova que o "braco BF16 por GGUF" e o BF16 original, no peso e na conta.

Contexto (medido 2026-09-14): o loader normal do ComfyUI nao consegue abrir o transformer BF16 de
39 GiB do LTX 2.3 nesta maquina -- `safetensors.safe_open` cobra 2x o arquivo em commit e o
`torch.empty` do modelo mais 1x, e sob expansao do pagefile o primeiro toque da access violation.
A saida foi escrever o mesmo BF16 num GGUF (`tools/safetensors_to_gguf_bf16.py`) e carregar por
`UnetLoaderGGUF`, que le por memmap somente-leitura e nao faz `torch.empty`. Isso troca DUAS coisas
ao mesmo tempo -- o container e as ops (`GGMLOps.Linear` em vez de `comfy.ops.manual_cast.Linear`)
-- e um braco de referencia que trocou duas coisas sem medir nenhuma e uma suposicao.

O que este probe mede, numa amostra deterministica de camadas Linear 2-D:
  P1  bytes do peso no GGUF == bytes do peso no safetensors (lido por faixa de bytes, sem mmap);
  P2  `GGMLLayer.get_weight` (o dequantize BF16 do ComfyUI-GGUF) == o tensor bf16 original, bit a bit;
  P3  saida de `GGMLOps.Linear` == saida de `comfy.ops.manual_cast.Linear`, mesma entrada, mesmo
      device, mesmo dtype (bias incluido: no GGUF ele esta em F32 exato, no safetensors em bf16).
Previsao escrita antes de rodar: P1, P2 e P3 identicos (diferenca maxima 0.0) em todas as camadas.
Refutacao: qualquer camada com bytes diferentes ou diferenca > 0 na saida.

Nao mede: camadas que nao sao Linear (norms, embeddings, scale_shift_table -- estas vao em F32 no
GGUF e o modelo as usa direto), o modelo inteiro carregado pelo ComfyUI, atencao, nem o render.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import struct
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "ComfyUI"))
sys.path.insert(0, os.path.join(RAIZ, "ComfyUI", "custom_nodes"))


def le_cabecalho(caminho):
    with open(caminho, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    hdr.pop("__metadata__", None)
    return 8 + hl, hdr


def le_bytes(caminho, ini, fim):
    with open(caminho, "rb") as fh:
        fh.seek(ini)
        return fh.read(fim - ini)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--safetensors", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--prefix", default="model.diffusion_model.")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    import numpy as np
    import torch
    import gguf
    import comfy.ops
    ops_mod = importlib.import_module("ComfyUI-GGUF.ops")
    GGMLOps, GGMLTensor = ops_mod.GGMLOps, ops_mod.GGMLTensor

    base, hdr = le_cabecalho(a.safetensors)
    lin = [k for k, v in hdr.items() if k.endswith(".weight") and len(v["shape"]) == 2 and v["dtype"] == "BF16"
           and (k[:-len(".weight")] + ".bias") in hdr]
    rng = random.Random(20260914)
    maior = max(lin, key=lambda k: hdr[k]["shape"][0] * hdr[k]["shape"][1])
    amostra = sorted(set(rng.sample(lin, min(a.n - 1, len(lin))) + [maior]))
    print(f"safetensors: {len(hdr)} tensores, {len(lin)} Linear 2-D BF16 com bias; amostra {len(amostra)} (inclui o maior, {maior})")

    r = gguf.GGUFReader(a.gguf)
    por_nome = {t.name: t for t in r.tensors}
    dev = torch.device(a.device)
    res = []
    t0 = time.time()
    for k in amostra:
        nome = k[len(a.prefix):]
        kb = k[:-len(".weight")] + ".bias"
        nb = nome[:-len(".weight")] + ".bias"
        out_f, in_f = hdr[k]["shape"]
        # --- P1: bytes ---
        st_w = le_bytes(a.safetensors, base + hdr[k]["data_offsets"][0], base + hdr[k]["data_offsets"][1])
        st_b = le_bytes(a.safetensors, base + hdr[kb]["data_offsets"][0], base + hdr[kb]["data_offsets"][1])
        gt_w, gt_b = por_nome[nome], por_nome[nb]
        assert gt_w.tensor_type == gguf.GGMLQuantizationType.BF16, (nome, gt_w.tensor_type)
        assert gt_b.tensor_type == gguf.GGMLQuantizationType.F32, (nb, gt_b.tensor_type)
        p1 = np.asarray(gt_w.data).tobytes() == st_w
        # bias: F32 no gguf, bf16 no safetensors -> compara depois de converter exato
        b_st = torch.frombuffer(bytearray(st_b), dtype=torch.bfloat16).clone()
        b_gg = torch.from_numpy(np.asarray(gt_b.data).astype(np.float32, copy=True)).reshape(-1)
        p1b = torch.equal(b_gg.to(torch.bfloat16), b_st) and torch.equal(b_gg, b_st.float())
        # --- P2: dequantize do ComfyUI-GGUF == bf16 original ---
        w_st = torch.frombuffer(bytearray(st_w), dtype=torch.bfloat16).reshape(out_f, in_f).clone().to(dev)
        tw = torch.from_numpy(np.asarray(gt_w.data))
        gw = GGMLTensor(tw, tensor_type=gt_w.tensor_type, tensor_shape=torch.Size((out_f, in_f))).to(dev)
        gb = GGMLTensor(torch.from_numpy(np.asarray(gt_b.data).copy()).reshape(out_f), tensor_type=gt_b.tensor_type,
                        tensor_shape=torch.Size((out_f,))).to(dev)
        lg = GGMLOps.Linear(in_f, out_f, bias=True)
        lg.weight = torch.nn.Parameter(gw, requires_grad=False)
        lg.bias = torch.nn.Parameter(gb, requires_grad=False)
        w_deq = lg.get_weight(lg.weight, torch.bfloat16)
        p2 = torch.equal(w_deq, w_st)
        # --- P3: saida ---
        ls = comfy.ops.manual_cast.Linear(in_f, out_f, bias=True, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            ls.weight.copy_(w_st)
            ls.bias.copy_(b_st.to(dev))
        g = torch.Generator(device="cpu").manual_seed(20260914)
        x = (torch.randn(8, in_f, generator=g) * 2.0).to(torch.bfloat16).to(dev)
        with torch.no_grad():
            y_s = ls(x)
            y_g = lg(x)
        dmax = float((y_s.float() - y_g.float()).abs().max())
        p3 = torch.equal(y_s, y_g)
        res.append({"camada": nome, "forma": [out_f, in_f], "P1_bytes_peso": bool(p1), "P1_bias_exato": bool(p1b),
                    "P2_dequant_bit_a_bit": bool(p2), "P3_saida_identica": bool(p3), "P3_dif_max": dmax,
                    "dtype_saida": [str(y_s.dtype), str(y_g.dtype)]})
        print(f"  {nome:<70} {str([out_f, in_f]):<14} P1 {'ok' if p1 and p1b else 'DIFERE'}  P2 {'ok' if p2 else 'DIFERE'}  P3 {'ok' if p3 else 'DIFERE'} dif_max {dmax:.3g}")
    ok = all(x["P1_bytes_peso"] and x["P1_bias_exato"] and x["P2_dequant_bit_a_bit"] and x["P3_saida_identica"] for x in res)
    print(f"\n{len(res)} camadas em {time.time() - t0:.1f}s, device {dev}: "
          f"P1 {sum(x['P1_bytes_peso'] and x['P1_bias_exato'] for x in res)}/{len(res)}  "
          f"P2 {sum(x['P2_dequant_bit_a_bit'] for x in res)}/{len(res)}  "
          f"P3 {sum(x['P3_saida_identica'] for x in res)}/{len(res)}  -> {'IDENTICO' if ok else 'DIFERE'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"safetensors": a.safetensors, "gguf": a.gguf, "device": str(dev), "camadas": res, "identico": ok}, fh, indent=1)
        print(f"json em {a.json}")
    print("NAO COBERTO: so Linear 2-D com bias, amostra e nao todas as camadas; norms, embeddings e "
          "scale_shift_table (F32 no GGUF, usados direto pelo modelo) nao entram; nada de atencao, "
          "nada de render; uma entrada aleatoria por camada.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
