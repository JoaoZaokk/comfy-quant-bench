"""Equivalencia da ESCRITA do ajusta_denso_diffusers: trecho antigo (save_file) x grava_saida nova.

O script inteiro nao roda sem diffusers + professor + treino; isto isola a escrita, que e a parte
que mudou. So CPU.
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
from gera_fontes import grava  # noqa: E402
from prova import equivalente  # noqa: E402

import _conversion as C  # noqa: E402
import ajusta_denso_diffusers as A  # noqa: E402

D = AQUI / "ajusta"
if D.exists():
    shutil.rmtree(D)
(D / "antes").mkdir(parents=True)
(D / "depois").mkdir(parents=True)
g = torch.Generator().manual_seed(21)
aluno = {f"double_blocks.{i}.img_attn.qkv.weight": (torch.randn(64, 128, generator=g) * 0.02).to(torch.bfloat16)
         for i in range(2)}
aluno |= {"img_in.weight": (torch.randn(64, 32, generator=g) * 0.02).to(torch.bfloat16),
          "final_layer.linear.weight": (torch.randn(16, 64, generator=g) * 0.02).to(torch.bfloat16),
          "time_in.in_layer.bias": torch.randn(64, generator=g),
          "double_blocks.0.norm.scale": torch.rand(64, generator=g).to(torch.float16)}
fonte = D / "aluno.safetensors"
grava(fonte, aluno, {"format": "pt", "origem": "sint\u00e9tico", "z": "1"})
treinaveis = [(k, torch.nn.Parameter((aluno[k].float() + 0.001).to(torch.float32)))
              for k in ("img_in.weight", "final_layer.linear.weight", "time_in.in_layer.bias")]
rel = {"modo": "ajuste", "mestre": "fp32", "elementos_mudados_frac": 0.5, "desvios_por_tensor": {"a": 1e-3}}

# ---- trecho ANTIGO, copiado de .scratch/antes_tools_conv_20260929/ajusta_denso_diffusers.py:314-338
from safetensors.torch import load_file, save_file  # noqa: E402

sai = D / "antes" / "aluno_aj.safetensors"
base = load_file(str(fonte))
for k, v in treinaveis:
    base[k] = v.detach().to("cpu", base[k].dtype).clone()
with fonte.open("rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n)).get("__metadata__")
parcial = sai.with_suffix(sai.suffix + ".partial")
save_file(base, str(parcial), metadata=meta)
parcial.replace(sai)
sai.with_suffix(".json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
del base

# ---- NOVO
sai2 = D / "depois" / "aluno_aj.safetensors"
conv = C.Conversion(fonte, sai2, sai2.with_suffix(".json"))
conv.refuse_unsafe(allow_quantized_source=True)
A.grava_saida(conv, treinaveis, rel)

ok_st = equivalente(sai, sai2)
ok_js = sai.with_suffix(".json").read_bytes() == sai2.with_suffix(".json").read_bytes()
print(f"safetensors equivalente (header parseado + dados identicos): {ok_st}")
print(f"relatorio .json byte a byte igual: {ok_js}")
sys.exit(0 if ok_st and ok_js else 1)
