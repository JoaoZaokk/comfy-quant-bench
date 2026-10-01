"""Exporta o ultimo checkpoint do QAT (mestre, nomes diffusers) para ternario desempacotado bf16.

Mesma regra de `exporta` do tool: corpo = ternariza(mestre), resto = mestre em bf16. Le com mmap.
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, r"F:/COMFY_PORTABLE/tools")
from qat_ternario_klein import BLOCO, pilhas_reais, ternariza
from safetensors.torch import save_file
ck, saida = Path(sys.argv[1]), Path(sys.argv[2])
e = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
m = e["mestre"]
pil = pilhas_reais(m.keys())
corpo = {k for k, v in m.items() if v.ndim == 2 and (b := BLOCO.match(k)) and b.group("pilha") in pil}
sd = {k: (ternariza(v.float(), 128) if k in corpo else v).to(torch.bfloat16).contiguous() for k, v in m.items()}
save_file(sd, str(saida), metadata={"format": "pt", "quantizado_por": "tools/qat_ternario_klein.py",
                                     "passo": str(e["passo"]), "otim": str(e["extra"])})
print(f"passo {e['passo']}  corpo {len(corpo)}  total {len(sd)}  -> {saida}")
