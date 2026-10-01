"""Roda verify_w4a4 --structural-only (antes e depois do codigo) sobre as saidas da prova. So CPU."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = Path("F:/COMFY_PORTABLE")
PY = RAIZ / "python_embeded" / "python.exe"
F = AQUI / "fontes"
CASOS = {
    "w4a4_gemma": "gemma_sint_bf16", "w4a4_ltx": "ltx_sint_bf16", "w4a8_gemma": "gemma_sint_bf16",
    "w4a8_gemma_nocb": "gemma_sint_bf16", "w4a8_ltx": "ltx_sint_bf16", "int8_gemma": "gemma_sint_bf16",
    "int8_gemma_noconvrot": "gemma_sint_bf16", "int8_ltx": "ltx_sint_bf16",
    "mixed_zimage": "zimage_sint_native_bf16", "mixed_zimage_somente": "zimage_sint_native_bf16",
    "smooth_gemma": "gemma_sint_bf16", "te_int8": "gemma_sint_bf16", "te_fp8": "gemma_sint_bf16",
}
linhas = []
for caso, fonte in list(CASOS.items()) + [("awq_gemma", "gemma_sint_bf16")]:
    d = AQUI / "depois" / caso
    saida = d / ("gemma_sint_bf16_q4_1_awq.safetensors" if caso == "awq_gemma" else "saida.safetensors")
    res = []
    for rot, tools in (("antes", RAIZ / ".scratch" / "antes_tools_conv_20260929"), ("depois", RAIZ / "tools")):
        p = subprocess.run([str(PY), "-s", str(tools / "verify_w4a4.py"), str(saida), "--source",
                            str(F / f"{fonte}.safetensors"), "--structural-only"],
                           capture_output=True, text=True, env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"})
        pas = [ln for ln in p.stdout.splitlines() if ln.startswith(("Structural", "ERROR", "Source comparison"))]
        res.append(f"{rot}: rc={p.returncode} {' | '.join(pas)[:220]}")
    linhas.append(f"{caso}\n    " + "\n    ".join(res))
txt = "\n".join(linhas)
print(txt)
(AQUI / "verifica_saidas.txt").write_text(txt + "\n", encoding="utf-8")
