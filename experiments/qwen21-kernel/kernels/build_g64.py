"""Fase 7: compila g64_gemm.cu (GEMM ConvRot W4A4 com escala por linha ou g64) isolado para sm_86 em build_g64/<nome>.

    python -s build_g64.py [nome]     (nome padrao g64_gemm; fontes = <nome>.cu)
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
import torch  # noqa: E402,F401
from torch.utils.cpp_extension import load  # noqa: E402

nome = sys.argv[1] if len(sys.argv) > 1 else "g64_gemm"
bd = HERE / "build_g64" / nome
bd.mkdir(parents=True, exist_ok=True)
t0 = time.time()
mod = load(name=nome, sources=[str(HERE / f"{nome}.cu")],
           extra_cflags=["/O2", "/Zc:preprocessor"],
           extra_cuda_cflags=["-O3", "-Xcompiler", "/Zc:preprocessor", "--expt-relaxed-constexpr", "--resource-usage",
                              "-lineinfo", "-gencode", "arch=compute_86,code=sm_86"],
           build_directory=str(bd), verbose=True)
print(f"ok: {mod.__file__} em {time.time() - t0:.0f} s", flush=True)
(bd / "OK").write_text(mod.__file__, encoding="utf-8")
