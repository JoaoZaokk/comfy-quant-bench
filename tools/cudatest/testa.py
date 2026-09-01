"""Compila e roda uma extensao CUDA minima. Responde UMA pergunta: da para compilar aqui?

A duvida que este arquivo existe para matar: `nvcc` nesta maquina e **13.2** e o torch e
**2.13.0+cu130** (13.0). Eu disse que versao diferente "costuma dar dor de cabeca" e nao testei --
exatamente o padrao que o dono mandou parar de repetir. Entao: testar.

O que uma passagem aqui prova: cabecalhos do torch achados, ABI do MSVC compativel, ligacao com
`torch_cuda` funcionando, kernel lancado de verdade e resultado correto. Ou seja, todo o caminho de
que qualquer kernel 2:4 proprio precisaria.

O que NAO prova: nada sobre CUTLASS, nada sobre cuSPARSELt, nada sobre desempenho. Um kernel de
soma nao usa template pesado nem tensor core -- e justamente template pesado que costuma derrubar o
MSVC.
"""
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402


def main() -> int:
    print(f"torch {torch.__version__}   cuda do torch {torch.version.cuda}")
    print(f"placa {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
    print(f"nvcc que sera usado: CUDA_HOME={torch.utils.cpp_extension.CUDA_HOME}")
    print()
    (AQUI / "build").mkdir(parents=True, exist_ok=True)
    print("compilando...", flush=True)
    inicio = time.perf_counter()
    try:
        mod = load(name="soma_cuda_teste", sources=[str(AQUI / "soma.cu")],
                   build_directory=str(AQUI / "build"), verbose=True,
                   # `/Zc:preprocessor` NAO e ajuste cosmetico: o CCCL do CUDA 13.2 recusa o
                   # pre-processador tradicional do MSVC com um `#error` explicito, e a propria
                   # mensagem manda passar esta flag. Sem ela a compilacao morre no primeiro
                   # cabecalho, antes de ver uma linha do nosso codigo -- o que e facil de ler
                   # como "toolkit incompativel com o torch" e nao e.
                   extra_cuda_cflags=["-gencode=arch=compute_86,code=sm_86",
                                      "-Xcompiler", "/Zc:preprocessor"])
    except Exception as e:
        print(f"\nFALHOU NA COMPILACAO em {time.perf_counter()-inicio:.0f}s")
        print(f"{type(e).__name__}: {str(e)[-1500:]}")
        return 1
    print(f"\ncompilou em {time.perf_counter()-inicio:.0f}s")

    a = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    b = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    saida = mod.soma(a, b)
    ref = a + b * 2.0
    erro = (saida - ref).abs().max().item()
    print(f"resultado: erro maximo contra o torch = {erro:.3e}")
    # Um kernel que nao rodasse devolveria `empty_like` sem escrever -- lixo, e o erro seria enorme.
    # Comparar contra zero e o que separa "compilou" de "compilou E executou certo".
    ok = erro == 0.0
    print(f"VEREDITO: {'compila e executa correto nesta maquina' if ok else 'compilou mas o resultado esta errado'}")
    print()
    print("NAO COBERTO: um kernel trivial. Nada aqui diz que CUTLASS ou cuSPARSELt compilariam --")
    print("  esses usam template pesado, que e justamente o que costuma derrubar o MSVC, e no caso")
    print("  do 2:4 do PyTorch o codigo esta excluido no fonte por `#if defined(_MSC_VER)`.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
