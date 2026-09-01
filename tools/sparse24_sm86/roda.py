r"""Compila o GEMM 2:4 do CUTLASS com tiles que cabem na sm_86, confere e cronometra.

O QUE ESTA FERRAMENTA RESPONDE
------------------------------
Uma pergunta, e ela e de TEMPO: esparsidade 2:4 paga? Esta bancada ja mediu que a poda 2:4 guiada
por ativacao custa MENOS erro que quantizar a 4 bits (0,0794 contra 0,0923, `probe_esparso_vs_quant`),
e todo aquele numero e erro numerico. Erro menor num formato mais lento nao compra nada.

E responde uma segunda, que apareceu no caminho: por que o kernel 2:4 do xformers recusa esta placa.
`smem_de_cada_config()` devolve `sizeof(GemmKernel::SharedStorage)` calculado pelo COMPILADOR para a
config exata que eles distribuem e para tres alternativas. Isso transforma "acho que pede ~139 KB"
em leitura.

COMO RODAR
----------
Precisa do ambiente do MSVC no PATH e dos headers do CUTLASS (nao ficam neste repo -- sao ~43 MB):

    curl -sL -o cutlass.tgz https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.8.0dev
    tar -xzf cutlass.tgz
    set CUTLASS_INCLUDE=<...>\cutlass-4.8.0dev\include
    tools\sparse24_sm86\roda.bat

`roda.bat` carrega o `vcvars64` e o CUDA 13.2, que e o que faz `cl.exe` existir.

O LOCK
------
A cronometragem passa por `_timing.compare()`, que pega o `GPU_BENCH.lock` sozinho. NAO tome
`Assert-GpuLock` antes: a ferramenta recusa a propria run se o lock ja estiver preso.

NAO COBERTO
-----------
GEMM isolado, nao render. bf16, A e B row-major, sem split-k, sem bias. Quatro configs escolhidas
para isolar um eixo por vez, NAO uma varredura de tuning -- a config mais rapida que cabe nao foi
procurada, entao qualquer razao aqui e um piso do que 2:4 pode dar, nao o teto. Nao mede o custo de
podar nem de empacotar, porque num modelo servido isso acontece uma vez. Nao mede memoria. E nao
diz NADA sobre qualidade de imagem.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "tools"))

# Shapes reais do Z-Image, contados no header do checkpoint em 2026-09-01; o numero e quantas
# camadas do modelo tem aquela forma.
SHAPES = [((10240, 3840), 60, "feed_forward.w1/w3"),
          ((11520, 3840), 30, "attention.qkv"),
          ((3840, 10240), 30, "feed_forward.w2"),
          ((3840, 3840), 30, "attention.out")]
M_PADRAO = (128, 512, 1024, 5856, 16384)
NOMES_CFG = ["256x128x64 s4 (a do xformers)", "256x128x64 s2", "128x128x64 s3", "128x128x64 s2"]


def poda24(w):
    import torch
    n, k = w.shape
    g = w.abs().reshape(n, k // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    m = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
    return w.reshape(n, k // 4, 4).mul(m).reshape(n, k)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # `type=str` e nao `Path`: um caminho do Windows atravessa cmd/bash e chega com aspas
    # literais ou espaco sobrando conforme a camada que o passou, e `Path('"C:\x"')` nao
    # existe. Normalizar aqui e uma linha; diagnosticar "headers nao achados" com o caminho
    # certo impresso na mensagem de erro custou tres tentativas.
    p.add_argument("--cutlass", type=str, default=os.environ.get("CUTLASS_INCLUDE", ""))
    p.add_argument("--m", type=int, nargs="+", default=list(M_PADRAO))
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--so-smem", action="store_true",
                   help="compila e imprime a memoria compartilhada de cada config, sem cronometrar")
    a = p.parse_args()

    a.cutlass = Path(a.cutlass.strip().strip('"').strip("'")) if a.cutlass else None
    if not (a.cutlass and (a.cutlass / "cutlass/cutlass.h").exists()):
        print(f"headers do CUTLASS nao achados em {a.cutlass!s}. "
              f"Passe --cutlass <dir>/include ou defina CUTLASS_INCLUDE.", file=sys.stderr)
        return 2

    import torch
    from torch.utils.cpp_extension import load
    aqui = Path(__file__).resolve().parent
    (aqui / "build").mkdir(exist_ok=True)
    print("compilando (CUTLASS e template pesado; minutos, nao segundos)...", flush=True)
    t0 = time.perf_counter()
    ext = load(name="sp24_sm86", sources=[str(aqui / "sp24_gemm.cu")],
               build_directory=str(aqui / "build"), verbose=False,
               extra_include_paths=[str(a.cutlass)],
               extra_cuda_cflags=["-gencode=arch=compute_86,code=sm_86",
                                  # O CCCL do CUDA 13.2 recusa o pre-processador tradicional do
                                  # MSVC com um #error explicito; sem esta flag a compilacao morre
                                  # no primeiro header, o que e facil de ler como "toolkit
                                  # incompativel" e nao e.
                                  "-Xcompiler", "/Zc:preprocessor",
                                  "--expt-relaxed-constexpr"])
    print(f"compilou em {time.perf_counter()-t0:.0f}s\n")

    prop = torch.cuda.get_device_properties(0)
    limite = prop.shared_memory_per_block_optin
    print(f"{prop.name}  cc {prop.major}.{prop.minor}")
    print(f"limite de shared dinamica desta placa: {limite:,} bytes "
          f"(a A100/sm_80 aceita 163.840)\n")
    print("=== sizeof(GemmKernel::SharedStorage), calculado pelo compilador ===")
    print(f"{'config':32s} {'smem (bytes)':>13s}  {'cabe?':>6s}")
    print("-" * 56)
    smems = ext.smem_de_cada_config()
    for nome, s in zip(NOMES_CFG, smems):
        print(f"{nome:32s} {s:>13,}  {'SIM' if s <= limite else 'NAO':>6s}")
    cabem = [i for i, s in enumerate(smems) if s <= limite]
    print(f"\ncabem na sm_86: {len(cabem)} de {len(smems)}")
    if a.so_smem:
        return 0

    from _bench_guard import BenchGuard
    from _timing import compare, provenance, wall_ms
    from torch.sparse._semi_structured_conversions import (
        sparse_semi_structured_from_dense_cutlass as empacota,
    )

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    guarda = BenchGuard("bench:sp24_sm86_tempo")
    guarda.__enter__()

    # ------------------------------------------------------------------ controle de correcao
    # Um kernel que devolve lixo rapido tambem "ganha" numa tabela de tempo. A referencia e
    # mm(W_PODADO, x) -- o mesmo produto -- e o controle negativo e mm(W_DENSO, x), que TEM de
    # divergir: se ele nao divergisse, a poda nao teria acontecido e o teste passaria por engano.
    print("\n=== controle: cada config computa o produto do peso PODADO? ===")
    Wc = torch.randn(256, 512, device=dev, dtype=torch.bfloat16)
    Wp = poda24(Wc)
    xc = torch.randn(512, 256, device=dev, dtype=torch.bfloat16)
    pk, mt = empacota(Wp)
    ref = torch.mm(Wp, xc).float()
    dense_ref = torch.mm(Wc, xc).float()
    bons = []
    for i in cabem:
        try:
            got = ext.mm(pk, xc, mt, i).float()
        except Exception as e:
            print(f"  {NOMES_CFG[i]:32s} FALHOU: {str(e).splitlines()[0][:60]}")
            continue
        rel = ((got - ref).norm() / ref.norm()).item()
        rel_d = ((got - dense_ref).norm() / dense_ref.norm()).item()
        ok = rel < 1e-2 < rel_d
        print(f"  {NOMES_CFG[i]:32s} contra podado {rel:.2e}   contra denso {rel_d:.2e}  "
              f"{'OK' if ok else 'SUSPEITO'}")
        if ok:
            bons.append(i)
    if not bons:
        print("nenhuma config passou o controle. Nao cronometro o que nao esta correto.",
              file=sys.stderr)
        guarda.__exit__(None, None, None)
        return 3

    ultimo = None
    for (N, K), quantas, apelido in SHAPES:
        W = poda24(torch.randn(N, K, device=dev, dtype=torch.bfloat16))
        packed, meta = empacota(W)
        print(f"\n=== W [{N}, {K}]  ({apelido}, {quantas} camadas no Z-Image) ===")
        print(f"{'M':>7s} {'denso ms':>10s} " +
              " ".join(f"{NOMES_CFG[i].split()[0]:>14s}" for i in bons))
        print("-" * (19 + 15 * len(bons)))
        for M in a.m:
            torch.cuda.empty_cache()
            x = torch.randn(K, M, device=dev, dtype=torch.bfloat16).contiguous()
            caminhos = {"denso": lambda W=W, x=x: torch.mm(W, x)}
            for i in bons:
                caminhos[f"c{i}"] = (lambda packed=packed, x=x, meta=meta, i=i:
                                     ext.mm(packed, x, meta, i))
            try:
                r = compare(caminhos, iters=a.iters, repeats=a.repeats, timer=wall_ms,
                            baseline="denso", owner=f"bench:sp24_sm86_{N}x{K}_M{M}",
                            better="mais rapido", worse="MAIS LENTO")
            except Exception as e:
                print(f"{M:>7d}   compare falhou: {type(e).__name__}: {e}")
                continue
            ultimo = r
            cols = []
            for i in bons:
                chave = f"c{i}"
                cols.append(f"{'--':>14s}" if chave in r.failed
                            else f"{r.ratios[chave].value:>13.3f}x")
            print(f"{M:>7d} {r.times['denso']:10.4f} " + " ".join(cols))

    guarda.__exit__(None, None, None)
    if ultimo is not None:
        print()
        print(provenance(ultimo))
    print()
    print("NAO COBERTO: GEMM isolado, nao render. bf16, row-major, sem split-k, sem bias. As")
    print("  quatro configs isolam UM eixo por vez e nao sao uma varredura de tuning -- a config")
    print("  mais rapida que cabe na sm_86 nao foi procurada, entao estas razoes sao um PISO do")
    print("  que 2:4 pode dar, nao o teto. Nao mede podar nem empacotar (acontece uma vez), nao")
    print("  mede memoria, e nao diz nada sobre qualidade de imagem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
