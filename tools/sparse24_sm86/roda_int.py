r"""2:4 esparso em INT8 e INT4: as linhas de 5,0 e 2,5 bits/peso, cronometradas.

O QUE FALTAVA
-------------
`sp24_gemm.cu` fechou o 2:4 em bf16 -- 1,7x a 1,95x sobre o denso, medido em duas passadas. Mas
aquilo custa 9,0 bits/peso, contra os 4,0 do W4A4 que esta bancada ja entrega. O registro daquele
dia deixou uma celula explicitamente vazia:

    formato          bits/peso   erro     contra o denso bf16
    2:4 + bf16            9,0    0,0794   1,7x-1,95x  MEDIDO
    W4A4 ConvRot          4,0    0,0923   1,83x-1,93x
    2:4 + W4A4            3,0    0,1306   SEM KERNEL, nao medido   <- esta

Esta ferramenta tira a celula do vazio, e leva o INT8 junto: 5,0 bits/peso e o braco C de
`probe_esparso_promocao.py` (promover a 8 bits so as camadas que aceitam), que tambem nunca foi
cronometrado. O INT8 nao e escada -- e o CONTROLE que separa "o INT4 quebrou" de "o caminho inteiro
quebrou". Sem ele um INT4 que falha nao se distingue de esparso-inteiro nao funcionar.

O EMPACOTAMENTO, E O QUE PERGUNTAR AO COMPILADOR EVITOU
--------------------------------------------------------
2:4 precisa de tres numeros que dependem do tipo do elemento. `params_de_cada()` os pergunta ao
COMPILADOR em vez de deduzi-los:

    tipo   kSparse   kElementsPerElementE   sizeof(ElementE)
    bf16         2                      8                  2
    int8         2                     16                  4
    int4         2                     32                  4

O plano original era reusar o empacotador do PyTorch para os dois inteiros; o gate que comparava os
parametros disparou (16 contra 32) e impediu isso. **Foi o gate que estava certo.** `k/2/32` significa
64 elementos logicos por uint32, ou seja **4 bits de metadata por 8 valores** -- metade da densidade
do int8, o que so fecha se a unidade de mascara for o PAR de int4 e nao o valor.

Isso nao foi deduzido ate o fim: com o kernel na mao e a referencia inteira exata, as duas
granularidades foram TESTADAS. Por elemento falha em 6 de 6 padroes; por par bate exato em 6 de 6, e
a codificacao saiu junto -- `nibble = idx0 | (idx1 << 2)`, oito nibbles por uint32:

    {(0,1): 4, (0,2): 8, (0,3): 12, (1,2): 9, (1,3): 13, (2,3): 14}

**Consequencia que muda a contabilidade e a acuracia:** 4 valores de 4 bits mais 4 bits de indice
por 8 pesos da **2,5 bits/peso**, nao os 3,0 que este repo estimou. Em troca, podar em pares e uma
restricao mais APERTADA que podar em valores, entao o erro do int4 esparso NAO e comparavel ao 2:4
por elemento que as outras tabelas desta bancada mediram.

O torch nao tem dtype int4, entao os valores vao dois por byte (par no nibble baixo, impar no alto,
que e o layout do `cutlass::int4b_t`).

A REFERENCIA E EXATA, DE PROPOSITO
-----------------------------------
Valores int4 cabem em int8, e `torch._int_mm` acumula em int32. Entao a referencia do braco INT4 e
uma matmul inteira **exata** do mesmo peso podado e quantizado: qualquer diferenca e defeito do
kernel ou do empacotamento, nunca arredondamento. Em bf16 isso nao era possivel e o controle tinha
de tolerar 1e-2.

NAO COBERTO
-----------
GEMM isolado, nao render. Uma placa. As configs isolam estagios e nao sao varredura de tuning. O
INT4 aqui e int4 SIMETRICO por linha, **nao e ConvRot** -- a rotacao do W4A4 desta bancada nao esta
neste caminho, entao o tempo medido e do formato, e o erro de 0,1306 daquela tabela veio de outro
lugar e nao se soma a este numero sem cuidado. Nao mede podar, quantizar nem empacotar.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "tools"))

SHAPES = [((10240, 3840), 60, "feed_forward.w1/w3"),
          ((11520, 3840), 30, "attention.qkv"),
          ((3840, 3840), 30, "attention.out")]
M_PADRAO = (512, 1024, 5856, 16384)
NOMES = ["int8 128x128x128 s3", "int8 128x128x128 s2",
         "int4 128x128x256 s3", "int4 128x128x256 s2"]


def poda24(w):
    import torch
    n, k = w.shape
    g = w.abs().reshape(n, k // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    m = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
    return w.reshape(n, k // 4, 4).mul(m).reshape(n, k)


def quantiza_int(w, bits: int):
    """Simetrico por linha, para o intervalo de `bits`. Devolve os valores em int8."""
    import torch
    lim = 2 ** (bits - 1) - 1
    escala = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / lim
    return (w / escala).round().clamp(-lim - 1, lim).to(torch.int8)


def empacota_nibbles(v):
    """[m, k] int8 com valores em [-8,7] -> [m, k/2] int8, dois por byte.

    Layout do `cutlass::int4b_t`: elemento par no nibble BAIXO, impar no ALTO. O `& 0xF` e o que
    faz o negativo virar complemento de dois de 4 bits em vez de estender o sinal por cima do
    vizinho -- sem ele todo peso negativo corrompe o peso seguinte.
    """
    baixo = (v[:, 0::2].to(int) & 0xF)
    alto = (v[:, 1::2].to(int) & 0xF)
    import torch
    return ((baixo | (alto << 4)) & 0xFF).to(torch.uint8).view(torch.int8).contiguous()


def poda24_pares(w):
    """2:4 em PARES, que e o que o tensor core esparso do INT4 exige.

    DESCOBERTO POR EXPERIMENTO em 2026-09-01, nao lido em documentacao: com o kernel na mao e a
    referencia inteira exata, testar as duas granularidades separou-as em um comando. Por elemento
    (o que serve para bf16 e int8) falha em 6 de 6 padroes; por par bate exato em 6 de 6.

    Grupo = 8 valores = 4 pares; sobrevivem 2 pares. Devolve (peso podado, mascara dos pares) -- a
    mascara vai JUNTO de proposito: rededuzi-la de `w != 0` marcaria como podado um par que a
    quantizacao zerou legitimamente, e o erro so apareceria como resultado errado muito depois.
    """
    import torch
    n, k = w.shape
    g = w.reshape(n, k // 8, 4, 2)
    escore = g.abs().sum(-1)                                   # L1 do par
    fracos = escore.argsort(dim=-1)[..., :2]
    manter = torch.ones_like(escore, dtype=torch.bool).scatter_(-1, fracos, False)
    return (g * manter.unsqueeze(-1)).reshape(n, k), manter


def empacota_int4(wq, manter):
    """(valores int4 em int8 [n,k], mascara [n,k/8,4]) -> (A [n,k/4] int8, meta [n,k/64] int32).

    Codificacao medida: nibble = idx0 | (idx1 << 2) com idx0 < idx1, oito nibbles por uint32.
    """
    import torch
    n, k = wq.shape
    g = wq.reshape(n, k // 8, 4, 2)
    # Os dois indices mantidos, em ordem crescente. `stable=True` e o que garante idx0 < idx1 --
    # sem isso a ordem dentro do grupo depende da implementacao do sort e o nibble sai trocado.
    ordem = manter.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[..., :2]
    idx, _ = ordem.sort(dim=-1)
    vals = torch.gather(g, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))
    A = empacota_nibbles(vals.reshape(n, k // 2).contiguous())
    nib = (idx[..., 0] | (idx[..., 1] << 2)).to(torch.int64)   # [n, k/8]
    nib = nib.reshape(n, k // 64, 8)
    desloc = (torch.arange(8, device=wq.device, dtype=torch.int64) * 4)
    palavra = (nib << desloc).sum(-1)
    # int32 e sinalizado; envolver preserva os bits, que e o que o kernel le.
    palavra = torch.where(palavra >= (1 << 31), palavra - (1 << 32), palavra)
    palavra = palavra.to(torch.int32)

    # O REORDENAMENTO, que quase ficou de fora e que a primeira versao desta funcao nao tinha.
    # `LayoutE` e `ColumnMajorInterleaved<2>`: a metadata nao vai em ordem de linha, vai espalhada.
    # A sonda que descobriu a codificacao usou um padrao UNIFORME, e um padrao uniforme e invariante
    # a reordenamento -- ou seja, ela nao PODIA detectar isto, e eu escrevi essa propriedade no
    # docstring dela e mesmo assim tratei o 6/6 como se cobrisse o layout. Sem este scatter o kernel
    # roda e erra 65407 de 65536 elementos: nao estoura, nao avisa, so devolve numero errado.
    from torch.sparse._semi_structured_conversions import (
        _calculate_meta_reordering_scatter_offsets as _offsets,
    )
    linhas, cols = palavra.shape
    reord = palavra.new_empty((linhas * cols,))
    reord.scatter_(0, _offsets(linhas, cols, torch.int32, wq.device), palavra.reshape(-1))
    return A, reord.view(linhas, cols).contiguous()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cutlass", type=str, default=os.environ.get("CUTLASS_INCLUDE", ""))
    p.add_argument("--m", type=int, nargs="+", default=list(M_PADRAO))
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    a.cutlass = Path(a.cutlass.strip().strip('"').strip("'")) if a.cutlass else None
    if not (a.cutlass and (a.cutlass / "cutlass/cutlass.h").exists()):
        print(f"headers do CUTLASS nao achados em {a.cutlass!s}", file=sys.stderr)
        return 2

    import torch
    from torch.utils.cpp_extension import load
    from torch.sparse._semi_structured_conversions import (
        sparse_semi_structured_from_dense_cutlass as empacota,
    )
    aqui = Path(__file__).resolve().parent
    (aqui / "build_int").mkdir(exist_ok=True)
    print("compilando...", flush=True)
    t0 = time.perf_counter()
    ext = load(name="sp24_int", sources=[str(aqui / "sp24_int.cu")],
               build_directory=str(aqui / "build_int"), verbose=False,
               extra_include_paths=[str(a.cutlass)],
               extra_cuda_cflags=["-gencode=arch=compute_86,code=sm_86",
                                  "-Xcompiler", "/Zc:preprocessor", "--expt-relaxed-constexpr"])
    print(f"compilou em {time.perf_counter()-t0:.0f}s\n")

    prop = torch.cuda.get_device_properties(0)
    limite = prop.shared_memory_per_block_optin
    smems, params = ext.smems(), ext.params_de_cada()
    print(f"limite desta placa: {limite:,} bytes")
    print(f"{'config':22s} {'smem':>9s} {'cabe':>5s}  {'kSparse':>7s} {'porE':>5s} {'sizeofE':>7s}")
    print("-" * 62)
    for nome, s, pr in zip(NOMES, smems, params):
        print(f"{nome:22s} {s:>9,} {'SIM' if s <= limite else 'NAO':>5s}  "
              f"{pr[0]:>7d} {pr[1]:>5d} {pr[2]:>7d}")
    cabem = [i for i, s in enumerate(smems) if s <= limite]
    print(f"\ncabem: {len(cabem)} de {len(smems)}")
    if not cabem:
        return 3

    # Reusar o empacotador int8 do PyTorch para o int4 so e valido se os tres parametros do
    # CUTLASS conferirem. Verificado, nao assumido.
    p8 = next((params[i] for i in (0, 1)), None)
    p4 = next((params[i] for i in (2, 3)), None)
    print(f"parametros (kSparse, porE, sizeofE): int8 {p8[:3]}   int4 {p4[:3]}")
    print("  divergem, e por isso o int4 NAO usa o empacotador do PyTorch: `kElementsPerElementE`")
    print("  32 contra 16 forca 4 bits de metadata por 8 valores logicos, ou seja a unidade de")
    print("  mascara e o PAR. Medido em 2026-09-01 testando as duas granularidades contra a")
    print("  referencia exata: por elemento falha 6/6, por par bate 6/6.")

    from _bench_guard import BenchGuard
    from _timing import compare, provenance, wall_ms

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    guarda = BenchGuard("bench:sp24_int_tempo")
    guarda.__enter__()

    def prepara(N, K, M, bits):
        """Devolve (A empacotado, B buffer, meta, referencia int32 exata).

        Os dois caminhos divergem no que a poda 2:4 significa, nao so no empacotamento: o int8
        poda 2 de cada 4 VALORES e o int4 poda 2 de cada 4 PARES. Sao restricoes diferentes e a
        do int4 e a mais apertada -- o custo disso em erro esta em `--erro`, nao aqui.
        """
        Wf = torch.randn(N, K, device=dev, dtype=torch.float32)
        x = torch.randint(-8 if bits == 4 else -127, 8 if bits == 4 else 128,
                          (K, M), device=dev, dtype=torch.int8)
        if bits == 8:
            Wq = quantiza_int(poda24(Wf), 8)
            A, meta = empacota(Wq)                        # empacotador do PyTorch, valido aqui
            xk = x.t().contiguous()
        else:
            Wp, manter = poda24_pares(Wf)
            Wq = quantiza_int(Wp, 4) * manter.repeat_interleave(2, -1).reshape(N, K)
            A, meta = empacota_int4(Wq, manter)
            xk = empacota_nibbles(x.t().contiguous())
        ref = torch._int_mm(Wq, x)                        # int32, EXATA
        return A, xk, meta, ref

    print("\n=== controle: cada config computa a matmul inteira EXATA do peso podado? ===")
    bons = []
    for i in cabem:
        bits = 8 if i < 2 else 4
        A, B, meta, ref = prepara(256, 512, 256, bits)
        try:
            got = ext.mm(A, B, meta, i, 256, 256, 512)
        except Exception as e:
            print(f"  {NOMES[i]:22s} FALHOU: {str(e).splitlines()[0][:70]}")
            continue
        iguais = torch.equal(got, ref)
        difs = (got != ref).sum().item()
        print(f"  {NOMES[i]:22s} {'EXATO' if iguais else f'{difs} elementos diferem'}")
        if iguais:
            bons.append(i)
    if not bons:
        print("nenhuma config passou. Nao cronometro o que nao esta correto.", file=sys.stderr)
        guarda.__exit__(None, None, None)
        return 3

    ultimo = None
    for (N, K), quantas, apelido in SHAPES:
        print(f"\n=== W [{N}, {K}]  ({apelido}, {quantas} camadas no Z-Image) ===")
        print(f"{'M':>7s} {'bf16 ms':>9s} {'int8 denso':>11s} " +
              " ".join(f"{NOMES[i].split()[0] + NOMES[i].split()[-1]:>12s}" for i in bons))
        print("-" * (30 + 13 * len(bons)))
        for M in a.m:
            torch.cuda.empty_cache()
            Wb = poda24(torch.randn(N, K, device=dev, dtype=torch.bfloat16))
            xb = torch.randn(K, M, device=dev, dtype=torch.bfloat16)
            preparados = {i: prepara(N, K, M, 8 if i < 2 else 4) for i in bons}
            xi8 = torch.randint(-127, 128, (K, M), device=dev, dtype=torch.int8)
            Wi8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
            caminhos = {"bf16": lambda Wb=Wb, xb=xb: torch.mm(Wb, xb),
                        "i8den": lambda Wi8=Wi8, xi8=xi8: torch._int_mm(Wi8, xi8)}
            for i in bons:
                A, B, mt, _ = preparados[i]
                caminhos[f"c{i}"] = (lambda A=A, B=B, mt=mt, i=i, N=N, M=M, K=K:
                                     ext.mm(A, B, mt, i, N, M, K))
            try:
                r = compare(caminhos, iters=a.iters, repeats=a.repeats, timer=wall_ms,
                            baseline="bf16", owner=f"bench:sp24_int_{N}x{K}_M{M}",
                            better="mais rapido", worse="MAIS LENTO")
            except Exception as e:
                print(f"{M:>7d}   compare falhou: {type(e).__name__}: {str(e)[:70]}")
                continue
            ultimo = r
            def col(ch):
                return f"{'--':>12s}" if ch in r.failed else f"{r.ratios[ch].value:>11.3f}x"
            print(f"{M:>7d} {r.times['bf16']:9.4f} {col('i8den'):>11s} " +
                  " ".join(col(f"c{i}") for i in bons))

    guarda.__exit__(None, None, None)
    if ultimo is not None:
        print()
        print(provenance(ultimo))
    print()
    print("NAO COBERTO: GEMM isolado, nao render. Uma placa, configs sem varredura de tuning. O")
    print("  int4 aqui e SIMETRICO POR LINHA e NAO e ConvRot -- a rotacao nao esta neste caminho,")
    print("  entao o erro 0,1306 da tabela de composicao veio de outro lugar e nao se soma a este")
    print("  tempo sem cuidado. Nao mede podar, quantizar nem empacotar. Todas as razoes sao")
    print("  contra o denso bf16, que e o mesmo denominador da tabela do 2:4 em bf16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
