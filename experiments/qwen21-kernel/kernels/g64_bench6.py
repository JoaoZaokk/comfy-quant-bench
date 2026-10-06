"""Fase 7 (v6): GEMM ConvRot W4A4 com peso em razao inteira (pol 5 = g64, pol 7 = g128; 15/17 = as mesmas com a razao em
uint8), g128/g128 FP (pol 4) e o controle por linha (pol 0),
contra o CUTLASS por linha do ck e contra o v3 na mesma sessao (mesmos dados, bias, relogio).

Correcao: 256 linhas do inicio e 256 do fim contra referencia fp32 com as mesmas escalas (rel < 3e-3 e erro maximo
< 4 % do desvio da saida; o arredondamento bf16 da ate ~2 % em valores de 5 desvios) e determinismo (duas execucoes bit a bit iguais na saida inteira: pega corrida de smem).
As 15/17 tem de sair bit a bit iguais as 5/7 (mesmas razoes). Shapes reais (N x K): 4096x4096, 24576x4096, 4096x12288; M = 4096 e 16384. Tempo: 9 rodadas intercaladas (todos os
kernels em cada rodada, ordem girada; 2 aquecimentos + 5 chamadas por medida), mediana por kernel.

    python -s g64_bench6.py [cfgs v6] [cfgs v3]      ex.: g64_bench6.py 0,1,2,3,4 1,2
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
HERE = os.path.dirname(os.path.abspath(__file__))
RODADAS, REP = 9, 5                                   # tempo: mediana de 9 rodadas de 5 chamadas

import torch  # noqa: E402


def carrega(nome):
    pyd = open(os.path.join(HERE, "build_g64", nome, "OK"), encoding="utf-8").read().strip()
    spec = importlib.util.spec_from_file_location(nome, pyd)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timeit(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def pack(v):
    lo = v[:, 0::2].to(torch.int16) & 0xF
    hi = v[:, 1::2].to(torch.int16) & 0xF
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8)


def ref(a, b, e, bias, pol, rows):
    """saida fp32 das linhas `rows` (slice) com as escalas da politica."""
    af, bf = a[rows].float(), b.float()
    K = a.shape[1]
    if pol == 0:
        out = af @ bf.t()
    elif pol == 4:
        out = torch.zeros(af.shape[0], b.shape[0], device=a.device)
        for g in range(K // 128):
            sl = slice(g * 128, (g + 1) * 128)
            out += (af[:, sl] @ bf[:, sl].t()) * e["sa"][g][rows][:, None] * e["sb"][g][None, :]
        return out + bias.float()[None, :]
    else:
        G = 64 if pol % 10 == 5 else 128
        out = torch.zeros(af.shape[0], b.shape[0], device=a.device)
        for g in range(K // G):
            sl = slice(g * G, (g + 1) * G)
            out += (af[:, sl] @ bf[:, sl].t()) * e["cw"][g].float()[None, :]
    return out * e["sa"][rows][:, None] * e["sb"][None, :] + bias.float()[None, :]


def confere(c, a, b, e, bias, pol):
    M = a.shape[0]
    pior_rel, pior_max = 0.0, 0.0
    for rows in (slice(0, 256), slice(M - 256, M)):
        r = ref(a, b, e, bias, pol, rows)
        d = c[rows].float() - r
        pior_rel = max(pior_rel, float(d.norm() / r.norm()))
        pior_max = max(pior_max, float(d.abs().max() / r.std()))
    return pior_rel, pior_max


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    cfgs6 = [int(c) for c in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    cfgs3 = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2]
    m6, m3 = carrega("g64_gemm6"), carrega("g64_gemm3")
    from comfy_kitchen.backends.cuda import int4_linear
    vazio = torch.empty(0, dtype=torch.int32, device="cuda")
    res = {}
    for (N, K) in ((4096, 4096), (24576, 4096), (4096, 12288)):
        for M in (4096, 16384):
            a = torch.randint(-7, 8, (M, K), device="cuda", dtype=torch.int8)
            b = torch.randint(-7, 8, (N, K), device="cuda", dtype=torch.int8)
            ap, bp = pack(a), pack(b)
            bias = (torch.randn(N, device="cuda") * 0.5).to(torch.bfloat16)
            row_a = torch.rand(M, device="cuda") * 0.01 + 0.005
            row_b = torch.rand(N, device="cuda") * 0.01 + 0.005
            esc = {
                0: dict(sa=row_a, sb=row_b, cw=vazio),
                5: dict(sa=row_a, sb=row_b / 255, cw=torch.randint(1, 256, (K // 64, N), device="cuda", dtype=torch.int32)),
                7: dict(sa=row_a, sb=row_b / 255, cw=torch.randint(1, 256, (K // 128, N), device="cuda", dtype=torch.int32)),
            }
            esc[4] = dict(sa=torch.rand(K // 128, M, device="cuda") * 0.01 + 0.005,
                          sb=(torch.rand(K // 128, N, device="cuda") * 0.01 + 0.005).to(torch.bfloat16).float().contiguous(), cw=vazio)
            esc[15] = dict(esc[5], cw=esc[5]["cw"].to(torch.uint8))
            esc[17] = dict(esc[7], cw=esc[7]["cw"].to(torch.uint8))
            linha = {"cutlass_row": {}}
            fns = {"cutlass_row": (lambda: int4_linear(ap, bp, row_a, row_b, bias, torch.bfloat16))}
            casos = [("v3", m3, p, c) for p in (0, 4, 5) for c in cfgs3] + [("v6", m6, p, c) for p in (0, 4, 5, 7, 15, 17) for c in cfgs6]
            for ver, mod, pol, cfg in casos:
                e = esc[pol]
                k = f"{ver}p{pol}c{cfg}"
                try:
                    fn = (lambda mod=mod, e=e, pol=pol, cfg=cfg: mod.gemm(ap, bp, e["sa"], e["sb"], vazio, e["cw"], bias, pol, cfg))
                    c1 = fn(); c2 = fn()
                    torch.cuda.synchronize()
                    rel, mx = confere(c1, a, b, e, bias, pol)
                    det = bool(torch.equal(c1, c2))
                    if pol >= 10:                                       # uint8 = int32 bit a bit
                        e32 = esc[pol - 10]
                        det = det and bool(torch.equal(c1, mod.gemm(ap, bp, e32["sa"], e32["sb"], vazio, e32["cw"], bias, pol - 10, cfg)))
                    fin = bool(torch.isfinite(c1).all())
                    del c1, c2
                    linha[k] = dict(rel=round(rel, 6), max=round(mx, 5), det=det, finito=fin)
                    fns[k] = fn
                except Exception as ex:  # noqa: BLE001
                    linha[k] = dict(erro=repr(ex)[:200])
            # tempo em rodadas intercaladas (todos os kernels por rodada, ordem girada): a deriva de relogio cai em todos
            amostras = {k: [] for k in fns}
            ordem = list(fns)
            for r in range(RODADAS):
                for k in ordem[r % len(ordem):] + ordem[:r % len(ordem)]:
                    fn = fns[k]
                    fn(); fn()
                    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(REP):
                        fn()
                    t1.record(); torch.cuda.synchronize()
                    amostras[k].append(t0.elapsed_time(t1) / REP)
            for k, v in amostras.items():
                v.sort()
                linha[k]["ms"] = round(v[len(v) // 2], 4)
                linha[k]["disp"] = round((v[-1] - v[0]) / v[len(v) // 2], 4)
            linha["cutlass_row"] = linha["cutlass_row"]["ms"]
            base = linha["cutlass_row"]
            partes = [f"cutlass={base:.3f}"]
            for k, v in linha.items():
                if k == "cutlass_row":
                    continue
                if "ms" not in v:
                    partes.append(f"{k}=ERRO({v.get('erro', '')[:60]})"); continue
                ok = v["rel"] < 3e-3 and v["max"] < 0.04 and v["det"] and v["finito"]
                flag = "" if ok else f"!rel{v['rel']:.1e}/max{v['max']:.1e}/det{int(v['det'])}"
                partes.append(f"{k}={v['ms']:.3f}({(v['ms'] / base - 1) * 100:+.0f}%){flag}")
            print(f"N={N} K={K} M={M}: " + " ".join(partes), flush=True)
            res[f"{N}x{K}_M{M}"] = linha
            del a, b, ap, bp
            torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(HERE, "g64_bench6.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("comfy_portable:fase7_g64_bench6") as gd:
        if gd.refused:
            print(gd.refused); raise SystemExit(1)
        main()
