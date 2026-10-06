"""Fase 7, etapa 2: confere o quantizador g64_quant (rotacao ConvRot + escala por grupo 64/128, SwiGLU opcional) contra
uma referencia em fp32 e mede o tempo contra o quantizador ConvRot por linha do ck (producao) e o SwiGLU fundido do ck.

Criterio (criterio_fase7.md): codigos iguais a referencia exceto empates de arredondamento (<= 0,1 % dos codigos, |dif| 1),
escalas rel < 1e-3; linhas de padding com q = 0 e s = 0.
Emenda de 05/10 (depois da primeira execucao, registrada em criterio_fase7.md): a escala e arredondada para CIMA em bf16
(ulp relativo ate 2^-7 = 7,8e-3); a FWHT do kernel soma em outra ordem que a matmul da referencia, entao quando am/7 cai
a ~1e-6 de um valor bf16 os dois arredondam para vizinhos. A escala passa se: <= 0,1 % das escalas diferentes, todas a
exatamente 1 ulp bf16 e com am/7 da referencia a <= 1e-5 (relativo) da fronteira; as demais iguais bit a bit.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
HERE = os.path.dirname(os.path.abspath(__file__))

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


def bf16_up(t):
    u = t.float().contiguous().view(torch.int32)
    low = u & 0xFFFF
    u = (u & ~0xFFFF) + (low != 0).to(torch.int32) * 0x10000
    return u.view(torch.float32)


def unpack(q, k):
    u = q.view(torch.uint8).to(torch.int16)
    lo, hi = u & 0xF, (u >> 4) & 0xF
    v = torch.stack([lo, hi], -1).reshape(q.shape[0], k)
    return torch.where(v >= 8, v - 16, v)


def ref(x, G, swiglu):
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard
    h = _build_hadamard(256, device=x.device, dtype=torch.float32)
    xf = x.float()
    if swiglu:
        k = xf.shape[1] // 2
        g, u = xf[:, :k], xf[:, k:]
        xf = g * torch.sigmoid(g) * u
    m, k = xf.shape
    xr = (xf.reshape(m, k // 256, 256) @ h).reshape(m, k)
    am = xr.reshape(m, k // G, G).abs().amax(-1)
    t = (am / 7).float()
    s = bf16_up(t)
    q = torch.clamp(torch.round(xr.reshape(m, k // G, G) / s[..., None]), -7, 7).reshape(m, k)
    return q.to(torch.int16), s.t().contiguous(), t.t().contiguous()


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    mod = carrega("g64_quant")
    ok = True
    for (M, K, swiglu) in ((300, 4096, False), (4096, 4096, False), (4096, 12288, True), (517, 12288, True)):
        x = torch.randn(M, 2 * K if swiglu else K, device="cuda") * (1 + 8 * (torch.rand(1, 2 * K if swiglu else K, device="cuda") > 0.99))
        x = x.to(torch.bfloat16)
        for G in (64, 128):
            q, s = mod.quant(x, G, swiglu, 128)
            torch.cuda.synchronize()
            qr, sr, tr = ref(x, G, swiglu)
            qk = unpack(q[:M], K)
            dif = (qk - qr).abs()
            frac = float((dif > 0).float().mean())
            sk = s[:, :M].contiguous()
            srel = float(((sk - sr).abs() / sr.clamp_min(1e-30)).max())
            dm = sk != sr
            n_ds = int(dm.sum())
            passos = (sk.view(torch.int32) - sr.view(torch.int32))[dm].abs()
            ulp1 = bool((passos == 0x10000).all())
            tb = tr.view(torch.int32)
            baixo = (tb & ~0xFFFF).view(torch.float32)
            dist = torch.minimum(tr - baixo, bf16_up(tr) - tr) / tr.clamp_min(1e-30)
            dist_max = float(dist[dm].max()) if n_ds else 0.0
            esc_ok = n_ds / sr.numel() <= 1e-3 and ulp1 and dist_max <= 1e-5
            pad_ok = bool((q[M:] == 0).all() and (s[:, M:] == 0).all())
            passa = frac <= 1e-3 and int(dif.max()) <= 1 and esc_ok and pad_ok
            ok &= passa
            print(f"M={M} K={K} swiglu={swiglu} G={G}: codigos diferentes {frac * 100:.4f} % (max {int(dif.max())}), "
                  f"escala rel max {srel:.2e} ({n_ds} de {sr.numel()} diferentes, todas a 1 ulp={ulp1}, distancia max "
                  f"a fronteira {dist_max:.1e}), padding ok={pad_ok} -> {'OK' if passa else 'FALHA'}", flush=True)
    # tempo: M = 4096 e 16384, K = 4096 (q/k/v/out) e K = 12288 com SwiGLU (mlp.out)
    from comfy_kitchen.backends.cuda import quantize_int4_rowwise_convrot64
    try:
        from comfy_kitchen.backends.cuda import _swiglu_quant
    except Exception:  # noqa: BLE001
        _swiglu_quant = None
    for M in (4096, 16384):
        x = (torch.randn(M, 4096, device="cuda")).to(torch.bfloat16)
        t_ck = timeit(lambda: quantize_int4_rowwise_convrot64(x, 256))
        t_64 = timeit(lambda: mod.quant(x, 64, False, 128))
        t_128 = timeit(lambda: mod.quant(x, 128, False, 128))
        print(f"tempo M={M} K=4096: ck por linha {t_ck * 1000:.0f} us, g64 {t_64 * 1000:.0f} us, g128 {t_128 * 1000:.0f} us", flush=True)
        h = (torch.randn(M, 2 * 12288, device="cuda")).to(torch.bfloat16)
        t_s64 = timeit(lambda: mod.quant(h, 64, True, 128))
        t_s128 = timeit(lambda: mod.quant(h, 128, True, 128))
        txt = f"tempo M={M} K=12288 SwiGLU: g64 fundido {t_s64 * 1000:.0f} us, g128 fundido {t_s128 * 1000:.0f} us"
        if _swiglu_quant is not None:
            t_sck = timeit(lambda: _swiglu_quant.swiglu_quantize_int4_rowwise_convrot64(h))
            txt += f", ck SwiGLU por linha fundido {t_sck * 1000:.0f} us"
        print(txt, flush=True)
    print("RESULTADO:", "OK" if ok else "FALHA", flush=True)


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("comfy_portable:fase7_g64_quant_test") as gd:
        if gd.refused:
            print(gd.refused); raise SystemExit(1)
        main()
