"""Fase 7, item 10: confere o g64_quant2 (recorte da ativacao, escala = amax * ratio / 7 para cima em bf16).

Criterio (escrito antes de rodar):
  1. ratio = 1: q e s identicos bit a bit ao g64_quant (mesmo kernel com a multiplicacao por 1).
  2. ratio = 0,9: contra a referencia fp32 (g64_quant_test.ref com amax * 0,9), o criterio emendado do g64_quant_test:
     escalas iguais bit a bit exceto <= 0,1 % a exatamente 1 ulp bf16 com t a <= 1e-5 (relativo) da fronteira; nos
     grupos de escala igual, codigos iguais exceto <= 0,1 % com |dif| = 1 (empates da FWHT em outra ordem).
  3. Linhas de padding com q = 0 e s = 0; mais codigos saturados em +-7 que com ratio = 1 (o recorte atua).
  4. Tempo igual ao do g64_quant (+-5 %), mediana de 9 rodadas intercaladas (emenda de 05/10, depois da 1a execucao).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import g64_quant_test as T  # noqa: E402


def ref_r(x, G, swiglu, ratio):
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard
    h = _build_hadamard(256, device=x.device, dtype=torch.float32)
    xf = x.float()
    if swiglu:
        k = xf.shape[1] // 2
        g, u = xf[:, :k], xf[:, k:]
        xf = g * torch.sigmoid(g) * u
    m, k = xf.shape
    xr = (xf.reshape(m, k // 256, 256) @ h).reshape(m, k)
    t = (xr.reshape(m, k // G, G).abs().amax(-1) * ratio / 7).float()
    s = T.bf16_up(t)
    q = torch.clamp(torch.round(xr.reshape(m, k // G, G) / s[..., None]), -7, 7).reshape(m, k)
    return q.to(torch.int16), s.t().contiguous(), t.t().contiguous()


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    q1 = T.carrega("g64_quant")
    q2 = T.carrega("g64_quant2")
    ok = True
    for (M, K, swiglu) in ((300, 4096, False), (4096, 4096, False), (4096, 12288, True), (517, 12288, True)):
        C = 2 * K if swiglu else K
        x = (torch.randn(M, C, device="cuda") * (1 + 8 * (torch.rand(1, C, device="cuda") > 0.99))).to(torch.bfloat16)
        for G in (64, 128):
            a_q, a_s = q1.quant(x, G, swiglu, 128)
            b_q, b_s = q2.quant(x, G, swiglu, 128, 1.0)
            c_q, c_s = q2.quant(x, G, swiglu, 128)                   # padrao do argumento = 1,0
            ident = torch.equal(a_q, b_q) and torch.equal(a_s, b_s) and torch.equal(a_q, c_q) and torch.equal(a_s, c_s)
            r_q, r_s = q2.quant(x, G, swiglu, 128, 0.9)
            Mp = r_q.shape[0]
            pad_ok = bool((r_q[M:] == 0).all() and (r_s[:, M:] == 0).all())
            fq, fs, ft = ref_r(x, G, swiglu, 0.9)
            ks = r_s[:, :M]
            dif_s = ks != fs
            n_s = int(dif_s.sum())
            ulp_ok = True
            if n_s:
                kb, fb = ks[dif_s].view(torch.int32), fs[dif_s].view(torch.int32)
                ulp_ok = bool(((kb - fb).abs() == 0x10000).all())
                fronteira = T.bf16_up(ft[dif_s] * (1 - 1e-5)) != T.bf16_up(ft[dif_s] * (1 + 1e-5))
                ulp_ok = ulp_ok and bool(fronteira.all())
            kq = T.unpack(r_q[:M], K).to(torch.int16)
            igual_g = (~dif_s).t().repeat_interleave(G, 1)               # (M, K): grupos de escala igual
            dq = (kq - fq).abs()[igual_g]
            n_q = int((dq > 0).sum())
            sat1 = float((T.unpack(a_q[:M], K).abs() == 7).float().mean())
            sat9 = float((kq.abs() == 7).float().mean())
            passa = (ident and pad_ok and n_s <= 1e-3 * ks.numel() and ulp_ok and n_q <= 1e-3 * dq.numel()
                     and int(dq.max()) <= 1 and sat9 > sat1)
            ok &= passa
            print(f"M={M} K={K} swiglu={swiglu} G={G}: ratio 1 identico={ident}; ratio 0,9: escalas dif {n_s}/{ks.numel()} "
                  f"(1 ulp na fronteira={ulp_ok}), codigos dif {n_q}/{dq.numel()} max {int(dq.max())}, padding={pad_ok}, "
                  f"saturados {100 * sat1:.2f} % -> {100 * sat9:.2f} %  {'ok' if passa else 'FALHA'}", flush=True)
    x = torch.randn(4096, 4096, device="cuda").to(torch.bfloat16)
    # emenda 05/10 (criterio_fase7.md): 9 rodadas intercaladas com ordem alternada, mediana -- a medida sequencial da
    # primeira execucao (88,1 contra 98,3 us) sofre deriva de relogio
    r1, r2 = [], []
    for r in range(9):
        par = [(r1, lambda: q1.quant(x, 128, False, 128)), (r2, lambda: q2.quant(x, 128, False, 128, 0.9))]
        for lista, fn in (par if r % 2 == 0 else par[::-1]):
            lista.append(T.timeit(fn))
    t1, t2 = sorted(r1)[4], sorted(r2)[4]
    t_ok = abs(t2 / t1 - 1) <= 0.05
    ok &= t_ok
    print(f"tempo 4096x4096 g128: g64_quant {t1 * 1000:.1f} us, g64_quant2(0,9) {t2 * 1000:.1f} us {'ok' if t_ok else 'FALHA'}")
    print("RESULTADO:", "PASSA" if ok else "FALHA", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
