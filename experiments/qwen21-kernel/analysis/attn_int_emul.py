"""Fase 7, pergunta do dono (05/10): (1) INT8 no P.V em vez de FP16/FP8 -- qual a perda?  (2) INT4 no Q.K com microescala ao
longo da dimensao da cabeca (o que o FP4 do Sage3/NVFP4 faz, so que com a grade inteira que o sm86 tem) -- e o proprio FP4
E2M1 como referencia. Emulacao (fake-quant) em fp32 na CPU, no dump real do Qwen-Image-2.1 1024^2 (otimizacao/
vdiag_qwen_1024_dump_52.pt: q 4096 x 32 x 128, k/v 4124 x 32 x 128); erro relativo da saida da atencao contra a atencao
exata em fp32 (mesma metrica do attn_qk_int4_emul.py da fase 3, onde o INT8 per_thread do Sage deu 0,0461 neste caso).

  QK (P.V exato):  int8/int4 por token (escala por linha nas 128 dimensoes), int4 em microescala de 64/32/16 canais,
                   FP4 E2M1 em blocos de 16 com escala E4M3 (NVFP4) e de 32 com escala potencia de 2 (MXFP4); smooth_k sempre
  P.V (Q.K exato): P em u8 com escala fixa 1/255 (P = exp(s - max) <= 1) ou por bloco de 64 chaves; V em int8 por canal ou
                   por bloco de 64 tokens; P em fp16 como o Sage hoje (V fp16)
  combinado:       int8 por token no QK + o melhor P.V int8

    python -s attn_int_emul.py [--threads N]
"""
import json
import math
import os
import sys
import time

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(AQUI, "..", "otimizacao", "vdiag_qwen_1024_dump_52.pt")
OUT = os.path.join(AQUI, "attn_int_emul.json")


def q_int(x, bits, bloco):
    """fake-quant simetrico int (+-(2^(bits-1)-1)) por linha, em blocos de `bloco` canais (None = linha inteira)."""
    qmax = 2 ** (bits - 1) - 1
    n, d = x.shape
    b = bloco or d
    xb = x.reshape(n, d // b, b)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.clamp(torch.round(xb / s), -qmax, qmax) * s).reshape(n, d)


def _e2m1(y):
    """arredonda |y| <= 6 para a grade E2M1 {0, .5, 1, 1.5, 2, 3, 4, 6} (com sinal)."""
    a = y.abs().clamp(max=6.0)
    r = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2))
    return torch.sign(y) * r


def _e4m3(s):
    """arredonda escala positiva para FP8 E4M3 (3 bits de mantissa)."""
    m, e = torch.frexp(s)                      # s = m * 2^e, m em [0,5, 1)
    return torch.ldexp(torch.round(m * 16) / 16, e)


def q_fp4(x, bloco, escala):
    n, d = x.shape
    xb = x.reshape(n, d // bloco, bloco)
    am = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    if escala == "e4m3":                       # NVFP4: escala = amax/6 em E4M3
        s = _e4m3(am / 6)
    else:                                      # MXFP4 (OCP): expoente compartilhado floor(log2(amax)) - 2
        s = torch.exp2(torch.floor(torch.log2(am)) - 2)
    return (_e2m1(xb / s) * s).reshape(n, d)


def q_p_u8(p, modo):
    """P = exp(s - max) em [0, 1]: u8 com escala fixa 1/255 ou por bloco de 64 chaves (escala = max do bloco / 255)."""
    if modo == "fixo":
        return torch.round(p * 255) / 255
    n, m = p.shape
    pad = (-m) % 64
    pp = torch.nn.functional.pad(p, (0, pad)).reshape(n, -1, 64)
    s = pp.amax(-1, keepdim=True).clamp_min(1e-30) / 255
    return (torch.round(pp / s) * s).reshape(n, -1)[:, :m]


def q_v_i8(v, modo):
    if modo == "canal":
        s = v.abs().amax(0, keepdim=True).clamp_min(1e-12) / 127
        return torch.clamp(torch.round(v / s), -127, 127) * s
    m, d = v.shape
    pad = (-m) % 64
    vv = torch.nn.functional.pad(v, (0, 0, 0, pad)).reshape(-1, 64, d)
    s = vv.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    return (torch.clamp(torch.round(vv / s), -127, 127) * s).reshape(-1, d)[:m]


QK = {
    "int8 por token": lambda x: q_int(x, 8, None),
    "int4 por token": lambda x: q_int(x, 4, None),
    "int4 micro 64": lambda x: q_int(x, 4, 64),
    "int4 micro 32": lambda x: q_int(x, 4, 32),
    "int4 micro 16": lambda x: q_int(x, 4, 16),
    "fp4 NVFP4 (16, E4M3)": lambda x: q_fp4(x, 16, "e4m3"),
    "fp4 MXFP4 (32, 2^k)": lambda x: q_fp4(x, 32, "mx"),
}
PV = {
    "P fp16, V fp16 (Sage hoje)": (lambda p: p.half().float(), lambda v: v.half().float()),
    "P u8 fixo, V fp16": (lambda p: q_p_u8(p, "fixo"), lambda v: v.half().float()),
    "P u8 bloco64, V fp16": (lambda p: q_p_u8(p, "bloco"), lambda v: v.half().float()),
    "P fp16, V int8 canal": (lambda p: p.half().float(), lambda v: q_v_i8(v, "canal")),
    "P u8 fixo, V int8 canal": (lambda p: q_p_u8(p, "fixo"), lambda v: q_v_i8(v, "canal")),
    "P u8 bloco64, V int8 canal": (lambda p: q_p_u8(p, "bloco"), lambda v: q_v_i8(v, "canal")),
    "P u8 bloco64, V int8 bloco64": (lambda p: q_p_u8(p, "bloco"), lambda v: q_v_i8(v, "bloco")),
}


def main():
    argv = sys.argv[1:]
    if "--threads" in argv:
        torch.set_num_threads(int(argv[argv.index("--threads") + 1]))
    t0 = time.time()
    d = torch.load(DUMP, map_location="cpu", weights_only=False)
    q, k, v = (d[n][0].float() for n in ("q", "k", "v"))        # (N, H, D)
    H, D = q.shape[1], q.shape[2]
    esc = 1 / math.sqrt(D)
    acc = {}                                                      # nome -> [soma |O - Oref|^2, soma |Oref|^2, [cos por cabeca]]
    for h in range(H):
        qh, kh, vh = q[:, h], k[:, h], v[:, h]
        kh_s = kh - kh.mean(0, keepdim=True)                      # smooth_k (nao muda o softmax)
        s = (qh @ kh.t()) * esc
        mx = s.amax(1, keepdim=True)
        p = torch.exp(s - mx)
        l = p.sum(1, keepdim=True)
        oref = (p @ vh) / l
        nref = float((oref ** 2).sum())

        def junta(nome, o):
            a = acc.setdefault(nome, [0.0, 0.0, []])
            a[0] += float(((o - oref) ** 2).sum()); a[1] += nref
            a[2].append(float(torch.nn.functional.cosine_similarity(o.flatten(), oref.flatten(), dim=0)))

        for nome, f in QK.items():
            sq = (f(qh) @ f(kh_s).t()) * esc
            pq = torch.exp(sq - sq.amax(1, keepdim=True))
            junta("QK " + nome, (pq @ vh) / pq.sum(1, keepdim=True))
        for nome, (fp, fv) in PV.items():
            junta("PV " + nome, (fp(p) @ fv(vh)) / l)
        sq = (q_int(qh, 8, None) @ q_int(kh_s, 8, None).t()) * esc
        pq = torch.exp(sq - sq.amax(1, keepdim=True)); lq = pq.sum(1, keepdim=True)
        junta("QK int8 por token + PV P fp16, V fp16", (pq.half().float() @ vh.half().float()) / lq)
        junta("QK int8 por token + PV P u8 bloco64, V int8 bloco64", (q_p_u8(pq, "bloco") @ q_v_i8(vh, "bloco")) / lq)
        if h % 8 == 7:
            print(f"[{h + 1}/{H} cabecas, {time.time() - t0:.0f} s]", flush=True)
    res = {n: {"erro_rel": math.sqrt(a[0] / a[1]), "cos_pior_cabeca": min(a[2]), "cos_medio": sum(a[2]) / len(a[2])}
           for n, a in acc.items()}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nerro relativo da saida da atencao contra fp32 exato (Qwen 2.1 1024^2, 32 cabecas); fase 3: INT8 per_thread do Sage = 0,0461")
    for n, r in res.items():
        print(f"  {n:58s} {r['erro_rel']:.4f}   cos pior cabeca {r['cos_pior_cabeca']:.5f}")


if __name__ == "__main__":
    main()
