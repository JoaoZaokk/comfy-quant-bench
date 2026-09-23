"""ONDE o Bonsai mudou os códigos em relação ao ternário ingênuo, e o que o BF16 original tinha ali.

Pergunta do dono (2026-09-23): "o que o original tinha nas camadas que o bonsai fez e depois
corrigiu?". Três tipos de mudança contra o ingênuo (absmean g128 no eixo K: código ≠ 0 sse
|w| > 0,5·d, d = média|w| do grupo):

    promovido   ingênuo 0  -> ±1     um peso "pequeno" que o treino julgou importante
    rebaixado   ingênuo ±1 -> 0      um peso "grande" que o treino desligou
    invertido   ingênuo ±1 -> ∓1     sinal trocado (passa pelo zero duas vezes)

Para cada tipo, a distribuição de r = |w|/d do PESO ORIGINAL (o limiar do ingênuo é r = 0,5), e a
fração de mudanças por tipo de camada e por profundidade. O mesmo para cada braço de QAT, para
comparar a ASSINATURA das mudanças, não só a quantidade.

Lê por offset (sem safe_open, ver compara_codigos_bonsai.py). Só CPU.

NÃO COBRE: ativações — um peso perto do limiar pode ser irrelevante se a entrada daquele canal é
pequena; isto é só geometria do peso.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compara_codigos_bonsai import BLOCO, Leitor, pilhas_reais  # noqa: E402

QS = (0.05, 0.25, 0.5, 0.75, 0.95)


def tipo_camada(k: str) -> str:
    pilha = BLOCO.match(k).group("pilha")
    resto = k.split(".", 2)[2].rsplit(".weight", 1)[0]
    return f"{'single' if pilha.startswith('single') else 'double'}:{resto}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16", required=True)
    p.add_argument("--bonsai", required=True)
    p.add_argument("--braco", action="append", default=[], metavar="ROTULO=ARQUIVO")
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--amostra-r", type=int, default=100_000,
                   help="r guardados por tipo e camada (amostra fixa). 100 camadas x 100k = 10M, abaixo do teto de 16M do torch.quantile")
    p.add_argument("--saida", type=Path)
    a = p.parse_args()

    bf = Leitor(a.bf16)
    bracos = [("bonsai", Leitor(a.bonsai))] + [(r, Leitor(f)) for r, _, f in
                                                 (s.partition("=") for s in a.braco)]
    pil = pilhas_reais(bf.h)
    corpo = sorted(k for k, m in bf.h.items() if len(m["shape"]) == 2
                   and (b := BLOCO.match(k)) and b.group("pilha") in pil)
    g = torch.Generator().manual_seed(20260923)
    tipos = ("promovido", "rebaixado", "invertido")
    cont = {r: {t: 0 for t in tipos} | {"n": 0} for r, _ in bracos}
    amostras = {r: {t: [] for t in tipos} for r, _ in bracos}
    base_r = []
    por_tipo_camada = {r: {} for r, _ in bracos}
    por_prof = {r: {} for r, _ in bracos}
    for i, k in enumerate(corpo):
        w = bf.get(k)
        n, kk = w.shape
        gw = w.reshape(n, kk // a.grupo, a.grupo)
        d = gw.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
        rr = (gw.abs() / d).reshape(n, kk)
        ci = (w.reshape(n, kk // a.grupo, a.grupo) / d).clamp(-1, 1).round().reshape(n, kk)
        idx = torch.randint(0, rr.numel(), (a.amostra_r,), generator=g)
        base_r.append(rr.flatten()[idx])
        tc, prof = tipo_camada(k), int(BLOCO.match(k).group("i"))
        pilha = BLOCO.match(k).group("pilha")
        for r, L in bracos:
            c = torch.sign(L.get(k))
            m = {"promovido": (ci == 0) & (c != 0), "rebaixado": (ci != 0) & (c == 0),
                 "invertido": (ci != 0) & (c != 0) & (c != ci)}
            cont[r]["n"] += c.numel()
            ptc = por_tipo_camada[r].setdefault(tc, {"n": 0, "mud": 0})
            pp = por_prof[r].setdefault(f"{pilha}.{prof}", {"n": 0, "mud": 0})
            ptc["n"] += c.numel(); pp["n"] += c.numel()
            for t in tipos:
                s = int(m[t].sum())
                cont[r][t] += s
                ptc["mud"] += s; pp["mud"] += s
                if s:
                    v = rr[m[t]]
                    if v.numel() > a.amostra_r:
                        v = v[torch.randint(0, v.numel(), (a.amostra_r,), generator=g)]
                    amostras[r][t].append(v)
        if i % 20 == 0:
            print(f"  {i}/{len(corpo)} {k}", flush=True)

    base = torch.cat(base_r)
    print("\nr = |w|/d do peso ORIGINAL (limiar do ingenuo: 0,5). Todos os pesos do corpo: quantis "
          + " ".join(f"{q:.0%}={float(base.quantile(q)):.3f}" for q in QS))
    res = {"quantis_todos": {str(q): float(base.quantile(q)) for q in QS}, "bracos": {}}
    for r, _ in bracos:
        n = cont[r]["n"]
        print(f"\n== {r} ==")
        res["bracos"][r] = {}
        for t in tipos:
            v = torch.cat(amostras[r][t]) if amostras[r][t] else torch.empty(0)
            qs = {str(q): float(v.quantile(q)) for q in QS} if v.numel() else {}
            perto = float(((v > 0.35) & (v < 0.65)).float().mean()) if v.numel() else float("nan")
            res["bracos"][r][t] = {"fracao": cont[r][t] / n, "quantis_r": qs, "perto_limiar_0_35_0_65": perto}
            print(f"  {t:>10}: {cont[r][t] / n:8.4%} das posicoes | r original: "
                  + " ".join(f"{q}={x:.3f}" for q, x in qs.items())
                  + f" | com r em [0,35; 0,65]: {perto:.1%}")
        top = sorted(por_tipo_camada[r].items(), key=lambda kv: -kv[1]["mud"] / kv[1]["n"])
        print("  tipo de camada (fracao mudada):", ", ".join(f"{k} {v['mud'] / v['n']:.2%}" for k, v in top))
        dup = [(k, v["mud"] / v["n"]) for k, v in por_prof[r].items()]
        dbl = sorted([x for x in dup if x[0].startswith("transformer_blocks")], key=lambda x: int(x[0].rsplit(".", 1)[1]))
        sgl = sorted([x for x in dup if x[0].startswith("single")], key=lambda x: int(x[0].rsplit(".", 1)[1]))
        print("  por profundidade, double:", " ".join(f"{v:.2%}" for _, v in dbl))
        print("  por profundidade, single:", " ".join(f"{v:.2%}" for _, v in sgl))
        res["bracos"][r]["por_tipo_camada"] = {k: v["mud"] / v["n"] for k, v in top}
        res["bracos"][r]["prof_double"] = [v for _, v in dbl]
        res["bracos"][r]["prof_single"] = [v for _, v in sgl]
    if a.saida:
        a.saida.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n=== NAO COBERTO ===\n  So geometria do peso; nenhuma ativacao. Quantis de amostra fixa (semente 20260923).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
