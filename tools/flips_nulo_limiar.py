"""Acaso casado por limiar para os flips ternários, e interseção de flips entre braços. Só CPU. (EXECUTADO)

`compara_codigos_bonsai.py` mede `acerto_da_mudanca` contra um acaso UNIFORME (~5,4%). Mas os flips de
um QAT curto e os do Bonsai se concentram perto do limiar r = |w|/d = 0,5 (d = absmean do grupo de 128
no BF16), então o acaso honesto é outro: flips sorteados nas MESMAS faixas de r. Aqui:

    faixa b      r em [0,02·b, 0,02·(b+1)), última faixa aberta (r >= 1,98)
    flip natural 0 -> sinal(w); ±1 -> 0   (o vizinho para onde r aponta)
    p_b          fração das posições da faixa em que o Bonsai fez o flip natural
    acerto_nulo  sum_b flips_braco(b) · p_b / flips_braco
    recall_nulo  sum_b flips_braco(b) · p_b / flips_bonsai
    inter_nula   sum_b flips_A(b) · flips_B(b) / N(b)      (flips independentes casados por faixa)

O nulo supõe flip natural; a fração natural de cada braço é impressa para conferir.
Lê por offset (Leitor de compara_codigos_bonsai), sem safe_open. Serve para BFL e diffusers: os grupos
são no eixo K e as fusões do mapa só concatenam linhas.

NÃO COBRE: nulo por camada (a faixa de r é a única variável casada); nenhuma ativação, nenhuma imagem.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from compara_codigos_bonsai import BLOCO, Leitor, pilhas_reais  # noqa: E402

G, NB = 128, 100


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bf16", required=True)
    p.add_argument("--ingenuo", required=True)
    p.add_argument("--bonsai", required=True)
    p.add_argument("--braco", action="append", required=True, metavar="ROTULO=ARQUIVO")
    p.add_argument("--pares", nargs="*", default=[], metavar="A:B")
    p.add_argument("--saida", type=Path, required=True)
    a = p.parse_args()

    bf, ing, bo = Leitor(a.bf16), Leitor(a.ingenuo), Leitor(a.bonsai)
    bracos = [(r, Leitor(f)) for r, _, f in (s.partition("=") for s in a.braco)]
    pil = pilhas_reais(bf.h)
    corpo = sorted(k for k, m in bf.h.items() if len(m["shape"]) == 2
                   and (b := BLOCO.match(k)) and b.group("pilha") in pil)
    rot = [r for r, _ in bracos]
    pares = [tuple(s.split(":")) for s in a.pares] or list(combinations(rot, 2))
    z = lambda: torch.zeros(NB, dtype=torch.float64)  # noqa: E731
    N, M, BM = z(), z(), z()
    ch = {r: z() for r in rot}
    hit = {r: z() for r in rot}
    nat = dict.fromkeys(rot, 0.0)
    rec = dict.fromkeys(rot, 0.0)
    inter = {pq: z() for pq in pares}
    mesmo = dict.fromkeys(pares, 0.0)
    ptq_ok = [0, 0]
    print(f"{len(corpo)} pesos de corpo; bracos {rot}", flush=True)
    for i, k in enumerate(corpo):
        w = bf.get(k)
        n, kk = w.shape
        g = w.reshape(n, kk // G, G)
        d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
        r = (g.abs() / d).reshape(n, kk)
        c0 = (w / d.expand(-1, -1, G).reshape(n, kk)).clamp(-1, 1).round()
        ci = torch.sign(ing.get(k))
        ptq_ok[0] += int((ci == c0).sum()); ptq_ok[1] += c0.numel()
        c0 = ci  # o arquivo manda; a conferência acima diz se a receita bate
        alvo = torch.where(c0 == 0, torch.sign(w), torch.zeros_like(w))
        fx = (r / 0.02).long().clamp(max=NB - 1).reshape(-1)
        cb = torch.sign(bo.get(k))
        bmud = cb != c0
        N += torch.bincount(fx, minlength=NB).double()
        M += torch.bincount(fx[(cb == alvo).reshape(-1) & bmud.reshape(-1)], minlength=NB).double()
        BM += torch.bincount(fx[bmud.reshape(-1)], minlength=NB).double()
        mud = {}
        cs = {}
        for rr, L in bracos:
            c = torch.sign(L.get(k))
            m = c != c0
            mud[rr], cs[rr] = m, c
            ch[rr] += torch.bincount(fx[m.reshape(-1)], minlength=NB).double()
            hit[rr] += torch.bincount(fx[(m & (c == cb)).reshape(-1)], minlength=NB).double()
            nat[rr] += int((m & (c == alvo)).sum())
            rec[rr] += int((bmud & (c == cb)).sum())
        for pq in pares:
            x, y = pq
            both = mud[x] & mud[y]
            inter[pq] += torch.bincount(fx[both.reshape(-1)], minlength=NB).double()
            mesmo[pq] += int((both & (cs[x] == cs[y])).sum())
        if i % 20 == 0:
            print(f"  {i}/{len(corpo)} {k}", flush=True)

    pb = torch.where(N > 0, M / N.clamp(min=1), torch.zeros_like(N))
    res = {"ptq_receita_confere": ptq_ok[0] / ptq_ok[1], "bonsai_flips": float(BM.sum()),
           "bonsai_flips_naturais": float(M.sum()), "bracos": {}, "pares": {}}
    print(f"\nPTQ do arquivo = receita absmean em {100 * ptq_ok[0] / ptq_ok[1]:.4f}% das posicoes")
    print(f"Bonsai: {BM.sum():.0f} flips, {100 * M.sum() / BM.sum():.2f}% naturais\n")
    print(f"{'braco':>16} {'flips':>12} {'natural':>8} {'acerto':>8} {'nulo_lim':>9} {'razao':>6}"
          f" {'recall':>7} {'rec_nulo':>8} {'razao':>6}")
    for rr in rot:
        f = float(ch[rr].sum())
        ac, acn = float(hit[rr].sum()) / f, float((ch[rr] * pb).sum()) / f
        re, ren = rec[rr] / float(BM.sum()), float((ch[rr] * pb).sum()) / float(BM.sum())
        res["bracos"][rr] = {"flips": f, "natural": nat[rr] / f, "acerto": ac, "acerto_nulo": acn,
                             "recall": re, "recall_nulo": ren,
                             "flips_por_faixa": ch[rr].tolist(), "acerto_por_faixa": hit[rr].tolist()}
        print(f"{rr:>16} {f:12.0f} {100 * nat[rr] / f:7.2f}% {100 * ac:7.2f}% {100 * acn:8.2f}%"
              f" {ac / acn:6.2f} {100 * re:6.2f}% {100 * ren:7.2f}% {re / ren:6.2f}")
    print(f"\n{'par':>28} {'inter':>12} {'inter_nula':>11} {'razao':>6} {'jaccard':>8} {'mesmo_alvo':>10}")
    for pq in pares:
        x, y = pq
        it, itn = float(inter[pq].sum()), float((ch[x] * ch[y] / N.clamp(min=1)).sum())
        jac = it / (float(ch[x].sum()) + float(ch[y].sum()) - it)
        res["pares"][f"{x}:{y}"] = {"inter": it, "inter_nula": itn, "jaccard": jac,
                                    "frac_de_A": it / float(ch[x].sum()), "frac_de_B": it / float(ch[y].sum()),
                                    "mesmo_alvo": mesmo[pq] / max(it, 1)}
        print(f"{x + ' x ' + y:>28} {it:12.0f} {itn:11.0f} {it / itn:6.1f} {jac:8.4f} {100 * mesmo[pq] / max(it, 1):9.2f}%")
    print("\nfaixa de r   N            p_bonsai   (so faixas com flips)")
    for b in range(NB):
        if BM[b] > 0.001 * BM.sum():
            print(f"  {0.02 * b:4.2f}-{0.02 * (b + 1):4.2f} {N[b]:12.0f} {100 * pb[b]:8.3f}%")
    res["faixas"] = {"N": N.tolist(), "p_bonsai": pb.tolist(), "bonsai_flips": BM.tolist()}
    a.saida.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n=== NAO COBERTO ===\n  Nulo casado so por faixa de r (nao por camada). Nenhuma ativacao, nenhuma imagem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
