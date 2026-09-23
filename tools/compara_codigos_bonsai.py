"""O QAT está andando na direção do Bonsai? Compara CÓDIGOS ternários, camada a camada, só pesos (CPU).

POR QUE CÓDIGO E NÃO DISTÂNCIA. A engenharia reversa (`bench/bonsai_image_engenharia_reversa.md`)
mediu que o Bonsai fica MAIS LONGE do BF16 original que um PTQ ingênuo (rel-L2 0,535 contra 0,466)
e mesmo assim renderiza igual ao BF16. Então "distância ao BF16" não é o alvo: o treino dele SE
AFASTA do ingênuo numa direção específica. O que diz se um QAT vai para o mesmo lugar é:

    mudou_vs_ingenuo    fração de posições cujo código difere do PTQ ingênuo (o quanto o corpo andou)
    acerto_da_mudanca   das posições que o QAT mudou, fração em que o código novo É o do Bonsai
    recall_bonsai       das posições que o BONSAI mudou vs o ingênuo, fração que o QAT também mudou
                        para o mesmo código
    concorda_bonsai     fração de posições com o mesmo código do Bonsai (o ingênuo dá a linha de base)
    inversao_sinal      entre não-nulos, fração com sinal oposto ao do BF16 (Bonsai: 0,116% no total)
    rel_l2_bf16         ||x - bf16|| / ||bf16|| no valor desempacotado

Código = sinal do peso desempacotado, 0 onde é exatamente 0 (os arquivos comparados são ternário
desempacotado: um valor por grupo de 128, zeros exatos — conferido na engenharia reversa).

LÊ POR OFFSET, sem `safe_open`: nesta máquina `safe_open` cobra 2x o arquivo em commit, e aqui há
cinco arquivos de 7,75 GB abertos. Nomes: todos em nomenclatura diffusers.

NÃO COBRE: nenhuma ativação, nenhuma imagem. Concordar com o Bonsai não é obrigatório — dois
treinos podem chegar a mínimos diferentes igualmente bons; é um sinal de direção, não um critério.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import torch

BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class Leitor:
    def __init__(self, p: Path):
        self.p = Path(p)
        with self.p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
            self.base = 8 + n
        self.h.pop("__metadata__", None)

    def get(self, k: str) -> torch.Tensor:
        m = self.h[k]
        a, b = m["data_offsets"]
        with self.p.open("rb") as f:
            f.seek(self.base + a)
            buf = bytearray(f.read(b - a))
        return torch.frombuffer(buf, dtype=DT[m["dtype"]]).reshape(m["shape"]).float()


def pilhas_reais(nomes) -> set[str]:
    ind: dict[str, set[str]] = {}
    for k in nomes:
        if (m := BLOCO.match(k)):
            ind.setdefault(m.group("pilha"), set()).add(m.group("i"))
    return {n for n, i in ind.items() if len(i) >= 2}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16", required=True)
    p.add_argument("--bonsai", required=True)
    p.add_argument("--ingenuo", required=True)
    p.add_argument("--braco", action="append", required=True, metavar="ROTULO=ARQUIVO")
    p.add_argument("--saida", type=Path)
    a = p.parse_args()

    bf, bo, ing = Leitor(a.bf16), Leitor(a.bonsai), Leitor(a.ingenuo)
    bracos = [(r, Leitor(f)) for r, _, f in (s.partition("=") for s in a.braco)]
    pil = pilhas_reais(bf.h)
    corpo = sorted(k for k, m in bf.h.items() if len(m["shape"]) == 2
                   and (b := BLOCO.match(k)) and b.group("pilha") in pil)
    for nome, L in [("bonsai", bo), ("ingenuo", ing), *bracos]:
        faltam = [k for k in corpo if k not in L.h]
        if faltam:
            print(f"RECUSADO: {nome} nao tem {len(faltam)} pesos de corpo, p.ex. {faltam[:2]}", file=sys.stderr)
            return 2
    print(f"{len(corpo)} pesos de corpo; bracos: {[r for r, _ in bracos]}\n", flush=True)

    rot = ["ingenuo", "bonsai", *[r for r, _ in bracos]]
    tot = {r: dict.fromkeys(["n", "nz", "inv", "conc", "mud", "mud_ok", "bmud", "bmud_ok",
                             "d2", "b2"], 0.0) for r in rot}
    por_camada = []
    for i, k in enumerate(corpo):
        w = bf.get(k)
        sw = torch.sign(w)
        cb = torch.sign(bo.get(k))
        ci = torch.sign(ing.get(k))
        bmudou = cb != ci
        linha = {"camada": k}
        for r, L in [("ingenuo", ing), ("bonsai", bo), *bracos]:
            x = L.get(k)
            c = torch.sign(x)
            nz = c != 0
            t = tot[r]
            t["n"] += c.numel()
            t["nz"] += int(nz.sum())
            t["inv"] += int((nz & (c != sw) & (sw != 0)).sum())
            t["conc"] += int((c == cb).sum())
            mud = c != ci
            t["mud"] += int(mud.sum())
            t["mud_ok"] += int((mud & (c == cb)).sum())
            t["bmud"] += int(bmudou.sum())
            t["bmud_ok"] += int((bmudou & (c == cb)).sum())
            t["d2"] += float(((x - w) ** 2).sum())
            t["b2"] += float((w ** 2).sum())
            linha[r] = {"concorda_bonsai": float((c == cb).float().mean()),
                        "mudou_vs_ingenuo": float(mud.float().mean()),
                        "recall_bonsai": float((bmudou & (c == cb)).sum() / bmudou.sum().clamp(min=1))}
        por_camada.append(linha)
        if i % 20 == 0:
            print(f"  {i}/{len(corpo)} {k}", flush=True)

    print(f"\n{'':>14}{'concorda_bonsai':>17}{'mudou_vs_ing':>14}{'acerto_mud':>12}{'recall_bonsai':>15}"
          f"{'inv_sinal':>11}{'rel_l2_bf16':>13}")
    res = {}
    for r in rot:
        t = tot[r]
        v = {"concorda_bonsai": t["conc"] / t["n"], "mudou_vs_ingenuo": t["mud"] / t["n"],
             "acerto_da_mudanca": t["mud_ok"] / max(t["mud"], 1),
             "recall_bonsai": t["bmud_ok"] / max(t["bmud"], 1),
             "inversao_sinal": t["inv"] / max(t["nz"], 1), "rel_l2_bf16": (t["d2"] / t["b2"]) ** 0.5}
        res[r] = v
        print(f"{r:>14}{v['concorda_bonsai']:>16.4%}{v['mudou_vs_ingenuo']:>13.4%}"
              f"{v['acerto_da_mudanca']:>12.2%}{v['recall_bonsai']:>14.2%}{v['inversao_sinal']:>10.4%}"
              f"{v['rel_l2_bf16']:>13.4f}")
    print("\nlinha de base: 'ingenuo' mudou 0% dele mesmo; 'bonsai' tem recall 100% por definicao.")
    base = tot["bonsai"]["mud"] / tot["bonsai"]["n"] / 2
    print(f"acaso para acerto_da_mudanca: ~{base:.1%} = (fracao que o Bonsai mudou) x 1/2. A primeira "
          "versao desta linha dizia ~50% -- errado: onde o Bonsai NAO mudou (a maioria), qualquer troca "
          "se afasta dele.")
    if a.saida:
        a.saida.write_text(json.dumps({"total": res, "por_camada": por_camada}, indent=1), encoding="utf-8")
    print("\n=== NAO COBERTO ===\n  So pesos. Concordar com o Bonsai e sinal de direcao, nao criterio: dois treinos")
    print("  podem chegar a minimos diferentes igualmente bons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
