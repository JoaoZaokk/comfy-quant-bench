r"""Quanto custa podar em PARES em vez de em valores, e quanto custa depois ir a 4 bits.

POR QUE A PERGUNTA EXISTE
-------------------------
Toda tabela de erro de esparsidade desta bancada usa 2:4 **por elemento**: de cada 4 pesos, ficam os
2 melhores. Medido em 2026-09-01, o tensor core esparso INT4 da sm_80 nao aceita isso -- a unidade
de mascara dele e o PAR: de cada 8 pesos (4 pares), ficam 2 pares. Descoberto testando as duas
granularidades contra a referencia inteira exata (por elemento falha 6/6, por par bate 6/6), e nao
lido em documentacao.

Isso importa porque o INT4 esparso e o formato mais rapido que esta bancada ja mediu -- 5,0x a 7,6x
sobre o denso bf16, a 2,5 bits/peso -- e as tabelas de erro que existiam **nao descrevem o que ele
faz**. Usar o 0,0794 do 2:4 por elemento para julgar o int4 esparso seria comparar duas restricoes
diferentes com o mesmo numero.

O DESENHO VARIA UM EIXO POR VEZ
--------------------------------
    W4A4                         o que a bancada entrega hoje, o ponto de referencia
    2:4 por ELEMENTO, bf16       o que as tabelas antigas mediram
    2:4 por PAR,      bf16       so a granularidade muda  <- isola o custo da restricao
    2:4 por PAR,      int4       + a largura                <- o que o kernel de fato roda

Sem a terceira linha, a diferenca entre a segunda e a quarta misturaria granularidade com largura, e
esta bancada ja registrou uma conclusao inteira nascida exatamente assim.

NAO COBERTO
-----------
Erro de SAIDA na ativacao calibrada real, **nao imagem** -- e esta bancada ja mediu que nenhum corte
nesse eixo separa usavel de inutilizavel. Um modelo, uma calibragem. Sem treino de recuperacao.

E o buraco que mais importa: o int4 destas linhas e **simetrico por linha, NAO e ConvRot**. A rotacao
e justamente o que faz o W4A4 chegar a 0,0956; sem ela a comparacao de erro pune o esparso por uma
razao que nao e a esparsidade. **ConvRot + poda por par + int4 nao foi medido**, nao tem obstaculo
conhecido, e e o proximo passo obvio.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import struct
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

import comfy_kitchen as ck
import torch

CG = 256


def ler_peso(modelo: Path, chave: str) -> torch.Tensor:
    with modelo.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        info = head[chave + ".weight"]
        a, b = info["data_offsets"]
        f.seek(base + a)
        cru = bytearray(f.read(b - a))
    return torch.frombuffer(cru, dtype=torch.bfloat16).reshape(info["shape"]).cuda()


def poda_elemento(w: torch.Tensor) -> torch.Tensor:
    """2 dos 4 valores de cada grupo de 4. O que as tabelas antigas mediram."""
    n, k = w.shape
    g = w.abs().reshape(n, k // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    m = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
    return w.reshape(n, k // 4, 4).mul(m).reshape(n, k)


def poda_par(w: torch.Tensor, norma: torch.Tensor | None = None) -> torch.Tensor:
    """2 dos 4 PARES de cada grupo de 8. O que o tensor core INT4 exige.

    Com `norma` (a norma L2 por coluna da ativacao real) o criterio vira Wanda estendido ao par:
    `sum_{j no par} |W_ij| * ||X_j||`. Sem ela e magnitude pura. As duas variantes existem porque
    trocar criterio e granularidade ao mesmo tempo mediria dois eixos de uma vez.
    """
    n, k = w.shape
    peso = w.abs() if norma is None else w.abs() * norma.unsqueeze(0)
    escore = peso.reshape(n, k // 8, 4, 2).sum(-1)
    idx = escore.argsort(dim=-1)[..., :2]
    m = torch.ones_like(escore, dtype=torch.bool).scatter_(-1, idx, False)
    return (w.reshape(n, k // 8, 4, 2) * m.unsqueeze(-1)).reshape(n, k)


def para_int4(w: torch.Tensor) -> torch.Tensor:
    """Quantiza simetrico por linha para [-8,7] e devolve o valor RECONSTRUIDO, em float.

    Devolver o reconstruido e nao o inteiro e o que torna esta linha comparavel as outras: todas
    medem erro de saida contra a mesma referencia float32, entao a quantizacao tem de aparecer como
    perturbacao do peso e nao como uma mudanca de unidade.
    """
    escala = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7
    return (w / escala).round().clamp(-8, 7) * escala


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", type=Path,
                   default=RAIZ / "ComfyUI/models/diffusion_models/"
                                  "beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--calib", type=Path, default=RAIZ / "calib/zimage_v2_sigma.calib.pt")
    p.add_argument("--camadas", type=int, default=24)
    a = p.parse_args()

    d = torch.load(a.calib, map_location="cpu", weights_only=False)
    chaves = [k for k in d["layers"] if k.startswith("layers.")][: a.camadas]

    print(f"{'camada':30s} {'W4A4':>7s} {'2:4elem':>8s} {'2:4par':>7s} {'par+int4':>9s} "
          f"{'parWanda':>9s} {'Wanda+int4':>11s}")
    print("-" * 92)
    acc = {"q": [], "e": [], "p": [], "i": [], "w": [], "wi": []}
    for k in chaves:
        x = d["layers"][k]["sample"].cuda().to(torch.bfloat16)
        w = ler_peso(a.modelo, k)
        if x.shape[-1] != w.shape[1] or w.shape[1] % 8:
            continue
        ref = x.float() @ w.float().T
        nrm = ref.norm()

        def rel(y):
            return ((y - ref).norm() / nrm).item()

        wf = w.float()
        norma = x.float().reshape(-1, x.shape[-1]).norm(dim=0)
        pw = poda_par(wf, norma)
        qw, qs = ck.quantize_convrot_w4a4_weight(w.contiguous(), CG, 64)
        r = {"q": rel(ck.convrot_w4a4_linear(x, qw, qs, None, CG).float()),
             "e": rel(x.float() @ poda_elemento(wf).T),
             "p": rel(x.float() @ poda_par(wf).T),
             "i": rel(x.float() @ para_int4(poda_par(wf)).T),
             "w": rel(x.float() @ pw.T),
             "wi": rel(x.float() @ para_int4(pw).T)}
        for kk, v in r.items():
            acc[kk].append(v)
        print(f"{k:30s} {r['q']:7.4f} {r['e']:8.4f} {r['p']:7.4f} {r['i']:9.4f} "
              f"{r['w']:9.4f} {r['wi']:11.4f}")

    if not acc["q"]:
        print("nenhuma camada medida", file=sys.stderr)
        return 2

    med = {kk: st.median(v) for kk, v in acc.items()}
    print()
    print(f"{'MEDIANA':30s} {med['q']:7.4f} {med['e']:8.4f} {med['p']:7.4f} {med['i']:9.4f} "
          f"{med['w']:9.4f} {med['wi']:11.4f}")
    print()
    print(f"{'formato':30s} {'bits/peso':>9s} {'erro':>8s} {'contra o W4A4':>26s} "
          f"{'velocidade vs denso bf16':>26s}")
    print("-" * 104)
    # As velocidades vem de tools/sparse24_sm86/, medidas na 3090; a coluna existe para que a
    # troca fique visivel numa linha so, e cada numero carrega de onde veio.
    linhas = [("W4A4 ConvRot (hoje)", 4.0, med["q"], "1,83x-1,93x, medido antes"),
              ("2:4 por elemento, bf16", 9.0, med["e"], "1,7x-1,95x, medido"),
              ("2:4 por PAR, bf16", 9.0, med["p"], "(sem kernel: o bf16 esparso e por elemento)"),
              ("2:4 por PAR + int4", 2.5, med["i"], "5,0x-7,6x, medido"),
              ("2:4 por PAR, Wanda + int4", 2.5, med["wi"], "5,0x-7,6x, medido")]
    for nome, bits, erro, vel in linhas:
        r = erro / med["q"]
        # Razao abaixo de 1 vai invertida e com a direcao no nome.
        txt = f"{1/r:.2f}x MAIS FIEL" if r <= 1.0 else f"{r:.2f}x menos fiel"
        print(f"{nome:30s} {bits:9.1f} {erro:8.4f} {txt:>26s} {vel:>26s}")

    print()
    custo = med["p"] / med["e"]
    print(f"custo isolado da GRANULARIDADE (par contra elemento, mesmo bf16): {custo:.2f}x de erro")
    print(f"custo isolado da LARGURA (int4 sobre par, mesma granularidade):   "
          f"{med['i']/med['p']:.2f}x de erro")
    g = med["p"] / med["w"]
    print(f"ganho do criterio WANDA sobre magnitude, na granularidade de par:  {g:.2f}x mais fiel")
    print()
    print("NAO COBERTO: erro de saida na ativacao calibrada, NAO imagem -- e esta bancada ja mediu")
    print("  que nenhum corte nesse eixo separa usavel de inutilizavel. Um modelo, uma calibragem,")
    print("  sem treino de recuperacao. O int4 aqui e SIMETRICO POR LINHA e NAO e ConvRot -- e a")
    print("  rotacao e o que faz o W4A4 chegar a 0,0956, entao esta coluna pune o esparso por uma")
    print("  razao que nao e a esparsidade. ConvRot + poda por par + int4 NAO foi medido e nao tem")
    print("  obstaculo conhecido. As velocidades da ultima coluna vem de tools/sparse24_sm86/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
