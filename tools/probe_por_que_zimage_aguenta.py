"""Por que o Z-Image tolera W4A4 e o HunyuanVideo 1.5 nao?

A PERGUNTA. Em 2026-08-31 esta bancada mediu que o MESMO formato, o MESMO kernel e o MESMO
conversor produzem uma imagem boa no Z-Image (1,50x mais rapido por passo) e lixo no HunyuanVideo
(1,055x mais LENTO, saida destruida). A explicacao chata -- "o Z-Image quantiza menos" -- morreu na
direcao contraria: o Z-Image tem **97,7%** dos parametros em 4 bits e o Hunyuan **65,3%**.

A HIPOTESE QUE ESTE PROBE TESTA, e e a que o README ja acusa: o caminho de ativacao do ConvRot e
`quantize_signed_int4_rowwise(rotate(x))` -- **um absmax por token**, cobrindo TODOS os canais, 15
niveis. Um unico canal outlier fixa a escala do vetor inteiro. Se as ativacoes do Hunyuan tiverem
outliers piores que as do Z-Image, o mesmo formato destroi um e nao o outro, e nao ha mais misterio.

O QUE ISTO USA. As duas calibracoes de 2026-08-19, que ja existiam e nunca foram comparadas: 128
linhas de ativacao REAL por camada, capturadas com forward-pre-hook durante amostragem de verdade.
Nada e recapturado aqui, e nao ha GPU envolvida.

    python_embeded\\python.exe -s tools/probe_por_que_zimage_aguenta.py

EIXOS QUE ESTE PROBE NAO SEGURA, e sao reais: as duas calibracoes foram tiradas com passos (8 e 6),
resolucao (1024 e 512) e numero de execucoes (4 e 2) diferentes. A FORMA da distribuicao de
ativacao e menos sensivel a isso do que um tempo seria, mas nao e insensivel. Se o resultado sair
apertado, refazer com os dois eixos casados antes de acreditar.

NAO COBERTO: mede a ENTRADA de cada camada isolada, nunca o acumulo pelo residual. Nao explica
POR QUE uma familia de modelo teria ativacao mais comportada -- so mede se tem. E a rotacao aqui e
uma Hadamard normalizada implementada neste arquivo, nao a do kernel: se o kernel usar uma
permutacao ou um sinal diferente, os numeros mudam de escala, mas a COMPARACAO entre os dois
modelos (que e a pergunta) sobrevive, porque os dois passam pela mesma.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent.parent
CALIB = RAIZ / "calib"

MODELOS = [
    ("Z-Image", CALIB / "xfer_beyond-reality-zimage-v2_native.calib.pt", "BOA"),
    ("HunyuanVideo 1.5", CALIB / "xfer_hunyuan15_v2.calib.pt", "DESTRUIDA"),
]


def hadamard(x: torch.Tensor, grupo: int) -> torch.Tensor:
    """Walsh-Hadamard normalizada, em grupos de `grupo` colunas.

    E o que a rotacao do ConvRot faz conceitualmente: espalha a energia de um canal por todos os
    outros do grupo, para que nenhum canal sozinho fixe a escala. `grupo` tem de ser potencia de 2.
    """
    assert grupo & (grupo - 1) == 0, f"grupo precisa ser potencia de 2: {grupo}"
    n, c = x.shape
    if c % grupo:
        grupo = 1 << int(math.log2(c & -c))          # maior potencia de 2 que divide c
    y = x.reshape(n, c // grupo, grupo).clone()
    passo = 1
    while passo < grupo:
        y = y.reshape(n, c // grupo, grupo // (2 * passo), 2, passo)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        passo *= 2
    return y.reshape(n, c) / math.sqrt(grupo)


def q_rowwise(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Uma escala absmax por LINHA (token), niveis uniformes. E o que o ConvRot faz na ativacao."""
    niveis = 2 ** (bits - 1) - 1
    escala = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / niveis
    codigo = torch.round(x / escala).clamp(-niveis - 1, niveis)
    return codigo, escala


def q_grupo(x: torch.Tensor, bits: int, grupo: int) -> tuple[torch.Tensor, torch.Tensor]:
    niveis = 2 ** (bits - 1) - 1
    n, c = x.shape
    g = grupo if c % grupo == 0 else 1
    v = x.reshape(n, c // g, g)
    escala = v.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / niveis
    codigo = torch.round(v / escala).clamp(-niveis - 1, niveis)
    return codigo.reshape(n, c), escala


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def entropia(codigo: torch.Tensor, bits: int) -> float:
    niveis = 2 ** bits
    h = torch.histc(codigo.float(), bins=niveis, min=-(2 ** (bits - 1)), max=2 ** (bits - 1) - 1)
    p = h / h.sum().clamp_min(1)
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def analisa(caminho: Path, cg: int, max_camadas: int) -> dict:
    d = torch.load(caminho, map_location="cpu", weights_only=False)
    meta, camadas = d["meta"], d["layers"]
    nomes = list(camadas)[:max_camadas] if max_camadas else list(camadas)

    acc = {k: [] for k in ("crest_rot", "crest_cru", "outlier", "zeros", "bits",
                           "int4_row", "int8_row", "int4_g16")}
    for nome in nomes:
        e = camadas[nome]
        x = e["sample"].float()
        if x.ndim != 2 or x.shape[0] < 8:
            continue
        xr = hadamard(x, cg)

        # crest por token, antes e depois da rotacao: e o numero que a rotacao existe para baixar
        for chave, v in (("crest_cru", x), ("crest_rot", xr)):
            rms = v.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
            acc[chave].append(float((v.abs().amax(dim=1) / rms).median()))

        # razao entre o pior canal e o canal mediano: a causa direta do crest por token
        ca = e["channel_absmax"].float()
        acc["outlier"].append(float(ca.max() / ca.median().clamp_min(1e-12)))

        c4, s4 = q_rowwise(xr, 4)
        acc["zeros"].append(float((c4 == 0).float().mean()))
        acc["bits"].append(entropia(c4, 4))
        acc["int4_row"].append(rel(c4 * s4, xr))

        c8, s8 = q_rowwise(xr, 8)
        acc["int8_row"].append(rel(c8 * s8, xr))

        cg16, sg16 = q_grupo(xr, 4, 16)
        n, c = xr.shape
        g = 16 if c % 16 == 0 else 1
        acc["int4_g16"].append(rel((cg16.reshape(n, c // g, g) * sg16).reshape(n, c), xr))

    def med(k):
        v = sorted(acc[k])
        return v[len(v) // 2] if v else float("nan")

    return {"meta": meta, "n": len(acc["crest_rot"]),
            **{k: med(k) for k in acc}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--convrot-groupsize", type=int, default=256)
    p.add_argument("--max-camadas", type=int, default=0, help="0 = todas")
    a = p.parse_args()

    print("=" * 84)
    print("Por que o mesmo W4A4 destroi um modelo e nao o outro?")
    print("Ativacoes REAIS, capturadas em 2026-08-19 durante amostragem de verdade. Sem GPU.")
    print("=" * 84)

    res = []
    for rotulo, caminho, desfecho in MODELOS:
        if not caminho.is_file():
            print(f"{rotulo}: calibracao ausente ({caminho.name})")
            continue
        r = analisa(caminho, a.convrot_groupsize, a.max_camadas)
        r["rotulo"], r["desfecho"] = rotulo, desfecho
        res.append(r)
        m = r["meta"]
        print(f"\n{rotulo}  ({desfecho})")
        print(f"  {r['n']} camadas   perfil={m['profile']}  passos={m['steps']}  "
              f"lado={m['size']}  execucoes={m['runs']}")

    if len(res) < 2:
        return 1

    print("\n" + "=" * 84)
    print(f"{'mediana sobre as camadas':38} {res[0]['rotulo']:>16} {res[1]['rotulo']:>18}   razao")
    print("-" * 84)
    linhas = [
        ("crest por token, SEM rotacao", "crest_cru", "maior = pior"),
        ("crest por token, COM rotacao", "crest_rot", "e isto que a rotacao deveria baixar"),
        ("pior canal / canal mediano", "outlier", "maior = pior"),
        ("fracao de ativacoes no codigo 0", "zeros", "maior = pior"),
        ("bits efetivos dos 3,907 pagos", "bits", "MENOR = pior"),
        ("erro int4 uma escala por token", "int4_row", "o que o ConvRot faz"),
        ("erro int8 uma escala por token", "int8_row", "o que o W4A8 faz"),
        ("erro int4 escala por grupo de 16", "int4_g16", "controle"),
    ]
    for rotulo, chave, nota in linhas:
        a_, b_ = res[0][chave], res[1][chave]
        if min(abs(a_), abs(b_)) > 0:
            r_ = max(a_, b_) / min(a_, b_)
            quem = res[1]["rotulo"] if b_ > a_ else res[0]["rotulo"]
            razao = f"{r_:5.2f}x maior no {quem.split()[0]}"
        else:
            razao = ""
        print(f"{rotulo:38} {a_:16.4f} {b_:18.4f}   {razao}")
        print(f"{'':38} {nota}")

    print("\n" + "=" * 84)
    print("EIXOS NAO SEGURADOS: as duas calibracoes tem passos, resolucao e numero de execucoes")
    print("diferentes (8/1024/4 contra 6/512/2). A forma da distribuicao e menos sensivel a isso")
    print("que um tempo seria, mas se a diferenca sair apertada, refazer casando os eixos antes de")
    print("acreditar.")
    print("NAO COBERTO: mede a entrada de cada camada isolada, nunca o acumulo pelo residual. A")
    print("rotacao aqui e uma Hadamard normalizada deste arquivo, nao a do kernel -- a escala dos")
    print("numeros pode diferir, a COMPARACAO entre os dois modelos nao, porque ambos passam pela")
    print("mesma. E nao explica por que uma familia teria ativacao mais comportada: so mede se tem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
