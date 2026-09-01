"""Tres bracos de esparsidade 2:4, e uma promocao que so gasta onde o erro exige.

DESENHO, PEDIDO PELO DONO
-------------------------
    A  2:4 sem Wanda  + W4A4     criterio de poda por |W| puro
    B  2:4 com Wanda  + W4A4     criterio |W| * ||X_j||, sem treino nenhum
    C  B, promovendo a W8        so as camadas que "aceitam" -- as piores primeiro

O braco A e o CONTROLE que faz os outros valerem: se B e C sairem bem sem ele, nao da para saber se
o criterio de poda importou ou se o modelo simplesmente tolera 2:4. Ele nao esta ai por simetria.

A PROMOCAO E POR ERRO E POR BYTE, NAO POR NOME
-----------------------------------------------
Ordena as camadas por erro removido POR BYTE EXTRA e promove enquanto couber no orcamento. Promover
a camada de pior erro absoluto seria pior: uma camada gigante pode custar muitos bytes para pouco
ganho, e uma pequena e barata pode devolver quase tudo. E a mesma logica que o `quant_mixed` usa
entre W4A4 e W4A8, aplicada a outro eixo.

CONTABILIDADE DE TAMANHO, DECLARADA
------------------------------------
Bits por peso, contando o indice do 2:4 (2 bits por valor guardado, 2 valores por grupo de 4):

    W4A4              4 bits/peso
    2:4 + W4A4        (2 valores x 4 bits + 4 bits de indice) / 4 pesos = 3 bits/peso
    2:4 + W8          (2 valores x 8 bits + 4 bits de indice) / 4 pesos = 5 bits/peso

As escalas por linha sao ignoradas nos tres -- sao o mesmo termo em todos e nao mudam a comparacao.

NAO COBERTO
-----------
Erro de saida na ativacao calibrada, NAO imagem. Sem treino de recuperacao (a proposta original do
dono; Wanda e o atalho sem treino). Sem SparseGPT.

CORRECAO 2026-09-01: este bloco dizia **nada disto executa em tensor core esparso -- falta
cuSPARSELt nesta maquina**. As duas metades estavam erradas. Executa: `tools/sparse24_sm86/`
compila o SparseGemm do CUTLASS aqui e mede **1,7x a 1,95x mais rapido que o denso bf16** nestes
mesmos shapes. E o bloqueio nunca foi o cuSPARSELt -- o kernel do CUTLASS nao o usa. O que barrava
era um tile dimensionado para a A100: `sizeof(GemmKernel::SharedStorage)` da config que o xformers
distribui e **139.264 bytes** e esta placa aceita **101.376**. Mesmo tile com 2 estagios pede
69.632 e roda. Era config, nao biblioteca, e nao a placa.

O que continua verdade e limita esta tabela: os numeros abaixo sao erro de SAIDA na ativacao
calibrada, nao imagem, e o tempo medido e do 2:4 em bf16 -- **a composicao 2:4 + W4A4 destas linhas
nao tem kernel**, entao a coluna de bits/peso dela e uma conta de tamanho e nao uma medicao de
velocidade.
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

import torch  # noqa: E402
import comfy_kitchen as ck  # noqa: E402

CG = 256
BITS = {"w4": 4.0, "s4": 3.0, "s8": 5.0}


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


def poda24(w: torch.Tensor, escore: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    g = escore.reshape(n, k // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    m = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
    return w.reshape(n, k // 4, 4).mul(m).reshape(n, k)


def saida_w4(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    q, s = ck.quantize_convrot_w4a4_weight(w.contiguous(), CG, 64)
    return ck.convrot_w4a4_linear(x, q, s, None, CG).float()


def saida_w8(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """W8 com ativacao em alta precisao: peso int8 tensorwise, matmul em bf16.

    Nao usa `int8_linear` de proposito -- aquele caminho quantiza a ATIVACAO tambem, e o pedido era
    W8A16. Aqui o peso e quantizado de verdade pelo kernel e desfeito para o matmul, que e o que
    W8A16 significa: o erro vem do peso, nao da ativacao.
    """
    q, s = ck.quantize_int8_tensorwise(w.contiguous())
    return x.float() @ (q.float() * s.float().reshape(-1, *([1] * (q.dim() - 1)))).T


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", type=Path,
                   default=RAIZ / "ComfyUI/models/diffusion_models/beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--calib", type=Path, default=RAIZ / "calib/zimage_v2_sigma.calib.pt")
    p.add_argument("--camadas", type=int, default=40)
    p.add_argument("--orcamento-bits", type=float, default=3.5,
                   help="media de bits/peso permitida no braco C")
    a = p.parse_args()

    d = torch.load(a.calib, map_location="cpu", weights_only=False)
    chaves = [k for k in d["layers"] if k.startswith("layers.")][: a.camadas]

    linhas = []
    for k in chaves:
        x = d["layers"][k]["sample"].cuda().to(torch.bfloat16)
        w = ler_peso(a.modelo, k)
        if x.shape[-1] != w.shape[1]:
            continue
        ref = x.float() @ w.float().T
        nrm = ref.norm()
        rel = lambda y: ((y - ref).norm() / nrm).item()  # noqa: E731

        wf = w.float()
        norma = x.float().norm(dim=0)
        cru = poda24(wf, wf.abs())
        wan = poda24(wf, wf.abs() * norma.unsqueeze(0))
        wan_bf = wan.to(torch.bfloat16)

        linhas.append({
            "k": k, "params": wf.numel(),
            # O BRACO QUE DECIDE: o que ja entregamos hoje, denso, nas MESMAS camadas e ativacoes.
            # Sem ele os tres bracos esparsos so se comparam entre si e nao respondem "vale a pena
            # trocar?" -- que e a unica pergunta que importa.
            "D": rel(saida_w4(x, w)),
            "A": rel(saida_w4(x, cru.to(torch.bfloat16))),
            "B": rel(saida_w4(x, wan_bf)),
            "C8": rel(saida_w8(x, wan_bf)),
        })

    if not linhas:
        print("nenhuma camada medida", file=sys.stderr)
        return 2

    # Promocao: ordena por erro removido POR BYTE EXTRA e gasta ate o orcamento.
    total = sum(l["params"] for l in linhas)
    extra_por_peso = BITS["s8"] - BITS["s4"]
    orcado = (a.orcamento_bits - BITS["s4"]) * total
    ranking = sorted(linhas, key=lambda l: -(l["B"] - l["C8"]) / (l["params"] * extra_por_peso))
    gasto, promovidas = 0.0, set()
    for l in ranking:
        custo = l["params"] * extra_por_peso
        if l["C8"] < l["B"] and gasto + custo <= orcado:
            promovidas.add(l["k"])
            gasto += custo

    for l in linhas:
        l["C"] = l["C8"] if l["k"] in promovidas else l["B"]

    bits_c = (sum(l["params"] * (BITS["s8"] if l["k"] in promovidas else BITS["s4"])
                  for l in linhas) / total)
    print(f"{len(linhas)} camadas, orcamento {a.orcamento_bits} bits/peso, "
          f"{len(promovidas)} promovidas a W8\n")
    print(f"{'braco':44s} {'mediana':>9s} {'media':>9s} {'pior':>9s} {'bits/peso':>10s}")
    print("-" * 86)
    for nome, campo, bits in (("D  W4A4 DENSO -- o que entregamos hoje", "D", BITS["w4"]),
                              ("A  2:4 SEM Wanda + W4A4  (controle)", "A", BITS["s4"]),
                              ("B  2:4 com Wanda + W4A4", "B", BITS["s4"]),
                              ("C  B, promovendo a W8 o que aceita", "C", bits_c)):
        v = [l[campo] for l in linhas]
        print(f"{nome:44s} {st.median(v):9.4f} {sum(v)/len(v):9.4f} {max(v):9.4f} {bits:10.2f}")

    import statistics as _st
    md = _st.median([l["D"] for l in linhas]); mc = _st.median([l["C"] for l in linhas])
    print()
    print(f"  C contra D: erro {mc/md:.2f}x, tamanho {bits_c/BITS['w4']:.2f}x "
          f"({bits_c:.2f} contra {BITS['w4']:.2f} bits/peso)")
    print()
    print("promovidas (erro B -> C8):")
    for l in sorted(linhas, key=lambda l: -(l["B"] - l["C8"]))[:8]:
        marca = "PROMOVIDA" if l["k"] in promovidas else "   --    "
        print(f"  {marca} {l['k']:34s} {l['B']:.4f} -> {l['C8']:.4f}")

    print()
    print("NAO COBERTO: erro de saida na ativacao calibrada, NAO imagem. Sem treino de recuperacao")
    print("  e sem SparseGPT. Estas colunas sao ERRO, nao velocidade -- para o tempo do 2:4 em")
    print("  bf16 (1,7x a 1,95x sobre o denso, medido) ver tools/sparse24_sm86/. A composicao")
    print("  2:4 + W4A4 destas linhas NAO tem kernel, entao os bits/peso dela sao conta de")
    print("  tamanho, nao medicao de velocidade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
