"""Quantos bits POR PESO o pack de deploy do Bonsai realmente gasta? (EXECUTADO)

Conta sobre as camadas QUANTIZADAS apenas -- os tensores densos bf16 (camadas puladas e normas) ficam
de fora, porque misturar os dois e o que produz o numero de marketing. `orig_shape` vem do proprio
pack, entao o denominador nao e estimado.

Le com `torch.load(mmap=True, weights_only=True)`: mmap para nao pagar o arquivo em RAM,
`weights_only` porque e pickle de terceiro e nao se executa codigo de terceiro para ler peso.

Nao cobre: nenhum kernel, nenhuma imagem. Diz o que o arquivo gasta, nao o que ele vale.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

CAMPOS = ("W_q", "scales", "zeros", "metadata", "orig_shape")


def razao(orig, forma) -> int:
    """Quantos elementos de K cabem em cada elemento de `forma`, no layout do pack gemlite.

    O layout foi LIDO dos shapes, nao assumido, e ele TRANSPOE. `orig_shape` segue a convencao do
    Linear do PyTorch, `(N_saida, K_entrada)`; o pack grava `(K/r, N)`. Confirmado nos 5 padroes
    distintos dos dois packs, por exemplo `orig=(27648, 3072) -> W_q=(768, 27648)`, que e
    `(3072/4, 27648)`.

    Duas versoes anteriores desta funcao erraram por nao saber disso: a primeira dividia sempre o
    eixo 0 e imprimia "36 pesos por byte" e "grupo 42"; a segunda procurava o eixo que casava e
    devolvia -1 em 60 das 100 camadas. As duas foram corrigidas olhando os shapes crus, nao
    remendando a heuristica.

    Devolve -1 se N nao casar ou K nao dividir -- o que significa layout diferente do medido aqui,
    e vai impresso como -1 em vez de virar um numero plausivel.
    """
    n, k = int(orig[0]), int(orig[1])
    linhas, colunas = int(forma[0]), int(forma[1])
    if colunas != n or linhas == 0 or k % linhas:
        return -1
    return k // linhas


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, action="append",
                   help="caminho do state_dict.pt; repetir para comparar packs")
    a = p.parse_args()

    for caminho in a.pack:
        sd = torch.load(caminho, map_location="cpu", mmap=True, weights_only=True)
        por_campo: Counter[str] = Counter()
        params = 0
        n_camadas = 0
        grupos = Counter()
        pesos_por_byte = Counter()

        fqns = sorted({k.rsplit(".", 1)[0] for k in sd if k.rsplit(".", 1)[-1] == "W_q"})
        for fqn in fqns:
            forma = sd[f"{fqn}.orig_shape"].tolist()
            n = int(forma[0]) * int(forma[1])
            params += n
            n_camadas += 1
            # QUAL eixo esta empacotado nao e dado: a primeira versao assumiu o eixo 0 sempre e
            # cuspiu "36 pesos por byte" e "grupo 42", que sao impossiveis. Descobre pelo eixo que
            # NAO mudou de tamanho.
            pesos_por_byte[razao(forma, sd[f"{fqn}.W_q"].shape)] += 1
            grupos[razao(forma, sd[f"{fqn}.scales"].shape)] += 1
            for c in CAMPOS:
                t = sd.get(f"{fqn}.{c}")
                if t is not None:
                    por_campo[c] += t.numel() * t.element_size()

        total = sum(por_campo.values())
        print(f"\n=== {Path(caminho).parent.name} ===")
        print(f"  {n_camadas} camadas quantizadas, {params:,} parametros")
        print(f"  {'campo':12s} {'bytes':>16s} {'bits/peso':>11s}")
        for c in CAMPOS:
            if por_campo[c]:
                print(f"  {c:12s} {por_campo[c]:16,d} {por_campo[c]*8/params:11.4f}")
        print(f"  {'TOTAL':12s} {total:16,d} {total*8/params:11.4f}")
        print(f"  pesos por byte de W_q: {dict(pesos_por_byte)}  -> slot de "
              f"{8/max(pesos_por_byte):.4f} bit" if pesos_por_byte else "")
        print(f"  tamanho de grupo (eixo empacotado): {dict(grupos)}")
        densos = sum(t.numel() * t.element_size() for k, t in sd.items()
                     if k.endswith(".weight") and torch.is_tensor(t))
        n_densos = sum(1 for k in sd if k.endswith(".weight"))
        print(f"  fora da conta: {n_densos} tensores densos, {densos:,} B "
              f"({densos/(densos+total)*100:.2f}% do arquivo)")

    print("\n=== NAO COBERTO ===")
    print("  Contabilidade de bytes do arquivo. Nenhum kernel, nenhuma imagem, GPU nao tocada.")
    print("  Nao diz se o gemlite faz GEMM na largura do slot ou desempacota antes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
