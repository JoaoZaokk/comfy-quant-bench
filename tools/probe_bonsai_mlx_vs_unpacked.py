"""O pack MLX carrega os MESMOS pesos treinados que o repo unpacked? (EXECUTADO)

Pergunta de controle, nao de descoberta: as tres distribuicoes do Bonsai Image (unpacked, gemlite,
mlx) deveriam ser vistas do mesmo modelo. Se o MLX desempacotado bater com o unpacked, batem; se nao
bater, alguem requantizou por pack e a frase "mesmos pesos, empacotadores diferentes" cai.

O layout do MLX e LIDO, nao ajustado ao resultado -- e a convencao publicada do `mx.quantize`, com os
shapes confirmando cada divisor:

    weight  U32  [N, K/r]        r = 32/bits valores por palavra, os bits BAIXOS primeiro
    scales  BF16 [N, K/128]      grupo 128 no eixo K
    biases  BF16 [N, K/128]
    w = scales[n, k//128] * codigo + biases[n, k//128]

Confirmado pelos shapes do proprio arquivo: 3072/192 = 16 valores por palavra de 32 bits no pack de
2 bits, 3072/96 = 32 no de 1 bit, 3072/24 = 128 no grupo. **Se este layout estiver errado o
resultado vem ruim e fica ruim** -- escolher layout pelo resultado que ele produz e ajuste ao alvo e
invalidaria a comparacao inteira.

Nao cobre: nenhuma imagem, nenhuma ativacao, nenhum kernel. GPU nao tocada.
"""
from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path

import numpy as np
import torch

DT = {"F16": np.float16, "BF16": None, "F32": np.float32, "U32": np.uint32, "I32": np.int32}


class Arquivo:
    """Leitor por mmap SOMENTE LEITURA: custa 0 de commit, ao contrario de `safe_open`."""

    def __init__(self, caminho: Path):
        # SIM115 silenciado de proposito: o handle TEM de sobreviver ao __init__; quem fecha e fecha()
        self.f = open(caminho, "rb")  # noqa: SIM115
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.mm[:8], "little")
        self.header = json.loads(self.mm[8:8 + n])
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def tensor(self, chave: str) -> torch.Tensor:
        e = self.header[chave]
        a, b = e["data_offsets"]
        bruto = bytearray(memoryview(self.mm)[self.base + a:self.base + b])
        if e["dtype"] == "BF16":
            return torch.frombuffer(bruto, dtype=torch.bfloat16).view(*e["shape"])
        return torch.from_numpy(
            np.frombuffer(bruto, dtype=DT[e["dtype"]]).reshape(e["shape"]).copy())

    def fecha(self):
        self.mm.close()
        self.f.close()


def desempacota(palavras: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """U32 [N, K/r] -> codigos inteiros [N, K], bits baixos primeiro (convencao do mx.quantize)."""
    r = 32 // bits
    p = palavras.to(torch.int64)                       # int64: >> em uint32 do torch e traicoeiro
    desloc = (torch.arange(r, dtype=torch.int64) * bits).view(1, 1, r)
    cod = (p.unsqueeze(-1) >> desloc) & ((1 << bits) - 1)
    return cod.reshape(p.shape[0], -1)[:, :k]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mlx", required=True)
    p.add_argument("--unpacked", required=True)
    p.add_argument("--bits", type=int, required=True)
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--camadas", type=int, default=8)
    a = p.parse_args()

    mlx, ref = Arquivo(Path(a.mlx)), Arquivo(Path(a.unpacked))
    fqns = sorted({k.rsplit(".", 1)[0] for k in mlx.header if k.endswith(".scales")})
    # amostra espalhada, nao os primeiros: os primeiros blocos podem nao representar o resto
    passo = max(1, len(fqns) // a.camadas)
    amostra = fqns[::passo][:a.camadas]
    print(f"{len(fqns)} camadas quantizadas no pack MLX; medindo {len(amostra)}\n")
    print(f"{'camada':46s} {'N':>6s} {'K':>6s} {'niveis':>7s} {'rel-L2':>10s} {'max|d|':>9s} {'iguais':>8s}")

    linhas = []
    for fqn in amostra:
        w = mlx.tensor(f"{fqn}.weight")
        s = mlx.tensor(f"{fqn}.scales").float()
        b = mlx.tensor(f"{fqn}.biases").float()
        chave_ref = f"{fqn}.weight"
        if chave_ref not in ref.header:
            print(f"{fqn:46s}  AUSENTE no unpacked")
            continue
        wr = ref.tensor(chave_ref).float()
        n, k = wr.shape
        cod = desempacota(w, a.bits, k)
        g = torch.repeat_interleave(torch.arange(k // a.grupo), a.grupo)[:k]
        wm = s[:, g] * cod.float() + b[:, g]

        d = (wm - wr)
        rel = float(d.norm() / wr.norm())
        iguais = float((wm == wr).double().mean())
        niveis = int(torch.unique(cod).numel())
        linhas.append({"fqn": fqn, "rel_l2": rel, "iguais": iguais, "niveis": niveis,
                       "max_abs": float(d.abs().max())})
        print(f"{fqn:46s} {n:6d} {k:6d} {niveis:7d} {rel:10.3e} {float(d.abs().max()):9.2e} "
              f"{iguais:8.5f}", flush=True)
        del w, s, b, wr, cod, wm, d

    if not linhas:
        print("\nNADA MEDIDO")
        return 2
    med = sorted(x["rel_l2"] for x in linhas)[len(linhas) // 2]
    ig = sorted(x["iguais"] for x in linhas)[len(linhas) // 2]
    niveis = sorted({x["niveis"] for x in linhas})
    print(f"\n=== MEDIANAS sobre {len(linhas)} camadas ===")
    print(f"  rel-L2 do MLX desempacotado contra o unpacked   {med:.4e}")
    print(f"  fracao de elementos BIT A BIT identicos          {ig:.6f}")
    print(f"  niveis distintos no codigo: {niveis}  (esperado {2 ** a.bits} slots, "
          f"{'3' if a.bits == 2 else '2'} usados se o nome do repo nao mentir)")
    print()
    if med < 1e-3:
        print("  LEITURA: o MLX desempacota para o MESMO peso do unpacked. As distribuicoes sao")
        print("  vistas do mesmo modelo treinado, e nenhuma delas requantizou por conta propria.")
    else:
        print(f"  LEITURA: NAO bate ({med:.3e}). Ou o layout que eu li esta errado, ou uma das duas")
        print("  distribuicoes foi requantizada. NAO ajusto o layout para fechar: isso seria ajuste")
        print("  ao alvo. Fica como divergencia medida.")

    print("\n=== NAO COBERTO ===")
    print(f"  Amostra de {len(linhas)} de {len(fqns)} camadas, nao o arquivo todo.")
    print("  Nenhuma imagem, nenhuma ativacao, nenhum kernel, GPU nao tocada.")
    print("  Os 69 tensores densos bf16 nao entram nesta comparacao; so as camadas quantizadas.")
    mlx.fecha()
    ref.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
