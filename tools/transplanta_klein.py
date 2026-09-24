"""Transplante entre dois ternários do klein: códigos, escalas por grupo e tensores não-ternários. Só CPU. (EXECUTADO)

Pergunta que responde (com render, depois): a recuperação do QAT mora nos códigos ternários ou nos
parâmetros contínuos em volta deles? No nosso QAT os "contínuos" são DUAS coisas diferentes:
    resto     os tensores fora do corpo ternário (9 lineares densos + 60 norm_q/norm_k), mestre bf16
    escala    uma por grupo de 128 no eixo K, NÃO é parâmetro: recalculada do mestre (ótima L2 dado o
              código) e gravada embutida no valor desempacotado (x = código · escala)
Então há três transplantes, a partir de um PTQ (A) e de um braço treinado (B):
    T1  corpo de A              + resto de B
    T2  códigos de A · escalas de B + resto de B
    T3  corpo de B              + resto de A (= originais, no PTQ ingênuo)

CONTROLE OBRIGATÓRIO (C0): o mesmo construtor, pedido "códigos de X · escalas de X" para X = A e X = B,
tem de reproduzir X byte a byte em todo tensor de corpo. Se falhar, nenhum arquivo é escrito.

Escrita em streaming (.partial + fsync + os.replace), um tensor por vez, sem safe_open (2x commit aqui).
O cabeçalho é o do braço B (mesmos nomes, dtypes, formas e offsets). Corpo = 2-D dentro das pilhas de
blocos (a mesma regra de compara_codigos_bonsai); grupos no eixo K, então BFL fundido serve.

NÃO COBRE: nenhuma imagem. T2 usa a escala de B, que é ótima para os códigos de B, não para os de A.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from compara_codigos_bonsai import BLOCO, Leitor, pilhas_reais  # noqa: E402

G = 128


def bruto(L: Leitor, k: str) -> bytes:
    a, b = L.h[k]["data_offsets"]
    with L.p.open("rb") as f:
        f.seek(L.base + a)
        return f.read(b - a)


def cod_esc(xc: torch.Tensor, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """código de xc, escala por grupo de xs (max|x| do grupo; grupo todo zero cai na escala de xc)."""
    n, k = xc.shape
    gc, gs = xc.reshape(n, k // G, G), xs.reshape(n, k // G, G)
    s = gs.abs().amax(dim=2, keepdim=True)
    s = torch.where(s > 0, s, gc.abs().amax(dim=2, keepdim=True))
    # zero fica o próprio zero de xc: o PTQ ingênuo grava -0.0 em ~metade dos zeros, e sign() o
    # reescreveria como +0.0 -- mesmo valor, outro byte, e o C0 byte a byte acusaria (medido 24/09)
    return torch.where(gc == 0, gc, torch.sign(gc) * s).reshape(n, k), s


def b16(t: torch.Tensor) -> bytes:
    return t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="PTQ (fonte dos códigos/corpo 'morto')")
    p.add_argument("--b", required=True, help="braço treinado")
    p.add_argument("--rotulo-b", required=True)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--variantes", nargs="+", default=["T1", "T2", "T3"], choices=["T1", "T2", "T3"])
    a = p.parse_args()
    A, B = Leitor(a.a), Leitor(a.b)
    assert set(A.h) == set(B.h), "nomes diferentes entre A e B"
    pil = pilhas_reais(B.h)
    corpo = {k for k, m in B.h.items() if len(m["shape"]) == 2 and (x := BLOCO.match(k)) and x.group("pilha") in pil}
    resto = sorted(set(B.h) - corpo)
    for k in B.h:
        assert B.h[k]["dtype"] == "BF16" and A.h[k]["shape"] == B.h[k]["shape"], k
    print(f"corpo {len(corpo)} tensores, resto {len(resto)}", flush=True)

    # C0 antes de escrever qualquer coisa
    razoes, falhas = [], []
    for k in sorted(corpo):
        xa, xb = A.get(k), B.get(k)
        for rot, x in (("A", xa), ("B", xb)):
            if b16(cod_esc(x, x)[0]) != bruto(A if rot == "A" else B, k):
                falhas.append(f"{rot}:{k}")
        sa, sb = cod_esc(xa, xa)[1], cod_esc(xb, xb)[1]
        m = (sa > 0) & (sb > 0)
        razoes.append((sb[m] / sa[m]).float())
    if falhas:
        print(f"C0 FALHOU em {len(falhas)} tensores, p.ex. {falhas[:3]} -- nada escrito", file=sys.stderr)
        return 2
    print("C0 ok: codigo x escala reproduz A e B byte a byte em todo o corpo", flush=True)
    rz = torch.cat(razoes)
    q = torch.quantile(rz[torch.randperm(rz.numel())[:2_000_000]], torch.tensor([0.05, 0.5, 0.95]))
    lr = rz.log().abs().mean()
    print(f"escala B/A por grupo: p5 {q[0]:.4f}  mediana {q[1]:.4f}  p95 {q[2]:.4f}  |log| medio {lr:.4f}"
          f"  ({rz.numel()} grupos)")
    print("resto, ||B - A|| / ||A||:")
    for k in resto:
        xa, xb = A.get(k), B.get(k)
        print(f"  {k:60s} {float((xb - xa).norm() / xa.norm().clamp(min=1e-12)):.4f}")

    saidas = {
        "T1": (f"corpo do PTQ + resto do {a.rotulo_b}", lambda k, xa, xb: bruto(A, k) if k in corpo else bruto(B, k)),
        "T2": (f"codigos do PTQ x escalas do {a.rotulo_b} + resto do {a.rotulo_b}",
               lambda k, xa, xb: b16(cod_esc(xa, xb)[0]) if k in corpo else bruto(B, k)),
        "T3": (f"corpo do {a.rotulo_b} + resto do PTQ (originais)", lambda k, xa, xb: bruto(B, k) if k in corpo else bruto(A, k)),
    }
    saidas = {t: v for t, v in saidas.items() if t in a.variantes}
    ordem = sorted(B.h, key=lambda k: B.h[k]["data_offsets"][0])
    arqs = {}
    for t, (desc, _) in saidas.items():
        dest = a.dir / f"klein4b_transp_{t}_{a.rotulo_b}_bfl.safetensors"
        if dest.exists() or dest.with_suffix(".safetensors.partial").exists():
            print(f"RECUSADO: {dest} ja existe", file=sys.stderr)
            return 2
        hd = dict(B.h)
        hd["__metadata__"] = {"transplante": desc, "a": str(a.a), "b": str(a.b)}
        hb = json.dumps(hd, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        f = dest.with_suffix(".safetensors.partial").open("wb")
        f.write(struct.pack("<Q", len(hb)) + hb)
        arqs[t] = (f, dest)
    for i, k in enumerate(ordem):
        xa = A.get(k) if k in corpo else None
        xb = B.get(k) if k in corpo else None
        for t, (_, fn) in saidas.items():
            by = fn(k, xa, xb)
            assert len(by) == B.h[k]["data_offsets"][1] - B.h[k]["data_offsets"][0], (t, k)
            arqs[t][0].write(by)
        if i % 30 == 0:
            print(f"  {i}/{len(ordem)}", flush=True)
    for t, (f, dest) in arqs.items():
        f.flush(); os.fsync(f.fileno()); f.close()
        os.replace(dest.with_suffix(".safetensors.partial"), dest)
        print(f"escrito {dest}  {dest.stat().st_size:,} B  ({saidas[t][0]})")
    print("\n=== NAO COBERTO ===\n  Nenhuma imagem. T2 usa a escala de B, otima para os codigos de B, nao os de A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
