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

Escrita em streaming pelo nucleo `_conversion` (.partial exclusivo + fsync + os.replace), um tensor por
vez, sem safe_open (2x commit aqui); uma passada por variante.
O cabeçalho é o do braço B (mesmos nomes, dtypes, formas e offsets). Corpo = 2-D dentro das pilhas de
blocos (a mesma regra de compara_codigos_bonsai); grupos no eixo K, então BFL fundido serve.

NÃO COBRE: nenhuma imagem. T2 usa a escala de B, que é ótima para os códigos de B, não para os de A.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import _conversion as C  # noqa: E402
from compara_codigos_bonsai import Leitor  # noqa: E402
from lowbit_canon import BLOCO, pilhas_reais  # noqa: E402  (fonte canônica)

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
    # Pelo nucleo desde 2026-09-29 (revisao, achado 2). A escrita a mao abria os tres `.partial` em
    # "wb" depois de um `exists()` (janela de corrida), conferia tamanho com `assert` (que some com
    # `-O`) e deixava os parciais no disco se falhasse no meio. Custo da troca: cada variante e uma
    # passada propria sobre A e B, em vez de uma passada escrevendo as tres ao mesmo tempo.
    convs = {}
    for t in saidas:
        dest = a.dir / f"klein4b_transp_{t}_{a.rotulo_b}_bfl.safetensors"
        conv = C.Conversion(Path(a.b), dest)
        try:
            conv.refuse_unsafe(allow_quantized_source=True)
        except SystemExit as exc:
            print(f"RECUSADO: {exc}", file=sys.stderr)
            return 2
        convs[t] = conv

    def produtor(fn, k: str):
        def produz() -> torch.Tensor:
            xa = A.get(k) if k in corpo else None
            xb = B.get(k) if k in corpo else None
            by = fn(k, xa, xb)
            return torch.frombuffer(bytearray(by), dtype=C.TORCH_DTYPES[B.h[k]["dtype"]]).reshape(B.h[k]["shape"])
        return produz

    for t, (desc, fn) in saidas.items():
        conv = convs[t]
        entradas = [C.plan_lazy(k, B.h[k]["dtype"], B.h[k]["shape"],
                                B.h[k]["data_offsets"][1] - B.h[k]["data_offsets"][0], produtor(fn, k))
                    for k in ordem]
        meta = {"transplante": desc, "a": str(a.a), "b": str(a.b)}

        def progresso(i: int, _n: int, _k: str) -> None:
            if (i - 1) % 30 == 0:
                print(f"  {t} {i - 1}/{len(ordem)}", flush=True)

        conv.guard(conv.planned_size(entradas, meta))
        # O cabecalho e o do braco B (mesmos nomes, dtypes, formas e offsets); este script sempre
        # gravou o `__metadata__` DEPOIS dos tensores e com o `json.dumps` padrao -- mantido, para
        # a saida sair byte a byte igual a de antes.
        conv.commit(entradas, meta, progress=progresso, metadata_last=True, ensure_ascii=True)
        print(f"escrito {conv.output}  {conv.output.stat().st_size:,} B  ({desc})")
    print("\n=== NAO COBERTO ===\n  Nenhuma imagem. T2 usa a escala de B, otima para os codigos de B, nao os de A.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
