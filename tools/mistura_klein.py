"""Híbrido de dois checkpoints BFL do Klein com as MESMAS chaves e shapes: cada tensor vem de `--base`,
exceto os que casam `--regex`, que vêm de `--doador`. Escrita em streaming pelo núcleo `_conversion`
(`.partial` exclusivo -> fsync -> `os.replace`), recusa saída existente. Não quantiza nada.

    python_embeded\\python.exe -s tools/mistura_klein.py --base b6.safetensors --doador bf16.safetensors \\
        --regex "double_blocks\\.\\d+\\.txt_(attn\\.(qkv|proj)|mlp\\.\\d)\\.weight" --saida H1.safetensors --esperadas 20
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402

DT = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def rtn4(raw: bytes, dtype: str, shape: list[int], grupo: int, bits: int = 4) -> bytes:
    """RTN absmax por grupo (`lowbit_canon.rtn_simetrico`), de bytes para bytes no dtype da fonte."""
    import torch
    from lowbit_canon import rtn_simetrico
    t = torch.frombuffer(bytearray(raw), dtype=getattr(torch, DT[dtype])).reshape(shape)
    try:
        q = rtn_simetrico(t, grupo, bits)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    out = q.to(getattr(torch, DT[dtype])).contiguous()
    return out.view(torch.uint8).numpy().tobytes() if dtype != "F32" else out.numpy().tobytes()


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--doador", type=Path, required=True)
    a.add_argument("--regex", required=True)
    a.add_argument("--saida", type=Path, required=True)
    a.add_argument("--esperadas", type=int, help="recusa se o regex casar outro numero de tensores")
    a.add_argument("--rtn4", type=int, metavar="GRUPO",
                   help="quantiza os tensores do doador em 4 bits RTN simulado (absmax simetrico, -7..7, "
                        "grupo no eixo K) e grava dequantizado no dtype original")
    a.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8),
                   help="bits do RTN simulado de --rtn4 (niveis simetricos +-(2^(b-1)-1))")
    args = a.parse_args()

    hb, meta_b = C.read_header(args.base)
    hd, _ = C.read_header(args.doador)
    meta = meta_b or None  # sem __metadata__ na base, a saida tambem sai sem
    if set(hb) != set(hd) or any(hb[k]["shape"] != hd[k]["shape"] or hb[k]["dtype"] != hd[k]["dtype"] for k in hb):
        raise SystemExit("recusado: base e doador nao tem as mesmas chaves/shapes/dtypes")
    rx = re.compile(args.regex)
    do_doador = sorted(k for k in hb if rx.fullmatch(k))
    if args.esperadas is not None and len(do_doador) != args.esperadas:
        raise SystemExit(f"recusado: regex casou {len(do_doador)} tensores, esperado {args.esperadas}")
    # Pelo nucleo desde 2026-09-29 (revisao, achado 2): recusa saida existente, parcial antigo e
    # fonte == saida; `.partial` exclusivo, fsync, bytes conferidos, os.replace. Era uma copia a
    # mao correta do mesmo contrato.
    conv = C.Conversion(args.base, args.saida)
    conv.refuse_unsafe(allow_quantized_source=True)
    doador = C.LazyTensors(args.doador, hd)

    def rtn_do_doador(k: str):
        def produz():
            ini, fim = hd[k]["data_offsets"]
            with args.doador.open("rb") as f:
                f.seek(doador.start + ini)
                dados = rtn4(f.read(fim - ini), hd[k]["dtype"], hd[k]["shape"], args.rtn4, args.bits)
            return torch.frombuffer(bytearray(dados), dtype=C.TORCH_DTYPES[hd[k]["dtype"]]).reshape(hd[k]["shape"])
        return produz

    ordem = sorted(hb, key=lambda k: hb[k]["data_offsets"][0])
    entradas = []
    for k in ordem:
        if k not in do_doador:
            entradas.append(C.plan_copy(k, hb[k]))
        elif args.rtn4:
            entradas.append(C.plan_lazy(k, hd[k]["dtype"], hd[k]["shape"],
                                        hd[k]["data_offsets"][1] - hd[k]["data_offsets"][0], rtn_do_doador(k)))
        else:
            entradas.append(C.plan_copy_from(k, hd[k], args.doador))
    novo_meta = None
    if meta is not None:
        novo_meta = dict(meta, mistura_doador=args.doador.name, mistura_regex=args.regex,
                         **({"mistura_rtn_grupo": str(args.rtn4), "mistura_rtn_bits": str(args.bits)} if args.rtn4 else {}))
    conv.guard(conv.planned_size(entradas, novo_meta))
    conv.commit(entradas, novo_meta)
    print(f"{args.saida.name}: {len(do_doador)} tensores do doador, {len(ordem) - len(do_doador)} da base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
