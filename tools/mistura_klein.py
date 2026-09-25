"""Híbrido de dois checkpoints BFL do Klein com as MESMAS chaves e shapes: cada tensor vem de `--base`,
exceto os que casam `--regex`, que vêm de `--doador`. Escrita em streaming, `.partial` -> `os.replace`,
recusa saída existente. Não quantiza nada.

    python_embeded\\python.exe -s tools/mistura_klein.py --base b6.safetensors --doador bf16.safetensors \\
        --regex "double_blocks\\.\\d+\\.txt_(attn\\.(qkv|proj)|mlp\\.\\d)\\.weight" --saida H1.safetensors --esperadas 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path

BLOCO = 64 << 20


def cabecalho(p: Path):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, meta, 8 + n


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--doador", type=Path, required=True)
    a.add_argument("--regex", required=True)
    a.add_argument("--saida", type=Path, required=True)
    a.add_argument("--esperadas", type=int, help="recusa se o regex casar outro numero de tensores")
    args = a.parse_args()

    hb, meta, base_b = cabecalho(args.base)
    hd, _, base_d = cabecalho(args.doador)
    if set(hb) != set(hd) or any(hb[k]["shape"] != hd[k]["shape"] or hb[k]["dtype"] != hd[k]["dtype"] for k in hb):
        raise SystemExit("recusado: base e doador nao tem as mesmas chaves/shapes/dtypes")
    rx = re.compile(args.regex)
    do_doador = sorted(k for k in hb if rx.fullmatch(k))
    if args.esperadas is not None and len(do_doador) != args.esperadas:
        raise SystemExit(f"recusado: regex casou {len(do_doador)} tensores, esperado {args.esperadas}")
    if args.saida.exists():
        raise SystemExit(f"recusado: saida ja existe {args.saida}")
    parcial = args.saida.with_suffix(args.saida.suffix + ".partial")
    if parcial.exists():
        raise SystemExit(f"recusado: parcial antigo {parcial}")

    ordem = sorted(hb, key=lambda k: hb[k]["data_offsets"][0])
    novo, off = {}, 0
    if meta is not None:
        novo["__metadata__"] = dict(meta, mistura_doador=args.doador.name, mistura_regex=args.regex)
    for k in ordem:
        tam = hb[k]["data_offsets"][1] - hb[k]["data_offsets"][0]
        novo[k] = {"dtype": hb[k]["dtype"], "shape": hb[k]["shape"], "data_offsets": [off, off + tam]}
        off += tam
    blob = json.dumps(novo, separators=(",", ":"), ensure_ascii=False).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    out = open(parcial, "xb")
    try:
        with args.base.open("rb") as fb, args.doador.open("rb") as fd, out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for k in ordem:
                fonte, h, base = (fd, hd, base_d) if k in do_doador else (fb, hb, base_b)
                ini, fim = h[k]["data_offsets"]
                fonte.seek(base + ini)
                resta = fim - ini
                while resta:
                    pedaco = fonte.read(min(BLOCO, resta))
                    if not pedaco:
                        raise EOFError(k)
                    out.write(pedaco)
                    resta -= len(pedaco)
            out.flush()
            os.fsync(out.fileno())
        os.replace(parcial, args.saida)
    finally:
        if parcial.exists():
            parcial.unlink()
    print(f"{args.saida.name}: {len(do_doador)} tensores do doador, {len(ordem) - len(do_doador)} da base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
