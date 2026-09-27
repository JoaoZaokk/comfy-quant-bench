"""Junta os shards de um checkpoint safetensors (index.json do diffusers/transformers) num arquivo unico.

Os bytes de cada tensor sao copiados sem alteracao, em streaming (sem mmap, sem carregar tensores). A ordem das
chaves segue o index. Escrita em `<saida>.partial` e `os.replace` no fim; recusa saida ou partial existentes.

    python_embeded\\python.exe -s tools/junta_shards_safetensors.py <dir com *.index.json> <saida.safetensors>
"""
import json
import os
import struct
import sys
import time
from pathlib import Path

BLOCO = 64 * 2**20


def cabecalho(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


def main():
    origem, saida = Path(sys.argv[1]), Path(sys.argv[2])
    parcial = saida.with_name(saida.name + ".partial")
    for p in (saida, parcial):
        if p.exists():
            raise SystemExit(f"RECUSADO: {p} ja existe")
    idx = next(origem.glob("*.safetensors.index.json"))
    mapa = json.loads(idx.read_text())["weight_map"]
    cab = {s: cabecalho(origem / s) for s in sorted(set(mapa.values()))}
    for s, (_, h) in cab.items():  # o index e os shards tem de concordar
        chaves = {k for k in h if k != "__metadata__"}
        declaradas = {k for k, v in mapa.items() if v == s}
        if chaves != declaradas:
            raise SystemExit(f"RECUSADO: {s} tem {len(chaves)} tensores, o index declara {len(declaradas)}")
    novo, pos = {}, 0
    for k in mapa:
        info = cab[mapa[k]][1][k]
        n = info["data_offsets"][1] - info["data_offsets"][0]
        novo[k] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [pos, pos + n]}
        pos += n
    novo["__metadata__"] = {"junta_shards_de": idx.name,
                            "shards": json.dumps({s: (origem / s).stat().st_size for s in cab})}
    h = json.dumps(novo, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    t0 = time.time()
    with open(parcial, "xb") as out:
        out.write(struct.pack("<Q", len(h)))
        out.write(h)
        abertos = {s: open(origem / s, "rb") for s in cab}
        try:
            for i, k in enumerate(mapa):
                s = mapa[k]
                ini, info = cab[s][0], cab[s][1][k]
                a, b = info["data_offsets"]
                f = abertos[s]
                f.seek(ini + a)
                falta = b - a
                while falta:
                    buf = f.read(min(BLOCO, falta))
                    if not buf:
                        raise IOError(f"{s}: fim inesperado em {k}")
                    out.write(buf)
                    falta -= len(buf)
                if i % 50 == 0:
                    print(f"[{i}/{len(mapa)}] {time.time() - t0:.0f} s", flush=True)
        finally:
            for f in abertos.values():
                f.close()
        out.flush()
        os.fsync(out.fileno())
    esperado = 8 + len(h) + pos
    if parcial.stat().st_size != esperado:
        raise SystemExit(f"tamanho {parcial.stat().st_size} != esperado {esperado}; partial mantido para inspecao")
    os.replace(parcial, saida)
    print(f"OK {saida} {esperado} B, {len(mapa)} tensores, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
