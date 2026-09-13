"""Extrai SO o transformer (`model.diffusion_model.*`) de um checkpoint unico do LTX 2.x para um
arquivo em `diffusion_models/`, por copia de faixas de bytes -- nunca mmap.

POR QUE
-------
O checkpoint unico do LTX 2.3 (43 GiB: DiT + VAEs + vocoder + projecao) derrubou o servidor DUAS
vezes ao ser carregado pelo caminho de checkpoint com DisTorch2 -- `access violation` em
`torch/storage.py __getitem__` dentro de `load_torch_file`, primeiro lendo do SMB, depois lendo do
disco local com 40 GiB de RAM livre. O transformer sozinho do 2.5 (39 GiB) pelo `UNETLoader`
sobreviveu na mesma maquina com a mesma RAM. Este arquivo produz o equivalente para o 2.3: o
mesmo caminho que ja funcionou, sem mudar um byte de tensor.

O que sai e byte a byte o que entrou: mesmas chaves (com o prefixo `model.diffusion_model.`, que
`load_diffusion_model` remove sozinho), mesmos dtypes, mesmas formas, mesmo `__metadata__`.
Refusa a sobrescrever. Nao le o arquivo inteiro para a memoria: faixa por faixa, 16 MiB por vez.

    python_embeded\\python.exe -s tools\\extrai_transformer.py ENTRADA.safetensors SAIDA.safetensors
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

PREFIXO = "model.diffusion_model."
BLOCO = 16 << 20


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        raise SystemExit(f"recuso sobrescrever {dst}")
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    meta = header.pop("__metadata__", None)
    chaves = sorted(k for k in header if k.startswith(PREFIXO))
    if not chaves:
        raise SystemExit(f"nenhuma chave com prefixo {PREFIXO!r} em {src}")
    # novo cabecalho: mesmas chaves, offsets recomputados na ordem em que serao copiadas
    novo: dict = {}
    pos = 0
    for k in chaves:
        s, e = header[k]["data_offsets"]
        novo[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [pos, pos + (e - s)]}
        pos += e - s
    if meta is not None:
        novo["__metadata__"] = meta
    hb = json.dumps(novo, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    total = pos
    print(f"{len(chaves)} tensores, {total / 2**30:.2f} GiB de dados, de {len(header)} no original", flush=True)
    parcial = dst.with_suffix(dst.suffix + ".partial")
    t0 = time.time()
    feito = 0
    with open(src, "rb") as fi, open(parcial, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for i, k in enumerate(chaves):
            s, e = header[k]["data_offsets"]
            fi.seek(base + s)
            resta = e - s
            while resta:
                b = fi.read(min(resta, BLOCO))
                if not b:
                    raise SystemExit(f"fim inesperado da fonte em {k}")
                fo.write(b)
                resta -= len(b)
                feito += len(b)
            if i % 400 == 0:
                print(f"  [{i}/{len(chaves)}] {feito / 2**30:.1f} GiB  {feito / 2**20 / max(time.time() - t0, 1):.0f} MiB/s", flush=True)
        fo.flush()
    parcial.replace(dst)
    print(f"escrito {dst} ({dst.stat().st_size / 2**30:.2f} GiB) em {time.time() - t0:.0f} s", flush=True)
    print("NAO COBERTO: nao verifica hash contra a fonte (cada faixa e copiada crua, sem decodificar); "
          "nao carrega o resultado -- o loader do ComfyUI e quem diz se o arquivo serve.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
