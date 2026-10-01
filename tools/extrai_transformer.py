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
Refusa a sobrescrever. Nao le o arquivo inteiro para a memoria: faixa por faixa, 16 MiB por vez,
pelo contrato de escrita de `_conversion` (.partial exclusivo, fsync, bytes conferidos, os.replace).

    python_embeded\\python.exe -s tools\\extrai_transformer.py ENTRADA.safetensors SAIDA.safetensors
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402

PREFIXO = "model.diffusion_model."


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    # Pelo nucleo desde 2026-09-29 (revisao, achado 2). A copia a mao abria o `.partial` em "wb"
    # sem recusar um parcial antigo (e o sobrescrevia), nao fazia fsync, nao conferia bytes
    # escritos contra planejados e deixava o parcial no disco se falhasse no meio. A fonte pode
    # ser quantizada: isto so copia faixas.
    conv = C.Conversion(src, dst)
    conv.refuse_unsafe(allow_quantized_source=True)
    header, meta = conv.header, conv.metadata
    chaves = sorted(k for k in header if k.startswith(PREFIXO))
    if not chaves:
        raise SystemExit(f"nenhuma chave com prefixo {PREFIXO!r} em {src}")
    # novo cabecalho: mesmas chaves, offsets recomputados na ordem em que serao copiadas
    entradas = [C.plan_copy(k, header[k]) for k in chaves]
    total = sum(e.nbytes for e in entradas)
    print(f"{len(chaves)} tensores, {total / 2**30:.2f} GiB de dados, de {len(header)} no original", flush=True)
    conv.guard(conv.planned_size(entradas, meta or None))
    t0 = time.time()
    feito = [0]

    def progresso(i: int, _n: int, chave: str) -> None:
        feito[0] += header[chave]["data_offsets"][1] - header[chave]["data_offsets"][0]
        if (i - 1) % 400 == 0:
            print(f"  [{i - 1}/{len(chaves)}] {feito[0] / 2**30:.1f} GiB  "
                  f"{feito[0] / 2**20 / max(time.time() - t0, 1):.0f} MiB/s", flush=True)

    # `metadata_last` e `ensure_ascii=True`: este script sempre gravou o `__metadata__` DEPOIS dos
    # tensores e com o `json.dumps` padrao; manter isso deixa a saida byte a byte igual a de antes.
    conv.commit(entradas, meta or None, progress=progresso, metadata_last=True, ensure_ascii=True)
    print(f"escrito {dst} ({dst.stat().st_size / 2**30:.2f} GiB) em {time.time() - t0:.0f} s", flush=True)
    print("NAO COBERTO: nao verifica hash contra a fonte (cada faixa e copiada crua, sem decodificar); "
          "nao carrega o resultado -- o loader do ComfyUI e quem diz se o arquivo serve.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
