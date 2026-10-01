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
