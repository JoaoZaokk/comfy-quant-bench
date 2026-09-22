"""Braco 0 da fila de GPU: PTQ ternario ingenuo, conjunto denso intacto. (EXECUTADO, CPU)

Esmaga para ternario (absmean, grupo 128 no eixo K -- a mesma receita e o mesmo eixo que o Bonsai
declara) todo tensor 2-D que esteja DENTRO de uma pilha de blocos, e copia **byte a byte** tudo o
resto. A regra de selecao nao e chute: foi medida no klein-4B por dois times independentes, com
intersecao 69 de 69 no conjunto denso (`replicabilidade_fora_do_flux_2026-09-22.md`).

Escreve os pesos **ja desempacotados de volta para o dtype original**, porque nao existe kernel de
1,58 bit nesta maquina. Consequencia que vai impressa em toda execucao: **este arquivo nao economiza
memoria nem tempo**. Ele existe para medir FIDELIDADE com o mesmo caminho de execucao dos outros
bracos, e qualquer numero de velocidade tirado dele e invalido.

Escrita em streaming com `.partial` + `os.replace`, um tensor por vez: `safe_open` cobraria 2x o
arquivo em commit nesta maquina, e o arquivo tem 7,4 GiB.

    braco 0   este script                                    controle inferior
    braco 1   este arquivo, e depois SO o denso ajustado      a hipotese
    braco 2   o Bonsai publicado                              controle superior
    braco 3   o original                                      referencia

Nao cobre: nenhuma imagem, nenhuma ativacao, nenhum kernel, GPU nao tocada. E construcao de arquivo.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import struct
import sys
from pathlib import Path

import torch

BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")
DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
TAM = {"F16": 2, "BF16": 2, "F32": 4}


def pilhas_reais(nomes) -> set[str]:
    """Uma pilha precisa de >= 2 indices distintos: `adaLN_modulation.1` nao e pilha."""
    ind: dict[str, set[str]] = {}
    for k in nomes:
        m = BLOCO.match(k)
        if m:
            ind.setdefault(m.group("pilha"), set()).add(m.group("i"))
    return {n for n, i in ind.items() if len(i) >= 2}


def ternariza(w: torch.Tensor, grupo: int) -> torch.Tensor:
    """absmean estilo BitNet b1.58 por grupo de `grupo` no eixo K (o ultimo), escala por minimo L2."""
    n, k = w.shape
    g = w.float().reshape(n, k // grupo, grupo)
    d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
    t = (g / d).clamp(-1, 1).round()
    # escala otima dado o codigo: s* = <w,t>/<t,t>, por grupo. Nao usar d cru: d nao minimiza L2.
    num = (g * t).sum(dim=2, keepdim=True)
    den = (t * t).sum(dim=2, keepdim=True)
    s = torch.where(den > 0, num / den, torch.zeros_like(num))
    return (t * s).reshape(n, k).to(w.dtype)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--entrada", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    ent, sai = Path(a.entrada), Path(a.saida)
    if sai.exists():
        print(f"RECUSADO: {sai} ja existe. Nao sobrescrevo saida.", file=sys.stderr)
        return 2
    parcial = sai.with_suffix(sai.suffix + ".partial")
    if parcial.exists():
        print(f"RECUSADO: {parcial} existe (conversao anterior morta). Apague a mao.", file=sys.stderr)
        return 2

    f = open(ent, "rb")  # noqa: SIM115 -- o mmap precisa do handle vivo; fechado no fim
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    nh = int.from_bytes(mm[:8], "little")
    header = json.loads(mm[8:8 + nh])
    meta = header.pop("__metadata__", None)
    base = 8 + nh

    pilhas = pilhas_reais(header)
    alvo, copia = [], []
    for k, v in header.items():
        forma = v["shape"]
        m = BLOCO.match(k)
        dentro = bool(m) and m.group("pilha") in pilhas
        if len(forma) == 2 and dentro and forma[1] % a.grupo == 0 and v["dtype"] in DT:
            alvo.append(k)
        else:
            copia.append(k)

    params_q = sum(header[k]["shape"][0] * header[k]["shape"][1] for k in alvo)
    print(f"entrada {ent}  {ent.stat().st_size:,} B  {len(header)} tensores")
    print(f"  ternarizar (2-D dentro de bloco, K%{a.grupo}==0): {len(alvo)} camadas, "
          f"{params_q:,} params")
    print(f"  copiar byte a byte:                              {len(copia)} tensores")
    if not alvo:
        print("RECUSADO: nenhuma camada selecionada. Nao escrevo arquivo vazio.", file=sys.stderr)
        mm.close(), f.close()
        return 2
    if a.dry_run:
        print("  --dry-run: nada escrito.")
        mm.close(), f.close()
        return 0

    # header de saida: mesmos nomes, mesmos dtypes, mesmos shapes -- so os valores mudam
    novo, off = {}, 0
    for k in header:
        v = header[k]
        n = TAM.get(v["dtype"])
        if n is None:                                  # dtype que nao sei medir: copia crua
            a0, b0 = v["data_offsets"]
            n_bytes = b0 - a0
        else:
            n_bytes = n
            for x in v["shape"]:
                n_bytes *= x
        novo[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + n_bytes]}
        off += n_bytes
    if meta is not None:
        novo["__metadata__"] = meta
    bruto = json.dumps(novo, separators=(",", ":")).encode()
    pad = (-len(bruto)) % 8
    bruto += b" " * pad

    alvo_set = set(alvo)
    feitos = 0
    with open(parcial, "wb") as out:
        out.write(struct.pack("<Q", len(bruto)))
        out.write(bruto)
        for k in header:                               # ordem do header de saida
            v = header[k]
            a0, b0 = v["data_offsets"]
            if k in alvo_set:
                cru = bytearray(memoryview(mm)[base + a0:base + b0])
                w = torch.frombuffer(cru, dtype=DT[v["dtype"]]).view(*v["shape"])
                # `.numpy()` nao aceita BFloat16 (TypeError: unsupported ScalarType) -- a primeira
                # versao morreu aqui, depois de escrever 1,6 GiB. Reinterpretar como uint8 sai pelos
                # bytes crus e serve para qualquer dtype de largura fixa.
                t = ternariza(w, a.grupo).contiguous()
                out.write(t.flatten().view(torch.uint8).numpy().tobytes())
                feitos += 1
                if feitos % 20 == 0:
                    print(f"    [{feitos}/{len(alvo)}] ternarizadas", flush=True)
            else:
                out.write(memoryview(mm)[base + a0:base + b0])
        out.flush()
        os.fsync(out.fileno())
    os.replace(parcial, sai)
    mm.close()
    f.close()
    print(f"\nescrito {sai}  {sai.stat().st_size:,} B  ({feitos} camadas ternarizadas)")

    print("\n=== NAO COBERTO ===")
    print("  Os pesos saem DESEMPACOTADOS no dtype original: este arquivo NAO economiza memoria")
    print("  nem tempo, e nenhum numero de velocidade tirado dele vale. Existe para medir")
    print("  fidelidade no mesmo caminho de execucao dos outros bracos.")
    print("  Nenhuma imagem, nenhuma ativacao, nenhum kernel, GPU nao tocada.")
    print("  A selecao usa a regra medida no klein-4B; noutra arquitetura e extrapolacao.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
