"""Dois checkpoints sao identicos? Responde em TRES niveis, porque os tres podem discordar.

POR QUE TRES E NAO UM. O criterio do braco 1 pede "arquivo byte a byte identico ao braco 0". Isso
e uma pergunta sobre o ARQUIVO, e um arquivo safetensors carrega tres coisas que falham
independentemente:

  1. bytes do arquivo inteiro  -- inclui a ordem das chaves no header e o `__metadata__`
  2. bytes de cada TENSOR      -- o que realmente decide se o peso mudou
  3. conjunto de chaves/shapes -- o que decide se e o mesmo modelo

Os dois lados desta comparacao foram escritos por escritores DIFERENTES -- o braco 0 por header
montado a mao em `constroi_ternario_ingenuo.py`, o braco 1 por `safetensors.torch.save_file` --
entao o nivel 1 pode falhar com o nivel 2 passando, e nesse caso o controle PASSOU: nenhum peso
mudou, so a serializacao. Reportar "FALHOU byte a byte" sem separar os niveis daria um veredito
errado sobre o experimento por causa da ordem das chaves.

O oposto tambem importa e e o motivo de o nivel 1 nao ser descartado: nivel 2 passando com nivel 1
falhando **por diferenca no `__metadata__`** e um arquivo que carrega proveniencia diferente, e isso
tem de aparecer.

LE EM BLOCOS. Os arquivos aqui tem 7,75 GiB cada; hashear os dois inteiros em RAM e o defeito que
este repo documenta como custo de commit. Bloco de 64 MiB, um arquivo por vez.

NAO COBRE: nada sobre qualidade. Diz se os bytes batem, nada mais.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

BLOCO = 64 << 20


def sha_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while (b := f.read(BLOCO)):
            h.update(b)
    return h.hexdigest()


def cabecalho(p: Path):
    """Devolve (dict do header, offset onde comecam os dados). So o header entra em RAM."""
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def sha_tensores(p: Path) -> dict[str, tuple[str, str, list[int]]]:
    hdr, base = cabecalho(p)
    saida = {}
    with p.open("rb") as f:
        for nome, meta in sorted(hdr.items()):
            if nome == "__metadata__":
                continue
            ini, fim = meta["data_offsets"]
            f.seek(base + ini)
            h = hashlib.sha256()
            restante = fim - ini
            while restante:
                b = f.read(min(BLOCO, restante))
                if not b:
                    raise RuntimeError(f"{p.name}: {nome} truncado, faltaram {restante} B")
                h.update(b)
                restante -= len(b)
            saida[nome] = (h.hexdigest(), meta["dtype"], meta["shape"])
    return saida


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--esperado", choices=["identico", "diferente"],
                   help="o que o CRITERIO exige. Sem isto o tool descreve; com isto ele julga, e o "
                        "julgamento e sobre os TENSORES (nivel 2), nao sobre a serializacao.")
    a = p.parse_args()
    pa, pb = Path(a.a), Path(a.b)
    for x in (pa, pb):
        if not x.exists():
            print(f"RECUSADO: {x} nao existe.", file=sys.stderr)
            return 2

    ta, tb = pa.stat().st_size, pb.stat().st_size
    print(f"A  {pa}\n   {ta:,} B")
    print(f"B  {pb}\n   {tb:,} B\n")

    print("--- nivel 1: bytes do arquivo inteiro ---")
    ha = sha_arquivo(pa)
    hb = sha_arquivo(pb)
    n1 = ha == hb
    print(f"  A sha256 {ha}")
    print(f"  B sha256 {hb}")
    print(f"  {'IDENTICOS' if n1 else 'DIFERENTES'}\n")

    print("--- nivel 3: chaves, dtypes e shapes ---")
    ca, cb = sha_tensores(pa), sha_tensores(pb)
    so_a, so_b = sorted(set(ca) - set(cb)), sorted(set(cb) - set(ca))
    comuns = sorted(set(ca) & set(cb))
    forma = [k for k in comuns if ca[k][1:] != cb[k][1:]]
    n3 = not so_a and not so_b and not forma
    print(f"  {len(ca)} e {len(cb)} tensores; so em A {len(so_a)}, so em B {len(so_b)}, "
          f"dtype/shape divergente {len(forma)}")
    for k in (so_a[:3] + so_b[:3] + forma[:3]):
        print(f"    {k}")
    print(f"  {'MESMA ESTRUTURA' if n3 else 'ESTRUTURA DIFERENTE'}\n")

    print("--- nivel 2: bytes de cada tensor ---")
    dif = [k for k in comuns if ca[k][0] != cb[k][0]]
    n2 = n3 and not dif
    print(f"  {len(comuns) - len(dif)}/{len(comuns)} tensores byte a byte identicos")
    for k in dif[:12]:
        print(f"    MUDOU  {k}  {ca[k][2]}")
    if len(dif) > 12:
        print(f"    ... e outros {len(dif) - 12}")
    print(f"  {'TODOS IDENTICOS' if not dif else f'{len(dif)} MUDARAM'}\n")

    ma = cabecalho(pa)[0].get("__metadata__")
    mb = cabecalho(pb)[0].get("__metadata__")
    print(f"--- __metadata__ ---\n  A {ma}\n  B {mb}")
    if n2 and not n1:
        print("\n  LEITURA: nenhum peso mudou; o arquivo difere por SERIALIZACAO (ordem de chave, "
              "\n  alinhamento ou metadata). Escritores diferentes produzem isso e nao e defeito "
              "\n  do experimento -- mas o metadata acima e' a proveniencia, e ela DIFERE.")

    print("\n=== NAO COBERTO ===")
    print("  Nada sobre qualidade: isto compara bytes. Dois arquivos identicos podem ser ambos")
    print("  lixo, e dois diferentes podem render igual.")

    if a.esperado:
        ok = n2 if a.esperado == "identico" else bool(dif)
        print(f"\nCRITERIO: esperado {a.esperado} nos TENSORES -> {'PASSOU' if ok else 'FALHOU'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
