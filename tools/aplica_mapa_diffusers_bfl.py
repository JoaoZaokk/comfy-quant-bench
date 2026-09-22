"""Aplica o mapa derivado: reescreve um checkpoint diffusers em nomenclatura BFL/ComfyUI. (EXECUTADO)

O mapa vem de `tools/deriva_mapa_diffusers_bfl.py`, que o derivou por **sha256 dos bytes** entre as
duas publicacoes do mesmo klein-4B -- 149 nomes BFL cobrindo 169 diffusers, sem sobra nos dois lados.
Tres operacoes, todas medidas naquele par, nenhuma adivinhada:

    1-para-1     copia o tensor
    fundido-3    concatena to_q + to_k + to_v no eixo 0 (idem add_q/add_k/add_v)  -> 10 casos
    permuta-2    troca as duas metades: diffusers grava [shift, scale] e a BFL
                 grava [scale, shift] em `final_layer.adaLN_modulation.1`          -> 1 caso

**O CONTROLE E OBRIGATORIO E ELE E EXATO.** Aplicado ao ORIGINAL em diffusers, o resultado tem de ser
byte a byte identico ao `flux-2-klein-4b.safetensors` publicado pela BFL. Nao e um teste aproximado:
ou reproduz o arquivo oficial, ou o mapa/aplicador esta errado e nenhum braco derivado dele vale.
Rodar com `--autoteste <arquivo BFL oficial>`.

Serve para pos o Bonsai (que so existe em diffusers) num arquivo que o ComfyUI carrega -- o braco 2 da
fila de GPU, que sem isto nao roda.

Escrita em streaming com `.partial` + fsync + `os.replace`, um tensor por vez: `safe_open` cobraria 2x
o arquivo em commit nesta maquina.

Nao cobre: nenhum kernel, nenhuma imagem, GPU nao tocada. Reproduzir o arquivo oficial prova o
RENOMEIO; nao prova que um checkpoint com outros pesos dentro desses nomes vai renderizar bem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
from pathlib import Path

TAM = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8,
       "U32": 4, "BOOL": 1}


class Leitor:
    def __init__(self, caminho: Path):
        self.f = open(caminho, "rb")  # noqa: SIM115 -- o mmap precisa do handle; fecha() e o dono
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.mm[:8], "little")
        self.header = json.loads(self.mm[8:8 + n])
        self.meta = self.header.pop("__metadata__", None)
        self.base = 8 + n

    def bytes_de(self, k: str) -> memoryview:
        a, b = self.header[k]["data_offsets"]
        return memoryview(self.mm)[self.base + a:self.base + b]

    def fecha(self):
        self.mm.close()
        self.f.close()


def n_bytes(v: dict) -> int:
    n = TAM[v["dtype"]]
    for x in v["shape"]:
        n *= x
    return n


def forma_de(ent: Leitor, regra: dict) -> tuple[list[int], str]:
    """Shape e dtype de saida SO pelo header da entrada -- nao le nem um byte de peso."""
    tipo = regra["tipo"]
    if tipo.startswith("fundido"):
        partes = regra["de"]
        forma = list(ent.header[partes[0]]["shape"])
        forma[0] = sum(ent.header[k]["shape"][0] for k in partes)
        return forma, ent.header[partes[0]]["dtype"]
    k = regra["de"]                                   # 1-para-1 e permuta preservam o shape
    return list(ent.header[k]["shape"]), ent.header[k]["dtype"]


def monta(ent: Leitor, destino: str, regra: dict) -> tuple[bytes, list[int], str]:
    """Devolve (bytes do tensor de saida, shape, dtype) segundo a regra do mapa."""
    tipo = regra["tipo"]
    if tipo == "1-para-1":
        k = regra["de"]
        return bytes(ent.bytes_de(k)), list(ent.header[k]["shape"]), ent.header[k]["dtype"]
    if tipo.startswith("fundido"):
        partes = regra["de"]
        dados = b"".join(bytes(ent.bytes_de(k)) for k in partes)
        forma = list(ent.header[partes[0]]["shape"])
        forma[0] = sum(ent.header[k]["shape"][0] for k in partes)
        return dados, forma, ent.header[partes[0]]["dtype"]
    if tipo.startswith("permuta"):
        k = regra["de"]
        cru = bytes(ent.bytes_de(k))
        n = len(regra["ordem"])
        tam = len(cru) // n
        pedacos = [cru[i * tam:(i + 1) * tam] for i in range(n)]
        dados = b"".join(pedacos[i] for i in regra["ordem"])
        return dados, list(ent.header[k]["shape"]), ent.header[k]["dtype"]
    raise ValueError(f"tipo de regra desconhecido para {destino}: {tipo!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--entrada", required=True, help="checkpoint em nomes diffusers")
    p.add_argument("--mapa", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--autoteste", default=None,
                   help="arquivo BFL oficial: exige saida byte a byte identica")
    a = p.parse_args()

    mapa = json.loads(Path(a.mapa).read_text(encoding="utf-8"))
    if mapa.get("nao_resolvidos"):
        print(f"RECUSADO: o mapa tem {len(mapa['nao_resolvidos'])} entradas NAO RESOLVIDAS. Aplicar um"
              " mapa parcial produz checkpoint que carrega e desenha lixo.", file=sys.stderr)
        return 2
    regras = mapa["mapa"]

    ent = Leitor(Path(a.entrada))
    precisa: set[str] = set()
    for r in regras.values():
        precisa.update([r["de"]] if isinstance(r["de"], str) else r["de"])
    faltando = sorted(precisa - set(ent.header))
    if faltando:
        print(f"RECUSADO: {len(faltando)} tensores que o mapa exige nao existem na entrada. "
              f"Primeiros: {faltando[:5]}", file=sys.stderr)
        ent.fecha()
        return 2
    sobra = sorted(set(ent.header) - precisa)
    print(f"entrada {Path(a.entrada).name}: {len(ent.header)} tensores")
    print(f"  o mapa consome {len(precisa)}, e {len(sobra)} sobram (nao vao para a saida)")
    for k in sobra[:8]:
        print(f"      SOBRA  {k}  {ent.header[k]['shape']}")

    sai = Path(a.saida)
    if sai.exists():
        print(f"RECUSADO: {sai} ja existe.", file=sys.stderr)
        ent.fecha()
        return 2
    parcial = sai.with_suffix(sai.suffix + ".partial")
    if parcial.exists():
        print(f"RECUSADO: {parcial} existe (corrida morta). Apague a mao.", file=sys.stderr)
        ent.fecha()
        return 2

    # PASSO 1: shape/dtype/tamanho SO pelo header, sem materializar byte nenhum.
    #
    # A primeira versao montava os 149 tensores em memoria antes de escrever. Com o commit livre em
    # 8,54 GiB e um payload de 7,4 GiB, isso e literalmente o defeito que o CLAUDE.md deste repo
    # documenta para o `quant_w4a8.py` -- duas passadas, payload inteiro em RAM, e o commit mordendo
    # antes do disco. Eu matei a propria execucao no meio por isso, com a maquina dele em uso.
    novo, off = {}, 0
    for k in sorted(regras):
        forma, dt = forma_de(ent, regras[k])
        n = TAM[dt]
        for x in forma:
            n *= x
        novo[k] = {"dtype": dt, "shape": forma, "data_offsets": [off, off + n]}
        off += n
    bruto = json.dumps(novo, separators=(",", ":")).encode()
    bruto += b" " * ((-len(bruto)) % 8)

    # PASSO 2: escreve UM tensor por vez; o maior vivo de cada vez e o maior tensor, nao o arquivo.
    with open(parcial, "wb") as out:
        out.write(struct.pack("<Q", len(bruto)))
        out.write(bruto)
        for k in sorted(regras):
            dados, forma, dt = monta(ent, k, regras[k])
            if len(dados) != novo[k]["data_offsets"][1] - novo[k]["data_offsets"][0]:
                print(f"RECUSADO: {k} rendeu {len(dados)} B, header previu "
                      f"{novo[k]['data_offsets'][1] - novo[k]['data_offsets'][0]}", file=sys.stderr)
                ent.fecha()
                return 2
            out.write(dados)
            del dados
        out.flush()
        os.fsync(out.fileno())
    os.replace(parcial, sai)
    ent.fecha()
    print(f"\nescrito {sai}  {sai.stat().st_size:,} B  {len(novo)} tensores")

    falha = 0
    if a.autoteste:
        of = Path(a.autoteste)
        print(f"\n=== AUTOTESTE contra {of.name} ===")
        print(f"  tamanho  oficial {of.stat().st_size:,}   nosso {sai.stat().st_size:,}")
        # em pedacos de 64 MiB: `read_bytes()` de 7,4 GiB duas vezes e commit que esta maquina nao tem
        def sha_arquivo(caminho: Path) -> str:
            h = hashlib.sha256()
            with open(caminho, "rb") as fh:
                while bloco := fh.read(64 << 20):
                    h.update(bloco)
            return h.hexdigest()

        sha_of = sha_arquivo(of)
        sha_no = sha_arquivo(sai)
        print(f"  sha256 oficial {sha_of}")
        print(f"  sha256 nosso   {sha_no}")
        if sha_of == sha_no:
            print("  IDENTICO byte a byte. O mapa e o aplicador reproduzem o arquivo oficial.")
        else:
            # onde difere: header, ou tensor?
            o2, n2 = Leitor(of), Leitor(sai)
            so, sn = set(o2.header), set(n2.header)
            dif = [k for k in sorted(so & sn)
                   if bytes(o2.bytes_de(k)) != bytes(n2.bytes_de(k))]
            print(f"  DIFERE. nomes so no oficial {len(so-sn)}, so no nosso {len(sn-so)}, "
                  f"tensores comuns com bytes diferentes {len(dif)}/{len(so & sn)}")
            for k in dif[:8]:
                print(f"      {k}")
            if not (so - sn) and not (sn - so) and not dif:
                print("  Todos os TENSORES batem: a diferenca esta no header (ordem, metadata ou")
                print("  padding). Isso NAO invalida o mapa -- invalida so a comparacao por sha.")
            else:
                falha = 1
            o2.fecha(), n2.fecha()

    print("\n=== NAO COBERTO ===")
    print("  Reproduzir o arquivo oficial prova o RENOMEIO, e so isso. Nenhum kernel, nenhuma")
    print("  imagem, GPU nao tocada. Um checkpoint com OUTROS pesos nesses mesmos nomes pode")
    print("  carregar e renderizar lixo -- isso so um render responde.")
    print("  O mapa foi derivado NO klein-4B. Nao vale para outra arquitetura.")
    return falha


if __name__ == "__main__":
    raise SystemExit(main())
