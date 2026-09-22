"""Deriva o mapa diffusers -> BFL do klein-4B por CONTEUDO, nao por nome. (EXECUTADO, CPU)

A BFL publica o mesmo transformer duas vezes: `transformer/diffusion_pytorch_model.safetensors` com
169 tensores em nomes diffusers, e `flux-2-klein-4b.safetensors` com 149 em nomes BFL. Sao os MESMOS
pesos -- medido: tamanhos 7.751.109.744 e 7.751.105.712, zero nomes em comum, e 149 < 169 porque a
nomenclatura BFL FUNDE tensores.

Isso e um par de Pedra de Roseta. Em vez de adivinhar o mapa por nome -- que e como eu errei tres
regras hoje -- casa-se por **sha256 dos bytes**:

  1. tensor BFL identico, byte a byte, a UM tensor diffusers          -> mapa 1 para 1
  2. tensor BFL igual a concatenacao de 2 ou 3 diffusers no eixo 0    -> mapa fundido, com a ORDEM
  3. nada casa                                                        -> reportado como NAO RESOLVIDO

O caso 3 vai impresso com nome e shape. **Um mapa parcial e um mapa parcial**, e quem usar isto sem
ler a contagem vai produzir um checkpoint que carrega e desenha lixo -- que e exatamente o modo de
falha que este repo registra para o `to_native.py` do Z-Image, onde `weight_scale` passava sem ser
renomeado e a camada carregava sem escala e sem erro.

Para que serve: aplicar o mapa ao Bonsai (que so existe em diffusers) e obter um arquivo que o
ComfyUI carrega -- o braco 2 da fila de GPU, que sem isto nao roda.

Nao cobre: nenhum kernel, nenhuma imagem, GPU nao tocada. E casamento de bytes entre dois arquivos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def cabecalho(caminho: Path) -> tuple[dict, int]:
    import mmap
    with open(caminho, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            n = int.from_bytes(mm[:8], "little")
            h = json.loads(mm[8:8 + n])
        finally:
            mm.close()
    h.pop("__metadata__", None)
    return h, 8 + n


class Leitor:
    def __init__(self, caminho: Path):
        import mmap
        self.f = open(caminho, "rb")  # noqa: SIM115 -- o mmap precisa do handle; fecha() e o dono
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.mm[:8], "little")
        self.header = json.loads(self.mm[8:8 + n])
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def bytes_de(self, k: str) -> memoryview:
        a, b = self.header[k]["data_offsets"]
        return memoryview(self.mm)[self.base + a:self.base + b]

    def sha(self, k: str) -> str:
        return hashlib.sha256(self.bytes_de(k)).hexdigest()

    def fecha(self):
        self.mm.close()
        self.f.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--diffusers", required=True)
    p.add_argument("--bfl", required=True)
    p.add_argument("--saida", required=True, help="JSON com o mapa derivado")
    p.add_argument("--max-fusao", type=int, default=4)
    a = p.parse_args()

    d = Leitor(Path(a.diffusers))
    b = Leitor(Path(a.bfl))
    print(f"diffusers {len(d.header)} tensores    BFL {len(b.header)} tensores")
    print(f"nomes em comum: {len(set(d.header) & set(b.header))}  (esperado 0)\n")

    print("indexando sha256 de cada tensor diffusers...", flush=True)
    por_sha: dict[str, list[str]] = defaultdict(list)
    for k in d.header:
        por_sha[d.sha(k)].append(k)
    print(f"  {len(por_sha)} sha distintos para {len(d.header)} tensores")
    dup = {s: v for s, v in por_sha.items() if len(v) > 1}
    if dup:
        print(f"  AVISO: {len(dup)} sha repetidos -- tensores identicos entre si, o mapa nesses")
        print("  casos e ambiguo por conteudo e vai marcado como tal.")

    # indice por shape, para tentar fusao no eixo 0
    por_forma: dict[tuple, list[str]] = defaultdict(list)
    for k, v in d.header.items():
        por_forma[tuple(v["shape"][1:])].append(k)

    mapa: dict[str, dict] = {}
    um_para_um = fundidos = ambiguos = nao = 0
    nao_resolvidos: list[tuple[str, list]] = []

    for k in sorted(b.header):
        s = b.sha(k)
        if s in por_sha:
            cands = por_sha[s]
            mapa[k] = {"tipo": "1-para-1", "de": cands[0],
                       "ambiguo": cands if len(cands) > 1 else None}
            um_para_um += 1
            ambiguos += 1 if len(cands) > 1 else 0
            continue
        # tenta fusao: concatenacao no eixo 0 de tensores com o MESMO resto de shape
        forma = b.header[k]["shape"]
        resto = tuple(forma[1:])
        achou = False
        if resto in por_forma:
            brutos = bytes(b.bytes_de(k))
            pool = [c for c in por_forma[resto] if d.header[c]["dtype"] == b.header[k]["dtype"]]
            # procura uma particao: acumula prefixos que casem os bytes na ordem
            for n_partes in range(2, a.max_fusao + 1):
                alvo = [c for c in pool if d.header[c]["shape"][0] * n_partes == forma[0]]
                if not alvo:
                    continue
                tam = len(brutos) // n_partes
                pedacos = [brutos[i * tam:(i + 1) * tam] for i in range(n_partes)]
                sha_pedacos = [hashlib.sha256(x).hexdigest() for x in pedacos]
                escolha = []
                for sp in sha_pedacos:
                    cand = [c for c in por_sha.get(sp, []) if c in alvo]
                    if not cand:
                        escolha = []
                        break
                    escolha.append(cand[0])
                if escolha:
                    mapa[k] = {"tipo": f"fundido-{n_partes}", "de": escolha, "eixo": 0}
                    fundidos += 1
                    achou = True
                    break
        if not achou:
            nao += 1
            nao_resolvidos.append((k, forma))

    print("\n=== MAPA DERIVADO ===")
    print(f"  1-para-1 por sha256 exato   {um_para_um}")
    print(f"    dos quais AMBIGUOS        {ambiguos}  (mais de um diffusers com o mesmo conteudo)")
    print(f"  fundidos (concat no eixo 0) {fundidos}")
    print(f"  NAO RESOLVIDOS              {nao}")
    for k, forma in nao_resolvidos[:20]:
        print(f"      {k}  {forma}  {b.header[k]['dtype']}")
    if nao > 20:
        print(f"      ... e outros {nao - 20}")

    cobertos = set()
    for v in mapa.values():
        if v["tipo"] == "1-para-1":
            cobertos.add(v["de"])
        else:
            cobertos.update(v["de"])
    sobrando = sorted(set(d.header) - cobertos)
    print(f"\n  tensores diffusers cobertos pelo mapa: {len(cobertos)}/{len(d.header)}")
    print(f"  diffusers que NENHUM tensor BFL usa:   {len(sobrando)}")
    for k in sobrando[:12]:
        print(f"      {k}  {d.header[k]['shape']}")

    Path(a.saida).write_text(json.dumps(
        {"mapa": mapa, "nao_resolvidos": [k for k, _ in nao_resolvidos],
         "diffusers_sobrando": sobrando}, indent=2), encoding="utf-8")
    print(f"\n  JSON em {a.saida}")

    print("\n=== NAO COBERTO ===")
    print("  Casamento de BYTES entre dois arquivos. Nenhum kernel, nenhuma imagem, GPU nao tocada.")
    print("  Um mapa com NAO RESOLVIDOS > 0 e PARCIAL: aplicar assim produz checkpoint que carrega")
    print("  e desenha lixo, sem erro -- o mesmo modo de falha que o `to_native.py` do Z-Image tem")
    print("  registrado neste repo (scale passando sem renomear, camada sem escala e sem erro).")
    print("  Fusao testada so como concatenacao no eixo 0 em partes IGUAIS; qkv com cabecas de")
    print("  tamanho diferente, permutacao interna ou transposicao NAO seriam detectadas aqui.")
    d.fecha()
    b.fecha()
    return 1 if nao else 0


if __name__ == "__main__":
    raise SystemExit(main())
