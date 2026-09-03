r"""A alavanca do `convrot_groupsize`: ela anda, e para que lado? A celula vazia do Z-Image.

A PERGUNTA
----------
A tabela de tolerancia publicada por esta bancada tem uma celula vazia:

    modelo               parametros   tolerado   NAO tolerado
    Wan 2.1 VACE             1,3 B     0,0546        0,0793
    Z-Image v2                ~6 B     0,1241     NAO MEDIDO
    HunyuanVideo 1.5         ~13 B     0,1837        0,2147

O `zimage-v2-w4a4` ja e 170 de 170 camadas em 4 bits e a imagem presta, entao nao existe build mais
agressivo no eixo da promocao -- nao ha o que promover. A unica alavanca restante e a granularidade
da rotacao de Hadamard: grupo MAIOR = rotacao mais grossa = menos capacidade de espalhar outlier =
mais erro esperado.

O tamanho tem de ser potencia de 4, entao os degraus em volta do 256 usado hoje sao 64 e 1024.

A ARMADILHA DO CONJUNTO DE CAMADAS, E POR QUE ESTA FERRAMENTA EXISTE
----------------------------------------------------------------------
Rodando o conversor nos tres valores, ele seleciona **populacoes diferentes**: uma camada so entra
se `shape[1] % convrot_groupsize == 0`. Medido: 170 camadas em cg 64, 170 em cg 256 e **34** em
cg 1024. Comparar a mediana de 170 camadas com a mediana de 34 outras nao mede granularidade -- mede
qual subconjunto calhou de ser divisivel, que e o eixo errado segurado no lugar certo.

Aqui o conjunto e a INTERSECAO: so as camadas que os tres valores aceitam, medidas nas mesmas
ativacoes, contra a mesma referencia float32. Um eixo varia, o `convrot_groupsize`, e nada mais.

O CONTROLE
----------
`cg 64` tem de medir ABAIXO de `cg 256`. Uma alavanca que so anda para um lado nao foi demonstrada:
sem esse braco, um `cg 1024` que sobe poderia estar subindo por qualquer motivo -- um erro meu de
chamada, por exemplo -- e eu leria como confirmacao da hipotese.

NAO COBERTO
-----------
Erro de saida por camada nas ativacoes calibradas. **Nao imagem**, e nesta bancada nenhum corte
nesse eixo separou usavel de inutilizavel: 0,1837 correto contra 0,2147 destruido no Hunyuan.
Isto diz se a alavanca existe e para que lado ela anda; a foto e outro passo, e e ela que decide.
Um modelo, uma calibragem, uma placa.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import struct
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

import comfy_kitchen as ck
import torch
import torch.nn.functional as F


def cabecalho(modelo: Path) -> tuple[dict, int]:
    with modelo.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def ler_peso(modelo: Path, head: dict, base: int, chave: str) -> torch.Tensor:
    info = head[chave + ".weight"]
    with modelo.open("rb") as f:
        a, b = info["data_offsets"]
        f.seek(base + a)
        cru = bytearray(f.read(b - a))
    return torch.frombuffer(cru, dtype=torch.bfloat16).reshape(info["shape"]).cuda()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", type=Path,
                   default=RAIZ / "ComfyUI/models/diffusion_models/"
                                  "beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--calib", type=Path, default=RAIZ / "calib/zimage_v2_sigma.calib.pt")
    p.add_argument("--grupos", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--saida", type=Path, default=RAIZ / "bench" / "teto_groupsize.json")
    a = p.parse_args()

    head, base = cabecalho(a.modelo)
    d = torch.load(a.calib, map_location="cpu", weights_only=False)
    maior = max(a.grupos)

    # A INTERSECAO, montada antes de medir qualquer coisa: uma camada so entra se TODOS os valores
    # a aceitam. Sem isto os bracos mediriam populacoes diferentes.
    chaves = []
    for k in d["layers"]:
        if not k.startswith("layers."):
            continue
        nome = k + ".weight"
        if nome not in head:
            continue
        forma = head[nome]["shape"]
        if d["layers"][k].get("sample") is None:
            continue
        if all(forma[1] % g == 0 for g in a.grupos):
            chaves.append(k)
    print(f"{len(chaves)} camadas aceitas por TODOS os grupos {a.grupos} "
          f"(a maior restricao e {maior})")
    if not chaves:
        print("intersecao vazia: nada a medir", file=sys.stderr)
        return 2

    cols = [f"cg{g}" for g in a.grupos]
    cab = f"{'camada':30s} " + " ".join(f"{c:>9s}" for c in cols)
    print()
    print(cab)
    print("-" * len(cab))
    acc = {c: [] for c in cols}
    linhas = []
    for k in chaves:
        x = d["layers"][k]["sample"].cuda().to(torch.bfloat16)
        w = ler_peso(a.modelo, head, base, k)
        if x.shape[-1] != w.shape[1]:
            continue
        ref = F.linear(x.float(), w.float())
        nrm = ref.norm()
        linha = {}
        for g, c in zip(a.grupos, cols):
            q, s = ck.quantize_convrot_w4a4_weight(w, g, a.group_size)
            got = ck.convrot_w4a4_linear(x, q, s, None, g, a.group_size)
            linha[c] = ((got.float() - ref).norm() / nrm).item()
            acc[c].append(linha[c])
            del q, s, got
        linhas.append({"layer": k, "shape": list(w.shape), **linha})
        print(f"{k:30s} " + " ".join(f"{linha[c]:9.4f}" for c in cols))
        del ref, w, x
        torch.cuda.empty_cache()

    med = {c: st.median(v) for c, v in acc.items()}
    print()
    print(f"{'MEDIANA':30s} " + " ".join(f"{med[c]:9.4f}" for c in cols))

    # CONTROLE: a alavanca tem de andar nos DOIS sentidos em torno do 256 de hoje.
    print()
    base_c = "cg256"
    if base_c in med:
        menor = [c for c in cols if int(c[2:]) < 256]
        maiores = [c for c in cols if int(c[2:]) > 256]
        ok_baixo = all(med[c] < med[base_c] for c in menor) if menor else None
        ok_alto = all(med[c] > med[base_c] for c in maiores) if maiores else None
        for c in cols:
            if c == base_c:
                continue
            r = med[c] / med[base_c]
            direcao = "MAIS erro" if r > 1 else "menos erro"
            print(f"  {c:>7s} contra cg256: {max(r, 1/r):.3f}x {direcao}")
        print()
        if ok_baixo is False:
            print("CONTROLE FALHOU: grupo menor NAO reduz o erro. A alavanca nao foi demonstrada,")
            print("  e um grupo maior que suba nao prova nada -- pode ser erro de chamada minha.")
        elif ok_alto is False:
            print("A HIPOTESE MORREU no sentido esperado: grupo maior NAO aumenta o erro.")
            print("  O controle de baixo passou, entao a alavanca existe e anda para o outro lado.")
        elif ok_baixo and ok_alto:
            print("CONTROLE PASSOU nos dois sentidos: menor->menos erro, maior->mais erro.")

    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(
        {"camadas": linhas, "mediana": med, "grupos": a.grupos,
         "n_camadas": len(linhas), "group_size": a.group_size,
         "modelo": str(a.modelo), "calib": str(a.calib)}, indent=2), encoding="utf-8")
    print()
    print("NAO COBERTO: erro de saida por camada nas ativacoes calibradas, NAO imagem. Nesta")
    print("  bancada nenhum corte nesse eixo separou usavel de inutilizavel. Isto diz se a")
    print("  alavanca existe e para que lado anda; so o render decide se quebra. Um modelo, uma")
    print("  calibragem, uma placa. A intersecao restringe as camadas -- a mediana aqui NAO e")
    print("  comparavel com a mediana publicada sobre as 170.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
