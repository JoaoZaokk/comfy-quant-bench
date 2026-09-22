"""O pack gemlite INT2 do Bonsai: o que esta DENTRO do arquivo de deploy, e em que precisao.

EXECUTADO. Abre `transformer-gemlite-int2/state_dict.pt` com `torch.load(mmap=True,
weights_only=True)` -- mmap para nao pagar o arquivo inteiro em RAM, `weights_only` porque e pickle
de terceiro e nao se executa codigo de terceiro para ler pesos.

Responde tres coisas que o README deles nao responde:

  1. Quais tensores existem por camada quantizada, em que dtype, e quantos bytes cada um. Isso diz
     se a escala e fp16 por grupo de 128 como eles afirmam, e se ha zero-point (assimetrico).
  2. Quanto do arquivo e peso de 2 bits e quanto e overhead. Eles publicam "1,21 GB representacao
     Bonsai" e "1,54 GB pack CUDA"; o arquivo tem 1.540.457.482 B. A conta de onde vem a diferenca
     sai daqui.
  3. As 9 camadas puladas aparecem no pack em que dtype? Se vierem em BF16/FP16 densas, confirma
     que o deploy carrega precisao mista, e da o tamanho real do "menos de 5% dos parametros".

Nao cobre: nao executa kernel nenhum, entao **nao diz se gemlite faz math de 2 bits em tensor core
ou desempacota para fp16 antes do GEMM**. Isso exige rodar o gemlite, que nao esta instalado aqui.
O que este script da e a ESTRUTURA do arquivo, nao o caminho de execucao.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch


def bytes_de(t) -> int:
    if torch.is_tensor(t):
        return t.numel() * t.element_size()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--autotune", default=None)
    p.add_argument("--saida", default=None)
    a = p.parse_args()

    caminho = Path(a.pack)
    tam = caminho.stat().st_size
    print(f"pack {caminho}")
    print(f"  {tam:,} B = {tam / 2**30:.3f} GiB")

    sd = torch.load(caminho, map_location="cpu", mmap=True, weights_only=True)
    print(f"  chaves de topo: {len(sd)}")

    # Achatar: o state_dict pode ser {fqn: {campo: tensor}} ou {fqn.campo: tensor}
    plano: dict[str, object] = {}
    def achata(d, prefixo=""):
        for k, v in d.items():
            nome = f"{prefixo}{k}"
            if isinstance(v, dict):
                achata(v, nome + ".")
            else:
                plano[nome] = v
    achata(sd)
    print(f"  entradas achatadas: {len(plano)}")

    # Que SUFIXOS existem (o vocabulario do formato), com dtype e bytes
    suf = defaultdict(lambda: {"n": 0, "bytes": 0, "dtypes": Counter(), "shapes": set()})
    nao_tensor = Counter()
    for nome, v in plano.items():
        s = nome.split(".")[-1]
        if not torch.is_tensor(v):
            nao_tensor[f"{s} :: {type(v).__name__}"] += 1
            continue
        d = suf[s]
        d["n"] += 1
        d["bytes"] += bytes_de(v)
        d["dtypes"][str(v.dtype)] += 1
        if len(d["shapes"]) < 3:
            d["shapes"].add(tuple(v.shape))

    print(f"\n=== VOCABULARIO DO FORMATO: um sufixo por linha ===")
    print(f"{'sufixo':28s} {'n':>5s} {'GiB':>8s} {'% do pack':>10s}  dtypes / shapes de exemplo")
    total_t = sum(d["bytes"] for d in suf.values())
    for s, d in sorted(suf.items(), key=lambda x: -x[1]["bytes"]):
        print(f"{s:28s} {d['n']:5d} {d['bytes']/2**30:8.4f} {100*d['bytes']/tam:9.2f}%  "
              f"{dict(d['dtypes'])}  {sorted(d['shapes'])[:2]}")
    print(f"{'TOTAL EM TENSOR':28s} {'':5s} {total_t/2**30:8.4f} {100*total_t/tam:9.2f}%")
    print(f"{'overhead do container':28s} {'':5s} {(tam-total_t)/2**30:8.4f} "
          f"{100*(tam-total_t)/tam:9.2f}%  (pickle, nomes, alinhamento)")
    if nao_tensor:
        print(f"\n  entradas que NAO sao tensor: {dict(nao_tensor)}")

    # Quantizada vs pulada: quem tem peso empacotado e quem veio denso
    PULADAS = ["x_embedder", "context_embedder", "proj_out", "time_guidance_embed",
               "double_stream_modulation_img", "double_stream_modulation_txt",
               "single_stream_modulation", "norm_out"]
    por_camada = defaultdict(dict)
    for nome, v in plano.items():
        if not torch.is_tensor(v):
            continue
        partes = nome.split(".")
        por_camada[".".join(partes[:-1])][partes[-1]] = v

    q, dens = [], []
    for fqn, campos in por_camada.items():
        eh_pulada = any(pat in fqn for pat in PULADAS)
        b = sum(bytes_de(v) for v in campos.values())
        (dens if eh_pulada else q).append((fqn, sorted(campos), b))

    print(f"\n=== CAMADAS ===")
    print(f"  com peso empacotado (nao-puladas): {len(q)}")
    print(f"  nas familias puladas:              {len(dens)}")
    if q:
        campos_q = Counter(tuple(c) for _, c, _ in q)
        for c, n in campos_q.most_common():
            print(f"    {n:4d}x campos {list(c)}")
        bq = sum(b for _, _, b in q)
        print(f"    bytes: {bq/2**30:.4f} GiB ({100*bq/tam:.2f}% do pack)")
    if dens:
        campos_d = Counter(tuple(c) for _, c, _ in dens)
        for c, n in campos_d.most_common():
            print(f"    {n:4d}x campos {list(c)}   <- familia pulada")
        bd = sum(b for _, _, b in dens)
        print(f"    bytes: {bd/2**30:.4f} GiB ({100*bd/tam:.2f}% do pack)")
        print(f"    (o README deles diz 'menos de 5% dos parametros' em FP16)")

    # Bits efetivos por peso, se der para inferir
    pesos = [(f, c, b) for f, c, b in q]
    if pesos:
        print(f"\n=== EXEMPLO DETALHADO DE UMA CAMADA EMPACOTADA ===")
        fqn, campos, _ = sorted(pesos, key=lambda x: -x[2])[0]
        for campo in campos:
            v = por_camada[fqn][campo]
            print(f"  {fqn}.{campo:22s} {str(v.dtype):16s} {tuple(v.shape)}  "
                  f"{bytes_de(v):,} B  min {float(v.flatten()[:4096].float().min()):.4g} "
                  f"max {float(v.flatten()[:4096].float().max()):.4g}")

    if a.autotune:
        at = json.loads(Path(a.autotune).read_text(encoding="utf-8"))
        print(f"\n=== gemlite_autotune.json ===")
        print(f"  chaves de topo: {len(at)}")
        amostra = list(at.items())[:3]
        for k, v in amostra:
            print(f"  {k[:110]}")
            print(f"    -> {json.dumps(v)[:200]}")
        shapes = sorted({m.group(0) for k in at for m in [re.search(r"\d+_\d+_\d+", k)] if m})
        print(f"  shapes distintos citados nas chaves: {len(shapes)}  {shapes[:6]}")

    print(f"\n=== NAO COBERTO ===")
    print("  Nenhum kernel executado. Este script NAO diz se gemlite faz GEMM de 2 bits em tensor")
    print("  core ou desempacota para fp16 antes -- que e a pergunta que decide se o ganho deles e")
    print("  memoria ou tempo. Para responder, teria de instalar gemlite e contar o despacho, como")
    print("  `probe_quant_dispatch.py` faz aqui para o ConvRot.")
    print("  O pack binario (INT1) nao foi aberto, nem o MLX.")

    if a.saida:
        Path(a.saida).write_text(json.dumps({
            "pack": str(caminho), "bytes": tam,
            "sufixos": {s: {"n": d["n"], "bytes": d["bytes"], "dtypes": dict(d["dtypes"]),
                            "shapes": [list(x) for x in sorted(d["shapes"])]}
                        for s, d in suf.items()},
            "total_em_tensor": total_t,
            "camadas_empacotadas": len(q), "camadas_em_familia_pulada": len(dens),
        }, indent=2), encoding="utf-8")
        print(f"\n  JSON em {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
