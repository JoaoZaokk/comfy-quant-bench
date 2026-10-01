"""Quem fica denso num release de baixo bit, e o denso foi REESCRITO? (EXECUTADO)

Duas perguntas, e a segunda e a que separa PTQ de treino sem precisar desempacotar bit nenhum:

  1. **A regra de selecao vale aqui?** Medida no FLUX.2-klein-4B por dois times independentes
     (Prism/Bonsai treinando a 1,58 bit, Nunchaku/SVDQuant fazendo PTQ int4), intersecao 69 de 69:
     fica denso tudo que e 1-D e todo 2-D FORA de uma pilha de blocos numerada. Este script testa a
     regra noutro release em vez de assumi-la.

  2. **Quantos dos densos sao byte a byte identicos ao original?** Um PTQ nao tem motivo para
     reescrever uma camada que ele nao quantizou. Medido no klein-4B: SVDQuant **66 de 69**, Bonsai
     **1 de 69**. As 3 do SVDQuant sao as modulacoes, que e o que dobrar um `smooth_factor` por canal
     exige. E a assinatura mais barata de treino que existe: nao precisa de layout, de escala, nem de
     desempacotamento -- so de `torch.equal`.

Le por mmap SOMENTE LEITURA: custa 0 de commit, ao contrario de `safe_open`, que na maquina desta
bancada cobra 2x o arquivo so para abrir.

Nao cobre: nenhum kernel, nenhuma imagem, nenhuma ativacao. E identidade de bytes e topologia de
nome. Um denso identico nao diz que a camada presta; um denso mudado nao diz COMO mudou.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")
# sufixos que sao o peso quantizado, por familia de quantizador
SUF_QUANT = ("qweight", "W_q", "weight_scale", "scales", "wscales", "comfy_quant", "qzeros")


def pilhas_reais(nomes) -> set[str]:
    """Uma pilha e um prefixo com >= 2 indices distintos: `adaLN_modulation.1` nao e pilha."""
    ind: dict[str, set[str]] = {}
    for k in nomes:
        m = BLOCO.match(k)
        if m:
            ind.setdefault(m.group("pilha"), set()).add(m.group("i"))
    return {n for n, i in ind.items() if len(i) >= 2}


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from probe_bonsai_mlx_vs_unpacked import Arquivo

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quantizado", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--esperado-densos", type=int, default=None, help="calibracao")
    p.add_argument("--esperado-identicos", type=int, default=None, help="calibracao")
    a = p.parse_args()

    q = Arquivo(Path(a.quantizado))
    o = Arquivo(Path(a.original))
    print(f"quantizado {Path(a.quantizado).name}  {len(q.header)} tensores")
    print(f"original   {Path(a.original).name}  {len(o.header)} tensores")

    # quem esta quantizado: a camada tem algum sufixo de quantizacao
    quant = {k.rsplit(".", 1)[0] for k in q.header if k.rsplit(".", 1)[-1] in SUF_QUANT}
    # quem ficou denso: `.weight` em dtype flutuante cuja camada NAO esta quantizada
    denso = [k for k, v in q.header.items()
             if k.endswith(".weight") and v["dtype"] in ("BF16", "F16", "F32")
             and k.rsplit(".", 1)[0] not in quant]
    print(f"\n  camadas quantizadas {len(quant)}    tensores densos (.weight flutuante) {len(denso)}")

    if not quant:
        print("\n  NENHUMA camada quantizada reconhecida. Sufixos vistos no arquivo:")
        import collections
        c = collections.Counter(k.rsplit(".", 1)[-1] for k in q.header)
        for s, n in c.most_common(12):
            print(f"      {n:6d}  .{s}")
        print("  Este script so reconhece " + ", ".join(SUF_QUANT) + " -- se o release usa outro")
        print("  nome, o veredito abaixo NAO vale e a lista acima diz o que adicionar.")
        q.fecha(), o.fecha()
        return 2

    # PERGUNTA 1: a regra de selecao vale?
    pilhas = pilhas_reais(q.header)
    dentro_mas_denso = [k for k in denso
                        if (m := BLOCO.match(k)) and m.group("pilha") in pilhas
                        and len(q.header[k]["shape"]) == 2]
    fora_mas_quant = [k for k in quant
                      if not ((m := BLOCO.match(k)) and m.group("pilha") in pilhas)]
    print("\n=== PERGUNTA 1: a regra 'denso = 1-D + 2-D fora dos blocos' vale? ===")
    print(f"  2-D DENTRO de bloco e ainda denso (viola):   {len(dentro_mas_denso)}")
    for k in dentro_mas_denso[:6]:
        print(f"      {k}  {q.header[k]['shape']}")
    print(f"  quantizada FORA de bloco (viola):            {len(fora_mas_quant)}")
    for k in fora_mas_quant[:6]:
        print(f"      {k}")
    vale = not dentro_mas_denso and not fora_mas_quant
    print(f"  -> regra {'VALE neste release' if vale else 'NAO VALE aqui'}")

    # PERGUNTA 2: o denso foi reescrito?
    ig = dif = ausente = 0
    mudados: list[tuple[str, float]] = []
    for k in denso:
        if k not in o.header or tuple(o.header[k]["shape"]) != tuple(q.header[k]["shape"]):
            ausente += 1
            continue
        wo = o.tensor(k)
        wq = q.tensor(k)
        if torch.equal(wq, wo):
            ig += 1
        else:
            dif += 1
            a_, b_ = wq.float(), wo.float()
            mudados.append((k, float((a_ - b_).norm() / b_.norm().clamp(min=1e-30))))
    comparaveis = ig + dif
    casam = comparaveis / len(denso) if denso else 0.0
    print("\n=== PERGUNTA 2: quantos densos sao byte a byte identicos ao original? ===")
    print(f"  densos com par no original (nome E shape): {comparaveis}/{len(denso)}  "
          f"{casam*100:.1f}%   <- concordancia de arquitetura")
    print(f"  identicos          {ig}/{comparaveis}"
          f"{f'  ({ig/comparaveis*100:.1f}%)' if comparaveis else ''}")
    print(f"  MUDADOS            {dif}/{comparaveis}")
    print(f"  sem par no original (nome ou shape): {ausente}")
    for k, r in sorted(mudados, key=lambda t: -t[1])[:10]:
        print(f"      rel-L2 {r:8.5f}  {k}")
    print()
    # GUARDA DE ARQUITETURA. Sem ela este script da veredito CONFIANTE E ERRADO em par trocado:
    # apontado para o SVDQuant do Qwen contra o original do klein-4B, 21 de 247 densos "casaram"
    # por nome e shape -- `norm_out.linear`, `proj_out`, `img_in`, `txt_in`, `time_text_embed`,
    # nomes genericos que as duas arquiteturas compartilham -- todos os 21 diferiam, e a saida
    # dizia "o denso foi REESCRITO, um PTQ nao faz isso". Era o controle que devia dar zero, e ele
    # nao deu zero: deu um veredito de treino sobre dois modelos sem parentesco. O controle nao
    # "passou", ele expos o defeito, que e o unico motivo de ele existir.
    if casam < 0.90:
        print(f"  SEM VEREDITO: so {casam*100:.1f}% dos densos tem par no original. Abaixo de 90%")
        print("  isto nao e 'o denso mudou', e sim DOIS MODELOS DIFERENTES -- nomes genericos como")
        print("  `proj_out.weight` e `norm_out.linear.weight` casam entre arquiteturas que nao tem")
        print("  parentesco nenhum. Confira se `--original` e de fato a base deste release.")
    elif comparaveis == 0:
        print("  SEM VEREDITO: nenhum denso teve par comparavel no original. Isto normalmente")
        print("  significa nomenclatura diferente entre os dois arquivos, nao ausencia de mudanca.")
    elif ig / comparaveis >= 0.90:
        print(f"  LEITURA: o denso esta praticamente intacto ({ig/comparaveis*100:.1f}%). Assinatura")
        print("  de PTQ: quem nao quantizou a camada nao tinha motivo para reescreve-la.")
    else:
        print(f"  LEITURA: o denso foi REESCRITO ({dif} de {comparaveis}). Um PTQ nao faz isso. Isto")
        print("  e o que se espera de quem TREINOU com o conjunto denso solto como capacidade de")
        print("  adaptacao -- e e o mesmo padrao medido no Bonsai (1 de 69 intacto).")

    falha = 0
    if a.esperado_densos is not None and len(denso) != a.esperado_densos:
        print(f"\n  CALIBRACAO FALHOU: {len(denso)} densos, esperado {a.esperado_densos}")
        falha = 1
    if a.esperado_identicos is not None and ig != a.esperado_identicos:
        print(f"  CALIBRACAO FALHOU: {ig} identicos, esperado {a.esperado_identicos}")
        falha = 1
    if not falha and a.esperado_densos is not None:
        print("\n  CALIBRACAO OK: reproduz os numeros medidos no klein-4B.")

    print("\n=== NAO COBERTO ===")
    print("  Identidade de bytes e topologia de nome. Nenhum kernel, nenhuma imagem, nenhuma")
    print("  ativacao, GPU nao tocada. Um denso identico nao diz que a camada presta, e um denso")
    print("  mudado nao diz COMO mudou -- pode ser treino, pode ser dobra de smooth factor.")
    print("  Releases com sufixo de quantizacao fora da lista SUF_QUANT saem como 'sem veredito'.")
    q.fecha()
    o.fecha()
    return falha


if __name__ == "__main__":
    raise SystemExit(main())
