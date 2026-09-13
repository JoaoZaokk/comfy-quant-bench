"""Mede uma edicao pelo que ela PRESERVOU, nao so pelo que mudou.

POR QUE ESTA METRICA, E NAO A DISTANCIA DE SEMPRE
-------------------------------------------------
Num gerador, comparar dois bracos e comparar duas saidas. Num EDITOR ha uma terceira imagem que
manda: a entrada. Uma edicao pode obedecer a instrucao perfeitamente e ainda estar errada, porque
destruiu o que devia ficar parado -- e foi exatamente isso que apareceu no primeiro teste aqui, a
maca virando pera com a textura da mesa virando mosaico.

Entao a comparacao util tem duas metades:

  MUDOU   distancia entre a saida e a ENTRADA. Uma edicao que nao muda nada desobedeceu.
  DERIVOU distancia entre os dois bracos, na regiao que NENHUM dos dois deveria ter tocado.

A segunda e aproximada por construcao e diz isso: nao existe mascara da regiao editada. Como
proxy, usa-se o quartil de pixels onde os DOIS bracos mudaram MENOS em relacao a entrada -- a
area que ambos concordaram em deixar quieta. Se os bracos divergem ali, a divergencia nao e a
edicao, e dano.

NAO COBERTO
-----------
Nao julga se a instrucao foi obedecida: isso precisa de alguem olhando. Nao usa metrica
perceptual. A regiao "quieta" e inferida dos proprios bracos, entao se os DOIS destruirem a mesma
area a metrica nao acusa -- ela mede divergencia entre bracos, nao dano absoluto. Para dano
absoluto seria preciso o braco BF16, que nesta maquina nao roda neste grafo (ver
bench/qwen_edit_bf16_inalcancavel.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def carrega(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saidas", default="ComfyUI/output")
    p.add_argument("--entradas", default="ComfyUI/input")
    p.add_argument("--bracos", nargs="+", default=["int8", "w4a8"])
    p.add_argument("--pares", nargs="+", default=["pera:edit_maca.png",
                                                  "cachecol:edit_pescador.png",
                                                  "closed:edit_placa.png"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--json", default=None)
    a = p.parse_args()

    saidas, entradas = Path(a.saidas), Path(a.entradas)
    linhas, registro = [], {}

    for par in a.pares:
        ident, nome_entrada = par.split(":", 1)
        entrada = carrega(entradas / nome_entrada)
        for semente in a.seeds:
            imgs = {}
            for braco in a.bracos:
                achados = sorted(saidas.glob(f"qedit_{braco}_{ident}_s{semente}_*.png"))
                if achados:
                    imgs[braco] = carrega(achados[-1])
            if len(imgs) < len(a.bracos):
                linhas.append((ident, semente, None, None, None,
                               f"faltam {set(a.bracos) - set(imgs)}"))
                continue

            # a entrada e reescalada para 1 MP pelo grafo; compara no tamanho da saida
            h, w = next(iter(imgs.values())).shape[:2]
            ent = np.asarray(Image.fromarray(entrada.astype(np.uint8)).resize((w, h),
                                                                             Image.LANCZOS),
                             dtype=np.float32)

            mudou = {b: float(np.abs(v - ent).mean()) for b, v in imgs.items()}
            # regiao quieta: onde AMBOS mudaram menos (quartil inferior do maximo por pixel)
            delta = np.maximum.reduce([np.abs(v - ent).mean(axis=2) for v in imgs.values()])
            corte = np.quantile(delta, 0.25)
            quieta = delta <= corte
            b1, b2 = a.bracos[0], a.bracos[1]
            dif = np.abs(imgs[b1] - imgs[b2]).mean(axis=2)
            derivou_quieta = float(dif[quieta].mean())
            derivou_tudo = float(dif.mean())
            linhas.append((ident, semente, mudou[b1], mudou[b2], derivou_quieta,
                           f"{derivou_tudo:.2f} global"))
            registro[f"{ident}_s{semente}"] = {
                "mudou": mudou, "derivou_regiao_quieta": derivou_quieta,
                "derivou_global": derivou_tudo, "corte_quartil": float(corte)}

    b1, b2 = a.bracos[0], a.bracos[1]
    print(f"{'par':<10}{'sem':>4}{f'mudou {b1}':>13}{f'mudou {b2}':>13}"
          f"{'derivou quieta':>16}   nota")
    for ident, s, m1, m2, dq, nota in linhas:
        if m1 is None:
            print(f"{ident:<10}{s:>4}{'--':>13}{'--':>13}{'--':>16}   {nota}")
        else:
            print(f"{ident:<10}{s:>4}{m1:>13.2f}{m2:>13.2f}{dq:>16.2f}   {nota}")

    if registro:
        vals = [v["derivou_regiao_quieta"] for v in registro.values()]
        print(f"\nderivacao media na regiao que ambos deixaram quieta: {np.mean(vals):.2f} "
              f"(min {min(vals):.2f} max {max(vals):.2f}) em {len(vals)} pares")
    if a.json:
        Path(a.json).write_text(json.dumps(registro, indent=2), encoding="utf-8")
        print(f"registro em {a.json}")

    print("\nNAO COBERTO: nao diz se a instrucao foi obedecida -- isso precisa de olho. A regiao "
          "'quieta' e inferida dos proprios bracos, entao dano que os DOIS causem na mesma area "
          "nao aparece. Sem braco BF16 neste grafo (ver bench/qwen_edit_bf16_inalcancavel.md), "
          "isto mede divergencia ENTRE bracos, nunca distancia ate o original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
