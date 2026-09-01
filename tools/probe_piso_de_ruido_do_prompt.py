"""Quantizar o encoder custa mais ou menos que reescrever o prompt?

A PERGUNTA, e ela veio do dono da bancada: se o encoder quantizado devolve condicionamento com
cosseno 0,99 contra o BF16, o que isso diz sobre difusores?

A HIPOTESE que isto testa. Um difusor e treinado com muitas legendas diferentes descrevendo
imagens parecidas, entao ele PRECISOU aprender que vetores diferentes podem querer dizer a mesma
coisa. Se isso for verdade existe um piso de ruido que ele ja engole todo dia -- e da para medir
qual e, sem tocar no difusor: e a distancia entre duas maneiras de escrever o MESMO pedido, no
mesmo encoder BF16, sem quantizacao nenhuma envolvida.

    quantizacao   |  mesmo prompt, encoder BF16 contra encoder de 4 bits
    parafrase     |  prompts diferentes com o mesmo sentido, os DOIS em BF16

Se a parafrase custar MAIS que a quantizacao, entao o erro da quantizacao esta dentro da variacao
que o modelo ja recebe de usuarios escrevendo a mesma coisa de jeitos diferentes. Se custar MENOS,
a quantizacao introduz um tipo de mudanca que o difusor nunca viu, e o argumento cai.

    python_embeded\\python.exe -s tools/probe_piso_de_ruido_do_prompt.py

TERCEIRO EIXO, de controle: prompts com sentidos DIFERENTES. Se parafrase e sentido-diferente
derem a mesma distancia, entao esta metrica nao esta medindo sentido nenhum e a comparacao toda
perde o chao. E o controle que impede este probe de se enganar sozinho.

NAO COBERTO: distancia de cosseno entre condicionamentos nao e "o difusor nota a diferenca" --
para saber isso e preciso gerar, com trajetoria imposta, e isto aqui nao gera nada. As parafrases
sao escritas a mao neste arquivo e sao um julgamento meu sobre o que "quer dizer o mesmo". E um
encoder, uma familia de modelo.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

_ARGV = sys.argv[:]
sys.argv = ["main.py", "--disable-dynamic-vram"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
sys.argv = _ARGV

import torch  # noqa: E402

# Cada grupo: o mesmo pedido escrito de jeitos diferentes. O primeiro de cada grupo e o canonico.
PARAFRASES = {
    "maca": [
        "a red apple on a weathered wooden table, soft window light",
        "a red apple sitting on an old wooden table, gentle light from a window",
        "soft window light falling on a red apple that rests on weathered wood",
        "on a worn wooden surface, one red apple, lit softly from a nearby window",
    ],
    "pescador": [
        "portrait of an elderly fisherman, weathered face, golden hour",
        "an old fisherman photographed at golden hour, his face lined and weathered",
        "close portrait of an aged fisherman with a weather-beaten face, warm evening light",
    ],
}

# Controle: sentidos DIFERENTES. Se isto der a mesma distancia que parafrase, a metrica nao mede
# sentido e a comparacao inteira perde o chao.
DIFERENTES = [
    "a red apple on a weathered wooden table, soft window light",
    "a chrome sports car speeding through a rainy tunnel at night",
    "an aerial photograph of a glacier calving into black water",
    "a bowl of steaming ramen with soft-boiled egg, overhead shot",
]


def carrega(nome: str, tipo_clip: str):
    import comfy.sd
    import folder_paths
    caminho = folder_paths.get_full_path_or_raise("text_encoders", nome)
    return comfy.sd.load_clip(
        ckpt_paths=[caminho],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, tipo_clip),
        model_options={"disable_dynamic": True})


def cond(clip, prompt: str) -> torch.Tensor:
    return clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].detach().float().cpu()


def distancia(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float] | None:
    """rel-RMSE e cosseno. Formas diferentes (prompts de tamanhos diferentes) nao sao comparaveis.

    Isto e uma limitacao real e nao um detalhe: dois prompts com contagens de token diferentes
    produzem tensores de formas diferentes, entao a comparacao so existe entre prompts do mesmo
    tamanho em tokens. As parafrases abaixo foram escritas tentando casar tamanho, e as que nao
    casarem aparecem como puladas em vez de sumirem.
    """
    if a.shape != b.shape:
        return None
    return (float(torch.linalg.vector_norm(b - a) / torch.linalg.vector_norm(a)),
            float(torch.nn.functional.cosine_similarity(b.reshape(1, -1), a.reshape(1, -1))))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default="qwen_3_4b.safetensors")
    p.add_argument("--quant", default="qwen_3_4b_w4a4_convrot.safetensors")
    p.add_argument("--clip-type", default="LUMINA2")
    p.add_argument("--json", type=Path, default=RAIZ / "bench" / "piso_de_ruido_prompt.json")
    a = p.parse_args()

    print("=" * 78)
    print("Quantizar o encoder custa mais ou menos que reescrever o prompt?")
    print("=" * 78)

    clip_ref = carrega(a.ref, a.clip_type)
    todas = [q for g in PARAFRASES.values() for q in g] + DIFERENTES
    cond_ref = {q: cond(clip_ref, q) for q in dict.fromkeys(todas)}
    del clip_ref
    torch.cuda.empty_cache()

    clip_q = carrega(a.quant, a.clip_type)
    canonicos = [g[0] for g in PARAFRASES.values()]
    cond_quant = {q: cond(clip_q, q) for q in canonicos}
    del clip_q
    torch.cuda.empty_cache()

    resultado = {"quantizacao": [], "parafrase": [], "sentido_diferente": [], "pulados": 0}

    for q in canonicos:
        d = distancia(cond_ref[q], cond_quant[q])
        if d:
            resultado["quantizacao"].append({"a": q, "b": q + " [4 bits]",
                                             "rel_rmse": d[0], "cosseno": d[1]})

    for grupo, prompts in PARAFRASES.items():
        for x, y in combinations(prompts, 2):
            d = distancia(cond_ref[x], cond_ref[y])
            if d is None:
                resultado["pulados"] += 1
                continue
            resultado["parafrase"].append({"grupo": grupo, "a": x, "b": y,
                                           "rel_rmse": d[0], "cosseno": d[1]})

    for x, y in combinations(DIFERENTES, 2):
        d = distancia(cond_ref[x], cond_ref[y])
        if d is None:
            resultado["pulados"] += 1
            continue
        resultado["sentido_diferente"].append({"a": x, "b": y,
                                               "rel_rmse": d[0], "cosseno": d[1]})

    print(f"\n{'eixo':22} {'n':>3} {'rel-RMSE medio':>15} {'cosseno medio':>14}  faixa de cosseno")
    print("-" * 80)
    resumo = {}
    for eixo, rotulo in (("quantizacao", "quantizar o peso"),
                         ("parafrase", "reescrever o prompt"),
                         ("sentido_diferente", "CONTROLE: outro sentido")):
        v = resultado[eixo]
        if not v:
            print(f"{rotulo:22} {0:3}   (nenhum par comparavel -- formas diferentes)")
            continue
        rr = sum(x["rel_rmse"] for x in v) / len(v)
        cc = [x["cosseno"] for x in v]
        resumo[eixo] = {"n": len(v), "rel_rmse": rr, "cosseno": sum(cc) / len(cc)}
        print(f"{rotulo:22} {len(v):3} {rr:15.4e} {sum(cc)/len(cc):14.6f}  "
              f"{min(cc):.4f} a {max(cc):.4f}")

    if resultado["pulados"]:
        print(f"\n{resultado['pulados']} pares pulados: contagens de token diferentes produzem "
              "formas diferentes e nao sao comparaveis.")

    if "quantizacao" in resumo and "parafrase" in resumo:
        q, pf = resumo["quantizacao"]["rel_rmse"], resumo["parafrase"]["rel_rmse"]
        print()
        if pf > q:
            print(f"Reescrever o prompt custa {pf/q:.2f}x MAIS que quantizar o peso para 4 bits.")
            print("A quantizacao cai dentro da variacao que o difusor ja recebe de gente")
            print("escrevendo a mesma coisa de jeitos diferentes.")
        else:
            print(f"Quantizar custa {q/pf:.2f}x MAIS que reescrever o prompt.")
            print("A quantizacao introduz uma mudanca MAIOR que a variacao natural entre")
            print("maneiras de pedir a mesma coisa. O argumento do piso de ruido NAO se sustenta.")
        if "sentido_diferente" in resumo:
            sd = resumo["sentido_diferente"]["rel_rmse"]
            if sd <= pf * 1.15:
                print(f"\nCONTROLE FALHOU: sentido diferente ({sd:.4e}) nao esta acima de "
                      f"parafrase ({pf:.4e}).")
                print("Esta metrica nao esta separando sentido, e a comparacao acima perde o chao.")
            else:
                print(f"\nControle OK: sentido diferente custa {sd/pf:.2f}x a parafrase, entao a "
                      "metrica separa sentido.")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {a.json}")
    print("\nNAO COBERTO: distancia entre condicionamentos NAO e 'o difusor nota a diferenca' --")
    print("para isso seria preciso gerar com trajetoria imposta, e este probe nao gera nada. As")
    print("parafrases sao escritas a mao e sao um julgamento sobre o que quer dizer o mesmo. Um")
    print("encoder, uma familia de modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
