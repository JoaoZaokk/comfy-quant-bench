"""LoRAs de poucos passos na Arc (texto para imagem). python3 roda_turbo.py <saida> [tags]"""
import pathlib
import sys

import arc_grafos as G
import executa

PROMPTS = {
    "p2": 'A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a coffee cup, cream background',
    "p1": "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, soft window light, detailed skin texture, 85mm photo",
}


def grafo(tag, pk):
    g, latente = G.texto(PROMPTS[pk])
    return G.amostra(g, latente, tag, f"arc_turbo/{tag}_{pk}")


if __name__ == "__main__":
    tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ("viggle6", "viggle4", "viggle8", "turbo8")
    executa.bateria(pathlib.Path(sys.argv[1]) / "tempos.jsonl",
                    [({"tag": t, "prompt": pk}, grafo(t, pk)) for t in tags for pk in PROMPTS])
