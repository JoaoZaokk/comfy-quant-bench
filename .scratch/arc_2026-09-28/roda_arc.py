"""Roda na Arc (python3 do sistema, só stdlib): bateria Qwen 2.1 leve. VRAM: vigia_vram.sh (orçamento Vulkan).
    python3 roda_arc.py <saida_dir> [p2s42|todas]"""
import pathlib
import sys

import arc_grafos as G
import executa

PROMPTS = [
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement, a small cat sitting under the sign, cinematic photo',
    "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, soft window light, detailed skin texture, 85mm photo",
    'A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a coffee cup, cream background',
    "Aerial view of a winding river through an autumn forest at sunrise, mist over the water, highly detailed landscape photo",
    "Studio product photo of a translucent glass perfume bottle on black marble, rim light, tiny water droplets on the glass",
    "Three red apples and two green pears arranged on a wooden table, a hand reaching for one apple, natural light, photo",
]
SEEDS = [42, 7]


def grafo(i, prompt, seed):
    g, latente = G.texto(prompt, latente_vazio=True)
    return G.amostra(g, latente, "base25", f"arc_leve/p{i}_s{seed}", semente=seed)


if __name__ == "__main__":
    modo = sys.argv[2] if len(sys.argv) > 2 else "p2s42"
    lista = [(2, 42)] if modo == "p2s42" else [(i, s) for i in range(len(PROMPTS)) for s in SEEDS]
    executa.bateria(pathlib.Path(sys.argv[1]) / f"tempos_{modo}.jsonl",
                    [({"prompt": i, "seed": s}, grafo(i, PROMPTS[i], s)) for i, s in lista])
