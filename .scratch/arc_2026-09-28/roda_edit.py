"""Edição na Arc. python3 roda_edit.py <saida> [tags] [casos]"""
import pathlib
import sys

import arc_grafos as G
import executa

GORRO = "Put a red knitted beanie on the man in <image1>; keep his face, pose, clothing and background unchanged."
CASOS = {
    "gorro": (GORRO, ["edit_pescador.png"], 1024),
    "gorro768": (GORRO, ["edit_pescador.png"], 768),
    "frasco": ("The man in <image1> holds the perfume bottle from <image2> up next to his face; keep his face, clothing "
               "and background unchanged.", ["edit_pescador.png", "edit_frasco.png"], 768),
}


def grafo(tag, pk):
    prompt, imagens, res = CASOS[pk]
    g, latente = G.edicao(prompt, imagens, res)
    return G.amostra(g, latente, tag, f"arc_edit/{tag}_{pk}")


if __name__ == "__main__":
    tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ("viggle6", "v01_4", "turbo8", "base25")
    casos = sys.argv[3].split(",") if len(sys.argv) > 3 else ("gorro", "frasco")
    executa.bateria(pathlib.Path(sys.argv[1]) / "tempos.jsonl",
                    [({"tag": t, "prompt": pk}, grafo(t, pk)) for t in tags for pk in casos])
