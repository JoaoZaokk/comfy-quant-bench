"""Grafos API dos workflows de edição; sobem como userdata temporário (tmp_api/) para o frontend converter em UI."""
import arc_grafos as G
import executa

PROMPT = "Put a red knitted beanie on the man in <image1>; keep his face, pose, clothing and background unchanged."
TITULOS = ("Imagem 1 (a que vai ser editada)", "Imagem 2 (opcional, reative com Ctrl+M)")
COMUM = ("### Como usar\n* **Imagem 1** é a que vai ser editada; a saída segue o formato dela, com ~768² de área.\n"
         "* **Imagem 2** (referência extra) vem desativada: reative com Ctrl+M e cite no prompt como `<image2>`.\n"
         "* Escreva a instrução em inglês citando `<image1>`, `<image2>` (\"the first image\" não funciona).\n\n"
         "### Por que 768\nNesta máquina (DiT Q4_1 + encoder zen), referências em 1024 saem com aspecto \"HDR queimado\" e "
         "ignoram a instrução; em 768 a edição sai certa. Não suba o `resolution` do nó de instrução.\n\n"
         "Testado na Arc: 1 e 2 referências. Não rode junto com o llama.cpp.")
MEDIDO = "medido na interface da Arc em 28/09, 1 referência"


def edicao(tag, prefixo):
    g, latente = G.edicao(PROMPT, ["edit_pescador.png", "edit_frasco.png"], 768, titulos=TITULOS)
    return G.amostra(g, latente, tag, prefixo, area=768)


WF = {
    "Qwen-Image 2.1 edição (25 passos)": (edicao("base25", "qwen21_edicao"),
        f"## Edição — 25 passos\n\n~92 s por edição ({MEDIDO}).\n\n" + COMUM),
    "Qwen-Image 2.1 edição turbo 4 passos (Viggle v0.1)": (edicao("v01_4", "qwen21_edicao_turbo4"),
        f"## Edição turbo — 4 passos (Viggle v0.1)\n\n~24 s por edição ({MEDIDO}).\n\n" + COMUM),
    "Qwen-Image 2.1 edição turbo 6 passos (Viggle v0.2.1)": (edicao("viggle6", "qwen21_edicao_turbo6"),
        f"## Edição turbo — 6 passos (Viggle v0.2.1) — recomendado\n\n~34 s por edição ({MEDIDO}).\n\n" + COMUM),
    "Qwen-Image 2.1 edição turbo 8 passos (Turbo8)": (edicao("turbo8", "qwen21_edicao_turbo8"),
        f"## Edição turbo — 8 passos (Turbo8)\n\n~43 s por edição ({MEDIDO}). A agenda `ModelSamplingFlux` usa "
        "768×768: o shift só depende da área, e a saída tem ~768² qualquer que seja o formato da imagem 1.\n\n" + COMUM),
}

if __name__ == "__main__":
    executa.sobe_tmp_api(WF)
