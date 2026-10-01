"""Grafos API dos workflows turbo; sobem como userdata temporário (tmp_api/) para o frontend converter em UI."""
import arc_grafos as G
import executa

PROMPT = ('A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a '
          'coffee cup, cream background')
COMUM = ("Encoder zen + DiT Q4_1 GGUF + VAE, 1024², cfg 1, sem prompt negativo. Prompt em inglês; "
         "números em letreiros podem sair errados. Não rode junto com o llama.cpp.")


def turbo(tag, prefixo):
    g, latente = G.texto(PROMPT)
    return G.amostra(g, latente, tag, prefixo)


WF = {
    "Qwen-Image 2.1 turbo 4 passos (Viggle v0.1)": (turbo("v01_4", "qwen21_turbo4"),
        "## Turbo 4 passos — Viggle v0.1 (r64)\n\n~30 s por imagem na Arc (21 s de amostragem).\n\n"
        "LoRA de 4 passos da Viggle. Mais rápido, um pouco mais estilizado que o modelo base. "
        "Não troque pela LoRA v0.2.1 com 4 passos: o texto sai embaralhado.\n\n" + COMUM),
    "Qwen-Image 2.1 turbo 6 passos (Viggle v0.2.1)": (turbo("viggle6", "qwen21_turbo6"),
        "## Turbo 6 passos — Viggle v0.2.1 (r128) — recomendado\n\n~40 s por imagem na Arc (31 s de amostragem).\n\n"
        f"Para 8 passos (texto miúdo mais limpo, ~48 s), troque os sigmas por:\n`{G.RECEITAS['viggle8']['sigmas']}`\n\n"
        "Regra do autor: só mexa na ponta de ruído alto; mantenha `0.875, 0.75, 0.5, 0.25`.\n\n" + COMUM),
    "Qwen-Image 2.1 turbo 8 passos (Turbo8)": (turbo("turbo8", "qwen21_turbo8"),
        "## Turbo 8 passos — Turbo8 (chriswritescode)\n\n~46 s por imagem na Arc (44 s de amostragem).\n\n"
        "LoRA comum (LoraLoaderModelOnly) com a agenda `ModelSamplingFlux` 0.6935/0.5; mantenha largura/altura "
        "iguais às do nó de prompt. Mantenha 8 passos, cfg 1, euler/simple.\n\n" + COMUM),
}

if __name__ == "__main__":
    executa.sobe_tmp_api(WF)
