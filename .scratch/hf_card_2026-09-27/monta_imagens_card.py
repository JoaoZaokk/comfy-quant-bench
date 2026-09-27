"""Imagens do card HF do Qwen-Image-2.1: grades rotuladas a partir dos renders da bateria (so leitura dos PNG).

    python_embeded\\python.exe -s monta_imagens_card.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAIZ = Path("F:/COMFY_PORTABLE/ComfyUI/output/qwen21_bateria")
SAIDA = Path(__file__).parent / "card" / "images"
SAIDA.mkdir(parents=True, exist_ok=True)
FONTE = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
FONTE_P = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)

TODAS = [("bf16", "BF16 (reference)"), ("nosso_int8", "int8 ConvRot"), ("nosso_w4a8", "W4A8"),
         ("nosso_mixed_p010", "mixed 0.10"), ("w4a4_rtn_3080", "W4A4 RTN*"), ("w4a4_qat_3080", "W4A4 QAT*")]
QAT = [("bf16", "BF16 (reference)"), ("w4a4_rtn_3080", "W4A4 before QAT (RTN)"), ("w4a4_qat_3080", "W4A4 after QAT")]
PROMPTS = ["neon sign", "portrait", "poster", "aerial river", "perfume", "fruit + hand"]


def abre(dit, p, s, corte=None):
    im = Image.open(RAIZ / dit / f"p{p}_s{s}_00001_.png").convert("RGB")
    if corte:
        W, H = im.size
        x0, y0, x1, y1 = corte
        im = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    return im


def rotulo(draw, xy, texto, fonte=FONTE):
    x, y = xy
    l, t, r, b = draw.textbbox((x + 8, y + 4), texto, font=fonte)
    draw.rectangle((x, y, r + 8, b + 6), fill=(0, 0, 0))
    draw.text((x + 8, y + 4), texto, fill=(255, 255, 255), font=fonte)


def linha(builds, p, s, lado, corte=None):
    ims = [abre(d, p, s, corte) for d, _ in builds]
    w0, h0 = ims[0].size
    alt = round(lado * h0 / w0)
    folha = Image.new("RGB", (lado * len(ims), alt))
    d = ImageDraw.Draw(folha)
    for k, (im, (_, nome)) in enumerate(zip(ims, builds)):
        folha.paste(im.resize((lado, alt), Image.LANCZOS), (k * lado, 0))
        rotulo(d, (k * lado, 0), nome)
    return folha


def grade(builds, s, lado, nome):
    cab = 0
    linhas = [linha(builds, p, s, lado) for p in range(6)]
    folha = Image.new("RGB", (linhas[0].width, sum(l.height for l in linhas) + cab), "white")
    y = cab
    for l in linhas:
        folha.paste(l, (0, y))
        y += l.height
    folha.save(SAIDA / nome, quality=88)


def empilha(folhas, nome):
    W = max(f.width for f in folhas)
    folha = Image.new("RGB", (W, sum(f.height for f in folhas)), "black")
    y = 0
    for f in folhas:
        folha.paste(f, (0, y))
        y += f.height
    folha.save(SAIDA / nome, quality=90)


# visao geral: 6 prompts x 6 versoes, nas duas seeds
grade(TODAS, 42, 320, "overview_seed42.jpg")
grade(TODAS, 7, 320, "overview_seed7.jpg")

# detalhes em recorte (mesmo recorte relativo em todas as versoes)
CORTES = {
    "detail_neon.jpg": (0, (0.05, 0.12, 0.75, 0.62)),
    "detail_skin.jpg": (1, (0.25, 0.2, 0.75, 0.7)),
    "detail_poster.jpg": (2, (0.1, 0.0, 0.9, 0.5)),
    "detail_glass.jpg": (4, (0.25, 0.12, 0.75, 0.62)),
    "detail_fruit.jpg": (5, (0.0, 0.1, 0.9, 0.9)),
}
for nome, (p, corte) in CORTES.items():
    empilha([linha(TODAS, p, 42, 320, corte), linha(TODAS, p, 7, 320, corte)], nome)

# QAT antes / depois (mesma placa: RTX 3080 Ti nos dois braços W4A4)
for nome, (p, corte) in {"qat_skin.jpg": (1, (0.25, 0.2, 0.75, 0.7)),
                         "qat_glass.jpg": (4, (0.25, 0.12, 0.75, 0.62)),
                         "qat_neon.jpg": (0, (0.05, 0.12, 0.75, 0.62)),
                         "qat_poster.jpg": (2, (0.1, 0.0, 0.9, 0.5))}.items():
    empilha([linha(QAT, p, 42, 560, corte), linha(QAT, p, 7, 560, corte)], nome)

# 27/09: W8A8 sem rotacao e weight-only (GGUF), contra BF16, int8 ConvRot e W4A8 (tudo na RTX 3090)
PESOS = [("bf16", "BF16 (reference)"), ("nosso_int8", "W8A8 ConvRot"), ("w8a8_rowwise", "W8A8 rowwise"),
         ("w8a16_q8_0", "W8A16 Q8_0"), ("nosso_w4a8", "W4A8"), ("w4a16_q4_1", "W4A16 Q4_1")]
grade(PESOS, 42, 320, "weightonly_overview_seed42.jpg")
grade(PESOS, 7, 320, "weightonly_overview_seed7.jpg")
empilha([linha(PESOS, 0, 42, 320, (0.05, 0.12, 0.75, 0.62)), linha(PESOS, 0, 7, 320, (0.05, 0.12, 0.75, 0.62)),
         linha(PESOS, 1, 42, 320, (0.25, 0.2, 0.75, 0.7)), linha(PESOS, 1, 7, 320, (0.25, 0.2, 0.75, 0.7))],
        "weightonly_detail.jpg")
# grafico velocidade x fidelidade (numeros medidos, fase 3/5 e QAT)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    pontos = json.loads((Path(__file__).parent / "pontos_grafico.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    for pt in pontos:
        ax.scatter(pt["its"], pt["msssim"], s=pt["gb"] * 40, alpha=0.85, facecolor="none" if pt.get("vazio") else pt["cor"], edgecolor=pt["cor"] if pt.get("vazio") else "black", linewidth=2 if pt.get("vazio") else 1)
        ax.annotate(f'{pt["nome"]}\n{pt["gb"]} GB', (pt["its"], pt["msssim"]), textcoords="offset points",
                    xytext=(pt.get("dx", 10), pt.get("dy", -4)), fontsize=9)
    ax.set_xlabel("it/s at 1024², 25 steps, RTX 3090")
    ax.set_ylabel("MS-SSIM vs BF16 (mean of 12 images)")
    ax.set_title("Qwen-Image-2.1 DiT: speed vs fidelity (bubble = weight size)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.6, 3.7)
    ax.set_ylim(0.78, 1.02)
    fig.tight_layout()
    fig.savefig(SAIDA / "speed_vs_fidelity.png")
print("ok", sorted(p.name for p in SAIDA.iterdir()))



