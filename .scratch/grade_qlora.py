"""Grade dos renders Qwen-Image-Edit Lightning 4 passos: linhas = pares, colunas = entrada + 5 bracos.
O controle (sem LoRA a 4 passos) tem de sair visivelmente pior; se nao sair, o teste nao distingue."""
from PIL import Image, ImageDraw
from pathlib import Path
OUT = Path("ComfyUI/output"); INP = Path("ComfyUI/input")
pares = [("pera", "edit_maca.png"), ("cachecol", "edit_pescador.png"), ("closed", "edit_placa.png")]
bracos = [("int8semlora", "INT8, no LoRA (control)"), ("int8lora", "INT8 + Lightning"),
          ("w4a8semlora", "W4A8, no LoRA (control)"), ("w4a8lora", "W4A8 + Lightning (merged)"),
          ("w4a8bypass", "W4A8 + Lightning (bypass)")]
S = 320; pad = 6; top = 28; left = 0
cols = 1 + len(bracos)
W = cols * (S + pad) + pad; H = len(pares) * (S + pad) + top + pad
g = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(g)
for ci, (_, rot) in enumerate([("in", "input")] + bracos):
    d.text((pad + ci * (S + pad) + 4, 8), rot, fill="black")
faltam = []
for ri, (par, img) in enumerate(pares):
    y = top + pad + ri * (S + pad)
    g.paste(Image.open(INP / img).convert("RGB").resize((S, S)), (pad, y))
    for ci, (arm, _) in enumerate(bracos, start=1):
        fs = sorted(OUT.glob(f"qedit_{arm}_{par}_s1_*.png"))
        x = pad + ci * (S + pad)
        if not fs:
            faltam.append(f"{arm}/{par}"); d.rectangle([x, y, x + S, y + S], outline="red"); continue
        g.paste(Image.open(fs[-1]).convert("RGB").resize((S, S)), (x, y))
Path("bench/qwen_edit_lora").mkdir(parents=True, exist_ok=True)
g.save("bench/qwen_edit_lora/grade_lightning_4passos.png")
print("grade em bench/qwen_edit_lora/grade_lightning_4passos.png; faltam:", faltam)
