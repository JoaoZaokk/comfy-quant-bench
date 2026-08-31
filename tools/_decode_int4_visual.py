"""Decodifica os latentes salvos por probe_int4_vs_int8_visual.py para pixel.

Separado do probe de proposito: o sampling ja rodou e custou a placa; um erro no decode
(foi um `.detach()` faltando) nao deve exigir re-amostrar. Le de bench/int4_vs_int8_visual/.

NAO COBERTO: so decodifica e compara pixel. Nada de metrica perceptual.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "int4_vs_int8_visual"
sys.path.insert(0, str(ROOT / "ComfyUI"))
sys.argv = ["main.py"]

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

VAE_NAME = sys.argv[1] if len(sys.argv) > 1 else "ae.safetensors"

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
    folder_paths.get_full_path_or_raise("vae", VAE_NAME)))

imgs = {}
for label in ("nativo_int4", "fallback_int8"):
    t = torch.load(OUT / f"latent_{label}.pt", map_location="cpu")
    with torch.no_grad():
        img = vae.decode(t)
    arr = (img[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(OUT / f"{label}.png")
    imgs[label] = arr.astype("float32")
    print(f"{label}: {arr.shape} -> {label}.png")

a, b = imgs["nativo_int4"], imgs["fallback_int8"]
d = np.abs(a - b)
mse = ((a - b) ** 2).mean()
psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
print()
print(f"diferenca media  {d.mean():6.2f} / 255")
print(f"diferenca maxima {d.max():6.0f} / 255")
print(f"pixels > 8/255   {(d > 8).mean() * 100:6.2f} %")
print(f"pixels > 32/255  {(d > 32).mean() * 100:6.2f} %")
print(f"PSNR entre os dois ramos = {psnr:.2f} dB")
print("  (referencia grosseira: >40 dB e diferenca invisivel; <30 dB e visivel a olho)")

Image.fromarray(np.concatenate([a, b], axis=1).astype("uint8")).save(OUT / "lado_a_lado.png")
Image.fromarray((d * 8).clip(0, 255).astype("uint8")).save(OUT / "diferenca_8x.png")
print(f"\nescrito: {OUT}\\lado_a_lado.png e diferenca_8x.png")
print("\nNAO COBERTO: uma semente, um prompt. Sem LPIPS/FID. So Z-Image.", file=sys.stderr)
