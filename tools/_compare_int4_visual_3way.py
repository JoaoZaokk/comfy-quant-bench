"""Qual dos dois ramos W4A4 fica mais perto do BF16 original?

A comparacao nativo-contra-int8 responde "sao diferentes", nao "qual e pior". Para dizer
pior e preciso uma referencia, e ela e o mesmo modelo antes de quantizar, na mesma semente.

Le os tres latentes ja amostrados em bench/int4_vs_int8_visual/ e decodifica para pixel.

NAO COBERTO: uma semente, um prompt. Sem LPIPS/FID. So Z-Image.
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

ARMS = {
    "bf16_referencia": "latent_beyond-reality-zimage-v2_native_nativo_int4.pt",
    "w4a4_nativo_int4": "latent_zimage-v2-w4a4_nativo_int4.pt",
    "w4a4_ramo_int8": "latent_zimage-v2-w4a4_fallback_int8.pt",
}

lat = {k: torch.load(OUT / v, map_location="cpu") for k, v in ARMS.items()}
ref = lat["bf16_referencia"]

print("=== no latente, contra a referencia BF16 ===")
for k in ("w4a4_nativo_int4", "w4a4_ramo_int8"):
    d = lat[k] - ref
    rel = float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    print(f"  {k:<18} rel-RMSE {rel:.4e}   norma {float(lat[k].norm()):8.2f} "
          f"(ref {float(ref.norm()):.2f})")

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
    folder_paths.get_full_path_or_raise("vae", "ae.safetensors")))

img = {}
for k, t in lat.items():
    with torch.no_grad():
        dec = vae.decode(t)
    arr = (dec[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(OUT / f"{k}.png")
    img[k] = arr.astype("float32")

print("\n=== no pixel, contra a referencia BF16 ===")
r = img["bf16_referencia"]
best = None
for k in ("w4a4_nativo_int4", "w4a4_ramo_int8"):
    d = np.abs(img[k] - r)
    mse = ((img[k] - r) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    print(f"  {k:<18} PSNR {psnr:6.2f} dB   dif media {d.mean():5.2f}/255   "
          f"pixels>32: {(d > 32).mean() * 100:5.2f}%")
    if best is None or psnr > best[1]:
        best = (k, psnr)

print(f"\n-> mais proximo do BF16: {best[0]} ({best[1]:.2f} dB)")
print("   ATENCAO: com 8 passos, uma perturbacao pequena reencaminha a trajetoria inteira.")
print("   PSNR baixo nos DOIS significa 'foi para outro lugar', nao necessariamente 'ficou")
print("   pior'. Olhe a folha antes de concluir qualidade.")

sheet = np.concatenate([img["bf16_referencia"], img["w4a4_nativo_int4"],
                        img["w4a4_ramo_int8"]], axis=1).astype("uint8")
Image.fromarray(sheet).save(OUT / "tres_vias.png")
print(f"\nescrito {OUT}\\tres_vias.png  (BF16 | W4A4 nativo | W4A4 ramo int8)")
print("\nNAO COBERTO: uma semente, um prompt, sem metrica perceptual, so Z-Image, so sm86.",
      file=sys.stderr)
