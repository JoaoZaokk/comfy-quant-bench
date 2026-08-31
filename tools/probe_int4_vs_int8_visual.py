"""1,49x de rel-RMSE por camada aparece na imagem, ou nao aparece?

DE ONDE VEM. Em 2026-08-30 mediu-se aqui que o ramo INT4 nativo do ConvRot perde para o
fallback INT8 em 24 de 24 camadas com ativacao real (1,49x mais erro, `probe_int4_vs_int8_
real_acts.py`). Aquilo e rel-RMSE de uma camada Linear isolada, e este repo ja tem escrito
que erro de camada NAO e metrica de qualidade. Entao a pergunta que decide qualquer coisa
pratica ficou aberta: **isso se ve?**

O QUE VARIA: `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, so isso. Mesmo checkpoint, mesmo
prompt, mesma semente, mesmo sampler, mesmo scheduler, mesmos passos, mesma placa.

POR QUE ESTE CHECKPOINT: `zimage-v2-w4a4.safetensors` tem, pelo proprio sidecar, **170
camadas convrot_w4a4 e zero em w4a8 ou bf16**. Nao ha camada que escape do despacho, entao
virar a variavel vira o modelo inteiro -- e nao uma fracao dele.

Os parametros de geracao sao os que a calibracao daquele checkpoint usou (euler, simple,
8 passos, cfg 1.0, 1024), para nao inventar um regime que o modelo nunca viu.

COMO LER: dois latentes identicos bit a bit significariam que o despacho nao mudou nada,
o que contradiria as duas medicoes anteriores. Esperado e que difiram; o numero que
interessa e QUANTO, no pixel, e se da para ver na folha lado-a-lado.

NAO COBERTO: uma semente e um prompt nao sao uma avaliacao de qualidade -- sao uma amostra.
Nao ha metrica perceptual (LPIPS/FID), so diferenca numerica no pixel e a folha para olho
humano. Nada sobre video. So Z-Image, so sm86.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "int4_vs_int8_visual"

ARM = r'''
import json, os, sys, time
sys.path.insert(0, "ComfyUI")
sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd, comfy.sample, comfy.samplers
from comfy_kitchen.backends import cuda as ckc

UNET = %(UNET)r; CLIP = %(CLIP)r; PROMPT = %(PROMPT)r; NEG = %(NEG)r
SEED = %(SEED)d; STEPS = %(STEPS)d; CFG = %(CFG)s; SIDE = %(SIDE)d; OUTPT = %(OUTPT)r

model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", UNET))
clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.LUMINA2)
positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0])]
negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(NEG))[0])]
del clip
import comfy.model_management as mm; mm.soft_empty_cache()

lf = model.model.latent_format
side = SIDE // 8
shape = [1, lf.latent_channels, side, side]
latent = torch.zeros(shape, device="cpu")
noise = comfy.sample.prepare_noise(latent, SEED, None)

probe = torch.randn(4, model.model.diffusion_model.__dict__.get("_probe_k", 256))
t0 = time.perf_counter()
out = comfy.sample.sample(model, noise, STEPS, CFG, "euler", "simple",
                          positive, negative, latent, denoise=1.0,
                          disable_noise=False, start_step=None, last_step=None,
                          force_full_denoise=False, noise_mask=None, callback=None,
                          disable_pbar=True, seed=SEED)
el = time.perf_counter() - t0
out = out.cpu().float()
torch.save(out, OUTPT)
print("RESULT " + json.dumps({
    "force": os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag": bool(ckc._FORCE_INT4_INT8_FALLBACK),
    "seconds": el, "shape": list(out.shape),
    "mean": float(out.mean()), "std": float(out.std()),
    "absmax": float(out.abs().max()),
}))
'''


def run_arm(force, a, outpt):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if force:
        env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1"
    else:
        env.pop("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", None)
    src = ARM % {"UNET": a.unet, "CLIP": a.clip, "PROMPT": a.prompt, "NEG": a.negative,
                 "SEED": a.seed, "STEPS": a.steps, "CFG": repr(a.cfg), "SIDE": a.size,
                 "OUTPT": str(outpt)}
    r = subprocess.run([str(ROOT / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=3600)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print("  ARM FALHOU\n  stdout:", r.stdout.strip()[-700:])
    print("  stderr:", r.stderr.strip()[-2000:])
    return None


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--unet", default="zimage-v2-w4a4.safetensors")
p.add_argument("--clip", default="qwen_3_4b.safetensors")
p.add_argument("--vae", default="ae.safetensors")
p.add_argument("--prompt", default="a red apple on a weathered wooden table, soft window "
                                   "light, shallow depth of field, fine skin texture")
p.add_argument("--negative", default="")
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--steps", type=int, default=8)
p.add_argument("--cfg", type=float, default=1.0)
p.add_argument("--size", type=int, default=1024)
a = p.parse_args()

OUT.mkdir(parents=True, exist_ok=True)
print("Varia UM eixo: COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK.")
print(f"checkpoint {a.unet} -- 170/170 camadas convrot_w4a4, entao a variavel vira o modelo inteiro")
print(f"seed {a.seed}, {a.steps} passos, euler/simple, cfg {a.cfg}, {a.size}px\n")

lat = {}
for force, label in ((False, "nativo_int4"), (True, "fallback_int8")):
    print(f"--- {label} ---", flush=True)
    # Namespace por checkpoint. Sem isto, rodar a referencia BF16 sobrescreve os latentes
    # quantizados -- aconteceu em 2026-08-30 e custou uma re-amostragem.
    res = run_arm(force, a, OUT / f"latent_{Path(a.unet).stem}_{label}.pt")
    if not res:
        sys.exit(f"braco {label} falhou")
    print(f"  flag={res['flag']}  {res['seconds']:.1f}s  latent {res['shape']}  "
          f"mean {res['mean']:+.4f}  std {res['std']:.4f}  absmax {res['absmax']:.3f}")
    lat[label] = res

import torch  # noqa: E402
stem = Path(a.unet).stem
A = torch.load(OUT / f"latent_{stem}_nativo_int4.pt", map_location="cpu")
B = torch.load(OUT / f"latent_{stem}_fallback_int8.pt", map_location="cpu")
same = bool(torch.equal(A, B))
d = (A - B)
rel = float(d.pow(2).mean().sqrt() / B.pow(2).mean().sqrt())
print(f"\nlatentes identicos: {'SIM' if same else 'NAO'}")
print(f"rel-RMSE(nativo vs int8) no latente = {rel:.4e}")
print(f"norma do latente: nativo {float(A.norm()):.2f}  int8 {float(B.norm()):.2f}")

try:
    sys.path.insert(0, str(ROOT / "ComfyUI"))
    sys.argv = ["main.py"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths, comfy.sd
    from PIL import Image
    import numpy as np
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", a.vae))) if False else None
    import comfy.utils
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", a.vae)))
    imgs = []
    for t, name in ((A, "nativo_int4"), (B, "fallback_int8")):
        img = vae.decode(t)
        arr = (img[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(OUT / f"{name}.png")
        imgs.append(arr.astype("float32"))
    pd = np.abs(imgs[0] - imgs[1])
    print(f"\npixel: diferenca media {pd.mean():.2f}/255, maxima {pd.max():.0f}/255, "
          f"{(pd > 8).mean() * 100:.2f}% dos pixels acima de 8/255")
    sheet = np.concatenate([imgs[0], imgs[1]], axis=1).astype("uint8")
    Image.fromarray(sheet).save(OUT / "lado_a_lado.png")
    amp = (np.abs(imgs[0] - imgs[1]) * 8).clip(0, 255).astype("uint8")
    Image.fromarray(amp).save(OUT / "diferenca_8x.png")
    print(f"escrito em {OUT}: nativo_int4.png, fallback_int8.png, lado_a_lado.png, "
          f"diferenca_8x.png")
except Exception as exc:  # noqa: BLE001
    print(f"\ndecode falhou ({type(exc).__name__}: {exc}). Os latentes estao salvos em "
          f"{OUT} e a comparacao no latente acima permanece valida.")

print("\nNAO COBERTO: uma semente e um prompt sao uma amostra, nao uma avaliacao de"
      " qualidade. Sem metrica perceptual (LPIPS/FID) -- so diferenca de pixel e a folha"
      " para olho humano. So Z-Image, so sm86, so imagem.", file=sys.stderr)
