"""O ramo VACE em forca 1.0 sobre quadros zerados e o que destroi a geracao do Wan 2.1?

DE ONDE VEM. Em 2026-09-01 o `quality_ladder` nao conseguiu renderizar o Wan 2.1 VACE 1.3B: a
referencia FP16, sem quantizacao nenhuma, saiu como uma trama tecida em tres configuracoes
diferentes (6 passos/1 quadro, 25 passos/33 quadros, cfg 6 e cfg 1, com e sem shift 8). Quando o
braco nao-quantizado tambem quebra, nao ha nada de quantizacao no resultado.

`comfy/model_base.py:1710-1737` mostra o mecanismo: sem no VACE, o ComfyUI PREENCHE `vace_frames`
com zeros, passa cada bloco por `process_latent_in` -- que subtrai a media do formato latente, e
portanto transforma zero em um valor NAO nulo -- concatena uma mascara toda de UNS, e injeta isso
com `vace_strength = 1.0` em cada bloco VACE. Isso nao e "sem controle": e um controle constante em
forca total.

Varia UM eixo: `vace_strength`, 1.0 contra 0.0, mesmo modelo, mesma semente, mesmo tudo.
"""
import sys
from pathlib import Path

RAIZ = Path(r"F:\COMFY_PORTABLE")
sys.path.insert(0, str(RAIZ / "ComfyUI"))

_ARGV = sys.argv[:]
sys.argv = ["main.py"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: F401
sys.argv = _ARGV

import numpy as np
import torch
from PIL import Image

import comfy.sd
import comfy.sample
import comfy.utils
import folder_paths
import node_helpers

SAIDA = Path(r"F:\COMFY_PORTABLE\bench\vace_strength")
SAIDA.mkdir(parents=True, exist_ok=True)

PROMPT = ("a cinematic shot of an elderly watchmaker in a small workshop, sitting at a walnut "
          "workbench, warm window light, brass gears on the table, photorealistic")

clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise(
        "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.WAN)
def _clonar(cond):
    """Sai do inference mode. `extra_conds` do VACE escreve in-place em `vf[:, i:i+16]`, e um
    tensor nascido sob inference mode recusa isso com RuntimeError -- medido nesta mesma probe."""
    return [[c[0].clone(), {k: (v.clone() if torch.is_tensor(v) else v)
                            for k, v in c[1].items()}] for c in cond]


pos = _clonar(clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT)))
neg = _clonar(clip.encode_from_tokens_scheduled(clip.tokenize("")))
del clip
torch.cuda.empty_cache()

model = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", "wan2.1_vace_1.3B_fp16.safetensors"),
    disable_dynamic=True)

latentes = {}
for forca in (1.0, 0.0):
    p = node_helpers.conditioning_set_values(pos, {"vace_strength": [forca]})
    n = node_helpers.conditioning_set_values(neg, {"vace_strength": [forca]})
    latent = torch.zeros([1, 16, 9, 60, 60], device="cpu")
    noise = comfy.sample.prepare_noise(latent, 1, None)
    out = comfy.sample.sample(model, noise, 25, 1.0, "euler", "simple", p, n, latent,
                              denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                              force_full_denoise=False, noise_mask=None, callback=None,
                              disable_pbar=True, seed=1)
    lat = out.detach().float().cpu()
    latentes[forca] = lat
    print(f"vace_strength={forca}  |latente|={float(lat.norm()):.2f}", flush=True)

# O modelo sai da placa ANTES do decode. Com ele residente o decode do VAE do Wan estourou a VRAM,
# caiu no caminho ladrilhado, e la o `process_output` escreve in-place num tensor de inference mode
# e morre com uma RuntimeError que nao fala de memoria nenhuma.
del model
torch.cuda.empty_cache()

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
    folder_paths.get_full_path_or_raise("vae", "wan_2.1_vae.safetensors")))

for forca, lat in latentes.items():
    with torch.no_grad():
        img = vae.decode(lat.cuda())
    if img.ndim == 5:
        img = img[0, 0]
    elif img.ndim == 4:
        img = img[0]
    arr = (img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    if arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    alvo = SAIDA / f"vace_strength_{forca:.1f}.png"
    Image.fromarray(arr).save(alvo)
    print(f"vace_strength={forca}  -> {alvo.name}", flush=True)

d = (latentes[0.0] - latentes[1.0]).norm() / latentes[1.0].norm()
print(f"\ndistancia relativa entre os dois latentes: {float(d):.4f}")

print("\nNAO COBERTO: uma semente, um prompt, um tamanho. Isto testa se a forca do ramo VACE")
print("explica a saida destruida -- nao diz qual e a forca CERTA para um uso real de VACE.")
