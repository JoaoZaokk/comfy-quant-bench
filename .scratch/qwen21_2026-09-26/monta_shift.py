"""Shift do Qwen-Image-2.1 a 2048^2 (issue ComfyUI #16447): padrao do ComfyUI (mu fixo 0,69) contra o mu do scheduler
oficial para 16384 tokens (0,5 + 0,4*(16384-256)/7936 = 1,3129), aplicado so com o node nativo ModelSamplingFlux
(base 0,5, max 0,6935 a 2048x2048 da o mesmo mu; o node usa a reta 256..4096 tokens). O shift_terminal 0,02 do
scheduler oficial NAO e reproduzido (mexe so no ultimo sigma). DiT int8 da Comfy-Org, 25 passos, euler/simple, cfg 1.
Grava bateria/shift_<braco>_p<i>_s<s>.json e bateria/ordem_shift.txt."""
import ast
import json
from pathlib import Path

AQUI = Path(__file__).parent
arv = ast.parse((AQUI / "monta_bateria.py").read_text(encoding="utf-8"))
PROMPTS = next(ast.literal_eval(n.value) for n in arv.body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PROMPTS")
ESCOLHIDOS = [0, 1, 3]   # letreiro, retrato, paisagem
SEEDS = [42, 7]
lista = []
for braco in ("padrao", "mu131"):
    for i in ESCOLHIDOS:
        for s in SEEDS:
            modelo = ["1", 0]
            g = {
                "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
                "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": PROMPTS[i], "negative_prompt": "", "resolution": 2048}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 2048, "height": 2048, "batch_size": 1}},
            }
            if braco == "mu131":
                g["9"] = {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0], "max_shift": 0.6935, "base_shift": 0.5,
                                                                        "width": 2048, "height": 2048}}
                modelo = ["9", 0]
            g["6"] = {"class_type": "KSampler", "inputs": {"model": modelo, "seed": s, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                      "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "denoise": 1.0}}
            g["7"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["6", 0], "vae": ["3", 0], "tile_size": 1024, "overlap": 64,
                      "temporal_size": 64, "temporal_overlap": 8}}
            g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"qwen21_shift/{braco}/p{i}_s{s}"}}
            f = f"bateria/shift_{braco}_p{i}_s{s}.json"
            (AQUI / f).write_text(json.dumps(g, indent=1))
            lista.append(f)
(AQUI / "bateria/ordem_shift.txt").write_text("\n".join(lista) + "\n")
print(len(lista), "grafos")
