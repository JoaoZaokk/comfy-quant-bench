"""Runtime do Qwen-Image-2.1 no ComfyUI 0.37.4: --disable-dynamic-vram (como os testes) x dynamic VRAM (o .bat do dono),
com e sem os commits da master 2f7c6d47 (local do cache KV) e 1d61dcc3 (prefetch/malloc por bloco). Mesmo grafo da
bateria; 3 DiTs x 3 prompts, seed 42. Grava bateria/rt_<cfg>_<dit>_p<i>_s42.json e bateria/ordem_rt_<cfg>.txt."""
import ast
import json
import sys
from pathlib import Path

AQUI = Path(__file__).parent
arv = ast.parse((AQUI / "monta_bateria.py").read_text(encoding="utf-8"))
PROMPTS = next(ast.literal_eval(n.value) for n in arv.body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PROMPTS")
DITS = {"bf16": "qwen_image_2.1_bf16.safetensors", "int8": "qwen_image_2.1_int8_convrot.safetensors",
        "mixed": "qwen_image_2.1_mixed_balanced.safetensors"}
for cfg in sys.argv[1:]:
    lista = []
    for dit, arquivo in DITS.items():
        for i in (0, 1, 3):
            g = {
                "1": {"class_type": "UNETLoader", "inputs": {"unet_name": arquivo, "weight_dtype": "default"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
                "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": PROMPTS[i], "negative_prompt": "", "resolution": 1024}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 42, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                      "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "denoise": 1.0}},
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"qwen21_runtime/{cfg}/{dit}/p{i}_s42"}},
            }
            f = f"bateria/rt_{cfg}_{dit}_p{i}_s42.json"
            (AQUI / f).write_text(json.dumps(g, indent=1))
            lista.append(f)
    (AQUI / f"bateria/ordem_rt_{cfg}.txt").write_text("\n".join(lista) + "\n")
    print(cfg, len(lista))
