"""Bateria de metricas do Qwen-Image-2.1: 6 prompts x 2 seeds x cada DiT, grafo igual ao template oficial
(1024^2, 25 passos, cfg 1, euler/simple, encoder qwen3vl_8b_w4a8). Gera um grafo de API por imagem em bateria/."""
import json, os
PROMPTS = [
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement, a small cat sitting under the sign, cinematic photo',
    "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, soft window light, detailed skin texture, 85mm photo",
    'A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a coffee cup, cream background',
    "Aerial view of a winding river through an autumn forest at sunrise, mist over the water, highly detailed landscape photo",
    "Studio product photo of a translucent glass perfume bottle on black marble, rim light, tiny water droplets on the glass",
    "Three red apples and two green pears arranged on a wooden table, a hand reaching for one apple, natural light, photo",
]
SEEDS = [42, 7]
DITS = {
    "bf16": ("UNETLoader", "qwen_image_2.1_bf16.safetensors"),
    "bf16junto": ("UNETLoader", "qwen_image_2.1_bf16_junto_dos_shards.safetensors"),
    "int8": ("UNETLoader", "qwen_image_2.1_int8_convrot.safetensors"),
    "mixed": ("UNETLoader", "qwen_image_2.1_mixed_balanced.safetensors"),
    "int4": ("Qwen21NunchakuLoader", "qwen-image-2.1-int4-r128.safetensors"),
}
os.makedirs("bateria", exist_ok=True)
lista = []
for nome, (classe, arquivo) in DITS.items():
    for i, p in enumerate(PROMPTS):
        for s in SEEDS:
            um = {"unet_name": arquivo} if classe != "UNETLoader" else {"unet_name": arquivo, "weight_dtype": "default"}
            g = {
                "1": {"class_type": classe, "inputs": um},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
                "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": p, "negative_prompt": "", "resolution": 1024}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": s, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                      "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "denoise": 1.0}},
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"qwen21_bateria/{nome}/p{i}_s{s}"}},
            }
            f = f"bateria/{nome}_p{i}_s{s}.json"
            json.dump(g, open(f, "w"), indent=1)
            lista.append(f)
open("bateria/ordem.txt", "w").write("\n".join(lista) + "\n")
print(len(lista), "grafos")
