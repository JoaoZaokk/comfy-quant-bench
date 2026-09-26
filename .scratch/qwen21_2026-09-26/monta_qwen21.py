"""Grafo t2i do Qwen-Image-2.1 igual ao template oficial (image_qwen_image_2_1_t2i): UNETLoader, CLIPLoader
qwen_image, TextEncodeQwenImage21, EmptyLatentImage, KSampler 25 passos cfg 1 euler/simple, VAEDecode, SaveImage.
Uso: monta_qwen21.py <nome> <unet> [seed] [lado]"""
import json, sys
nome, unet = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
lado = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
PROMPT = ('A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement, '
          'a small cat sitting under the sign, cinematic photo')
g = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
    "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": PROMPT, "negative_prompt": "", "resolution": 1024}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": lado, "height": lado, "batch_size": 1}},
    "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                                               "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1],
                                               "latent_image": ["5", 0], "denoise": 1.0}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
    "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"qwen21/{nome}_s{seed}"}},
}
json.dump(g, open(f"prompt_{nome}_s{seed}_api.json", "w"), indent=1)
print("ok", nome, unet, seed, lado)
