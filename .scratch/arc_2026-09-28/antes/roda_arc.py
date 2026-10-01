"""Roda na Arc (python3 do sistema, só stdlib): bateria Qwen 2.1 leve. VRAM: vigia_vram.sh (orçamento Vulkan).
    python3 roda_arc.py <saida_dir> [p2s42|todas]"""
import json
import pathlib
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8188"
OUT = pathlib.Path(sys.argv[1])
MODO = sys.argv[2] if len(sys.argv) > 2 else "p2s42"
M = "/mnt/comfy-models"
PROMPTS = [
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement, a small cat sitting under the sign, cinematic photo',
    "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, soft window light, detailed skin texture, 85mm photo",
    'A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a coffee cup, cream background',
    "Aerial view of a winding river through an autumn forest at sunrise, mist over the water, highly detailed landscape photo",
    "Studio product photo of a translucent glass perfume bottle on black marble, rim light, tiny water droplets on the glass",
    "Three red apples and two green pears arranged on a wooden table, a hand reaching for one apple, natural light, photo",
]
SEEDS = [42, 7]


def grafo(i, prompt, seed):
    return {
        "1": {"class_type": "ZenImage21AdapterLoader", "inputs": {
            "model_folder": "", "adapter_file": f"{M}/text_encoders/zen/adapter_v12.safetensors",
            "text_encoder": f"{M}/text_encoders/qwen3.5_0.8b", "dtype": "bf16"}},
        "2": {"class_type": "ZenImage21TextEncode", "inputs": {"adapter": ["1", 0], "prompt": prompt, "negative_prompt": "",
              "resolution": 1024, "width": 1024, "height": 1024}},
        "3": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen_image_2.1_bf16_Q4_1.gguf"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
              "scheduler": "simple", "positive": ["2", 0], "negative": ["2", 1], "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["4", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"arc_leve/p{i}_s{seed}"}},
    }


def chama(caminho, dados=None):
    req = urllib.request.Request(URL + caminho, data=json.dumps(dados).encode() if dados is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


OUT.mkdir(parents=True, exist_ok=True)
lista = [(2, 42)] if MODO == "p2s42" else [(i, s) for i in range(len(PROMPTS)) for s in SEEDS]
tempos = []
for i, s in lista:
    t0 = time.time()
    pid = chama("/prompt", {"prompt": grafo(i, PROMPTS[i], s)})["prompt_id"]
    while True:
        h = chama(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            msgs = {m[0]: m[1] for m in st.get("messages", [])}
            r = {"prompt": i, "seed": s, "t0": t0, "t1": time.time()}
            if st.get("status_str") != "success":
                r["erro"] = msgs.get("execution_error", st)
            else:
                r["s"] = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000
            print(json.dumps(r, default=str)[:600], flush=True)
            tempos.append(r)
            break
        time.sleep(0.5)
(OUT / f"tempos_{MODO}.json").write_text(json.dumps(tempos, indent=1, default=str))
