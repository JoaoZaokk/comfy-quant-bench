"""Testa as LoRAs de poucos passos na Arc: Viggle v0.2.1 r128 (4/6/8 passos) e Turbo8 (8 passos). python3 roda_turbo.py <saida>"""
import json, pathlib, sys, time, urllib.request

URL = "http://127.0.0.1:8188"
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
M = "/mnt/comfy-models"
PROMPTS = {
    "p2": 'A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a coffee cup, cream background',
    "p1": "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, soft window light, detailed skin texture, 85mm photo",
}
SIGMAS = {"viggle4": "1.0, 0.75, 0.5, 0.25", "v01_4": "1.0, 0.75, 0.5, 0.25", "viggle6": "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25",
          "viggle8": "1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125"}


def base(prompt):
    return {
        "1": {"class_type": "ZenImage21AdapterLoader", "inputs": {"model_folder": "", "adapter_file": f"{M}/text_encoders/zen/adapter_v12.safetensors",
              "text_encoder": f"{M}/text_encoders/qwen3.5_0.8b", "dtype": "bf16"}},
        "2": {"class_type": "ZenImage21TextEncode", "inputs": {"adapter": ["1", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024, "width": 1024, "height": 1024}},
        "3": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen_image_2.1_bf16_Q4_1.gguf"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["4", 0]}},
    }


def grafo(tag, pk):
    g = base(PROMPTS[pk])
    if tag.startswith(("viggle", "v01")):
        lora = "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors" if tag.startswith("v01") else "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
        g.update({
            "10": {"class_type": "ViggleTurboLora", "inputs": {"model": ["3", 0], "lora_name": lora, "strength": 1.0}},
            "11": {"class_type": "BasicGuider", "inputs": {"model": ["10", 0], "conditioning": ["2", 0]}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "13": {"class_type": "ViggleTurboSigmas", "inputs": {"latent": ["2", 2], "nodes": SIGMAS[tag]}},
            "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
            "6": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["2", 2]}},
        })
    else:
        g.update({
            "10": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": "turbo8_lora_step2500.safetensors", "strength_model": 1.0}},
            "11": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["10", 0], "max_shift": 0.6935, "base_shift": 0.5, "width": 1024, "height": 1024}},
            "6": {"class_type": "KSampler", "inputs": {"model": ["11", 0], "seed": 42, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                  "positive": ["2", 0], "negative": ["2", 1], "latent_image": ["2", 2], "denoise": 1.0}},
        })
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"arc_turbo/{tag}_{pk}"}}
    return g


def chama(caminho, dados=None):
    req = urllib.request.Request(URL + caminho, data=json.dumps(dados).encode() if dados is not None else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


res = []
for tag in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("viggle6", "viggle4", "viggle8", "turbo8")):
    for pk in ("p2", "p1"):
        pid = chama("/prompt", {"prompt": grafo(tag, pk)})["prompt_id"]
        while True:
            h = chama(f"/history/{pid}")
            if pid in h:
                st = h[pid]["status"]; msgs = {m[0]: m[1] for m in st.get("messages", [])}
                r = {"tag": tag, "prompt": pk}
                if st.get("status_str") != "success":
                    r["erro"] = str(msgs.get("execution_error", st))[:400]
                else:
                    r["s"] = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000
                print(json.dumps(r), flush=True); res.append(r); break
            time.sleep(0.5)
        if "erro" in r:
            break
(OUT / "tempos.json").write_text(json.dumps(res, indent=1))
