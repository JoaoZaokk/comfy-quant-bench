"""Bateria do loader low-bit no ComfyUI real (P3-P6): cada braco renderiza os mesmos prompts/sementes.

Usa um servidor ja no ar (sobe pelo roda_lowbit.ps1, que toma o lock). Grava imagens em
ComfyUI/output/lowbit_2026-09-27/<braco>/ e tempos por prompt em tempos_<fase>.json.

    python_embeded\\python.exe -s .scratch\\lowbit_2026-09-27\\roda_lowbit.py <porta> <fase>
"""
import json
import pathlib
import sys
import time
import urllib.request

PORTA, FASE = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{PORTA}"
D = pathlib.Path(__file__).parent
B = "F:/bonsai-re"
REF = "klein4b_braco2_bonsai_ternario_bfl.safetensors"

PROMPTS = [
    "a red apple on a weathered wooden table, soft window light",
    "a lighthouse on a rocky coast at dusk, waves crashing, dramatic clouds",
]
SEEDS = [11, 12]

BRACOS = {
    "ref_bf16_ternario": {"loader": "UNETLoader"},
    "ternario_mlx": {"path": f"{B}/bonsai-image-ternary-4B-mlx-2bit/transformer-packed-mflux/diffusion_pytorch_model.safetensors"},
    "ternario_gemlite": {"path": f"{B}/bonsai-image-ternary-4B-gemlite-2bit/transformer-gemlite-int2/state_dict.pt"},
    "ternario_unpacked": {"path": f"{B}/bonsai-image-ternary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors"},
    "binario_mlx": {"path": f"{B}/bonsai-image-binary-4B-mlx-1bit/transformer-packed-mflux/diffusion_pytorch_model.safetensors"},
    "binario_gemlite": {"path": f"{B}/bonsai-image-binary-4B-gemlite-1bit/transformer-gemlite-int1/state_dict.pt"},
    "binario_unpacked": {"path": f"{B}/bonsai-image-binary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors"},
}
FASES = {
    "principal": list(BRACOS),
    "residente": ["ref_bf16_ternario", "ternario_mlx", "binario_mlx"],
    "offload": ["ref_bf16_ternario", "ternario_mlx", "binario_mlx"],
    "gpu1": ["ternario_mlx"],
}


def grafo(braco, prompt, seed, device):
    conf = BRACOS[braco]
    if conf.get("loader") == "UNETLoader":
        modelo = {"class_type": "UNETLoader", "inputs": {"unet_name": REF, "weight_dtype": "default"}}
    else:
        modelo = {"class_type": "LowBitDiffusionLoader", "inputs": {
            "unet_name": REF, "device": device, "offload_device": "auto", "compute_dtype": "auto", "path": conf["path"]}}
    return {
        "1": modelo,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2_klein_vae_diffusers.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"lowbit_2026-09-27/{FASE}/{braco}/p{PROMPTS.index(prompt)}_s{seed}"}},
    }


def chama(caminho, dados=None):
    req = urllib.request.Request(URL + caminho, data=json.dumps(dados).encode() if dados is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def espera_servidor():
    for _ in range(300):
        try:
            return chama("/system_stats")
        except Exception:
            time.sleep(2)
    raise SystemExit("servidor nao subiu")


def roda(g):
    pid = chama("/prompt", {"prompt": g})["prompt_id"]
    while True:
        h = chama(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            msgs = {m[0]: m[1] for m in st.get("messages", [])}
            if st.get("status_str") != "success":
                return {"erro": msgs.get("execution_error", st)}
            t0, t1 = msgs["execution_start"]["timestamp"], msgs["execution_success"]["timestamp"]
            return {"s": (t1 - t0) / 1000}
        time.sleep(0.5)


espera_servidor()
device = "cuda:1" if FASE == "gpu1" else "cuda:0"
tempos = []
for braco in FASES[FASE]:
    for p in PROMPTS:
        for s in SEEDS:
            r = roda(grafo(braco, p, s, device))
            r.update({"braco": braco, "prompt": PROMPTS.index(p), "seed": s})
            print(json.dumps(r), flush=True)
            tempos.append(r)
    stats = chama("/system_stats")
    print("VRAM", braco, [(d["name"][:24], round((d["vram_total"] - d["vram_free"]) / 2**30, 2)) for d in stats["devices"]], flush=True)
(D / f"tempos_{FASE}.json").write_text(json.dumps(tempos, indent=1), encoding="utf-8")
