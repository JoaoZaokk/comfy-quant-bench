"""Bateria zen: mesmos 6 prompts x seeds 42/7 da bateria Qwen 2.1, so troca o no de texto.
    python_embeded\python.exe -s .scratch\zen_2026-09-27\roda_zen.py <porta>"""
import ast
import json
import pathlib
import sys
import time
import urllib.request

URL = f"http://127.0.0.1:{sys.argv[1]}"
D = pathlib.Path(__file__).parent
arv = ast.parse((D.parent / "qwen21_2026-09-26" / "monta_bateria.py").read_text(encoding="utf-8"))
C = {n.targets[0].id: ast.literal_eval(n.value) for n in arv.body
     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("PROMPTS", "SEEDS")}


def grafo(i, prompt, seed):
    return {
        "1": {"class_type": "ZenImage21AdapterLoader", "inputs": {
            "model_folder": "", "adapter_file": "P:/ComfyBench/zen_test/adapter_v12.safetensors",
            "text_encoder": "P:/ComfyBench/zen_test/qwen3.5_0.8b", "dtype": "bf16"}},
        "2": {"class_type": "ZenImage21TextEncode", "inputs": {"adapter": ["1", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024, "width": 1024, "height": 1024}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "seed": seed, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
              "scheduler": "simple", "positive": ["2", 0], "negative": ["2", 1], "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["4", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"qwen21_bateria/zen_v12/p{i}_s{seed}"}},
    }


def chama(caminho, dados=None):
    req = urllib.request.Request(URL + caminho, data=json.dumps(dados).encode() if dados is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


for _ in range(300):
    try:
        chama("/system_stats")
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("servidor nao subiu")
if "ZenImage21TextEncode" not in chama("/object_info/ZenImage21TextEncode"):
    raise SystemExit("no zen nao carregou")
tempos = []
for i, p in enumerate(C["PROMPTS"]):
    for s in C["SEEDS"]:
        pid = chama("/prompt", {"prompt": grafo(i, p, s)})["prompt_id"]
        while True:
            h = chama(f"/history/{pid}")
            if pid in h:
                st = h[pid]["status"]
                msgs = {m[0]: m[1] for m in st.get("messages", [])}
                r = {"prompt": i, "seed": s}
                if st.get("status_str") != "success":
                    r["erro"] = msgs.get("execution_error", st)
                else:
                    r["s"] = (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000
                print(json.dumps(r, default=str)[:400], flush=True)
                tempos.append(r)
                break
            time.sleep(0.5)
(D / "tempos_zen.json").write_text(json.dumps(tempos, indent=1, default=str), encoding="utf-8")
