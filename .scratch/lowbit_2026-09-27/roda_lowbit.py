"""Bateria do loader low-bit no ComfyUI real (P3-P6): cada braco renderiza os mesmos prompts/sementes.

Usa um servidor ja no ar (sobe pelo roda_lowbit.ps1, que toma o lock). Grava imagens em
ComfyUI/output/lowbit_2026-09-27/<braco>/, tempos por prompt em tempos_<fase>.json e um registro
por prompt_id em resultados_<fase>.jsonl (tools/comfy_client.py). Exit != 0 se algum grafo falhou.

    python_embeded\\python.exe -s .scratch\\lowbit_2026-09-27\\roda_lowbit.py <porta> <fase> [prazo_s_por_grafo]
"""
import json
import pathlib
import sys

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


# Cliente canonico (revisao 2026-09-29): o laco anterior era `while True` sem prazo sobre /history,
# e um KeyError se a execucao nao tivesse execution_start. Agora: prazo por grafo, node_errors num
# 200 recusa, cache hit marcado, um registro JSONL por prompt_id, exit != 0 se algum grafo falhou.
sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
import comfy_client as cc  # noqa: E402

PRAZO_S = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
comfy = cc.Comfy(URL)
try:
    comfy.wait_up(600)
except TimeoutError as e:
    raise SystemExit(f"servidor nao subiu: {e}")
ctl = cc.controles(comfy)
device = "cuda:1" if FASE == "gpu1" else "cuda:0"
saida_jsonl = D / f"resultados_{FASE}.jsonl"
tempos, registros = [], []
for braco in FASES[FASE]:
    for p in PROMPTS:
        for s in SEEDS:
            i = PROMPTS.index(p)
            reg = cc.roda_um(comfy, grafo(braco, p, s, device), PRAZO_S,
                             rotulo=f"{FASE}/{braco}/p{i}_s{s}",
                             extra={**ctl, "braco": braco, "prompt": i, "seed": s, "device": device})
            cc.grava_jsonl(saida_jsonl, reg)
            registros.append(reg)
            # formato antigo de tempos_<fase>.json, que metricas_lowbit.py le
            r = {"braco": braco, "prompt": i, "seed": s, "prompt_id": reg.get("prompt_id"),
                 "files": reg.get("files"), "cache_hit": reg.get("cache_hit")}
            if reg["status"] == "success":
                r["s"] = reg["server_side_s"]
            else:
                r["erro"] = {"status": reg["status"], "erro": reg.get("erro")}
            print(json.dumps(r), flush=True)
            tempos.append(r)
    stats = comfy.system_stats()
    print("VRAM", braco, [(d["name"][:24], round((d["vram_total"] - d["vram_free"]) / 2**30, 2)) for d in stats["devices"]], flush=True)
(D / f"tempos_{FASE}.json").write_text(json.dumps(tempos, indent=1), encoding="utf-8")
raise SystemExit(cc.codigo_de_saida(registros))
