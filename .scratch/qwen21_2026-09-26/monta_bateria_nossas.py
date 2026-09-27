"""Grafos da bateria para as NOSSAS quantizacoes do Qwen-Image-2.1: mesmos prompts, seeds e grafo de monta_bateria.py,
so muda o DiT. Grava bateria/<dit>_p<i>_s<s>.json e bateria/ordem_nossas_f32.txt."""
import ast
import json
from pathlib import Path

AQUI = Path(__file__).parent
arv = ast.parse((AQUI / "monta_bateria.py").read_text(encoding="utf-8"))
const = {n.targets[0].id: ast.literal_eval(n.value) for n in arv.body
         if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("PROMPTS", "SEEDS")}
PROMPTS, SEEDS = const["PROMPTS"], const["SEEDS"]
# Os `_f32` quantizam do peso em FP32 (correcao de 26/09; os primeiros builds rotacionavam em BF16).
DITS = {
    "nosso_int8": "qwen_image_2.1_bf16_int8_convrot_f32.safetensors",
    "nosso_w4a8": "qwen_image_2.1_bf16_w4a8_f32.safetensors",
    "nosso_w4a4": "qwen_image_2.1_bf16_w4a4_convrot_f32.safetensors",
    "nosso_mixed": "qwen_image_2.1_bf16_mixed.safetensors",
}
ORDEM = "bateria/ordem_nossas_f32.txt"
import sys
if sys.argv[1:] == ["escalas"]:  # fase 4: escalas refinadas por minimos quadrados (tools/refina_escalas.py)
    DITS = {"esc_mixed": "qwen_image_2.1_bf16_mixed_escalas.safetensors",
            "esc_w4a4": "qwen_image_2.1_bf16_w4a4_convrot_f32_escalas.safetensors",
            "esc_w4a8": "qwen_image_2.1_bf16_w4a8_f32_escalas.safetensors"}
    ORDEM = "bateria/ordem_escalas.txt"
if sys.argv[1:] == ["p010"]:  # fase 5: mixed com promote 0,10
    DITS = {"nosso_mixed_p010": "qwen_image_2.1_bf16_mixed_p010.safetensors"}
    ORDEM = "bateria/ordem_p010.txt"
if sys.argv[1:] == ["qat"]:  # 27/09: W4A4 treinada por bloco (Colab) x a mesma W4A4 sem treino, as duas na 3080 Ti
    DITS = {"w4a4_rtn_3080": "qwen_image_2.1_bf16_w4a4_convrot_f32.safetensors",
            "w4a4_qat_3080": "qwen_image_2.1_bf16_w4a4_qat.safetensors"}
    ORDEM = "bateria/ordem_qat.txt"
if sys.argv[1:] == ["pesos_so"]:  # 27/09: W8A8 sem rotacao e os weight-only GGUF (W8A16 Q8_0, W4A16 Q4_1)
    DITS = {"w8a8_rowwise": "qwen_image_2.1_bf16_int8.safetensors",
            "w8a16_q8_0": "qwen_image_2.1_bf16_Q8_0.gguf",
            "w4a16_q4_1": "qwen_image_2.1_bf16_Q4_1.gguf"}
    ORDEM = "bateria/ordem_pesos_so.txt"
lista = []
for nome, arquivo in DITS.items():
    for i, p in enumerate(PROMPTS):
        for s in SEEDS:
            g = {
                "1": {"class_type": "UNETLoader", "inputs": {"unet_name": arquivo, "weight_dtype": "default"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
                "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": p, "negative_prompt": "", "resolution": 1024}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": s, "steps": 25, "cfg": 1.0, "sampler_name": "euler",
                      "scheduler": "simple", "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "denoise": 1.0}},
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"qwen21_bateria/{nome}/p{i}_s{s}"}},
            }
            if arquivo.endswith(".gguf"):  # ComfyUI-GGUF: desquantiza o peso para BF16 antes de cada matmul
                g["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": arquivo}}
            f = f"bateria/{nome}_p{i}_s{s}.json"
            (AQUI / f).write_text(json.dumps(g, indent=1))
            lista.append(f)
(AQUI / ORDEM).write_text("\n".join(lista) + "\n")
print(len(lista), "grafos")
