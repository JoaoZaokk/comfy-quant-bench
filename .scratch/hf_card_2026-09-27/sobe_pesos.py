"""Sobe as versoes validadas do Qwen-Image-2.1 para o repo privado JoaoZaokk/qwen21-w4a4-qat (so leitura da fonte).

    set HF_TOKEN_PATH=F:\\COMFY_PORTABLE\\.hf\\token
    python_embeded\\python.exe -s sobe_pesos.py
"""
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "JoaoZaokk/qwen21-w4a4-qat"
FONTE = Path("P:/ComfyBench/diffusion_models")
# arquivo local -> nome no repo (a QAT ja esta la)
ARQUIVOS = [
    ("qwen_image_2.1_bf16_w4a4_convrot_f32.safetensors", "qwen_image_2.1_w4a4_convrot_rtn.safetensors"),
    ("qwen_image_2.1_bf16_w4a4_convrot_f32.quant.json", "qwen_image_2.1_w4a4_convrot_rtn.quant.json"),
    ("qwen_image_2.1_bf16_mixed_p010.safetensors", "qwen_image_2.1_mixed_p010.safetensors"),
    ("qwen_image_2.1_bf16_mixed_p010.quant.json", "qwen_image_2.1_mixed_p010.quant.json"),
    ("qwen_image_2.1_bf16_w4a8_f32.safetensors", "qwen_image_2.1_w4a8.safetensors"),
    ("qwen_image_2.1_bf16_w4a8_f32.quant.json", "qwen_image_2.1_w4a8.quant.json"),
    ("qwen_image_2.1_bf16_int8_convrot_f32.safetensors", "qwen_image_2.1_int8_convrot.safetensors"),
    ("qwen_image_2.1_bf16_int8_convrot_f32.quant.json", "qwen_image_2.1_int8_convrot.quant.json"),
]

import sys
if sys.argv[1:] == ["pesos_so"]:  # 27/09: W8A8 sem rotacao e os weight-only GGUF
    ARQUIVOS = [
        ("qwen_image_2.1_bf16_Q4_1.gguf", "qwen_image_2.1_w4a16_Q4_1.gguf"),
        ("qwen_image_2.1_bf16_Q4_1.quant.json", "qwen_image_2.1_w4a16_Q4_1.quant.json"),
        ("qwen_image_2.1_bf16_int8.safetensors", "qwen_image_2.1_w8a8_rowwise.safetensors"),
        ("qwen_image_2.1_bf16_int8.quant.json", "qwen_image_2.1_w8a8_rowwise.quant.json"),
        ("qwen_image_2.1_bf16_Q8_0.gguf", "qwen_image_2.1_w8a16_Q8_0.gguf"),
        ("qwen_image_2.1_bf16_Q8_0.quant.json", "qwen_image_2.1_w8a16_Q8_0.quant.json"),
    ]


if sys.argv[1:] == ["nativo_a16"]:  # 27/09: W8A16/W4A16 nativos (full_precision_matrix_mult)
    ARQUIVOS = [
        ("qwen_image_2.1_bf16_w4a8_f32_a16.safetensors", "qwen_image_2.1_w4a16.safetensors"),
        ("qwen_image_2.1_bf16_w4a8_f32_a16.quant.json", "qwen_image_2.1_w4a16.quant.json"),
        ("qwen_image_2.1_bf16_int8_convrot_f32_a16.safetensors", "qwen_image_2.1_w8a16.safetensors"),
        ("qwen_image_2.1_bf16_int8_convrot_f32_a16.quant.json", "qwen_image_2.1_w8a16.quant.json"),
    ]


if sys.argv[1:] == ["q4_1_nativo"]:  # 27/09: codigos Q4_1 no layout AWQ W4A16 nativo
    ARQUIVOS = [
        ("qwen_image_2.1_bf16_q4_1_awq.safetensors", "qwen_image_2.1_w4a16_q4_1.safetensors"),
        ("qwen_image_2.1_bf16_q4_1_awq.quant.json", "qwen_image_2.1_w4a16_q4_1.quant.json"),
    ]

api = HfApi()
no_repo = {f.path: getattr(f, "size", None) for f in api.list_repo_tree(REPO, recursive=True)}
for local, remoto in ARQUIVOS:
    src = FONTE / local
    tam = src.stat().st_size
    if no_repo.get(remoto) == tam:
        print(f"ja existe {remoto} ({tam} B)", flush=True)
        continue
    t0 = time.time()
    print(f"subindo {local} -> {remoto} ({tam / 2**30:.2f} GiB)", flush=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=remoto, repo_id=REPO,
                    commit_message=f"Add {remoto}")
    print(f"ok {remoto} em {time.time() - t0:.0f} s", flush=True)
print("FIM", flush=True)
