"""Sobe o build W4A8 do Qwen-Image-Edit 2511, com as provas.

O QUE SOBE E O QUE NAO SOBE
---------------------------
Sobe: o `w4a8` (10,79 GiB, FUNCIONA), o sidecar dele, o README em ingles, e as DUAS grades --
a que mostra o build bom contra o BF16, e a que mostra os dois builds que sairam ruido.

Nao sobem os arquivos `w4a4` e `mixed`. Eles sao um negativo medido, nao um modelo: publicar o
peso convida alguem a usar um checkpoint que renderiza estatica. A PROVA de que falharam sobe
como imagem, que e o que torna a afirmacao conferivel sem distribuir o arquivo quebrado.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import HfApi  # noqa: E402

REPO = "JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot"
RAIZ = r"F:\COMFY_PORTABLE"
CARD = os.path.join(RAIZ, "bench", "hf", "qwen-image-edit-2511-quant")
MODELOS = os.path.join(RAIZ, "ComfyUI", "models", "diffusion_models")

ARQUIVOS = [
    (os.path.join(MODELOS, "qwen_image_edit_2511_w4a8.safetensors"),
     "qwen_image_edit_2511_w4a8.safetensors"),
    (os.path.join(MODELOS, "qwen_image_edit_2511_w4a8.quant.json"),
     "qwen_image_edit_2511_w4a8.quant.json"),
    (os.path.join(CARD, "images", "grade_w4a8.png"), "images/grade_w4a8.png"),
    (os.path.join(CARD, "images", "grade_falhas.png"), "images/grade_falhas.png"),
    (os.path.join(CARD, "README.md"), "README.md"),
]

api = HfApi()
quem = api.whoami()
print(f"autenticado como {quem.get('name')}", flush=True)

api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
print(f"repo pronto: https://huggingface.co/{REPO}", flush=True)

for local, remoto in ARQUIVOS:
    if not os.path.exists(local):
        print(f"FALTA {local}", flush=True)
        sys.exit(1)
    n = os.path.getsize(local)
    print(f"subindo {remoto} ({n:,} B) ...", flush=True)
    t0 = time.time()
    api.upload_file(path_or_fileobj=local, path_in_repo=remoto,
                    repo_id=REPO, repo_type="model")
    print(f"  ok em {time.time() - t0:.0f}s", flush=True)

print("QWEN_EDIT_UPLOAD_OK", flush=True)
