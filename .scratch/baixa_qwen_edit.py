"""Baixa a fonte BF16 e o braco de referencia publico do Qwen-Image-Edit-2511.

POR QUE ESSES DOIS ARQUIVOS
---------------------------
`qwen_image_edit_2511_bf16` (38,05 GiB) e a FONTE: sem ela nao existe referencia contra a qual
medir erro por camada, e a quantizacao nao tem de onde sair.

`qwen_image_edit_2511_int8_convrot` (19,09 GiB) e o braco que ja venceu o nosso W4A4 em quatro
familias seguidas nesta bancada (Z-Image, MiniMax, epsilon por passo, e agora Krea2 por 10 de 10
corridas pareadas). Baixar o concorrente e mais barato do que descobrir depois que o nosso perde
e nao ter com o que comparar.

`fp8mixed` fica de fora por disco, e isso e uma escolha, nao um esquecimento.

A VERSAO
--------
2511 e a mais recente da familia Edit -- conferido na API do HF em 2026-09-12, listando a org
`Qwen` por data de criacao: Edit-2511 (17 dez 2025) e o ultimo, Qwen-Image-2512 (30 dez 2025) e
da familia BASE, nao Edit. Nao e leitura de post de blog.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
DESTINO = r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models"
ARQUIVOS = [
    "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
    "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
]

for remoto in ARQUIVOS:
    nome = remoto.rsplit("/", 1)[-1]
    alvo = os.path.join(DESTINO, nome)
    if os.path.exists(alvo):
        print(f"ja existe: {nome} ({os.path.getsize(alvo)} B)", flush=True)
        continue
    t = time.time()
    print(f"baixando {nome} ...", flush=True)
    caminho = hf_hub_download(REPO, remoto, local_dir=DESTINO + r"\_hf_qwen")
    tam = os.path.getsize(caminho)
    os.replace(caminho, alvo)
    print(f"  {nome}  {tam} B  em {time.time()-t:.0f}s", flush=True)

print("QWEN_DOWNLOAD_OK", flush=True)
sys.exit(0)
