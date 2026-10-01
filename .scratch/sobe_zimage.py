"""Sobe os quatro builds oficiais do Z-Image e CONFERE byte a byte contra o Hub depois.

A conferencia nao e zelo: um upload que termina sem erro nao prova que o arquivo do outro lado e o
do disco. A regra desta bancada e comparar o tamanho que o Hub reporta com o do arquivo local,
depois de subir, e falhar alto se diferirem.
"""
import os
import sys

from huggingface_hub import HfApi

D = r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models"
CARDS = r"F:\COMFY_PORTABLE\bench\hf"

REPOS = [
    {
        "repo": "JoaoZaokk/Z-Image-Turbo-W4A4-ConvRot",
        "card": os.path.join(CARDS, "zimage-turbo-quant"),
        "arquivos": ["zimage_turbo_w4a4.safetensors", "zimage_turbo_w4a4.quant.json",
                     "zimage_turbo_mixed.safetensors", "zimage_turbo_mixed.quant.json"],
    },
    {
        "repo": "JoaoZaokk/Z-Image-De-Turbo-W4A4-ConvRot",
        "card": os.path.join(CARDS, "zimage-deturbo-quant"),
        "arquivos": ["zimage_deturbo_w4a4.safetensors", "zimage_deturbo_w4a4.quant.json",
                     "zimage_deturbo_mixed.safetensors", "zimage_deturbo_mixed.quant.json"],
    },
]

api = HfApi()
falhas = []
for r in REPOS:
    print(f"\n=============== {r['repo']} ===============", flush=True)
    api.create_repo(r["repo"], repo_type="model", exist_ok=True, private=False)

    for nome in r["arquivos"]:
        caminho = os.path.join(D, nome)
        tam = os.path.getsize(caminho)
        print(f"  subindo {nome} ({tam:,} B) ...", flush=True)
        api.upload_file(path_or_fileobj=caminho, path_in_repo=nome,
                        repo_id=r["repo"], repo_type="model")

    api.upload_file(path_or_fileobj=os.path.join(r["card"], "README.md"),
                    path_in_repo="README.md", repo_id=r["repo"], repo_type="model")
    api.upload_folder(folder_path=os.path.join(r["card"], "images"),
                      path_in_repo="images", repo_id=r["repo"], repo_type="model")

    # A conferencia, contra o Hub e nao contra o proprio log do upload.
    info = api.model_info(r["repo"], files_metadata=True)
    remoto = {s.rfilename: s.size for s in info.siblings}
    for nome in r["arquivos"]:
        local = os.path.getsize(os.path.join(D, nome))
        hub = remoto.get(nome)
        igual = local == hub
        print(f"  {nome:42s} local={local:>14,} hub={str(hub):>14}  "
              f"{'IGUAL' if igual else 'DIFERE'}", flush=True)
        if not igual:
            falhas.append((r["repo"], nome, local, hub))

if falhas:
    print("\nFALHAS DE CONFERENCIA:", falhas, flush=True)
    sys.exit(1)
print("\nSOBE_ZIMAGE_OK -- todos os arquivos conferem byte a byte contra o Hub", flush=True)
