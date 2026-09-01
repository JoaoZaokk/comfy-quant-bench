"""Publica um checkpoint quantizado desta bancada no HuggingFace, com card e sidecar.

POR QUE UM SCRIPT E NAO O CLI. Tres coisas precisam acontecer na ordem certa e cada uma pode
falhar sozinha: criar o repo, subir o card (barato, e o que da o link imediato) e subir o peso
(caro, minutos). Se o peso falhar no meio, o repo tem que continuar existindo e documentado --
um repo com README e sem peso e recuperavel, um peso sem README nao explica nada a ninguem.

LICENCA E O UNICO PORTAO. Este script NAO decide licenca: quem chama passa `--licenca` e ela tem
que casar com a do modelo de origem, conferida na fonte e nao de memoria. Derivar um checkpoint
nao apaga a licenca de quem treinou.

    python_embeded\\python.exe -s tools/hf_publicar.py ^
        --repo JoaoZaokk/Qwen3-4B-W4A4-ConvRot ^
        --card bench/hf/qwen3-4b-w4a4/README.md ^
        --peso ComfyUI/models/text_encoders/qwen_3_4b_w4a4_convrot.safetensors ^
        --sidecar ComfyUI/models/text_encoders/qwen_3_4b_w4a4_convrot.quant.json

NAO COBERTO: nao valida o conteudo do card contra o arquivo que sobe, e nao confere se a licenca
passada e mesmo a da origem. Nao mede nada. Publicar e irreversivel na pratica -- um repo apagado
ja pode ter sido clonado.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="owner/nome")
    p.add_argument("--card", type=Path, required=True, help="README.md com frontmatter")
    p.add_argument("--peso", type=Path, action="append", default=[],
                   help="arquivo de peso; repetivel")
    p.add_argument("--sidecar", type=Path, action="append", default=[],
                   help="arquivo pequeno junto do peso; repetivel")
    p.add_argument("--imagens", type=Path, default=None,
                   help="pasta enviada inteira para images/ no repo")
    p.add_argument("--privado", action="store_true",
                   help="cria privado. Default e publico, que e o ponto de publicar.")
    p.add_argument("--so-card", action="store_true",
                   help="cria o repo e sobe so o card. Use para reservar o nome e revisar antes.")
    a = p.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    quem = api.whoami()["name"]
    print(f"autenticado como {quem}")
    if not a.card.is_file():
        raise SystemExit(f"card nao existe: {a.card}")
    for f in a.peso + a.sidecar:
        if not f.is_file():
            raise SystemExit(f"arquivo nao existe: {f}")

    url = api.create_repo(a.repo, repo_type="model", private=a.privado, exist_ok=True)
    print(f"repo  {url}")

    api.upload_file(path_or_fileobj=str(a.card), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="model",
                    commit_message="Model card: what was measured, and what was not")
    print("card  enviado")

    for f in a.sidecar:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=a.repo, repo_type="model",
                        commit_message=f"Conversion provenance sidecar: {f.name}")
        print(f"side  {f.name}")

    if a.imagens is not None:
        if not a.imagens.is_dir():
            raise SystemExit(f"pasta de imagens nao existe: {a.imagens}")
        api.upload_folder(folder_path=str(a.imagens), path_in_repo="images",
                          repo_id=a.repo, repo_type="model",
                          commit_message="Comparison images: reference against each build")
        print(f"imgs  {len(list(a.imagens.iterdir()))} arquivos")

    if a.so_card:
        print("\n--so-card: pesos NAO enviados. O repo existe e esta documentado.")
        return 0

    for f in a.peso:
        gib = f.stat().st_size / 2 ** 30
        print(f"peso  {f.name} ({gib:.2f} GiB) subindo...", flush=True)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=a.repo, repo_type="model",
                        commit_message=f"ConvRot W4A4 weights: {f.name}")
        print(f"peso  {f.name} OK", flush=True)

    print(f"\npronto: https://huggingface.co/{a.repo}")
    print("\nNAO COBERTO: este script nao confere que a licenca declarada no card e a da origem,")
    print("nem que o card descreve o arquivo que subiu. Publicar e irreversivel na pratica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
