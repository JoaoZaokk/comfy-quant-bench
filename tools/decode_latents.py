"""Decodifica latentes gravados em disco, num processo separado de quem os produziu.

POR QUE ISTO EXISTE. `tools/quality_ladder.py` amostra e decodifica no mesmo processo, e sob o
ComfyUI 0.33 o decode morre ali com
`AttributeError: 'NoneType' object has no attribute 'hostbuf_allocate'` -- o `comfy.sd.VAE`
constroi um `CoreModelPatcher`, que aloca um HostBuffer do comfy-aimdo, e o aimdo nao esta
inicializado. Num processo LIMPO, com o mesmo preambulo, o mesmo VAE constroi sem reclamar: o
que quebra e o estado deixado pelos carregamentos anteriores daquele processo, nao o VAE.

Consertar isso dentro do ladder foi tentado e custou caro: inicializar o aimdo cedo derruba o
processo sem traceback, e religar `comfy_aimdo.host_buffer.lib` na mao termina em SIGSEGV
(exit 139). Separar o decode e mais barato e mais robusto -- e uma amostragem de dezenas de
minutos deixa de depender de um passo que nao tem nada a ver com a medicao.

    python_embeded\\python.exe -s tools/decode_latents.py bench/aceitacao_sigma/latents ^
        --vae ae.safetensors

NAO COBERTO: nao valida que os latentes vieram do modelo que o nome do arquivo diz. Nao mede
nada -- so decodifica. Um VAE errado produz imagem plausivel e silenciosamente errada.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

_ARGV = sys.argv[:]
sys.argv = ["main.py"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401  -- o parse acontece aqui, com o argv neutro
sys.argv = _ARGV

import torch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("latents", type=Path, help="diretorio com arquivos .pt")
    p.add_argument("--vae", default="ae.safetensors", help="arquivo em models/vae")
    p.add_argument("--out", type=Path, default=None, help="default: o pai de <latents>")
    a = p.parse_args()

    import folder_paths
    import comfy.sd
    import comfy.utils
    import numpy as np
    from PIL import Image

    arquivos = sorted(a.latents.glob("*.pt"))
    if not arquivos:
        raise SystemExit(f"nenhum .pt em {a.latents}")
    saida = a.out or a.latents.parent
    saida.mkdir(parents=True, exist_ok=True)

    vae_path = folder_paths.get_full_path_or_raise("vae", a.vae)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
    print(f"VAE {a.vae} carregado. {len(arquivos)} latentes.")

    for arq in arquivos:
        t = torch.load(arq, map_location="cpu", weights_only=True)
        with torch.no_grad():
            img = vae.decode(t.cuda())
        if img.ndim == 4:
            img = img[0]
        elif img.ndim == 5:      # video: so o primeiro quadro
            img = img[0, 0]
        arr = (img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        if arr.shape[0] in (1, 3, 4):        # canal primeiro
            arr = arr.transpose(1, 2, 0)
        alvo = saida / f"{arq.stem}.png"
        Image.fromarray(arr).save(alvo)
        print(f"  {alvo.name}  {arr.shape[1]}x{arr.shape[0]}")

    print(f"\n{len(arquivos)} imagens em {saida}")
    print("NAO COBERTO: nada aqui confere que o latente veio do modelo que o nome diz, e um VAE "
          "errado produz imagem plausivel e silenciosamente errada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
