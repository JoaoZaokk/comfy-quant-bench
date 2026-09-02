r"""Decodifica os latentes do quality_ladder do Wan. Avulso, porque o decode DELE morre.

O `quality_ladder.py` grava os latentes e depois tenta decodificar no mesmo processo; nessa maquina
isso morre em `comfy_aimdo.host_buffer.HostBuffer` com `'NoneType' object has no attribute
'hostbuf_allocate'` -- o `aimdo` nao esta inicializado naquele caminho. A medicao nao depende do
decode e fica gravada; a IMAGEM ficava faltando, e imagem e a unica coisa que decide.

Mesma separacao que `_decode_int4_visual.py` ja fazia: amostrar custa a placa, decodificar nao deve
exigir re-amostrar.

NAO COBERTO: pega UM quadro de cada video (o do meio, onde o movimento ja aconteceu). Um quadro nao
julga um video. Sem metrica perceptual.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))
# `comfy.options.enable_args_parsing()` le sys.argv, entao ele precisa ser trocado ANTES
# do import. Mas isso apaga os argumentos DESTA ferramenta antes do argparse rodar --
# foi assim que um `--dir` foi ignorado em silencio e tres regimes diferentes
# decodificaram o mesmo diretorio. Guardar antes de trocar e o conserto.
ARGV_REAL = sys.argv[1:]
sys.argv = ["main.py"]

import comfy.options

comfy.options.enable_args_parsing()

import comfy.sd
import comfy.utils
import folder_paths
import torch
from PIL import Image, ImageDraw


def escreve(img, texto):
    faixa = 20
    nova = Image.new("RGB", (img.width, img.height + faixa), (0, 0, 0))
    nova.paste(img, (0, faixa))
    ImageDraw.Draw(nova).text((5, 4), texto, fill=(255, 255, 255))
    return nova


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=RAIZ / "bench" / "ladder_wan21_v8")
    p.add_argument("--vae", default="wan_2.1_vae.safetensors")
    p.add_argument("--quadro", type=int, default=None, help="indice; padrao = o do meio")
    a = p.parse_args(ARGV_REAL)

    lat_dir = a.dir / "latents"
    saida = a.dir / "imagens"
    saida.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(lat_dir.glob("*.pt"))
    if not arquivos:
        print(f"nenhum latente em {lat_dir}", file=sys.stderr)
        return 2

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", a.vae)))

    por_ckpt: dict[str, dict[str, Image.Image]] = {}
    for f in arquivos:
        ckpt, _, semente = f.stem.rpartition("__p0_s")
        with torch.no_grad():
            img = vae.decode(torch.load(f, map_location="cpu"))
        # A forma do decode do Wan nao e a mesma do decode de imagem: normalizo para
        # [T,H,W,C] achatando tudo que vier na frente, em vez de supor um layout.
        img = img.detach().cpu()
        if img.dim() > 4:
            img = img.reshape(-1, *img.shape[-3:])
        t = img.shape[0]
        idx = a.quadro if a.quadro is not None else t // 2
        q = img[min(idx, t - 1)]
        if q.shape[0] in (1, 3) and q.shape[-1] not in (1, 3):
            q = q.permute(1, 2, 0)          # [C,H,W] -> [H,W,C]
        arr = (q.numpy() * 255.0).clip(0, 255).astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr.repeat(3, axis=-1)
        im = Image.fromarray(arr)
        im.save(saida / f"{f.stem}_q{idx}.png")
        por_ckpt.setdefault(ckpt, {})[semente] = im
        print(f"{f.stem}: {t} quadros, usando o {idx} -> {f.stem}_q{idx}.png")

    # A ordem e a escada de agressividade, e a lista precisa conter TODOS os checkpoints:
    # o `misto015` foi renderizado e decodificado e ficou fora da folha porque nao estava aqui.
    # Qualquer checkpoint fora desta lista entra no fim, em vez de sumir em silencio.
    conhecidos = ("wan2.1_vace_1.3B_fp16", "wan21-vace-13b-misto005",
                  "wan21-vace-13b-misto015", "wan21-vace-13b-w4a4")
    ordem = [c for c in conhecidos if c in por_ckpt]
    ordem += [c for c in sorted(por_ckpt) if c not in conhecidos]
    sementes = sorted({s for d in por_ckpt.values() for s in d}, key=int)
    if ordem and sementes:
        w = next(iter(por_ckpt[ordem[0]].values())).width
        h = next(iter(por_ckpt[ordem[0]].values())).height + 20
        folha = Image.new("RGB", (w * len(sementes), h * len(ordem)), (0, 0, 0))
        for li, c in enumerate(ordem):
            for co, s in enumerate(sementes):
                im = por_ckpt[c].get(s)
                if im is None:
                    continue
                folha.paste(escreve(im, f"s{s}  {c}"), (co * w, li * h))
        cam = saida / "folha_wan.png"
        folha.save(cam)
        print(f"\nfolha -> {cam}  ({folha.width}x{folha.height})")
        print("linhas: " + " | ".join(ordem))
        print("colunas: sementes " + ", ".join(sementes))
    print()
    print("NAO COBERTO: UM quadro de cada video, o do meio. Um quadro nao julga um video, e o")
    print("  proprio movimento pode borrar um quadro que o video inteiro mostra nitido. Sem")
    print("  metrica perceptual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
