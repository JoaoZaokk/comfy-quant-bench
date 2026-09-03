r"""Decodifica os latentes de `probe_esparso_visual.py` e monta a folha rotulada.

SEPARADO DO PROBE DE PROPOSITO: a amostragem ja custou a placa, e um erro aqui nao deve exigir
re-amostrar. Mesma razao pela qual `_decode_int4_visual.py` existe.

A FOLHA CARREGA OS ROTULOS DENTRO DA IMAGEM, e isso nao e enfeite. Este repo publicou um card de
modelo em que a referencia FP16 de uma semente estava quebrada e ninguem percebeu, porque as
imagens viajaram soltas e o rotulo estava so no markdown. Um PNG que sai daqui tem de dizer o que
e sem o texto ao lado.

NAO COBERTO: so decodifica e compara pixel. Sem metrica perceptual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "bench" / "esparso_visual"
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
import numpy as np
import torch
from PIL import Image, ImageDraw

ROTULOS = {
    "bf16": "BF16 (referencia)  16 bits/peso",
    # bracos do probe de encoder: <prompt>-<braco>. O rotulo diz qual dos TRES caminhos rodou,
    # porque "w4a4" sozinho era ambiguo entre peso-de-4-bits-com-matematica-de-16 e kernel real.
    "curto_bf16": "prompt CURTO  TE BF16 7,49 GiB (referencia)",
    "curto_w4a4t": "prompt CURTO  TE W4A4 2,42 GiB TRAVADO (peso 4b, math BF16)",
    "curto_w4a4s": "prompt CURTO  TE W4A4 2,42 GiB SOLTO (kernel 4 bits real)",
    "longo_bf16": "prompt LONGO  TE BF16 7,49 GiB (referencia)",
    "longo_w4a4t": "prompt LONGO  TE W4A4 2,42 GiB TRAVADO (peso 4b, math BF16)",
    "longo_w4a4s": "prompt LONGO  TE W4A4 2,42 GiB SOLTO (kernel 4 bits real)",
    "w4a4": "W4A4 ConvRot (hoje)  4,0 bits/peso",
    "esp_peso": "2:4 par Wanda + int4, PESO  2,5 bits/peso",
    "esp_peso_ativ": "2:4 par Wanda + int4, PESO+ATIV  2,5 bits/peso",
    "convrot_peso": "ConvRot + 2:4 par Wanda + int4, PESO  2,5 bits/peso",
    "convrot_peso_ativ": "ConvRot + 2:4 par Wanda + int4, PESO+ATIV  2,5 bits/peso",
    "so_poda_elem": "SO PODA 2:4 por ELEMENTO, Wanda, bf16  9,0 bits/peso",
    "so_poda_par": "SO PODA 2:4 por PAR, Wanda, bf16  9,0 bits/peso",
}


def escreve(img: Image.Image, texto: str) -> Image.Image:
    """Faixa preta com o rotulo no topo. Sem fonte externa: a default basta e nunca falta."""
    faixa = 22
    nova = Image.new("RGB", (img.width, img.height + faixa), (0, 0, 0))
    nova.paste(img, (0, faixa))
    ImageDraw.Draw(nova).text((6, 5), texto, fill=(255, 255, 255))
    return nova


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae", default="ae.safetensors")
    # `--dir` existe porque a versao anterior tinha o diretorio fixo no modulo e tres regimes
    # diferentes foram decodificados para a mesma pasta sem ninguem perceber.
    p.add_argument("--dir", type=Path, default=SAIDA)
    a = p.parse_args(ARGV_REAL)
    saida = a.dir

    latentes = sorted(saida.glob("latent_*.pt"))
    if not latentes:
        print(f"nenhum latente em {saida}", file=sys.stderr)
        return 2
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", a.vae)))

    por_semente: dict[str, dict[str, np.ndarray]] = {}
    for lat in latentes:
        miolo = lat.stem[len("latent_"):]
        braco, _, semente = miolo.rpartition("_s")
        with torch.no_grad():
            img = vae.decode(torch.load(lat, map_location="cpu"))
        arr = (img[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(saida / f"{miolo}.png")
        por_semente.setdefault(semente, {})[braco] = arr
        print(f"{miolo}: {arr.shape} -> {miolo}.png")

    conhecidos = ["bf16", "w4a4", "esp_peso", "esp_peso_ativ", "convrot_peso",
                  "convrot_peso_ativ", "so_poda_elem", "so_poda_par",
                  "curto_bf16", "curto_w4a4t", "curto_w4a4s",
                  "longo_bf16", "longo_w4a4t", "longo_w4a4s"]
    vistos = {b for d in por_semente.values() for b in d}
    ordem = [b for b in conhecidos if b in vistos] + sorted(vistos - set(conhecidos))
    relatorio: dict[str, dict[str, float]] = {}
    folhas = []
    for semente in sorted(por_semente):
        tem = [b for b in ordem if b in por_semente[semente]]
        if not tem:
            continue
        # Distancia no pixel contra o BF16 da MESMA semente. Contra outra semente nao mediria nada:
        # a trajetoria do amostrador muda inteira e a distancia vira ruido de caos.
        # A tabela de pixel saiu VAZIA na primeira execucao com estes bracos porque a referencia
        # era procurada pelo nome literal "bf16" e aqui eles se chamam "curto_bf16"/"longo_bf16".
        # Um cabecalho sem linhas le-se como "nada a reportar" em vez de "nao achei a referencia".
        # A referencia certa e sempre a do MESMO prefixo, nunca uma global.
        def referencia_de(braco):
            prefixo = braco.rsplit("_", 1)[0] if "_" in braco else ""
            for cand in (f"{prefixo}_bf16", "bf16"):
                if cand in por_semente[semente]:
                    return cand
            return None
        sem_ref = [b for b in tem if referencia_de(b) is None]
        if sem_ref:
            print(f"  s{semente}: sem referencia bf16 para {sem_ref} -- sem distancia de pixel")
        for b in tem:
            base = por_semente[semente].get(referencia_de(b) or "")
            if base is not None:
                d = por_semente[semente][b].astype("float32") - base.astype("float32")
                mse = float((d ** 2).mean())
                relatorio.setdefault(semente, {})[b] = {
                    "rmse_pixel": mse ** 0.5,
                    "referencia": referencia_de(b),
                    "psnr_db": (10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf"),
                }
        tiras = [escreve(Image.fromarray(por_semente[semente][b]),
                         f"s{semente}  {ROTULOS.get(b, b)}") for b in tem]
        larg = sum(t.width for t in tiras)
        folha = Image.new("RGB", (larg, tiras[0].height), (0, 0, 0))
        x = 0
        for t in tiras:
            folha.paste(t, (x, 0))
            x += t.width
        cam = saida / f"folha_s{semente}.png"
        folha.save(cam)
        folhas.append(cam)
        print(f"folha da semente {semente} -> {cam.name}")

    if folhas:
        alt = sum(Image.open(f).height for f in folhas)
        larg = max(Image.open(f).width for f in folhas)
        tudo = Image.new("RGB", (larg, alt), (0, 0, 0))
        y = 0
        for f in folhas:
            im = Image.open(f)
            tudo.paste(im, (0, y))
            y += im.height
        tudo.save(saida / "folha_completa.png")
        print(f"folha completa -> folha_completa.png  ({larg}x{alt})")

    (saida / "pixel.json").write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    print()
    print(f"{'semente':>10s} {'braco':>16s} {'RMSE pixel':>11s} {'PSNR dB':>9s}")
    print("-" * 52)
    for semente in sorted(relatorio):
        for b in ordem:
            if b in relatorio[semente]:
                v = relatorio[semente][b]
                print(f"{semente:>10s} {b:>16s} {v['rmse_pixel']:11.2f} {v['psnr_db']:9.2f}")
    print()
    print("NAO COBERTO: distancia no PIXEL contra o BF16 da mesma semente. Nao e metrica")
    print("  perceptual, e esta bancada ja mediu que a imagem livre de dois formatos do MESMO")
    print("  modelo mede caos e nao fidelidade -- aqui ela serve para ordenar bracos que")
    print("  compartilham a semente e o prompt, e para nada alem disso. Olhe a folha.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
