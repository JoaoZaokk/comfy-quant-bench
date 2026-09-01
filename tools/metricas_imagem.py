"""Metricas de imagem contra uma referencia, para parar de dizer "o olho acha".

POR QUE ISTO EXISTE. Todo julgamento de imagem desta bancada ate 2026-08-31 foi "quebrou ou nao
quebrou", olhado por uma pessoa. Isso serve para catastrofe e nao serve para o resto: nao distingue
"granulada" de "limpa", nao ordena dois builds bons, e nao vira numero que outro alguem confira.

O QUE MEDE, e o que cada uma NAO responde:

  PSNR    erro por pixel em dB. Alto = parecido. Nao sabe nada de percepcao: um deslocamento de um
          pixel derruba o PSNR sem que ninguem veja diferenca.
  SSIM    estrutura local (luminancia, contraste, correlacao). Melhor que PSNR, ainda insensivel a
          textura que muda de carater sem mudar de estatistica local.
  MS-SSIM SSIM em cinco escalas. E a que menos se engana com granulacao fina.
  LPIPS   distancia entre ativacoes de uma rede treinada. E a que mais se aproxima do julgamento
          humano, e a UNICA aqui que precisa de pesos baixados -- VGG16, 528 MB, em
          ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth.

          ATENCAO, e este texto ja esteve errado: a primeira versao dizia que se os pesos nao
          estivessem em cache a coluna sairia vazia e a saida avisaria. FALSO -- o construtor do
          torchmetrics dispara o download do torchvision antes de qualquer except aqui ver
          alguma coisa, e ele baixou 528 MB sozinho na primeira execucao. Agora o download so
          acontece com --com-lpips explicito, e o padrao e NAO baixar nada.
  grao    desvio-padrao do laplaciano, dividido pelo da referencia. Nao e perceptual: e o numero
          que quantifica exatamente aquilo que eu vinha chamando de "granulada" no olho.

    python_embeded\\python.exe -s tools/metricas_imagem.py --ref REF.png IMG1.png IMG2.png ...
    python_embeded\\python.exe -s tools/metricas_imagem.py --dir bench/hunyuan_misto --ref-contem fp16

NAO COBERTO: nenhuma destas metricas foi validada contra julgamento humano NESTA bancada -- sao
padrao da literatura, usadas aqui como padrao da literatura. Todas comparam contra uma referencia
escolhida, entao dizem "quao longe do alvo", nunca "boa". Duas imagens podem estar igualmente longe
por motivos diferentes. E imagem de trajetoria livre continua sendo o que ja foi medido que e: um
teste de quebrou-ou-nao, e uma metrica melhor nao conserta a trajetoria ter divergido.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def carrega(p: Path) -> torch.Tensor:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]      # 1,3,H,W em [0,1]


def grao(x: torch.Tensor) -> float:
    """Desvio-padrao do laplaciano: quanto de detalhe de alta frequencia a imagem tem."""
    cinza = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    return float(torch.nn.functional.conv2d(cinza, k, padding=1).std())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("imagens", nargs="*", type=Path)
    p.add_argument("--ref", type=Path, help="a referencia")
    p.add_argument("--dir", type=Path, help="mede todo .png do diretorio")
    p.add_argument("--ref-contem", default="fp16",
                   help="com --dir, o que o nome da referencia contem")
    p.add_argument("--com-lpips", action="store_true",
                   help="baixa VGG16 (528 MB) na primeira vez, se nao estiver em cache")
    a = p.parse_args()

    if a.dir:
        todas = sorted(a.dir.glob("*.png"))
        ref = next((x for x in todas if a.ref_contem in x.name.lower()), None)
        if ref is None:
            raise SystemExit(f"nenhum .png em {a.dir} com {a.ref_contem!r} no nome")
        imagens = [x for x in todas if x != ref]
    else:
        ref, imagens = a.ref, list(a.imagens)
    if ref is None or not ref.is_file():
        raise SystemExit("passe --ref (ou --dir)")
    if not imagens:
        raise SystemExit("nenhuma imagem para medir")

    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure as msssim,
        peak_signal_noise_ratio as psnr,
        structural_similarity_index_measure as ssim,
    )

    lpips_fn = None
    aviso_lpips = ""
    if a.com_lpips:
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
            lpips_fn.eval()
        except Exception as exc:  # noqa: BLE001
            aviso_lpips = f"LPIPS falhou ({type(exc).__name__}: {str(exc)[:140]})"
    else:
        cache = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
        aviso_lpips = ("LPIPS nao medido. Passe --com-lpips para incluir; ele exige VGG16 (528 MB)"
                       + (", que JA esta em cache aqui." if cache.is_file()
                          else ", e ele SERA BAIXADO na primeira vez."))

    r = carrega(ref)
    g_ref = grao(r)
    print(f"referencia: {ref.name}   grao {g_ref:.5f}")
    print()
    cab = f"{'imagem':46} {'PSNR':>7} {'SSIM':>7} {'MS-SSIM':>8} {'grao/ref':>9}"
    if lpips_fn is not None:
        cab += f" {'LPIPS':>7}"
    print(cab)
    print("-" * len(cab))

    linhas = []
    for img in imagens:
        x = carrega(img)
        if x.shape != r.shape:
            print(f"{img.name[:46]:46} forma diferente da referencia, pulando")
            continue
        v = {"nome": img.name,
             "psnr": float(psnr(x, r, data_range=1.0)),
             "ssim": float(ssim(x, r, data_range=1.0)),
             "msssim": float(msssim(x, r, data_range=1.0)),
             "grao": grao(x) / g_ref}
        if lpips_fn is not None:
            with torch.no_grad():
                v["lpips"] = float(lpips_fn(x.clamp(0, 1), r.clamp(0, 1)))
        linhas.append(v)

    for v in sorted(linhas, key=lambda z: -z["msssim"]):
        s = (f"{v['nome'][:46]:46} {v['psnr']:7.2f} {v['ssim']:7.4f} "
             f"{v['msssim']:8.4f} {v['grao']:9.3f}")
        if "lpips" in v:
            s += f" {v['lpips']:7.4f}"
        print(s)

    print()
    if aviso_lpips:
        print(aviso_lpips)
    print("Ordenado por MS-SSIM, que e a menos enganada por granulacao fina. 'grao/ref' acima de 1")
    print("significa MAIS detalhe de alta frequencia que a referencia -- que em imagem quantizada")
    print("costuma ser ruido, nao detalhe.")
    print("NAO COBERTO: nenhuma destas metricas foi validada contra julgamento humano nesta")
    print("bancada. Todas medem distancia ate a referencia escolhida, nunca qualidade absoluta. E")
    print("imagem de trajetoria livre segue sendo teste de quebrou-ou-nao: metrica melhor nao")
    print("conserta a trajetoria ter divergido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
