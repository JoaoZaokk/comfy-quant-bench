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

COMO BIBLIOTECA (revisao 2026-09-29): `medir(x, r)` devolve as mesmas colunas que o `main` imprime,
`identica(x, r)` compara pixels, `cria_lpips(politica)` decide se LPIPS baixa pesos, e
`imagem_unica(pasta, prefixo)` acha a imagem de um prompt sem supor `_00001_` (um grafo rodado de
novo grava `_00002_`, e ler `_00001_` fixo mede a imagem ANTIGA). Quatro scripts reimportavam o
torchmetrics e recalculavam por conta propria, com politicas de LPIPS diferentes.
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


VGG16_CACHE = Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
METRICAS = ("psnr", "ssim", "msssim", "grao")


def identica(x: torch.Tensor, r: torch.Tensor) -> bool:
    """Mesmos pixels. (O PNG leva o grafo nos metadados, entao bytes do arquivo sempre diferem.)"""
    return x.shape == r.shape and bool(torch.equal(x, r))


def sha_pixels(x: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def cria_lpips(politica: str = "nunca") -> tuple[object | None, str]:
    """(fn, aviso). `nunca`: nao mede. `so_cache`: mede so se os pesos VGG16 ja estao em cache --
    NUNCA baixa. `baixar`: pode baixar 528 MB (so com pedido explicito, ver docstring do modulo)."""
    if politica not in ("nunca", "so_cache", "baixar"):
        raise ValueError(f"politica de LPIPS desconhecida: {politica!r}")
    if politica == "nunca" or (politica == "so_cache" and not VGG16_CACHE.is_file()):
        return None, ("LPIPS nao medido. Ele exige VGG16 (528 MB)"
                      + (", que JA esta em cache aqui." if VGG16_CACHE.is_file()
                         else ", fora do cache; so --com-lpips baixa."))
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        fn.eval()
        return fn, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"LPIPS falhou ({type(exc).__name__}: {str(exc)[:140]})"


def medir(x: torch.Tensor, r: torch.Tensor, lpips_fn=None, quais=METRICAS,
          grao_ref: float | None = None) -> dict:
    """As metricas de `x` contra a referencia `r`, na ordem de `quais` (+ `lpips` se `lpips_fn`).
    Levanta ValueError se as formas diferem -- medir imagens de tamanhos diferentes nao e uma
    distancia, e um erro de quem montou a comparacao."""
    if x.shape != r.shape:
        raise ValueError(f"forma diferente da referencia: {tuple(x.shape)} contra {tuple(r.shape)}")
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure as msssim,
        peak_signal_noise_ratio as psnr,
        structural_similarity_index_measure as ssim,
    )
    calc = {"psnr": lambda: float(psnr(x, r, data_range=1.0)),
            "ssim": lambda: float(ssim(x, r, data_range=1.0)),
            "msssim": lambda: float(msssim(x, r, data_range=1.0)),
            "grao": lambda: grao(x) / (grao_ref if grao_ref is not None else grao(r))}
    v = {m: calc[m]() for m in quais}
    if lpips_fn is not None:
        with torch.no_grad():
            v["lpips"] = float(lpips_fn(x.clamp(0, 1), r.clamp(0, 1)))
    return v


def _sha_arquivo_pixels(p: Path) -> str:
    import hashlib
    return hashlib.sha256(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest()


def imagem_unica(pasta: Path, prefixo: str) -> Path:
    """A imagem `<prefixo>_NNNNN_.png` de `pasta`, sem supor o contador.

    Uma: devolve. Nenhuma: FileNotFoundError. Varias (o grafo rodou de novo): se todas tem os
    MESMOS pixels, tanto faz qual -- devolve a de maior contador; se diferem, ValueError, porque
    escolher uma seria escolher o resultado. Quem tem o JSONL do executor deve usar os `files`
    dele em vez disto."""
    achadas = sorted(Path(pasta).glob(f"{prefixo}_[0-9][0-9][0-9][0-9][0-9]_.png"))
    if not achadas:
        raise FileNotFoundError(f"nenhuma {prefixo}_NNNNN_.png em {pasta}")
    if len(achadas) > 1 and len({_sha_arquivo_pixels(a) for a in achadas}) > 1:
        raise ValueError(f"{len(achadas)} imagens DIFERENTES para {prefixo} em {pasta}: "
                         + ", ".join(a.name for a in achadas)
                         + " -- passe o JSONL do executor para saber qual e de qual execucao")
    return achadas[-1]


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

    lpips_fn, aviso_lpips = cria_lpips("baixar" if a.com_lpips else "nunca")
    if not a.com_lpips:
        aviso_lpips = ("LPIPS nao medido. Passe --com-lpips para incluir; ele exige VGG16 (528 MB)"
                       + (", que JA esta em cache aqui." if VGG16_CACHE.is_file()
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
        linhas.append({"nome": img.name, **medir(x, r, lpips_fn, grao_ref=g_ref)})

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
