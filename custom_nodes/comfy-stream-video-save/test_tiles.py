"""quadros_em_tiles tem de dar o mesmo tensor, bit a bit, que comfy.utils.tiled_scale_multidim.

    python_embeded\\python.exe -s -m pytest custom_nodes/comfy-stream-video-save/test_tiles.py -q --import-mode=importlib

(`--import-mode=importlib` porque o diretorio tem `__init__.py` e nome com hifen; sem isso o pytest tenta importar
o pacote do no, que puxa `nodes` e o ComfyUI inteiro.)
"""
import os
import sys

import pytest
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
# o ComfyUI e `../..` quando instalado em ComfyUI/custom_nodes/; `../../ComfyUI` na bancada de origem
for raiz in (os.path.join(AQUI, "..", ".."), os.path.join(AQUI, "..", "..", "ComfyUI")):
    if os.path.isfile(os.path.join(raiz, "comfy", "utils.py")):
        sys.path.insert(0, raiz)
        break

import comfy.utils  # noqa: E402
from tiles import quadros_em_tiles  # noqa: E402

UP = 4  # o LTX usa 32; 4 deixa o teste pequeno sem mudar a logica
UPSCALE = (lambda a: max(0, a * 8 - 7), UP, UP)  # tempo causal como o do LTX
INDICE = (8, UP, UP)


def decode_falso(a):
    """Tile latente -> pixels com a forma do VAE do LTX e valor que depende do tile inteiro (a sobreposicao importa)."""
    x = a[:, :3].float()
    x = torch.cat([x[:, :, :1], x[:, :, 1:].repeat_interleave(8, dim=2)], 2)
    x = x.repeat_interleave(UP, dim=3).repeat_interleave(UP, dim=4)
    return torch.sin(x * 3.1) * 0.7 + a.float().mean() * 0.3


@pytest.mark.parametrize("forma,tile,overlap", [
    ((1, 4, 12, 9, 10), (3, 4, 4), (1, 1, 1)),
    ((1, 4, 46, 11, 8), (8, 4, 4), (1, 2, 2)),    # proporcao do caso real (46 latentes, tile 8, sobreposicao 1)
    ((1, 4, 7, 5, 5), (2, 16, 16), (1, 2, 2)),    # so o tempo e fatiado
    ((1, 4, 5, 4, 4), (8, 8, 8), (1, 2, 2)),      # cabe num tile so
])
def test_igual_ao_tiled_scale_multidim(forma, tile, overlap):
    torch.manual_seed(0)
    lat = torch.randn(forma)
    ref = comfy.utils.tiled_scale_multidim(lat, decode_falso, tile=tile, overlap=overlap, upscale_amount=UPSCALE,
                                           out_channels=3, index_formulas=INDICE, output_device="cpu")
    trechos = list(quadros_em_tiles(lat, decode_falso, tile, overlap, UPSCALE, 3, "cpu", INDICE))
    assert all(t.shape[:2] == (1, 3) for t in trechos)
    got = torch.cat(trechos, 2)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), float((got - ref).abs().max())
    if forma[2] > tile[0]:
        assert len(trechos) > 1  # de fato soltou o video aos pedacos


def test_trecho_maximo_e_uma_janela():
    """No caso real (46 latentes, tile 8, sobreposicao 1) nenhum trecho passa de 57 quadros."""
    lat = torch.randn(1, 4, 46, 6, 6)
    tam = [t.shape[2] for t in quadros_em_tiles(lat, decode_falso, (8, 4, 4), (1, 2, 2), UPSCALE, 3, "cpu", INDICE)]
    assert sum(tam) == 46 * 8 - 7 and max(tam) <= 57, tam
