"""Cruzamento X: o mesmo braço fica à mesma distância do BF16 no ComfyUI e no diffusers?

Mede dentro de cada runtime (braço contra BF16, mesmo prompt e semente) e compara as distâncias
entre runtimes. Escrito sob a suposição, NÃO medida, de que o ruído inicial diferia entre runtimes;
medido depois, no klein ele é o mesmo (mesmo braço, dois runtimes: SSIM 0,98), então a comparação
direta imagem-contra-imagem também vale -- ver `render_klein_diffusers.py`. Lê os nomes:
  ComfyUI   <stem do arquivo>__p<i>_s<s>.png
  diffusers <rotulo>__p<i>_s<s>.png
Métricas: SSIM e PSNR (torchmetrics) e LPIPS-VGG se os pesos JÁ estiverem em cache -- nunca baixa.

NÃO COBRE: nenhuma destas métricas foi validada contra olho nesta bancada; ver metricas_imagem.py.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metricas_imagem import carrega, cria_lpips, medir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lado", action="append", required=True, metavar="NOME=DIR:REF:BRACO1,BRACO2",
                   help="um por runtime: diretorio, prefixo da referencia, prefixos dos braços")
    p.add_argument("--chaves", nargs="+", required=True, help="p0_s11 p0_s12 ...")
    p.add_argument("--saida", type=Path)
    a = p.parse_args()

    lp, _aviso = cria_lpips("so_cache")     # nunca baixa

    out = {}
    for spec in a.lado:
        nome, resto = spec.split("=", 1)
        d, ref, bracos = resto.rsplit(":", 2) if resto.count(":") > 2 else resto.split(":")
        d = Path(d)
        out[nome] = {}
        for b in bracos.split(","):
            linhas = []
            for k in a.chaves:
                r, x = carrega(d / f"{ref}__{k}.png"), carrega(d / f"{b}__{k}.png")
                linhas.append(medir(x, r, lp, quais=("ssim", "psnr")))
            out[nome][b] = {m: statistics.fmean(l[m] for l in linhas) for m in linhas[0]}
            out[nome][b]["por_chave"] = dict(zip(a.chaves, linhas))
            print(f"{nome:10} {b:40} " + "  ".join(f"{m} {out[nome][b][m]:.4f}" for m in linhas[0]))
    if lp is None:
        print("LPIPS NAO medido: pesos VGG16 fora do cache (esta ferramenta nunca baixa).")
    if a.saida:
        a.saida.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
