"""Metricas da bateria Qwen-Image-2.1: cada DiT contra o bf16 do mesmo prompt/seed, mais tempo e memoria do log.

    python_embeded\\python.exe -s metricas_bateria.py <saida.json> <log:ordem> [<log:ordem> ...]
Os logs (stderr do ComfyUI) sao casados com as ordens `bateria/ordem*.txt` pela sequencia de prompts executados.
"""
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, "F:/COMFY_PORTABLE/tools")
from metricas_imagem import carrega, grao  # noqa: E402
from torchmetrics.functional.image import (  # noqa: E402
    multiscale_structural_similarity_index_measure as msssim,
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)

AQUI = Path(__file__).parent
IMG = Path("F:/COMFY_PORTABLE/ComfyUI/output/qwen21_bateria")
NOME = re.compile(r"bateria/(\w+?)_p(\d+)_s(\d+)\.json")


def tempos(log: Path):
    """Uma entrada por prompt executado: it/s final do tqdm, segundos, e a ultima carga de modelo vista."""
    out, its, carga = [], None, None
    for linha in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"25/25 \[[^\]]*?([\d.]+)(it/s|s/it)\]", linha)
        if m:
            v = float(m.group(1))
            its = v if m.group(2) == "it/s" else 1 / v
        m = re.search(r"(loaded completely|loaded partially)[^\n]*", linha)
        if m:
            carga = m.group(0)[:200]
        # acima de um minuto o ComfyUI escreve hh:mm:ss em vez de "N seconds"
        m = re.search(r"Prompt executed in (?:([\d.]+) seconds|(\d+):(\d+):([\d.]+))", linha)
        if m:
            seg = float(m.group(1)) if m.group(1) else int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))
            out.append({"its": its, "seg": seg, "carga": carga})
            its = None
    return out


def main():
    saida = Path(sys.argv[1])
    # cada argumento e `log:ordem` (ordem relativa a este diretorio), ex. bateria_comfy.log:bateria/ordem.txt
    pares = [a.split(":", 1) for a in sys.argv[2:]]
    execucoes = []
    for log, ordem in ((Path(l), AQUI / o) for l, o in pares):
        grafos = [g for g in ordem.read_text().split() if g]
        ts = tempos(log)
        if len(ts) != len(grafos):
            print(f"AVISO {log.name}: {len(ts)} execucoes para {len(grafos)} grafos de {ordem.name}")
        execucoes += list(zip(grafos, ts))
    por_dit = {}
    for g, t in execucoes:
        dit, p, s = NOME.search(g).groups()
        por_dit.setdefault(dit, []).append({"p": int(p), "s": int(s), **t})

    res = {"por_imagem": {}, "resumo": {}}
    refs = {}
    for dit, linhas in por_dit.items():
        for L in linhas:
            f = IMG / dit / f"p{L['p']}_s{L['s']}_00001_.png"
            x = carrega(f)
            # hash dos PIXELS: o PNG leva o grafo nos metadados, entao bytes do arquivo sempre diferem
            L["sha256"] = hashlib.sha256(x.numpy().tobytes()).hexdigest()
            if dit == "bf16":
                refs[(L["p"], L["s"])] = (x, grao(x), L["sha256"])
    for dit, linhas in por_dit.items():
        for L in linhas:
            if dit == "bf16":
                continue
            r, g_ref, sha_ref = refs[(L["p"], L["s"])]
            x = carrega(IMG / dit / f"p{L['p']}_s{L['s']}_00001_.png")
            L.update({"identica": L["sha256"] == sha_ref, "psnr": float(psnr(x, r, data_range=1.0)),
                      "ssim": float(ssim(x, r, data_range=1.0)), "msssim": float(msssim(x, r, data_range=1.0)),
                      "grao": grao(x) / g_ref})
        res["por_imagem"][dit] = linhas
        # a 1a imagem de cada DiT inclui a carga do modelo: fica fora da media de tempo
        quentes = linhas[1:] or linhas  # controle de 1 imagem: sem media quente
        r = {"n": len(linhas), "its_mediana": st.median(L["its"] for L in quentes if L["its"]),
             "seg_mediana_quente": st.median(L["seg"] for L in quentes), "seg_primeira": linhas[0]["seg"],
             "carga": linhas[-1]["carga"]}
        if dit != "bf16":
            fin = [L["psnr"] for L in linhas if L["psnr"] != float("inf")]
            r.update({"identicas": sum(L["identica"] for L in linhas),
                      "psnr_media_finita": st.mean(fin) if fin else None,
                      "ssim_media": st.mean(L["ssim"] for L in linhas),
                      "msssim_media": st.mean(L["msssim"] for L in linhas),
                      "msssim_min": min(L["msssim"] for L in linhas),
                      "grao_media": st.mean(L["grao"] for L in linhas)})
        res["resumo"][dit] = r
    saida.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for dit, r in res["resumo"].items():
        print(dit, json.dumps(r))


if __name__ == "__main__":
    main()
