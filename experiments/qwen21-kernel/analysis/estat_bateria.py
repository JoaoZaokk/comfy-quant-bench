"""Fase 4, itens 6/7: estatistica pareada da bateria contra BF16 (criterio_fase4.md).

Le bateria_resultados.jsonl (registros da tag pedida), mede PSNR/MAE de cada render contra a referencia BF16 do mesmo
grafo e compara bracos dois a dois NO MESMO GRAFO (diferenca pareada). IC 95 % por bootstrap por prompt (cluster: todas
as seeds de um prompt saem juntas; 10 000 reamostragens, semente fixa), t pareado e Wilcoxon.

    python -s estat_bateria.py --tag g --refdir <dir> --pares a:b[,c:d] [--saida arq.md]

Para cada par a:b informa media e IC de (b - a): positivo = b mais perto do BF16 (PSNR) / menor MAE (sinal invertido).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

HERE = Path(__file__).resolve().parent


def carrega(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def bootstrap_cluster(d, grupos, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    chaves = sorted(set(grupos))
    por = {c: d[np.array(grupos) == c] for c in chaves}
    medias = np.empty(n)
    for i in range(n):
        amostra = rng.choice(chaves, size=len(chaves), replace=True)
        medias[i] = np.concatenate([por[c] for c in amostra]).mean()
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--refdir", required=True, type=Path)
    ap.add_argument("--pares", required=True)
    ap.add_argument("--saida", type=Path)
    a = ap.parse_args()

    tags = a.tag.split(",")
    regs = {}
    for linha in open(HERE / "bateria_resultados.jsonl", encoding="utf-8"):
        r = json.loads(linha)
        if r.get("tag") in tags and r.get("files"):
            braco = r["arm"] if len(tags) == 1 else f'{r["tag"]}.{r["arm"]}'       # varias tags: "tag.braco"
            regs[(braco, r["grafo"])] = r                                  # o ultimo registro de cada (braco, grafo) vale
    metr = {}
    tempos = {}
    for (arm, grafo), r in regs.items():
        ref = sorted(a.refdir.glob(f"{grafo}_*.png"))
        if not ref:
            continue
        ra, rb = carrega(ref[0]), carrega(r["files"][0])
        metr[(arm, grafo)] = (psnr(ra, rb), float(np.abs(ra - rb).mean()), bool(np.isfinite(rb).all()), float(rb.mean()))
        tempos.setdefault(arm, []).append(r["ksampler_s"] or 0)

    out = [f"# Bateria `{a.tag}` contra BF16 ({a.refdir})", ""]
    out.append("| braço | n | PSNR médio (mín–máx) | MAE médio | KSampler mediana | imagens escuras (média < 5) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for arm in sorted({k[0] for k in metr}):
        v = [metr[k] for k in metr if k[0] == arm]
        ps = np.array([x[0] for x in v]); ma = np.array([x[1] for x in v])
        escuras = sum(1 for x in v if x[3] < 5)
        out.append(f"| {arm} | {len(v)} | {ps.mean():.2f} ({ps.min():.2f}–{ps.max():.2f}) | {ma.mean():.2f} | "
                   f"{np.median(tempos[arm]):.2f} s | {escuras} |")
    out += ["", "| par (b − a) | n | ΔPSNR médio | IC 95 % (cluster) | t pareado p | Wilcoxon p | b vence | ΔMAE médio | IC 95 % ΔMAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    resumo = {}
    for par in a.pares.split(","):
        x, y = par.split(":")
        grafos = sorted(g for (arm, g) in metr if arm == x and (y, g) in metr)
        if not grafos:
            out.append(f"| {y} − {x} | 0 | — | — | — | — | — | — | — |"); continue
        dp = np.array([metr[(y, g)][0] - metr[(x, g)][0] for g in grafos])
        dm = np.array([metr[(x, g)][1] - metr[(y, g)][1] for g in grafos])   # positivo = b com MAE menor
        prompts = [re.match(r"(p\d+)_", g).group(1) for g in grafos]
        lo, hi = bootstrap_cluster(dp, prompts)
        lom, him = bootstrap_cluster(dm, prompts)
        t_p = float(stats.ttest_rel(dp, np.zeros_like(dp)).pvalue) if len(dp) > 1 else float("nan")
        w_p = float(stats.wilcoxon(dp).pvalue) if len(dp) > 5 and np.any(dp != 0) else float("nan")
        vence = int((dp > 0).sum())
        out.append(f"| {y} − {x} | {len(dp)} | {dp.mean():+.3f} dB | [{lo:+.3f}, {hi:+.3f}] | {t_p:.3g} | {w_p:.3g} | "
                   f"{vence}/{len(dp)} | {dm.mean():+.3f} | [{lom:+.3f}, {him:+.3f}] |")
        resumo[f"{y}-{x}"] = dict(n=len(dp), dpsnr=float(dp.mean()), ic=[lo, hi], t_p=t_p, w_p=w_p, vence=vence,
                                  dmae=float(dm.mean()), ic_mae=[lom, him], sd=float(dp.std(ddof=1)) if len(dp) > 1 else None)
    txt = "\n".join(out) + "\n"
    print(txt)
    if a.saida:
        a.saida.write_text(txt, encoding="utf-8")
        a.saida.with_suffix(".json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
