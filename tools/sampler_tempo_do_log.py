"""Tempo do SAMPLER por prompt, lido do log stderr de um servidor ComfyUI.

Por que existe (MEDIDO 2026-09-14): `tools/ltx_video.py` grava `s_por_passo` e `s_por_quadro` como
parede da corrida inteira dividida por N -- carga do modelo, encoder, sampler, VAEs e mux, tudo num
numero. Isso foi publicado por uma hora como "tempo por passo" e estava errado por 3x. O instrumento
por passo e a barra do tqdm que o sampler imprime no stderr do servidor: `8/8 [00:18<00:00, 2.30s/it]`.
Este script pareia, em ordem, cada `got prompt` com as barras completas que apareceram ate o
`Prompt executed in X seconds` seguinte, e imprime os dois tempos lado a lado.

Uso:
    python -s tools/sampler_tempo_do_log.py .scratch/comfy_8190_j.err [--json saida.json]

O que ele NAO cobre (impresso no fim de toda execucao): a barra e uma por chamada de sampler e o
tqdm a imprime duas vezes no fechamento (100% e close) -- o script deduplica barras identicas
consecutivas, entao dois samplers de MESMA duracao num prompt contam como um; prompts sem barra
(encode, decode, carga) saem com sampler vazio; o tempo da barra e o do tqdm (arredondado ao
segundo no campo `MM:SS`, com o s/it em duas casas); nada aqui identifica QUAL modelo rodou -- isso
vem do JSON da corrida ou da ordem da fila; um log de servidor reiniciado no meio comeca do zero.
"""
import argparse
import json
import re
import sys
from pathlib import Path

RE_GOT = re.compile(r"got prompt")
RE_BAR = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)<00:00, +([0-9.]+)(s/it|it/s)\]")
# Acima de 600 s o ComfyUI troca o formato (main.py:400-402): "Prompt executed in 00:12:41".
# A primeira versao desta ferramenta so lia "in X seconds" e marcava toda corrida longa como
# "(sem fim)" -- foi assim que o BF16 do 2.5 (761 s) ficou sem parede na leitura de 2026-09-14.
RE_END = re.compile(r"Prompt executed in (?:([0-9.]+) seconds|(\d+):(\d\d):(\d\d))")
RE_LOAD = re.compile(r"Requested to load ([A-Za-z_0-9]+)")


def parse(texto: str) -> list[dict]:
    prompts: list[dict] = []
    atual: dict | None = None
    # tqdm usa \r; trata \r como quebra para que cada snapshot vire uma "linha"
    for linha in texto.replace("\r", "\n").split("\n"):
        if RE_GOT.search(linha):
            if atual is not None:
                prompts.append(atual)
            atual = {"barras": [], "carregou": [], "parede_s": None}
            continue
        if atual is None:
            continue
        m = RE_LOAD.search(linha)
        if m:
            atual["carregou"].append(m.group(1))
        for m in RE_BAR.finditer(linha):
            feito, total, mm, ss, taxa, unid = m.groups()
            if feito != total:
                continue
            seg = int(mm) * 60 + int(ss)
            taxa = float(taxa)
            s_it = taxa if unid == "s/it" else (1.0 / taxa if taxa else float("nan"))
            barra = {"passos": int(total), "segundos": seg, "s_por_it": round(s_it, 3)}
            if atual["barras"] and atual["barras"][-1]["passos"] == barra["passos"] and \
                    atual["barras"][-1]["segundos"] == barra["segundos"]:
                continue  # o tqdm imprime a barra completa duas vezes (100% e close)
            atual["barras"].append(barra)
        m = RE_END.search(linha)
        if m:
            if m.group(1) is not None:
                atual["parede_s"] = float(m.group(1))
            else:
                h, mi, se = (int(x) for x in m.groups()[1:])
                atual["parede_s"] = float(h * 3600 + mi * 60 + se)
            prompts.append(atual)
            atual = None
    if atual is not None:
        prompts.append(atual)
    return prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("log", help="stderr do servidor ComfyUI (ex.: .scratch/comfy_8190_j.err)")
    p.add_argument("--json", default=None, help="grava a lista de prompts em JSON")
    p.add_argument("--so-com-barra", action="store_true", help="omite prompts sem sampler")
    a = p.parse_args()
    texto = Path(a.log).read_text(encoding="utf-8", errors="replace")
    prompts = parse(texto)
    print(f"{a.log}: {len(prompts)} prompt(s)")
    print(f"{'#':>3}  {'sampler (passos @ s/it = s)':34s}  {'parede s':>9s}  carregou")
    for i, pr in enumerate(prompts, 1):
        if a.so_com_barra and not pr["barras"]:
            continue
        samp = "; ".join(f"{b['passos']} @ {b['s_por_it']:.2f} s/it = {b['segundos']} s" for b in pr["barras"]) or "-"
        parede = f"{pr['parede_s']:.1f}" if pr["parede_s"] is not None else "(sem fim)"
        print(f"{i:>3}  {samp:34s}  {parede:>9s}  {','.join(pr['carregou'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(prompts, indent=1), encoding="utf-8")
        print(f"json em {a.json}")
    print("NAO COBERTO: barras identicas consecutivas sao deduplicadas (dois samplers de mesma duracao num "
          "prompt contam como um); prompts sem barra saem com '-'; tempo da barra e o do tqdm (segundo "
          "inteiro); qual modelo rodou vem do JSON da corrida ou da ordem da fila, nao deste log.")


if __name__ == "__main__":
    sys.exit(main())
