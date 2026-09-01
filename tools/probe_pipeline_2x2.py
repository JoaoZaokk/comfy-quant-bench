"""O erro do encoder e o do difusor somam, multiplicam, ou se cancelam?

A PERGUNTA, e ela veio do dono da bancada: quantizar um estagio da cadeia por vez e medir o erro
ACUMULADO. Um ladder cumulativo (so encoder, depois encoder+difusor) responde metade -- da o total
e nao diz de quem e. Uma grade 2x2 responde inteiro:

                       difusor BF16      difusor 4 bits
    encoder BF16          A (referencia)      C
    encoder 4 bits        B                   D

  D ~= B + C   os dois erros sao independentes e somam
  D  >  B + C   um amplifica o outro -- acumulacao de verdade
  D  <  B + C   se cancelam em parte

A CADEIA NAO FECHA, e o motivo importa. O terceiro estagio e o VAE, e ele **nao e quantizavel por
esta bancada**: medido, `ae.safetensors` tem 70 pesos 4D (convolucao) e ZERO 2D (Linear), e o
caminho inteiro daqui e `convrot_w4a4_linear`, um kernel de GEMM. Os tres perfis do conversor casam
0 camadas nele. Isso nao e escolha: nao ha kernel de convolucao em 4 bits aqui, e e por isso que
ninguem no ecossistema quantiza VAE para 4 bits. O custo de deixar de fora e pequeno -- o VAE e
0,31 de 19,27 GiB do pipeline do Z-Image, **1,6%**.

COMO ISOLA. Trajetoria imposta, como em `probe_epsilon_per_step.py`: o braco A amostra uma vez
gravando cada `(x, timestep)`, e B, C e D nao amostram -- reproduzem exatamente aquelas entradas e
devolvem a propria previsao. Cada passo vira comparacao casada e divergencia de trajetoria nao pode
existir por construcao. Cada braco calcula o proprio condicionamento com o proprio encoder, que e
justamente o eixo em teste.

    python_embeded\\python.exe -s tools/probe_pipeline_2x2.py

NAO COBERTO: mede o RUIDO PREVISTO a cada passo, nunca a imagem. O VAE fica fora por
impossibilidade, entao a cadeia medida e encoder+difusor e nao o pipeline inteiro. Uma semente, um
prompt, um par de modelos. E "somam" aqui e comparacao de medias sobre 8 passos, sem teste formal.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from probe_epsilon_per_step import QUANT_ARM, REF_ARM  # noqa: E402


def roda(src: str, dev: int, timeout: int = 3600) -> dict | None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(dev)
    env.pop("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", None)
    r = subprocess.run([str(ROOT / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print("  BRACO FALHOU\n  stdout:", r.stdout.strip()[-700:])
    print("  stderr:", r.stderr.strip()[-2000:])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet-bf16", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--unet-quant", default="zimage-v2-w4a4.safetensors")
    p.add_argument("--clip-bf16", default="qwen_3_4b.safetensors")
    p.add_argument("--clip-quant", default="qwen_3_4b_w4a4_convrot.safetensors")
    p.add_argument("--prompt", default="a red apple on a weathered wooden table, soft window light")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--json", type=Path, default=ROOT / "bench" / "pipeline_2x2.json")
    a = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="p2x2_"))
    refp = tmp / "ref.pkl"

    comum = {"PROMPT": a.prompt, "SEED": a.seed, "STEPS": a.steps, "CFG": repr(a.cfg),
             "SIDE": a.size, "DEV": str(a.device)}

    print("=" * 78)
    print("Grade 2x2: o erro do encoder e o do difusor somam, multiplicam ou se cancelam?")
    print("Trajetoria imposta -- divergencia de trajetoria nao pode existir por construcao.")
    print("=" * 78)
    print(f"  A  encoder {a.clip_bf16:34}  difusor {a.unet_bf16}")
    print(f"  B  encoder {a.clip_quant:34}  difusor {a.unet_bf16}")
    print(f"  C  encoder {a.clip_bf16:34}  difusor {a.unet_quant}")
    print(f"  D  encoder {a.clip_quant:34}  difusor {a.unet_quant}")
    print()

    print("A (referencia): amostrando e gravando a trajetoria...")
    ref = roda(REF_ARM % {**comum, "UNET": a.unet_bf16, "CLIP": a.clip_bf16, "OUTP": str(refp)},
               a.device)
    if ref is None:
        return 1
    print(f"   {ref['chamadas']} chamadas, forma {ref['shape']}")

    bracos = {
        "B  encoder 4b, difusor BF16": (a.unet_bf16, a.clip_quant),
        "C  encoder BF16, difusor 4b": (a.unet_quant, a.clip_bf16),
        "D  ambos 4 bits": (a.unet_quant, a.clip_quant),
    }
    saidas = {}
    for rotulo, (unet, clip) in bracos.items():
        print(f"{rotulo}: reproduzindo a trajetoria de A...")
        r = roda(QUANT_ARM % {**comum, "UNET": unet, "CLIP": clip, "REFP": str(refp),
                              "OUTP": str(tmp / "x.pkl")}, a.device)
        if r is None:
            continue
        saidas[rotulo] = r

    if len(saidas) < 3:
        print("\nMenos de tres bracos completaram; a soma nao e calculavel.")
        return 1

    def media(rot: str) -> float:
        return sum(x["rel_rmse"] for x in saidas[rot]["passos"]) / len(saidas[rot]["passos"])

    print()
    print(f"{'braco':30} {'eps medio':>11} {'min':>9} {'max':>9} {'cos medio':>10}")
    print("-" * 74)
    for rot in bracos:
        if rot not in saidas:
            continue
        ps = saidas[rot]["passos"]
        e = [x["rel_rmse"] for x in ps]
        c = [x["cos"] for x in ps]
        print(f"{rot:30} {sum(e)/len(e):11.4e} {min(e):9.4e} {max(e):9.4e} "
              f"{sum(c)/len(c):10.6f}")

    rb, rc, rd = (media(k) for k in bracos)
    soma = rb + rc
    print()
    print(f"  B (so encoder)   {rb:.4e}")
    print(f"  C (so difusor)   {rc:.4e}   {rc/rb:.2f}x o do encoder")
    print(f"  B + C            {soma:.4e}   se os erros fossem independentes")
    print(f"  D (os dois)      {rd:.4e}   {rd/soma:.3f} da soma")
    print()
    if rd > soma * 1.10:
        print(f"  D e {rd/soma:.2f}x a soma: um erro AMPLIFICA o outro. Acumulacao de verdade.")
    elif rd < soma * 0.90:
        print(f"  D e {soma/rd:.2f}x MENOR que a soma: os erros se cancelam em parte.")
    else:
        print("  D esta dentro de 10% da soma: os dois erros sao praticamente independentes.")

    print()
    print("Por passo (sigma alto primeiro):")
    print(f"{'sigma':>8} " + " ".join(f"{k.split()[0]:>10}" for k in bracos))
    n = min(len(saidas[k]["passos"]) for k in bracos if k in saidas)
    for i in range(n):
        sig = saidas[list(bracos)[0]]["passos"][i]["sigma"]
        col = " ".join(f"{saidas[k]['passos'][i]['rel_rmse']:10.4e}"
                       for k in bracos if k in saidas)
        print(f"{sig:8.3f} {col}")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({"config": vars(a) | {"json": str(a.json)},
                                  "referencia": {k: ref[k] for k in ("chamadas", "shape")},
                                  "bracos": saidas}, indent=1, default=str), encoding="utf-8")
    print(f"\nescrito {a.json}")
    print("\nNAO COBERTO: mede o ruido previsto por passo, nunca a imagem. O VAE fica de fora por")
    print("impossibilidade -- 100% convolucional, e o kernel daqui e de GEMM -- entao a cadeia")
    print("medida e encoder+difusor e nao o pipeline inteiro. Uma semente, um prompt. E 'somam' e")
    print("comparacao de medias sobre os passos, sem teste formal de hipotese.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
