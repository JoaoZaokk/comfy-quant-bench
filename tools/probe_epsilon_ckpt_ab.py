"""Dois checkpoints quantizados contra o MESMO BF16, passo a passo, entradas casadas.

DE ONDE VEM. `tools/probe_epsilon_per_step.py` compara UM checkpoint por dois caminhos de
dispatch. Esta variante compara DOIS checkpoints pelo mesmo caminho -- que e o que responde
"o criterio A escolhe melhor que o criterio B?", e nao "o kernel A erra menos que o B".

A tecnica e a mesma e e a razao de este arquivo existir: o braco BF16 grava cada
`(x, timestep)` que recebe via `model_options["model_function_wrapper"]`, e cada braco
quantizado **reproduz exatamente aquelas entradas**. Divergencia de trajetoria deixa de
existir por construcao. Sem isso, com 8 passos, uma perturbacao minuscula desvia o sampler e
o que se mede e caos, nao fidelidade -- registrado em 2026-08-30, quando a imagem final foi
usada como juiz e nao carregava sinal nenhum.

O relatorio separa sigma ALTO de sigma BAIXO porque e ai que a pergunta vive: um criterio
ponderado por sigma alto deve ganhar la e pagar aqui. Se ganhar nos dois, nao era uma troca;
se perder nos dois, o peso so adicionou ruido.

NAO COBERTO: um prompt, uma semente, uma placa, sem metrica perceptual. E o BF16 e o alvo,
nao a verdade -- ele nunca foi validado contra float32. Nada aqui diz que a imagem fica
melhor: diz que a previsao do modelo fica mais perto da do BF16, que e coisa diferente.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from probe_epsilon_per_step import QUANT_ARM, REF_ARM  # noqa: E402

OUT = ROOT / "bench" / "epsilon_ckpt_ab"


def rodar(src, device, timeout=3600):
    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env.pop("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", None)
    r = subprocess.run([str(ROOT / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print("  BRACO FALHOU\n  stdout:", r.stdout.strip()[-600:])
    print("  stderr:", r.stderr.strip()[-2500:])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-unet", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--arm", action="append", required=True, metavar="ROTULO=UNET[,CLIP]",
                   help="pode repetir; o primeiro e a base das razoes. Com um CLIP depois da "
                        "virgula, o braco usa esse text encoder em vez de --clip -- e assim o "
                        "eixo que varia passa a ser o ENCODER, com o mesmo modelo de difusao "
                        "e a mesma trajetoria imposta.")
    p.add_argument("--clip", default="qwen_3_4b.safetensors")
    p.add_argument("--prompt",
                   default="a red apple on a weathered wooden table, soft window light")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    arms = []
    for spec in a.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm precisa de ROTULO=UNET[,CLIP], recebi {spec!r}")
        rotulo, resto = spec.split("=", 1)
        unet, _, clip_do_braco = resto.partition(",")
        arms.append((rotulo, unet, clip_do_braco or a.clip))
    if len(arms) < 2:
        raise SystemExit("dois --arm no minimo; com um so nao ha comparacao")

    OUT.mkdir(parents=True, exist_ok=True)
    refp = OUT / "trajetoria_bf16.pkl"
    common = {"UNET": a.ref_unet, "CLIP": a.clip, "PROMPT": a.prompt, "SEED": a.seed,
              "STEPS": a.steps, "CFG": repr(a.cfg), "SIDE": a.size,
              "DEV": a.device, "OUTP": str(refp)}

    print(f"device cuda:{a.device}  |  a trajetoria do BF16 e imposta a TODOS os bracos\n")
    print("--- referencia BF16 (grava a trajetoria) ---", flush=True)
    ref = rodar(REF_ARM % common, a.device)
    if not ref:
        return 1
    print(f"  {ref['chamadas']} chamadas, saida {ref['shape']}")
    print(f"  sigmas: {', '.join(f'{s:.3f}' for s in ref['sigmas'])}\n")

    got = {}
    for rotulo, arquivo, clip_do_braco in arms:
        marca = arquivo if clip_do_braco == a.clip else f"{arquivo}  +  clip {clip_do_braco}"
        print(f"--- {rotulo}  ({marca}) ---", flush=True)
        q = dict(common)
        q.update({"UNET": arquivo, "CLIP": clip_do_braco, "REFP": str(refp), "OUTP": ""})
        res = rodar(QUANT_ARM % q, a.device)
        if not res:
            return 1
        got[rotulo] = res["passos"]
        print(f"  {len(res['passos'])} passos medidos")

    rotulos = [r for r, _, _ in arms]
    base = rotulos[0]
    n = len(got[base])
    sigmas = [s["sigma"] for s in got[base]]
    corte = sorted(sigmas)[n // 2]

    print()
    print("-" * 78)
    cab = f"{'passo':>6} {'sigma':>8}" + "".join(f"{r:>14}" for r in rotulos)
    print("rel-RMSE do epsilon contra o BF16, entradas casadas")
    print("-" * 78)
    print(cab)
    for i in range(n):
        linha = f"{i:>6} {sigmas[i]:>8.3f}"
        for r in rotulos:
            linha += f"{got[r][i]['rel_rmse']:>14.4e}"
        print(linha)

    def media(r, filtro):
        vs = [s["rel_rmse"] for s in got[r] if filtro(s["sigma"])]
        return sum(vs) / len(vs) if vs else float("nan")

    faixas = [("todos os passos", lambda s: True),
              (f"sigma ALTO (>= {corte:.3f})", lambda s: s >= corte),
              (f"sigma BAIXO (< {corte:.3f})", lambda s: s < corte)]
    print()
    print("-" * 78)
    print("media por faixa, e a razao sempre >= 1 com a direcao dita")
    print("-" * 78)
    print(f"{'faixa':>26}" + "".join(f"{r:>14}" for r in rotulos) + "   quem ganha")
    for nome, filtro in faixas:
        vals = {r: media(r, filtro) for r in rotulos}
        vencedor = min(vals, key=vals.get)
        perdedor = max(vals, key=vals.get)
        razao = vals[perdedor] / vals[vencedor] if vals[vencedor] else float("inf")
        linha = f"{nome:>26}" + "".join(f"{vals[r]:>14.4e}" for r in rotulos)
        print(f"{linha}   {vencedor} {razao:.4f}x mais fiel")

    print()
    print(f"{'passos vencidos':>26}", end="")
    contagem = {r: 0 for r in rotulos}
    for i in range(n):
        melhor = min(rotulos, key=lambda r: got[r][i]["rel_rmse"])
        contagem[melhor] += 1
    print("".join(f"{contagem[r]:>14}" for r in rotulos) + f"   de {n}")

    print()
    print("NAO COBERTO: um prompt, uma semente, uma placa, sem metrica perceptual. O BF16 e o")
    print("  alvo e nao a verdade -- nunca foi validado contra float32. E isto mede a PREVISAO")
    print("  do modelo, nao a imagem: a imagem livre ja foi medida e nao carrega este sinal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
