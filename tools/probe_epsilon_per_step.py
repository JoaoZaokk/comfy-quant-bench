"""Comparar a PREVISAO do modelo em entradas casadas, em vez do resultado de um processo livre.

O PROBLEMA QUE ISTO RESOLVE. Em 2026-08-30 mediu-se aqui que o ramo INT4 nativo perde para o
fallback INT8 em 24/24 camadas (1,49x), e depois que na imagem final o placar era 2x1 em 3
sementes, com os dois ramos a ~0,3-0,5 do BF16. As duas medicoes discordam porque nenhuma
delas responde a pergunta certa: erro de camada nao ve o modelo, e imagem final ve o
resultado de uma TRAJETORIA que diverge -- uma perturbacao minuscula no passo 0 manda o
sampler para outro ponto, e o destino continua sendo uma imagem boa.

A saida que o mundo de LLM achou para o mesmo muro foi medir a DISTRIBUICAO DE SAIDA do
modelo contra a precisao cheia (KL, `llama-perplexity --kl-divergence-base`), e nao o texto
gerado. O analogo em difusao e comparar o RUIDO PREVISTO, e o truque que torna isso possivel
e impor a mesma trajetoria a todos os bracos.

COMO: o braco BF16 roda a amostragem uma vez com um `model_function_wrapper` que GRAVA cada
chamada -- (x, timestep, cond) e a saida. Os bracos quantizados nao amostram: eles REPRODUZEM
exatamente aquelas entradas gravadas e devolvem sua propria previsao. Assim todo passo e uma
comparacao casada, e a divergencia de trajetoria -- que embaralhou a leitura visual -- deixa
de existir por construcao.

O QUE VARIA: `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` entre os dois bracos quantizados.
O QUE E MANTIDO: o modelo, a semente, o prompt, e agora tambem a TRAJETORIA INTEIRA.

O QUE UM RESULTADO SIGNIFICA:
  - se o erro por passo separa os ramos de forma estavel, e candidato a criterio para o
    `tools/quant_mixed.py`, que hoje escolhe formato por erro de camada -- e erro de camada
    ja foi medido aqui como nao-preditivo.
  - se os dois ramos tem erro parecido em todos os passos, entao a diferenca visual de ontem
    era mesmo so trajetoria, e nenhum dos dois e "pior".
  - o perfil ao longo dos passos e informacao propria: erro concentrado nos primeiros passos
    (onde a estrutura e decidida) pesa diferente de erro no fim (onde e textura).

NAO COBERTO: um prompt e uma semente; sem metrica perceptual; so Z-Image; so imagem; e o
BF16 e a referencia, o que assume que a referencia esta certa -- ela e o alvo, nao a verdade.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "epsilon_per_step"

REF_ARM = r'''
import json, os, pickle, sys
sys.path.insert(0, "ComfyUI"); sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd, comfy.sample

UNET=%(UNET)r; CLIP=%(CLIP)r; CLIPTYPE=%(CLIPTYPE)r; PROMPT=%(PROMPT)r; SEED=%(SEED)d
STEPS=%(STEPS)d; CFG=%(CFG)s; SIDE=%(SIDE)d; DEV=%(DEV)r; OUTP=%(OUTP)r

model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", UNET))
clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=getattr(comfy.sd.CLIPType, CLIPTYPE.upper()))
positive=[list(clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0])]
negative=[list(clip.encode_from_tokens_scheduled(clip.tokenize(""))[0])]
del clip
import comfy.model_management as mm; mm.soft_empty_cache()

gravado = []
def wrapper(apply_model, args):
    x = args["input"]; t = args["timestep"]; c = args["c"]
    out = apply_model(x, t, **c)
    gravado.append({
        "x": x.detach().to("cpu", torch.float32).clone(),
        "t": t.detach().to("cpu", torch.float32).clone(),
        "out": out.detach().to("cpu", torch.float32).clone(),
    })
    return out

model.model_options = dict(model.model_options)
model.model_options["model_function_wrapper"] = wrapper

lf = model.model.latent_format
latent = torch.zeros([1, lf.latent_channels, SIDE // 8, SIDE // 8], device="cpu")
noise = comfy.sample.prepare_noise(latent, SEED, None)
comfy.sample.sample(model, noise, STEPS, CFG, "euler", "simple", positive, negative,
                    latent, denoise=1.0, disable_pbar=True, seed=SEED)

# `c` guarda os condicionamentos; salvos a parte porque sao os mesmos em toda chamada.
with open(OUTP, "wb") as f:
    pickle.dump({"chamadas": gravado, "n": len(gravado)}, f, protocol=4)
print("RESULT " + json.dumps({"chamadas": len(gravado),
                              "shape": list(gravado[0]["out"].shape),
                              "sigmas": [float(g["t"].flatten()[0]) for g in gravado]}))
'''

QUANT_ARM = r'''
import json, os, pickle, sys
sys.path.insert(0, "ComfyUI"); sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd
from comfy_kitchen.backends import cuda as ckc

UNET=%(UNET)r; CLIP=%(CLIP)r; CLIPTYPE=%(CLIPTYPE)r; PROMPT=%(PROMPT)r; DEV=%(DEV)r
REFP=%(REFP)r; OUTP=%(OUTP)r

with open(REFP, "rb") as f:
    ref = pickle.load(f)

model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", UNET))
clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=getattr(comfy.sd.CLIPType, CLIPTYPE.upper()))
positive=[list(clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0])]
negative=[list(clip.encode_from_tokens_scheduled(clip.tokenize(""))[0])]
del clip
import comfy.model_management as mm; mm.soft_empty_cache()

# Reproduz as entradas gravadas: mesma trajetoria, previsao propria.
capturado = []
def wrapper(apply_model, args):
    x = args["input"]; t = args["timestep"]; c = args["c"]
    i = len(capturado)
    if i < ref["n"]:
        x = ref["chamadas"][i]["x"].to(x.device, x.dtype)
        t = ref["chamadas"][i]["t"].to(t.device, t.dtype)
    out = apply_model(x, t, **c)
    capturado.append(out.detach().to("cpu", torch.float32).clone())
    return out

model.model_options = dict(model.model_options)
model.model_options["model_function_wrapper"] = wrapper

import comfy.sample
lf = model.model.latent_format
side = ref["chamadas"][0]["x"].shape[-1]
latent = torch.zeros([1, lf.latent_channels, side, side], device="cpu")
noise = comfy.sample.prepare_noise(latent, %(SEED)d, None)
comfy.sample.sample(model, noise, ref["n"], %(CFG)s, "euler", "simple", positive, negative,
                    latent, denoise=1.0, disable_pbar=True, seed=%(SEED)d)

linhas = []
for i, o in enumerate(capturado[:ref["n"]]):
    r = ref["chamadas"][i]["out"]
    d = o - r
    linhas.append({
        "passo": i,
        "sigma": float(ref["chamadas"][i]["t"].flatten()[0]),
        "rel_rmse": float(d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()),
        "cos": float(torch.nn.functional.cosine_similarity(
            o.flatten().unsqueeze(0), r.flatten().unsqueeze(0)).item()),
        "norma_prev": float(o.norm()), "norma_ref": float(r.norm()),
    })
print("RESULT " + json.dumps({
    "force": os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag": bool(ckc._FORCE_INT4_INT8_FALLBACK),
    "passos": linhas}))
'''


YAML_BOOT = """
import os.path as _op
import sys as _sys
try:
    import utils.extra_config
    _y = _op.join("ComfyUI", "extra_model_paths.yaml")
    if _op.isfile(_y):
        utils.extra_config.load_extra_path_config(_y)
    else:
        _sys.stderr.write("AVISO: extra_model_paths.yaml ausente; roots montados invisiveis\\n")
except Exception as _e:
    _sys.stderr.write("AVISO: extra_model_paths.yaml nao carregou: " + repr(_e) + "\\n")
"""

MARCA_BOOT = "import comfy.options; comfy.options.enable_args_parsing()"


def injeta_yaml(src: str) -> str:
    """Carrega o `extra_model_paths.yaml` DENTRO do subprocesso do braco.

    Sem isso o subprocesso ve so `ComfyUI/models` e um checkpoint que existe em qualquer root
    montado da `FileNotFoundError`. Terceiro tool desta bancada com o mesmo buraco: o
    `probe_quant_dispatch.py` foi consertado em 2026-09-21 e estes dois ficaram.

    A injecao acontece DEPOIS do `%`-format do template, nunca dentro dele. Duas tentativas minhas
    de embutir no proprio template quebraram, e as duas do mesmo jeito: os templates passam por
    formatacao printf, entao qualquer caractere de porcentagem no texto inserido vira
    especificador. A segunda quebrou no comentario em que eu explicava a primeira.
    """
    if MARCA_BOOT not in src:
        raise ValueError("template sem a marca de boot; injecao do yaml nao pode ser verificada")
    return src.replace(MARCA_BOOT, MARCA_BOOT + YAML_BOOT, 1)


def rodar(src, env_extra=None, timeout=3600):
    src = injeta_yaml(src)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(ARGS.device)
    env.pop("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", None)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([str(ROOT / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print("  BRACO FALHOU\n  stdout:", r.stdout.strip()[-600:])
    print("  stderr:", r.stderr.strip()[-2500:])
    return None


def main() -> int:
    """Corpo do script, atras de um guard para que os templates acima sejam importaveis.

    Estava tudo no escopo do modulo, entao `from probe_epsilon_per_step import QUANT_ARM`
    executava `parse_args()` do OUTRO programa e morria com 'unrecognized arguments'. Os
    templates ARM sao a parte reutilizavel deste arquivo -- e a tecnica de casar entradas
    que vale, nao a comparacao especifica que ele faz.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-unet", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--quant-unet", default="zimage-v2-w4a4.safetensors")
    p.add_argument("--clip", default="qwen_3_4b.safetensors")
    p.add_argument("--clip-type", default="lumina2",
                   help="tipo de text encoder para `comfy.sd.CLIPType`, sem diferenciar caixa. O default lumina2 e o do Z-Image; krea2 e o do Krea2 (Qwen3-VL-4B com tap de 12 camadas). Errar aqui nao levanta excecao: carrega um encoder que produz condicionamento de outra forma e o resultado parece uma medida.")
    p.add_argument("--prompt", default="a red apple on a weathered wooden table, soft window light")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--device", type=int, default=1, help="1 = RTX 3080 Ti (tambem cc 8.6)")
    global ARGS
    ARGS = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    refp = OUT / "trajetoria_bf16.pkl"

    print(f"device cuda:{ARGS.device}  |  a trajetoria do BF16 e imposta aos dois bracos quantizados\n")

    print("--- referencia BF16 (grava a trajetoria) ---", flush=True)
    common = {"UNET": ARGS.ref_unet, "CLIP": ARGS.clip, "CLIPTYPE": ARGS.clip_type, "PROMPT": ARGS.prompt, "SEED": ARGS.seed,
              "STEPS": ARGS.steps, "CFG": repr(ARGS.cfg), "SIDE": ARGS.size,
              "DEV": ARGS.device, "OUTP": str(refp)}
    ref = rodar(REF_ARM % common)
    if not ref:
        sys.exit("referencia falhou")
    print(f"  {ref['chamadas']} chamadas ao modelo, saida {ref['shape']}")
    print(f"  sigmas: {', '.join(f'{s:.3f}' for s in ref['sigmas'])}\n")

    arms = {}
    for force, label in ((False, "nativo_int4"), (True, "fallback_int8")):
        print(f"--- {label} (reproduz a trajetoria) ---", flush=True)
        q = dict(common)
        q.update({"UNET": ARGS.quant_unet, "REFP": str(refp), "OUTP": ""})
        res = rodar(QUANT_ARM % q,
                    {"COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK": "1"} if force else None)
        if not res:
            sys.exit(f"braco {label} falhou")
        arms[label] = res
        print(f"  flag={res['flag']}  {len(res['passos'])} passos medidos")

    a, b = arms["nativo_int4"]["passos"], arms["fallback_int8"]["passos"]
    print(f"\n{'passo':>6} {'sigma':>9} {'rmse nativo':>12} {'rmse int8':>11} "
          f"{'cos nativo':>11} {'cos int8':>10} {'ganha':>8}")
    vit = 0
    for x, y in zip(a, b):
        quem = "nativo" if x["rel_rmse"] < y["rel_rmse"] else "int8"
        vit += quem == "nativo"
        print(f"{x['passo']:>6} {x['sigma']:>9.3f} {x['rel_rmse']:>12.4e} {y['rel_rmse']:>11.4e} "
              f"{x['cos']:>11.6f} {y['cos']:>10.6f} {quem:>8}")

    mn = sum(x["rel_rmse"] for x in a) / len(a)
    mf = sum(y["rel_rmse"] for y in b) / len(b)
    print(f"\nmedia por passo   nativo {mn:.4e}   int8 {mf:.4e}")
    print(f"nativo ganha em {vit}/{len(a)} passos")
    print(f"-> {'int8' if mn > mf else 'nativo'} e "
          f"{max(mn, mf) / min(mn, mf):.2f}x mais fiel ao BF16 na previsao")
    print("\ncomparar com: erro por camada deu int8 1,49x melhor (24/24);")
    print("a imagem final deu 2x1 com os dois a ~0,3-0,5 do BF16.")

    print("\nNAO COBERTO: um prompt, uma semente; sem metrica perceptual; so Z-Image; so imagem."
          " O BF16 e o alvo, nao a verdade -- ele proprio nao foi validado contra float32.",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())