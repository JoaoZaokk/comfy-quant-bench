"""A desvantagem de precisao do ramo INT4 nativo sobrevive a ativacao REAL?

DE ONDE VEM A PERGUNTA. Em 2026-08-30 mediu-se aqui (`probe_int4_mma_dispatch.py`) que o
ramo nativo INT4 do ConvRot executa na sm86 e e o MENOS preciso dos dois: rel-RMSE 2,2e-1
contra 1,56e-1 do fallback INT8. Mas aquela medicao usou entrada GAUSSIANA ALEATORIA, e
gaussiana nao tem outlier -- justamente o que a rotacao do ConvRot existe para suprimir.
A conclusao foi publicada com essa ressalva, e este probe e a ressalva sendo testada.

O QUE VARIA: `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, e so isso. Mesma camada, mesmo peso
real, mesma ativacao real, mesma placa, mesma ordem.

O QUE E MANTIDO, e e o ponto: peso REAL lido de `z_image_turbo_bf16.safetensors` por faixa
de bytes (nao mmap -- este host quebra com `os error 1455`), e ativacao REAL amostrada em
`calib/xfer_z_image_turbo_bf16.calib.pt`, capturada com forward-pre-hook durante sampling de
verdade, reservoir por Algoritmo R.

TRES DESFECHOS, e dizem coisas diferentes:
  nativo continua pior  -> o custo do A4 e real e nao era artefato da gaussiana.
  a diferenca encolhe   -> parte da desvantagem era a entrada sintetica; a rotacao ajuda
      quando ha outlier, so nao ha outlier em ruido branco.
  nativo passa a ganhar -> a medicao gaussiana estava invertendo o resultado, e a conclusao
      de 30/08 precisa ser corrigida no arquivo que a carrega.

Tambem se cruza a diferenca com o crest factor gravado na calibracao, porque crest e a
medida de outlier e a rotacao e o mecanismo que os trata. AVISO: este repo ja mediu que o
crest NAO prediz o erro W4A4 (Spearman +0,10 sobre 170 camadas). A pergunta aqui e outra --
se ele prediz a DIFERENCA entre os dois ramos -- e um Spearman baixo aqui tambem seria
resultado.

NAO COBERTO: uma camada Linear isolada, sem o resto do modelo; nada de imagem gerada, entao
nada sobre qualidade visual; sem inspecao de SASS; so Z-Image, so sm86.
"""
import json
import os
import struct
import subprocess
import sys

SRC = os.path.join("ComfyUI", "models", "diffusion_models", "z_image_turbo_bf16.safetensors")
CALIB = os.path.join("calib", "xfer_z_image_turbo_bf16.calib.pt")
N_LAYERS = 24
CONVROT_GROUPSIZE = 256

ARM_SRC = r'''
import json, os, struct, sys
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as ckc

SRC = %(SRC)r
CALIB = %(CALIB)r
NAMES = %(NAMES)s
CG = %(CG)d

with open(SRC, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
DATA_START = 8 + n

def read_weight(name):
    """Faixa de bytes, nunca mmap: este host quebra torch_cpu com 0xc0000005 em mmap."""
    e = header[name]
    a, b = e["data_offsets"]
    with open(SRC, "rb") as f:
        f.seek(DATA_START + a)
        raw = f.read(b - a)
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
    return t.reshape(e["shape"])

calib = torch.load(CALIB, map_location="cpu", weights_only=False)["layers"]
dev = torch.device("cuda:0")
out = {
    "force": os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag": bool(ckc._FORCE_INT4_INT8_FALLBACK),
    "layers": [],
}

for name in NAMES:
    w = read_weight(name + ".weight").to(dev)
    x = calib[name]["sample"].to(dev).to(torch.bfloat16)
    if x.shape[-1] != w.shape[1] or w.shape[1] %% CG != 0:
        continue
    qw, ws = ck.quantize_convrot_w4a4_weight(w, convrot_groupsize=CG)
    if out["flag"] is False and not out["layers"]:
        out["native_supported"] = bool(ckc._cuda_device_supports_native_int4_mma(x))
    y = ck.convrot_w4a4_linear(x, qw, ws, None, convrot_groupsize=CG)
    torch.cuda.synchronize()
    ref = torch.nn.functional.linear(x.float(), w.float())
    d = y.float() - ref
    out["layers"].append({
        "name": name,
        "rel_rmse": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
        "out_sum": float(y.float().sum()),
        "crest_mean": float(calib[name]["crest_mean"]),
        "crest_p99": float(calib[name]["crest_p99"]),
        "shape": list(w.shape),
    })
    del w, x, qw, ws, y, ref, d
    torch.cuda.empty_cache()

print("RESULT " + json.dumps(out))
'''


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else float("nan")


def run_arm(force, names):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if force:
        env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1"
    else:
        env.pop("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", None)
    src = ARM_SRC % {"SRC": SRC, "CALIB": CALIB, "NAMES": repr(names), "CG": CONVROT_GROUPSIZE}
    r = subprocess.run([os.path.join("python_embeded", "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print("ARM FALHOU\n stdout:", r.stdout.strip()[-500:], "\n stderr:", r.stderr.strip()[-1200:])
    return None


if not os.path.exists(SRC) or not os.path.exists(CALIB):
    sys.exit(f"faltando: {SRC if not os.path.exists(SRC) else CALIB}")

import torch  # noqa: E402  (so para ler a calibracao no processo pai)
calib = torch.load(CALIB, map_location="cpu", weights_only=False)["layers"]
# Espalha pelo crest, para nao amostrar so um regime de outlier.
by_crest = sorted(calib, key=lambda k: float(calib[k]["crest_mean"]))
step = max(1, len(by_crest) // N_LAYERS)
names = by_crest[::step][:N_LAYERS]
print(f"{len(names)} camadas de {len(by_crest)}, espalhadas por crest "
      f"({float(calib[names[0]]['crest_mean']):.1f} a "
      f"{float(calib[names[-1]]['crest_mean']):.1f})")
print("Varia UM eixo: COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK. Peso e ativacao sao reais.\n")

nat = run_arm(False, names)
fb = run_arm(True, names)
if not nat or not fb:
    sys.exit("braco faltando")

print(f"native_supported = {nat.get('native_supported')}   "
      f"(flag nativo={nat['flag']}, flag fallback={fb['flag']})\n")
print(f"{'camada':<44} {'crest':>7} {'nativo':>10} {'int8':>10} {'quem ganha':>11}")
gaps, crests, nat_wins = [], [], 0
for a, b in zip(nat["layers"], fb["layers"]):
    win = "nativo" if a["rel_rmse"] < b["rel_rmse"] else "int8"
    nat_wins += win == "nativo"
    gaps.append(a["rel_rmse"] - b["rel_rmse"])
    crests.append(a["crest_mean"])
    print(f"{a['name'][:44]:<44} {a['crest_mean']:>7.1f} {a['rel_rmse']:>10.4e} "
          f"{b['rel_rmse']:>10.4e} {win:>11}")

n = len(gaps)
mn = sum(x["rel_rmse"] for x in nat["layers"]) / n
mf = sum(x["rel_rmse"] for x in fb["layers"]) / n
print(f"\nmedia rel-RMSE   nativo {mn:.4e}   int8 {mf:.4e}")
print(f"nativo ganha em {nat_wins}/{n} camadas")
if mn > mf:
    print(f"-> o int8 e {mn / mf:.2f}x mais fiel na media")
else:
    print(f"-> o nativo e {mf / mn:.2f}x mais fiel na media")
print(f"Spearman(crest, nativo-menos-int8) = {spearman(crests, gaps):+.3f}")
print("  (este repo ja mediu crest x erro W4A4 em +0,10; aqui a pergunta e outra)")

print("\nComparacao com a medicao gaussiana de 2026-08-30: la o int8 foi ~1,4x mais fiel"
      " (2,2e-1 contra 1,56e-1).")

print("\nNAO COBERTO: camada Linear isolada, sem o resto do modelo; nenhuma imagem gerada,"
      " logo nada sobre qualidade visual; sem inspecao de SASS; so Z-Image; so sm86;"
      " tempo nao foi medido aqui, so erro.", file=sys.stderr)
