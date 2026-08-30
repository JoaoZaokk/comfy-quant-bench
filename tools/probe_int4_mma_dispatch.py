"""O MMA de 4 bits do ConvRot executa mesmo na sm86, ou o ramo so e escolhido?

O QUE ISTO RESPONDE. Em 2026-08-30 leu-se em
`comfy_kitchen/backends/cuda/__init__.py:293` que `_cuda_device_supports_native_int4_mma`
devolve `major == 8`, ou seja Ampere e Ada pegam o kernel `m16n8k64 s4` e Hopper/Blackwell
sao desviados para um fallback INT8. Isso foi LIDO, nao executado -- saber qual ramo o
Python escolhe nao prova que a instrucao emitiu nem que o resultado difere.

O EIXO QUE VARIA, e so ele: a variavel de ambiente
`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, lida em `:212`. Ela forca o mesmo
`convrot_w4a4_linear`, com o mesmo peso e a mesma entrada, a descer pelo ramo INT8. Tudo
mais e mantido: mesmo processo-irmao, mesma semente, mesmo tensor, mesma placa.

Isto e deliberadamente o oposto do erro registrado em `teste-varia-o-eixo-errado`: aqui o
eixo em teste E o dispatch, e ele e o unico que se mexe.

COMO LER O RESULTADO:
  saidas IDENTICAS bit a bit  -> os dois bracos tomaram o MESMO caminho. O `major == 8` nao
      esta mudando nada de fato, e a leitura do codigo nao se traduz em execucao.
  saidas DIFERENTES           -> sao dois kernels distintos. O ramo nativo existe e roda.
      Ai o erro contra a referencia float diz qual dos dois e mais fiel, e o tempo diz o
      preco.

NAO COBERTO por este probe: ele nao inspeciona SASS nem conta instrucoes, entao "o ramo
nativo roda" significa "produz numero diferente do fallback", e nao "a instrucao
m16n8k64.s4 foi observada emitindo". Nao mede modelo real, so uma camada Linear sintetica.
Nao diz nada sobre qualidade de imagem. E nao testa Hopper nem Blackwell, que esta bancada
nao tem.
"""
import json
import os
import subprocess
import sys

MIB = 1024 * 1024
SHAPES = [(4096, 4096), (8192, 4096)]
M_VALUES = [1, 64, 1024]
CONVROT_GROUPSIZE = 256
REPEATS = 5

ARM_SRC = r'''
import json, os, sys, time
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as ckc

dev = torch.device("cuda:0")
torch.manual_seed(1234)

N, K = %(N)d, %(K)d
M_VALUES = %(MS)s
CG = %(CG)d
REPEATS = %(REP)d

w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05)
qw, ws = ck.quantize_convrot_w4a4_weight(w, convrot_groupsize=CG)

out = {
    "force_fallback": os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag_seen_by_module": bool(ckc._FORCE_INT4_INT8_FALLBACK),
    "capability": list(torch.cuda.get_device_capability(0)),
    "device": torch.cuda.get_device_name(0),
    "qweight_shape": list(qw.shape),
    "qweight_dtype": str(qw.dtype),
    "arms": [],
}

for M in M_VALUES:
    torch.manual_seed(99)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
    out["supports_native_int4_mma"] = bool(ckc._cuda_device_supports_native_int4_mma(x))
    out["should_use_turing_int4"] = bool(ckc._should_use_turing_int4(x))

    y = ck.convrot_w4a4_linear(x, qw, ws, None, convrot_groupsize=CG)
    torch.cuda.synchronize()

    ref = torch.nn.functional.linear(x.float(), w.float())

    ts = []
    for _ in range(REPEATS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ck.convrot_w4a4_linear(x, qw, ws, None, convrot_groupsize=CG)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()

    d = (y.float() - ref)
    out["arms"].append({
        "M": M,
        "out_sha_first64": [round(float(v), 6) for v in y.float().flatten()[:8].tolist()],
        "out_sum": float(y.float().sum()),
        "rel_rmse_vs_float": float((d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())),
        "us_median": ts[len(ts)//2],
        "us_min": ts[0], "us_max": ts[-1],
        "dtype": str(y.dtype),
    })

print("RESULT " + json.dumps(out))
'''


def run_arm(force_fallback, N, K):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if force_fallback:
        env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1"
    else:
        env.pop("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", None)
    src = ARM_SRC % {"N": N, "K": K, "MS": repr(M_VALUES), "CG": CONVROT_GROUPSIZE,
                     "REP": REPEATS}
    r = subprocess.run(
        [os.path.join("python_embeded", "python.exe"), "-s", "-c", src],
        capture_output=True, text=True, env=env, timeout=900,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print("  ARM FALHOU. stdout:", r.stdout.strip()[-400:])
    print("  stderr:", r.stderr.strip()[-800:])
    return None


print("Varia UM eixo: COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK. Tudo mais e mantido.")
print()

verdicts = []
for (N, K) in SHAPES:
    print(f"=== peso [{N}, {K}], convrot_groupsize={CONVROT_GROUPSIZE} ===")
    nat = run_arm(False, N, K)
    fb = run_arm(True, N, K)
    if not nat or not fb:
        print("  arm faltando, pulando forma")
        continue

    print(f"  placa {nat['device']}  cc {nat['capability'][0]}.{nat['capability'][1]}")
    print(f"  supports_native_int4_mma: nativo={nat['supports_native_int4_mma']}"
          f"  fallback-forcado={fb['supports_native_int4_mma']}")
    print(f"  qweight {nat['qweight_shape']} {nat['qweight_dtype']}")
    print()
    print(f"  {'M':>6} {'identico?':>10} {'rmse nativo':>12} {'rmse int8':>11} "
          f"{'us nativo':>10} {'us int8':>9}")
    for a, b in zip(nat["arms"], fb["arms"]):
        same = a["out_sha_first64"] == b["out_sha_first64"] and \
            abs(a["out_sum"] - b["out_sum"]) < 1e-6
        verdicts.append(same)
        print(f"  {a['M']:>6} {('SIM' if same else 'NAO'):>10} "
              f"{a['rel_rmse_vs_float']:>12.4e} {b['rel_rmse_vs_float']:>11.4e} "
              f"{a['us_median']:>10.1f} {b['us_median']:>9.1f}")
    print()

print("=== VEREDITO ===")
if not verdicts:
    print("nenhum braco completou; nada a concluir")
elif all(verdicts):
    print("Saidas IDENTICAS em todos os M. Os dois bracos tomaram o MESMO caminho:")
    print("a leitura de `major == 8` nao se traduz em execucao distinta aqui.")
else:
    n = sum(1 for v in verdicts if not v)
    print(f"Saidas DIFERENTES em {n}/{len(verdicts)} casos. Sao dois kernels distintos,")
    print("logo o ramo nativo INT4 existe e roda nesta placa -- e nao apenas e escolhido.")

print()
print("NAO COBERTO: nenhuma inspecao de SASS, entao 'roda' aqui significa 'produz numero",
      "diferente do fallback', nao 'a instrucao m16n8k64.s4 foi vista emitindo'. Camada",
      "sintetica, nao modelo real. Nada sobre qualidade de imagem. Hopper e Blackwell",
      "nao testados: esta bancada nao os tem.", file=sys.stderr)
