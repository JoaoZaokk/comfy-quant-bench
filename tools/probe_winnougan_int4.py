"""Este checkpoint publico emite MMA de 4 bits, ou so se declara W4A4?

O QUE ISTO RESPONDE. Dois checkpoints ConvRot publicos com centenas de milhares de
downloads (`Abiray/...FL2VA`, `Abiray/...Ref2VA`) gravam `linear_dtype: "int8"` em CADA
camada, e essa e a PRIMEIRA condicao do desvio em `convrot_w4a4_linear`
(`comfy_kitchen/backends/cuda/__init__.py`), antes de qualquer teste de hardware. Eles
nunca emitem um GEMM de 4 bits, em placa nenhuma.

`Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI` e a excecao candidata: 350 camadas
`convrot_w4a4` com `linear_dtype` AUSENTE. Ausente cai no default da assinatura, que e
`"int4"`. Isso e o que se LE em:

    comfy_kitchen/backends/cuda/__init__.py   linear_dtype: str = "int4"   (assinatura)
    ComfyUI/comfy/ops.py:1202                 layer_conf.get("linear_dtype",
                                                params_conf.get("linear_dtype", "int4"))

LER a cadeia nao prova que a instrucao executou. Este probe EXECUTA, com os bytes
quantizados do proprio arquivo -- nada e requantizado aqui.

O EIXO QUE VARIA, e so ele: `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, lida no import do
backend (`:212`), por isso cada braco e um subprocesso. Mesmo peso empacotado, mesma
escala, mesma entrada, mesma semente, mesma placa. Se os dois bracos dessem o mesmo
numero, o `major == 8` nao estaria mudando nada.

NAO COBERTO por este probe:
  - Nao inspeciona SASS nem conta instrucoes. "O ramo nativo roda" aqui significa "produz
    numero diferente do fallback INT8", nao "m16n8k64.s4 foi observada emitindo".
  - Ativacao SINTETICA (gaussiana). Nao ha calibracao deste modelo nesta bancada, e
    gaussiana nao tem os outliers que a rotacao existe para suprimir.
  - NAO mede fidelidade contra o BF16 original: o BF16 deste modelo nao esta aqui. Diz
    qual ramo executa e quanto custa, nao qual erra menos.
  - Nao carrega o modelo no ComfyUI. Prova o kernel, nao o loader.
  - Uma placa (sm86), um processo por braco.
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:/COMFY_PORTABLE")
CKPT = ROOT / "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-int4_convrot.safetensors"
N_LAYERS = 8
M_VALUES = [1, 64, 1024]
REPEATS = 5


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def pick_layers(header, qmeta):
    """Uma camada convrot_w4a4 por FORMA distinta, ate N_LAYERS.

    Uma por forma, e nao as N primeiras, porque as N primeiras seriam todas do bloco 0 e
    o eixo que interessa aqui e a forma (o desvio testa `x.shape` e a memoria
    compartilhada). Se o modelo tiver menos formas que N_LAYERS, o resultado tem menos
    linhas que o pedido -- e isso e dito no cabecalho da saida, nao escondido.
    """
    out = []
    seen_shapes = set()
    for name, conf in qmeta["layers"].items():
        if conf.get("format") != "convrot_w4a4":
            continue
        wk, sk = f"{name}.weight", f"{name}.weight_scale"
        if wk not in header or sk not in header:
            continue
        shape = tuple(header[wk]["shape"])
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        out.append((name, conf, shape))
        if len(out) >= N_LAYERS:
            break
    return out


ARM_SRC = r'''
import json, os, struct, sys, time
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as ckc
from comfy_kitchen.backends.cuda import convrot_w4a4_linear, _cuda_device_supports_native_int4_mma

CKPT = %(CKPT)r
LAYERS = json.loads(%(LAYERS)r)
M_VALUES = %(MS)s
REPEATS = %(REP)d

dev = torch.device("cuda:0")
with open(CKPT, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
DATA0 = 8 + n

DT = {"I8": (torch.int8, 1), "U8": (torch.uint8, 1), "F32": (torch.float32, 4),
      "BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2)}

def load(name):
    """Faixa de bytes, nunca mmap: este host quebra torch_cpu com 0xc0000005 em mmap."""
    e = header[name]
    dtype, _ = DT[e["dtype"]]
    a, b = e["data_offsets"]
    with open(CKPT, "rb") as f:
        f.seek(DATA0 + a)
        raw = f.read(b - a)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(e["shape"])

out = {
    "force_fallback": os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag_seen_by_module": bool(ckc._FORCE_INT4_INT8_FALLBACK),
    "device": torch.cuda.get_device_name(0),
    "cc": list(torch.cuda.get_device_capability(0)),
    "layers": [],
}

for name, conf, _shape in LAYERS:
    qw = load(name + ".weight").to(dev)
    ws = load(name + ".weight_scale").to(dev)
    cg = conf.get("convrot_groupsize", 256)
    qgs = conf.get("quant_group_size", 64)
    # linear_dtype vem do arquivo; ausente = default da assinatura, exatamente como ops.py:1202
    ld = conf.get("linear_dtype", "int4")
    K = qw.shape[-1] * 2
    rec = {"name": name, "shape": list(qw.shape), "K": K, "convrot_groupsize": cg,
           "quant_group_size": qgs, "linear_dtype_used": ld, "M": {}}
    for M in M_VALUES:
        torch.manual_seed(1234)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        if M == M_VALUES[0]:
            rec["native_mma_supported_for_this_x"] = bool(_cuda_device_supports_native_int4_mma(x))
        try:
            y = convrot_w4a4_linear(x, qw, ws, None, cg, qgs, ld)
            torch.cuda.synchronize()
            ts = []
            for _ in range(REPEATS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                convrot_w4a4_linear(x, qw, ws, None, cg, qgs, ld)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
            ts.sort()
            yf = y.float()
            rec["M"][str(M)] = {
                "ok": True,
                "sum": float(yf.sum()), "absmean": float(yf.abs().mean()),
                "fingerprint": [float(v) for v in yf.reshape(-1)[:6]],
                "us_median": ts[len(ts)//2], "us_min": ts[0], "us_max": ts[-1],
            }
        except Exception as exc:
            rec["M"][str(M)] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out["layers"].append(rec)
    del qw, ws
    torch.cuda.empty_cache()

print("@@JSON@@" + json.dumps(out))
'''


def run_arm(force, layers):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1" if force else "0"
    src = ARM_SRC % {
        "CKPT": str(CKPT),
        "LAYERS": json.dumps(layers),
        "MS": repr(M_VALUES),
        "REP": REPEATS,
    }
    proc = subprocess.run(
        [str(ROOT / "python_embeded/python.exe"), "-s", "-c", src],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    print(proc.stdout[-3000:])
    print(proc.stderr[-3000:], file=sys.stderr)
    raise SystemExit(f"braco force={force} nao devolveu JSON (rc={proc.returncode})")


def main():
    header, _ = read_header(CKPT)
    qmeta = json.loads(header["__metadata__"]["_quantization_metadata"])
    picked = pick_layers(header, qmeta)

    fmts = {}
    lds = {}
    for conf in qmeta["layers"].values():
        fmts[conf.get("format")] = fmts.get(conf.get("format"), 0) + 1
        key = conf.get("linear_dtype", "<AUSENTE>")
        lds[key] = lds.get(key, 0) + 1

    print("=" * 78)
    print(f"arquivo      {CKPT.name}")
    print(f"             {CKPT.stat().st_size / 1024**3:.2f} GiB")
    print(f"converted_by {header['__metadata__'].get('converted_by')}")
    print(f"formatos     {fmts}")
    print(f"linear_dtype {lds}   <- ausente cai no default da assinatura: 'int4'")
    n_shapes = len({tuple(header[f"{n}.weight"]["shape"])
                    for n, c in qmeta["layers"].items()
                    if c.get("format") == "convrot_w4a4" and f"{n}.weight" in header})
    print(f"camadas probadas {len(picked)} de {N_LAYERS} pedidas "
          f"-- o modelo tem {n_shapes} formas convrot_w4a4 distintas, uma por forma")
    print("=" * 78)

    layers_arg = [[n, c, list(s)] for n, c, s in picked]
    nat = run_arm(False, layers_arg)
    fb = run_arm(True, layers_arg)

    print()
    print(f"placa        {nat['device']}  cc {tuple(nat['cc'])}")
    print(f"braco nativo   flag vista pelo modulo = {nat['flag_seen_by_module']}")
    print(f"braco fallback flag vista pelo modulo = {fb['flag_seen_by_module']}")
    print(f"native_mma_supported (executado)      = {nat['layers'][0].get('native_mma_supported_for_this_x')}")

    print()
    print("-" * 78)
    print("saidas diferem entre os bracos? (identico = mesmo kernel = ramo nativo inerte)")
    print("-" * 78)
    hdr = f"{'camada':<34} {'forma':>14} {'M':>5}  {'nativo':>13} {'fallback':>13}  dif"
    print(hdr)
    diff_count = tot = 0
    fail = []
    for a, b in zip(nat["layers"], fb["layers"]):
        for M in M_VALUES:
            ra, rb = a["M"][str(M)], b["M"][str(M)]
            if not ra.get("ok") or not rb.get("ok"):
                fail.append((a["name"], M, ra.get("error"), rb.get("error")))
                continue
            tot += 1
            d = ra["fingerprint"] != rb["fingerprint"]
            diff_count += bool(d)
            short = a["name"].replace("model.layers.", "L")
            print(f"{short:<34} {str(tuple(a['shape'])):>14} {M:>5}  "
                  f"{ra['fingerprint'][0]:>13.6f} {rb['fingerprint'][0]:>13.6f}  "
                  f"{'SIM' if d else 'identico'}")
    print()
    print(f"VEREDITO   {diff_count}/{tot} pares diferem")
    if tot and diff_count == tot:
        print("           os dois bracos sao kernels DISTINTOS. O ramo nativo INT4 executa")
        print("           com os bytes deste arquivo.")
    elif diff_count == 0:
        print("           IDENTICOS. Os dois bracos tomaram o mesmo caminho: o arquivo se")
        print("           declara W4A4 mas nao emite GEMM de 4 bits nesta placa.")
    else:
        print("           MISTO -- ler linha a linha antes de concluir qualquer coisa.")
    for name, M, ea, eb in fail:
        print(f"FALHA      {name} M={M}  nativo: {ea}  fallback: {eb}")

    print()
    print("-" * 78)
    print("preco: mediana de 5, us (razao sempre >= 1, com a direcao dita)")
    print("-" * 78)
    print(f"{'camada':<34} {'M':>5} {'nativo us':>11} {'fallback us':>12}  quem ganha")
    for a, b in zip(nat["layers"], fb["layers"]):
        for M in M_VALUES:
            ra, rb = a["M"][str(M)], b["M"][str(M)]
            if not (ra.get("ok") and rb.get("ok")):
                continue
            na, nb = ra["us_median"], rb["us_median"]
            if na <= nb:
                verd = f"nativo {nb / na:.2f}x mais rapido"
            else:
                verd = f"fallback {na / nb:.2f}x mais rapido"
            short = a["name"].replace("model.layers.", "L")
            print(f"{short:<34} {M:>5} {na:>11.1f} {nb:>12.1f}  {verd}")

    print()
    print("NAO COBERTO: sem SASS (diferenca numerica, nao instrucao observada); ativacao")
    print("  sintetica gaussiana, sem os outliers que a rotacao existe para suprimir; sem")
    print("  referencia BF16 deste modelo, entao NAO ha afirmacao de fidelidade aqui; nao")
    print("  carrega o modelo no ComfyUI, prova o kernel e nao o loader; uma placa sm86.")


if __name__ == "__main__":
    main()
