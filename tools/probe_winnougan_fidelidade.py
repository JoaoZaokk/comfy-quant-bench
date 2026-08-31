"""Dos dois ramos de 4 bits, qual erra menos contra o BF16 ORIGINAL deste checkpoint publico?

O QUE ISTO FECHA. `tools/probe_winnougan_int4.py` termina com esta ressalva, escrita por ele
mesmo: "NAO mede fidelidade contra o BF16 original: o BF16 deste modelo nao esta aqui. Diz qual
ramo executa e quanto custa, nao qual erra menos." O gemeo foi baixado em 2026-08-31
(`Comfy-Org/MiniMax-H3`, `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`, 47,97 GiB), e
351 de 351 nomes de camada casam com o arquivo quantizado do Winnougan. A ressalva caiu.

O EIXO QUE VARIA, e so ele: `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`, lido no import do backend,
por isso cada braco e um subprocesso separado. Mesmo peso empacotado, mesma escala, mesma entrada,
mesma semente, mesma placa, mesma referencia. O que muda e por qual ramo o MESMO
`convrot_w4a4_linear` desce.

A REFERENCIA e `F.linear(x, W_bf16)` em float32, com o peso lido do gemeo por faixa de bytes.
Nao e "a verdade": e o BF16 do publicador, que nunca foi validado contra float32 nesta bancada.
Chamar o erro contra ele de "fidelidade" e uma convencao, nao uma afirmacao sobre o mundo.

    python_embeded\\python.exe -s tools/probe_winnougan_fidelidade.py

NAO COBERTO:
  - Ativacao SINTETICA (gaussiana). Nao ha calibracao deste modelo aqui, e gaussiana nao tem os
    outliers que a rotacao existe para suprimir -- exatamente o caso onde este eixo ja mudou de
    resposta antes (ver W4A4_PROGRESS parte 27: sintetico deu 1,4x, ativacao real deu 1,49x).
  - Nao inspeciona SASS. "Ramo nativo" significa "produz numero diferente do fallback", nunca
    "m16n8k64.s4 foi observada emitindo".
  - Erro por CAMADA, nao erro do modelo. Esta bancada ja mediu que o erro por camada preve o erro
    de predicao do modelo mas NAO preve a imagem final (parte 29).
  - Uma placa (sm86). Nao carrega nada no ComfyUI: prova o kernel, nao o loader.
  - O BF16 do publicador e o alvo, nao a verdade.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-int4_convrot.safetensors"
REF = ROOT / "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
N_LAYERS = 20
M_VALUES = [1, 64, 1024]


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def pick_layers(header, ref_header, qmeta):
    """Camadas convrot_w4a4 espalhadas em FORMA **e** em PROFUNDIDADE.

    A primeira versao desta funcao pegava uma camada por forma distinta e parava. Num
    transformer as formas se repetem bloco a bloco, entao "primeira ocorrencia de cada forma"
    e "tudo do bloco 0" -- as cinco camadas medidas em 2026-08-31 sairam todas de
    `model.layers.0`. A forma variava e a profundidade ficava presa, que e exatamente o modo
    de falha que esta bancada chama de `teste-varia-o-eixo-errado`.

    Agora: agrupa por forma, e dentro de cada forma pega blocos espalhados pelo intervalo
    inteiro (primeiro, meio, ultimo, ...). Se o modelo tiver menos candidatos que N_LAYERS, a
    saida diz quantos saiu em vez de fingir a cota.
    """
    import re

    por_forma: dict[tuple, list] = {}
    for name, conf in qmeta["layers"].items():
        if conf.get("format") != "convrot_w4a4":
            continue
        wk, sk = f"{name}.weight", f"{name}.weight_scale"
        if wk not in header or sk not in header or wk not in ref_header:
            continue
        forma = tuple(header[wk]["shape"])
        m = re.search(r"\.layers\.(\d+)\.", name)
        prof = int(m.group(1)) if m else -1
        por_forma.setdefault(forma, []).append((prof, name, conf, forma))

    for v in por_forma.values():
        v.sort()

    # Dentro de cada forma, indices em linspace FECHADO nos dois extremos: o primeiro bloco e o
    # ULTIMO entram sempre. Um passo fixo a partir do inicio (0, 10, 20, 30 num modelo de 50
    # blocos) deixa o fim do modelo inteiro fora da amostra, e o fim e onde a representacao ja
    # esta formada -- justamente a metade que pode se comportar diferente da entrada.
    formas = sorted(por_forma, key=lambda f: -len(por_forma[f]))
    cota = max(1, -(-N_LAYERS // len(formas))) if formas else 0
    out, vistos = [], set()
    for f in formas:
        cands = por_forma[f]
        t = min(cota, len(cands))
        if t == 1:
            idxs = [0]
        else:
            idxs = [round(i * (len(cands) - 1) / (t - 1)) for i in range(t)]
        for i in dict.fromkeys(idxs):
            _prof, name, conf, forma = cands[i]
            if name in vistos:
                continue
            vistos.add(name)
            out.append((name, conf, forma))
            if len(out) >= N_LAYERS:
                return out
    return out


ARM_SRC = r'''
import json, os, struct
import torch
import torch.nn.functional as F
from comfy_kitchen.backends import cuda as ckc
from comfy_kitchen.backends.cuda import convrot_w4a4_linear

CKPT = %(CKPT)r
REF = %(REF)r
LAYERS = json.loads(%(LAYERS)r)
M_VALUES = %(MS)s

dev = torch.device("cuda:0")
DT = {"I8": torch.int8, "U8": torch.uint8, "F32": torch.float32,
      "BF16": torch.bfloat16, "F16": torch.float16}


def abrir(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


HQ, DQ = abrir(CKPT)
HR, DR = abrir(REF)


def load(path, header, data0, name):
    """Faixa de bytes, nunca mmap: este host quebra torch_cpu com 0xc0000005 em mmap."""
    e = header[name]
    a, b = e["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data0 + a)
        raw = f.read(b - a)
    return torch.frombuffer(bytearray(raw), dtype=DT[e["dtype"]]).reshape(e["shape"])


out = {
    "force_fallback": os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
    "flag_seen_by_module": bool(ckc._FORCE_INT4_INT8_FALLBACK),
    "device": torch.cuda.get_device_name(0),
    "cc": list(torch.cuda.get_device_capability(0)),
    "layers": [],
}

for name, conf, _forma in LAYERS:
    qw = load(CKPT, HQ, DQ, name + ".weight").to(dev)
    ws = load(CKPT, HQ, DQ, name + ".weight_scale").to(dev)
    wref = load(REF, HR, DR, name + ".weight").to(dev)
    cg = conf.get("convrot_groupsize", 256)
    qgs = conf.get("quant_group_size", 64)
    ld = conf.get("linear_dtype", "int4")
    K = qw.shape[-1] * 2
    rec = {"name": name, "shape": list(qw.shape), "K": K, "K_ref": list(wref.shape),
           "convrot_groupsize": cg, "quant_group_size": qgs, "linear_dtype_used": ld, "M": {}}
    if wref.shape[-1] != K or wref.shape[0] != qw.shape[0]:
        rec["forma_incompativel"] = True
        out["layers"].append(rec)
        del qw, ws, wref
        torch.cuda.empty_cache()
        continue
    wref32 = wref.float()
    for M in M_VALUES:
        torch.manual_seed(1234)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        try:
            yref = F.linear(x.float(), wref32)
            y = convrot_w4a4_linear(x, qw, ws, None, cg, qgs, ld).float()
            torch.cuda.synchronize()
            num = torch.linalg.vector_norm(y - yref)
            den = torch.linalg.vector_norm(yref)
            cos = F.cosine_similarity(y.reshape(1, -1), yref.reshape(1, -1)).item()
            rec["M"][str(M)] = {
                "ok": True,
                "rel_rmse": float(num / den),
                "cos": float(cos),
                "ref_absmean": float(yref.abs().mean()),
                "out_absmean": float(y.abs().mean()),
                "fingerprint": [float(v) for v in y.reshape(-1)[:4]],
            }
            del yref, y
        except Exception as exc:
            rec["M"][str(M)] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()
    out["layers"].append(rec)
    del qw, ws, wref, wref32
    torch.cuda.empty_cache()

print("@@JSON@@" + json.dumps(out))
'''


def run_arm(force: bool, layers) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1" if force else "0"
    src = ARM_SRC % {
        "CKPT": str(CKPT), "REF": str(REF),
        "LAYERS": json.dumps(layers), "MS": repr(M_VALUES),
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


def main() -> int:
    for p in (CKPT, REF):
        if not p.is_file():
            raise SystemExit(f"nao existe: {p}")
    header, _ = read_header(CKPT)
    ref_header, _ = read_header(REF)
    qmeta = json.loads(header["__metadata__"]["_quantization_metadata"])
    picked = pick_layers(header, ref_header, qmeta)

    casam = sum(1 for n in qmeta["layers"] if f"{n}.weight" in ref_header)
    print("=" * 78)
    print(f"quantizado   {CKPT.name}  ({CKPT.stat().st_size / 1024**3:.2f} GiB)")
    print(f"referencia   {REF.name}  ({REF.stat().st_size / 1024**3:.2f} GiB)")
    print(f"camadas no manifesto {len(qmeta['layers'])}, com gemeo no BF16: {casam}")
    print(f"medindo {len(picked)} camadas (uma por forma distinta), M em {M_VALUES}")
    print("=" * 78)

    nat = run_arm(False, picked)
    i8 = run_arm(True, picked)

    print(f"placa  {nat['device']}  cc {nat['cc']}")
    print(f"flag vista pelo modulo: nativo={nat['flag_seen_by_module']} "
          f"int8={i8['flag_seen_by_module']}")
    if nat["flag_seen_by_module"] == i8["flag_seen_by_module"]:
        print("AVISO: os dois bracos viram a MESMA flag. A comparacao abaixo nao tem eixo.")
    print()

    vitorias = {"nativo": 0, "int8": 0, "empate": 0}
    somas = {"nativo": [], "int8": []}
    for M in M_VALUES:
        print(f"--- M = {M} " + "-" * 60)
        print(f"{'camada':44} {'nativo':>10} {'int8':>10}  vence")
        for rn, ri in zip(nat["layers"], i8["layers"]):
            a = rn["M"].get(str(M), {})
            b = ri["M"].get(str(M), {})
            if not (a.get("ok") and b.get("ok")):
                print(f"{rn['name'][:44]:44} {'FALHOU':>10} {'FALHOU':>10}")
                continue
            ea, eb = a["rel_rmse"], b["rel_rmse"]
            somas["nativo"].append(ea)
            somas["int8"].append(eb)
            if abs(ea - eb) / max(ea, eb, 1e-12) < 1e-6:
                quem = "empate"
            else:
                quem = "nativo" if ea < eb else "int8"
            vitorias[quem] += 1
            print(f"{rn['name'][:44]:44} {ea:10.4e} {eb:10.4e}  {quem}")
        print()

    def media(v):
        return sum(v) / len(v) if v else float("nan")

    mn, mi = media(somas["nativo"]), media(somas["int8"])
    print("=" * 78)
    print(f"media rel-RMSE contra BF16   nativo {mn:.4e}   int8 {mi:.4e}")
    if mn == mn and mi == mi and min(mn, mi) > 0:
        pior, melhor = (mn, mi) if mn > mi else (mi, mn)
        quem = "int8" if mn > mi else "nativo"
        print(f"{quem} e {pior / melhor:.2f}x mais fiel na media")
    print(f"vitorias por camada-M: {vitorias}")
    print()
    print("NAO COBERTO: ativacao gaussiana sintetica, sem os outliers que a rotacao suprime -- este")
    print("eixo ja mudou de resposta entre sintetico e real nesta bancada. Sem SASS: 'ramo nativo'")
    print("significa 'numero diferente do fallback', nao 'instrucao observada'. Erro por CAMADA, que")
    print("preve o erro de predicao do modelo mas NAO preve a imagem final. Uma placa, sm86. E o")
    print("BF16 do publicador e o alvo, nao a verdade -- nunca foi validado contra float32 aqui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
