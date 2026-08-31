"""Por onde o encode de um text encoder quantizado passa de verdade -- contado, nao lido.

DE ONDE VEM A PERGUNTA. `probe_winnougan_load.py` mostrou que trocar
`COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` nao muda um bit do condicionamento. Se os dois
bracos dao o mesmo numero, nenhum dos dois passou por `convrot_w4a4_linear`. A leitura
seguinte apontou `comfy/sd1_clip.py:114`:

    operations = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)

`full_precision_mm=True`, fixo, para TODO text encoder. Em `comfy/ops.py:1375` isso zera
`_use_quantized`, e o forward dequantiza para BF16.

Isso e LEITURA. Este probe CONTA. Instrumenta tres pontos e roda um encode real:

    _convrot_w4a4_forward     chamadas ao caminho de 4 bits do comfy-kitchen
    QuantizedTensor.dequantize   chamadas ao caminho que materializa BF16
    _full_precision_mm        o valor com que cada Linear ficou, lido do modulo carregado

Contagem de convrot > 0  -> o encode emite 4 bits.
Contagem 0 e dequantize > 0 -> o peso fica 4 bits na VRAM e a MATEMATICA e BF16. Economia
    de memoria sem economia de tempo, e o kernel nativo nunca e alcancado.

NAO COBERTO: um modelo (MiniMax H3 / qwen3vl_32b), um prompt, uma placa sm86. Nao diz se
a escolha upstream esta certa -- so o que ela faz. Nao mede o modelo de difusao, cujo
caminho e outro (`comfy/ops.py:1667`, que passa `disabled=` e nao `full_precision_mm`).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:/COMFY_PORTABLE")
CKPT = ROOT / "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3-int4_convrot.safetensors"
PROMPT = "a red fox walking through snow at dusk"

SRC = r'''
import json, sys, traceback
from pathlib import Path
ROOT = Path(r"F:/COMFY_PORTABLE")
sys.path.insert(0, str(ROOT / "ComfyUI"))

import torch
import comfy.utils, comfy.sd, comfy.ops
from comfy_kitchen.tensor import convrot_w4a4 as ckt
from comfy_kitchen.tensor.base import QuantizedTensor

CKPT = %(CKPT)r
PROMPT = %(PROMPT)r
rep = {}

clip = comfy.sd.load_clip(ckpt_paths=[CKPT], clip_type=comfy.sd.CLIPType.MINIMAX)

# estado dos modulos DEPOIS de carregar, antes de qualquer forward
tr = clip.cond_stage_model.qwen3vl_32b.transformer
mods = []
for name, mod in tr.named_modules():
    if name.endswith(("self_attn.q_proj", "mlp.down_proj")):
        mods.append({
            "name": name,
            "class": type(mod).__name__,
            "class_qualname": type(mod).__qualname__,
            "full_precision_mm": getattr(mod, "_full_precision_mm", "<sem atributo>"),
            "full_precision_mm_config": getattr(mod, "_full_precision_mm_config", "<sem atributo>"),
            "quant_format": getattr(mod, "quant_format", "<sem atributo>"),
            "layout_type": getattr(mod, "layout_type", "<sem atributo>"),
            "weight_is_qt": isinstance(getattr(mod, "weight", None), QuantizedTensor),
        })
rep["n_linear_inspecionadas"] = len(mods)
rep["exemplo_modulo"] = mods[0] if mods else None
from collections import Counter
rep["classes"] = dict(Counter(m["class_qualname"] for m in mods))
rep["full_precision_mm"] = dict(Counter(str(m["full_precision_mm"]) for m in mods))
rep["quant_format"] = dict(Counter(str(m["quant_format"]) for m in mods))
rep["weight_is_quantized_tensor"] = dict(Counter(str(m["weight_is_qt"]) for m in mods))

# instrumenta SO depois do load: o load usa dequantize legitimamente
counts = {"convrot_forward": 0, "dequantize": 0}
_orig_fw = ckt._convrot_w4a4_forward
def counted_fw(*a, **k):
    counts["convrot_forward"] += 1
    return _orig_fw(*a, **k)
ckt._convrot_w4a4_forward = counted_fw

_orig_dq = QuantizedTensor.dequantize
def counted_dq(self, *a, **k):
    counts["dequantize"] += 1
    return _orig_dq(self, *a, **k)
QuantizedTensor.dequantize = counted_dq

import time
# Grava os termos de `_use_quantized` (ops.py:1372-1377) na PRIMEIRA passagem de uma Linear.
# Adivinhar qual termo e falso ja custou uma conclusao errada aqui.
probe_mod = dict(tr.named_modules())["model.layers.0.self_attn.q_proj"]
termos = {}
_orig_fcw = type(probe_mod).forward_comfy_cast_weights
def spying_fcw(self, input, *a, **k):
    if self is probe_mod and "fase" not in termos:
        termos["fase"] = {
            "input_ndim": int(input.ndim),
            "input_is_qt": isinstance(input, QuantizedTensor),
            "layout_type": getattr(self, "layout_type", None),
            "full_precision_mm": getattr(self, "_full_precision_mm", None),
            "comfy_force_cast_weights": bool(getattr(self, "comfy_force_cast_weights", False)),
            "n_weight_function": len(getattr(self, "weight_function", [])),
            "n_bias_function": len(getattr(self, "bias_function", [])),
            "weight_only_quant_kwarg": k.get("weight_only_quant"),
            "want_requant_kwarg": k.get("want_requant"),
        }
    return _orig_fcw(self, input, *a, **k)
type(probe_mod).forward_comfy_cast_weights = spying_fcw

def encode_once():
    tokens = clip.tokenize(PROMPT)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = clip.encode_from_tokens_scheduled(tokens)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1e3
    return out[0][0].float(), ms

REPEATS = 5
def encode_repeated():
    """Uma medicao nao e uma medicao. Devolve mediana e a lista bruta."""
    encode_once()  # aquecimento: a primeira chamada paga a subida do modelo para a VRAM
    ts = []
    for _ in range(REPEATS):
        cond, ms = encode_once()
        ts.append(round(ms, 1))
    return cond, sorted(ts)[len(ts) // 2], ts

try:
    encode_once()
    termos.pop("fase", None)
    counts["convrot_forward"] = 0
    counts["dequantize"] = 0
    cond, ms = encode_once()
    rep["termos_use_quantized"] = termos.pop("fase", None)
    rep["encoded"] = True
    rep["cond_shape"] = list(cond.shape)
    rep["cond_norm"] = float(cond.norm())
    rep["cond_fingerprint"] = [float(v) for v in cond.reshape(-1)[:8]]
    _, ms, ts = encode_repeated()
    rep["encode_ms"] = ms
    rep["encode_ms_all"] = ts
except Exception:
    rep["encoded"] = False
    rep["encode_traceback"] = traceback.format_exc()[-2000:]

rep["counts"] = dict(counts)

# FASE 2 -- unico eixo que muda: `_full_precision_mm` das Linear, de True para False.
# `sd1_clip.py:114` fixa True para todo text encoder; o arquivo pede False
# (`_full_precision_mm_config` = False). Isto pergunta o que aconteceria se a trava saisse.
if rep.get("encoded"):
    n_flipped = 0
    for _, mod in tr.named_modules():
        if getattr(mod, "layout_type", None) is None:
            continue
        # AS DUAS travas, nao uma. `_full_precision_mm` vem de `sd1_clip.py:114`;
        # `comfy_force_cast_weights` vem de `sd.py:269` via `set_model_compute_dtype(float32)`,
        # e e ELE que aparece True na instrumentacao da fase 1.
        mod._full_precision_mm = False
        mod.comfy_force_cast_weights = False
        n_flipped += 1
    rep["n_linear_destravadas"] = n_flipped
    counts["convrot_forward"] = 0
    counts["dequantize"] = 0
    termos.pop("fase", None)
    try:
        cond2, _ms2 = encode_once()
        termos_f2 = termos.pop("fase", None)
        counts_f2 = dict(counts)
        _, ms2, ts2 = encode_repeated()
        rep["fase2"] = {
            "encoded": True,
            "termos_use_quantized": termos_f2,
            "cond_norm": float(cond2.norm()),
            "cond_fingerprint": [float(v) for v in cond2.reshape(-1)[:8]],
            "encode_ms": ms2,
            "encode_ms_all": ts2,
            "counts": counts_f2,
            "rel_rmse_vs_fase1": float(((cond2 - cond) ** 2).mean().sqrt() / (cond ** 2).mean().sqrt()),
        }
    except Exception:
        rep["fase2"] = {"encoded": False, "traceback": traceback.format_exc()[-2000:],
                        "counts": dict(counts)}

print("@@JSON@@" + json.dumps(rep))
'''


def main():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    src = SRC % {"CKPT": str(CKPT), "PROMPT": PROMPT}
    proc = subprocess.run(
        [str(ROOT / "python_embeded/python.exe"), "-s", "-c", src],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    rep = None
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            rep = json.loads(line[len("@@JSON@@"):])
    if rep is None:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"nao devolveu JSON (rc={proc.returncode})")

    print("=" * 78)
    print(f"arquivo  {CKPT.name}   {CKPT.stat().st_size / 1024**3:.2f} GiB")
    print(f"prompt   {PROMPT!r}")
    print("=" * 78)
    print()
    print(f"Linear inspecionadas       {rep['n_linear_inspecionadas']}")
    print(f"classe do modulo           {rep['classes']}")
    print(f"_full_precision_mm         {rep['full_precision_mm']}")
    print(f"quant_format               {rep['quant_format']}")
    print(f"peso e QuantizedTensor     {rep['weight_is_quantized_tensor']}")
    print(f"exemplo                    {rep['exemplo_modulo']}")
    print()
    print(f"codificou                  {rep['encoded']}")
    if rep["encoded"]:
        print(f"cond                       shape {rep['cond_shape']}  norma {rep['cond_norm']:.4f}")
    else:
        print(rep.get("encode_traceback"))

    c = rep["counts"]
    print()
    print("-" * 78)
    print("CONTAGEM durante o encode (instrumentado APOS o load)")
    print("-" * 78)
    print(f"_convrot_w4a4_forward      {c['convrot_forward']}")
    print(f"QuantizedTensor.dequantize {c['dequantize']}")
    print(f"termos _use_quantized      {rep.get('termos_use_quantized')}")
    print()
    if c["convrot_forward"] > 0:
        print("VEREDITO  o encode EMITE o caminho de 4 bits do comfy-kitchen.")
    elif c["dequantize"] > 0:
        print("VEREDITO  ZERO chamadas ao caminho de 4 bits. O peso fica 4 bits na VRAM e a")
        print("          matematica e BF16 dequantizada. Economia de memoria, nao de tempo --")
        print("          e o kernel nativo nunca e alcancado por este caminho.")
    else:
        print("VEREDITO  nenhum dos dois contadores subiu. O peso pode nem ter chegado")
        print("          quantizado; ler `peso e QuantizedTensor` acima antes de concluir.")

    f2 = rep.get("fase2")
    if f2:
        print()
        print("-" * 78)
        print("FASE 2 -- as DUAS travas desligadas: `_full_precision_mm` e "
              f"`comfy_force_cast_weights` ({rep.get('n_linear_destravadas')} Linear)")
        print("-" * 78)
        if not f2.get("encoded"):
            print(f2.get("traceback"))
        else:
            c2 = f2["counts"]
            print(f"_convrot_w4a4_forward      {c2['convrot_forward']}")
            print(f"QuantizedTensor.dequantize {c2['dequantize']}")
            print(f"termos _use_quantized      {f2.get('termos_use_quantized')}")
            print(f"norma      fase1 {rep['cond_norm']:.4f}   fase2 {f2['cond_norm']:.4f}")
            print(f"rel-RMSE fase2 contra fase1 (BF16 dequantizado)   {f2['rel_rmse_vs_fase1']:.3e}")
            a, b = rep["encode_ms"], f2["encode_ms"]
            aa, bb = rep.get("encode_ms_all", []), f2.get("encode_ms_all", [])
            if a <= b:
                verd = f"BF16 {b / a:.2f}x mais rapido"
            else:
                verd = f"4 bits {a / b:.2f}x mais rapido"
            print(f"tempo do encode  fase1 {a:.1f} ms {aa}")
            print(f"                 fase2 {b:.1f} ms {bb}")
            print(f"                 -> {verd}  (medianas; a lista e o espalhamento bruto)")
            if c2["convrot_forward"] > 0:
                print()
                print("LEITURA  destravado, o checkpoint EMITE o caminho de 4 bits. E a trava")
                print("         que morde nao e `sd1_clip.py:114` -- a instrumentacao da fase 1")
                print("         mostra `comfy_force_cast_weights=True`, que vem de")
                print("         `comfy/sd.py:269`, `set_model_compute_dtype(torch.float32)`,")
                print("         aplicado a TODO objeto CLIP. Sao duas travas independentes, e")
                print("         so soltando as duas o kernel dispara.")

    print()
    print("NAO COBERTO: um modelo, um prompt, uma placa sm86. Nao avalia se a escolha")
    print("  upstream esta certa, so o que ela faz. Nao mede o modelo de difusao, cujo")
    print("  caminho e outro (`comfy/ops.py:1667`).")


if __name__ == "__main__":
    main()
