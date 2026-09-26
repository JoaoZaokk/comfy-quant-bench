"""Misto W4A8 + INT8: parte das camadas de um checkpoint W4A8 ja pronto troca para INT8 ConvRot.

Motivo (25/09): o 10Eros v1.5 W4A8 chia no audio e o BF16 nao (julgamento do dono, render no Colab). O palpite e
o 4 bits nas camadas de audio. Em vez de requantizar tudo, este conversor:

    camadas que casam --regex   INT8 ConvRot (int8_tensorwise, o mesmo de tools/quant_int8.py), quantizadas
                                A PARTIR DA FONTE BF16, uma por vez, dentro do laco de escrita (sem acumular)
    todo o resto                copiado BYTE A BYTE do W4A8 existente (pesos 4 bits, escalas, bias, VAE...)

`_quantization_metadata` sai do W4A8 com as entradas dessas camadas trocadas para
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}; o ComfyUI ja le misto por camada.

Recusas: saida/sidecar/partial existentes, fonte que nao e a do W4A8 (tamanho no sidecar do W4A8), tensor fora
das camadas com forma/dtype diferente da fonte, camada INT8 com K nao divisivel pelo grupo do ConvRot.
Cada camada INT8 e conferida pelo dequantizador real do ComfyUI (TensorWiseINT8Layout) contra a fonte; o erro
vai para o sidecar. Isso prova o formato, nao a qualidade -- qualidade e render.

    python_embeded\\python.exe -s tools/quant_misto_w4a8_int8.py --fonte P:/ComfyBench/checkpoints/10Eros_v1.5_bf16.safetensors \\
        --w4a8 P:/ComfyBench/checkpoints/10Eros_v1.5_bf16_w4a8.safetensors --dry-run
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import struct
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ComfyUI"))

import _conversion as C  # noqa: E402
from quant_int8 import quantize  # noqa: E402
from quant_w4a8 import read_header, read_tensor  # noqa: E402

REGEX_AUDIO = r"(?:^|\.)(?:audio_attn\d+|audio_ff|audio_to_video_attn|video_to_audio_attn)\."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fonte", required=True, type=Path, help="checkpoint BF16 de onde o W4A8 saiu")
    p.add_argument("--w4a8", required=True, type=Path, help="checkpoint W4A8 pronto (base da copia)")
    p.add_argument("--regex", default=REGEX_AUDIO, help="camadas (nome sem .weight) que viram INT8")
    p.add_argument("--convrot-groupsize", type=int, default=256)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    fonte, base = a.fonte.resolve(), a.w4a8.resolve()
    output = (a.output or base.with_name(base.stem + "_audioint8.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    conv = C.Conversion(base, output, sidecar)
    conv.refuse_unsafe(allow_quantized_source=True)  # a base E quantizada por construcao
    hb, mb = conv.header, conv.metadata
    hf, _ = read_header(fonte)

    side_base = base.with_suffix(".quant.json")
    if side_base.is_file():
        tam = json.loads(side_base.read_text(encoding="utf-8")).get("source_size")
        if tam != fonte.stat().st_size:
            raise SystemExit(f"RECUSADO: o W4A8 veio de uma fonte de {tam} B, esta tem {fonte.stat().st_size} B")
    else:
        raise SystemExit(f"RECUSADO: sem sidecar {side_base} para provar de que fonte o W4A8 saiu")

    qmeta = json.loads(mb["_quantization_metadata"])
    camadas = qmeta["layers"]
    rx = re.compile(a.regex)
    alvo = sorted(n for n in camadas if rx.search(n))
    if not alvo:
        raise SystemExit("RECUSADO: --regex nao casou nenhuma camada quantizada")
    for n in alvo:
        info = hf.get(n + ".weight")
        if info is None or len(info["shape"]) != 2:
            raise SystemExit(f"RECUSADO: {n}.weight ausente ou nao 2D na fonte")
        if info["shape"][1] % a.convrot_groupsize:
            raise SystemExit(f"RECUSADO: {n} K={info['shape'][1]} nao divide por {a.convrot_groupsize}")
    # tensores de cada camada alvo no W4A8 (peso 4 bits + escalas) saem; o bias fica
    sai = {k for n in alvo for k in hb if k.startswith(n + ".weight")}
    # fora das camadas quantizadas, a base tem de ser a fonte (mesma forma e dtype)
    prefixos = tuple(n + "." for n in camadas)
    for k, v in hb.items():
        if not k.startswith(prefixos):
            f = hf.get(k)
            if f is None or f["shape"] != v["shape"] or f["dtype"] != v["dtype"]:
                raise SystemExit(f"RECUSADO: {k} difere da fonte ({v['dtype']} {v['shape']} vs "
                                 f"{f and (f['dtype'], f['shape'])})")

    bytes_int8 = sum(hf[n + ".weight"]["shape"][0] * hf[n + ".weight"]["shape"][1] for n in alvo)
    bytes_sai = sum(hb[k]["data_offsets"][1] - hb[k]["data_offsets"][0] for k in sai)
    tam_saida = base.stat().st_size - bytes_sai + bytes_int8 + sum(hf[n + ".weight"]["shape"][0] * 4 for n in alvo)
    print(f"base {base.name}: {len(camadas)} camadas quantizadas; {len(alvo)} viram INT8 ConvRot")
    for n in alvo[:6]:
        print(f"  {n}  {hf[n + '.weight']['shape']}")
    print(f"  ... saida estimada {tam_saida / 2**30:.2f} GiB (base {base.stat().st_size / 2**30:.2f})  -> {output}")
    if a.dry_run:
        return 0
    conv.guard(tam_saida)

    # forma da escala: descoberta quantizando a primeira camada (nao presumida)
    fh = open(fonte, "rb")
    fdata = 8 + struct.unpack("<Q", fh.read(8))[0]

    def le(n):
        i = hf[n + ".weight"]
        s, e = i["data_offsets"]
        return read_tensor(fh, fdata + s, e - s, i["dtype"], i["shape"])

    _, esc0 = quantize(le(alvo[0]), True, a.convrot_groupsize)
    forma_esc = lambda n: [hf[n + ".weight"]["shape"][0], *esc0.shape[1:]]  # noqa: E731

    from comfy.quant_ops import QUANT_ALGOS, get_layout_class
    layout = get_layout_class(QUANT_ALGOS["int8_tensorwise"]["comfy_tensor_layout"])
    erros: dict[str, float] = {}
    pendente: dict[str, torch.Tensor] = {}

    def produz_peso(n):
        def f():
            w = le(n)
            q, s = quantize(w, True, a.convrot_groupsize)
            q, s = q.cpu().contiguous(), s.cpu().contiguous().float()
            params = layout.Params(scale=s, orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape),
                                   convrot=True, convrot_groupsize=a.convrot_groupsize)
            d = layout.dequantize(q, params).float()
            erros[n] = float((d - w.float()).norm() / w.float().norm().clamp_min(1e-30))
            pendente[n] = s
            return q
        return f

    def produz_escala(n):
        return lambda: pendente.pop(n)

    entradas, feitos = [], set()
    for k, v in hb.items():
        dono = next((n for n in alvo if k.startswith(n + ".weight")), None)
        if dono is None:
            entradas.append(C.plan_copy(k, v))
        elif dono not in feitos:  # no lugar do primeiro tensor do peso 4 bits, escreve peso INT8 + escala
            feitos.add(dono)
            N, K = hf[dono + ".weight"]["shape"]
            fe = forma_esc(dono)
            entradas.append(C.plan_lazy(dono + ".weight", "I8", [N, K], N * K, produz_peso(dono)))
            entradas.append(C.plan_lazy(dono + ".weight_scale", "F32", fe, 4 * int(torch.tensor(fe).prod()),
                                        produz_escala(dono)))
    assert feitos == set(alvo)

    for n in alvo:
        camadas[n] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": a.convrot_groupsize}
    meta = dict(mb)
    meta["_quantization_metadata"] = json.dumps(qmeta, separators=(",", ":"))
    meta["quantization"] = f"{mb.get('quantization', '?')}+int8_tensorwise_convrot({len(alvo)} camadas)"

    t0 = time.perf_counter()
    ultimo = [0.0]

    def progresso(i, tot, chave):
        if time.perf_counter() - ultimo[0] > 30 or i == tot:
            ultimo[0] = time.perf_counter()
            print(f"[{i}/{tot}] {len(erros)}/{len(alvo)} INT8  {time.perf_counter() - t0:.0f} s  {chave}", flush=True)

    conv.commit(entradas, meta, progress=progresso)
    fh.close()
    vals = sorted(erros.values())
    conv.write_sidecar({
        "fonte": str(fonte), "fonte_size": fonte.stat().st_size, "base_w4a8": str(base),
        "base_size": base.stat().st_size, "output": str(output), "output_size": output.stat().st_size,
        "regex_int8": a.regex, "camadas_int8": len(alvo), "camadas_w4a8": len(camadas) - len(alvo),
        "int8": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": a.convrot_groupsize},
        "erro_rel_int8_vs_fonte": {"mediana": vals[len(vals) // 2], "max": vals[-1],
                                   "pior": max(erros, key=erros.get)},
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__, "segundos": round(time.perf_counter() - t0, 1),
    })
    print(f"OK {output} ({output.stat().st_size / 2**30:.2f} GiB); erro INT8 mediana {vals[len(vals) // 2]:.4f} "
          f"max {vals[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
