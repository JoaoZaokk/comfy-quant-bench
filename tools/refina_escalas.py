"""Refina a escala POR LINHA de um checkpoint quantizado, em forma fechada, sobre ativacoes reais.

Nos tres formatos da bancada a saida de uma linha n de peso e `y[t, n] = a[t] * s[n] * z[t, n]`, com `z` o produto
dos codigos (inteiros) e `a` a escala da ativacao, que nao depende do peso:

    convrot_w4a4      s = `weight_scale`      [N]
    int8_tensorwise   s = `weight_scale`      [N, 1]
    asym_w4a8_int8    s = `weight_s_channel`  [N]   (a escala por grupo `s_rel` fica)

Com os codigos fixos, o `s[n]` que minimiza ||y_q - y_ref||^2 sobre as linhas de calibracao e exato:
`s*[n] = s[n] * <y_ref[:, n], y_q[:, n]> / <y_q[:, n], y_q[:, n]>`, onde `y_q` sai do kernel real com a escala atual.
E o "so escalas" do QAT do klein (b10) resolvido camada a camada, sem gradiente pela rede e sem treino: um
minimos-quadrados por linha. Metade das linhas da calibracao ajusta, a outra metade mede (held-out); uma camada so
recebe a escala nova se o erro held-out cair.

Nao mexe em codigo nenhum, nem em formato: a saida e o mesmo arquivo com os tensores de escala trocados.

    python_embeded\\python.exe -s tools/refina_escalas.py --model Q.safetensors --source BF16.safetensors \\
        --calib calib.pt [--output Q_escalas.safetensors]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _conversion as C  # noqa: E402
from verify_w4a4 import FORMATS, load_tensor_cuda  # noqa: E402
from erro_por_camada import configs_por_camada  # noqa: E402

ESCALA = {"convrot_w4a4": "weight_scale", "int8_tensorwise": "weight_scale", "asym_w4a8_int8": "weight_s_channel"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    model = args.model.resolve()
    output = (args.output or model.with_name(f"{model.stem}_escalas.safetensors")).resolve()
    conv = C.Conversion(model, output, output.with_suffix(".quant.json"))
    conv.refuse_unsafe(allow_quantized_source=True)
    header, metadata = conv.header, conv.metadata
    src_header, _ = C.read_header(args.source)
    camadas = configs_por_camada(header, metadata, model)
    calib = torch.load(args.calib, map_location="cpu", weights_only=False)["layers"]
    import comfy_kitchen as ck

    novas, relatorio = {}, {}
    t0 = time.time()
    for idx, (camada, cfg) in enumerate(sorted(camadas.items())):
        if camada not in calib:
            raise SystemExit(f"{camada}: sem calibracao")
        fmt = FORMATS[cfg["format"]]
        chave = f"{camada}.{ESCALA[cfg['format']]}"
        x = calib[camada]["sample"].to("cuda", torch.bfloat16)
        ajuste, medida = x[0::2], x[1::2]
        w = load_tensor_cuda(args.source, src_header[camada + ".weight"]).float()
        s_atual = load_tensor_cuda(model, header[chave])

        def saida(xx, s):
            def load(suf, view=None, optional=False):
                if suf == "." + ESCALA[cfg["format"]]:
                    return s
                info = header.get(camada + suf)
                if info is None:
                    if optional:
                        return None
                    raise KeyError(camada + suf)
                return load_tensor_cuda(model, info, view=view)
            return getattr(ck, fmt.linear_op)(**fmt.smoke_kwargs(camada, cfg, load, xx, {})).float()

        ref_a = torch.nn.functional.linear(ajuste.float(), w)
        ya = saida(ajuste, s_atual)
        c = (ref_a * ya).sum(0) / ya.square().sum(0).clamp_min(1e-30)   # [N]
        c = torch.where(torch.isfinite(c) & (c > 0), c, torch.ones_like(c))
        s_nova = (s_atual.float() * c.reshape(s_atual.shape)).to(s_atual.dtype)
        ref_m = torch.nn.functional.linear(medida.float(), w)
        antes = float((saida(medida, s_atual) - ref_m).norm() / ref_m.norm())
        depois = float((saida(medida, s_nova) - ref_m).norm() / ref_m.norm())
        aceita = depois < antes
        if aceita:
            novas[chave] = s_nova.cpu().contiguous()
        relatorio[camada] = {"formato": cfg["format"], "antes": antes, "depois": depois, "aceita": aceita,
                             "fator_mediana": float(c.median()), "fator_min": float(c.min()), "fator_max": float(c.max())}
        if idx % 24 == 0:
            print(f"[{idx}/{len(camadas)}] {camada} {antes:.4f} -> {depois:.4f}", flush=True)
        del x, w, ref_a, ref_m, ya
        torch.cuda.empty_cache()

    entradas = [C.plan_write(k, novas[k]) if k in novas else C.plan_copy(k, info) for k, info in header.items()]
    meta = dict(metadata)
    meta["escalas_refinadas"] = json.dumps({"ferramenta": "tools/refina_escalas.py", "calib": str(args.calib),
                                            "camadas_trocadas": len(novas), "de": len(camadas)})
    conv.commit(entradas, meta)
    ant = [r["antes"] for r in relatorio.values()]
    dep = [r["depois"] if r["aceita"] else r["antes"] for r in relatorio.values()]
    resumo = {"model": str(model), "output": str(output), "camadas": len(camadas), "trocadas": len(novas),
              "erro_heldout_medio_antes": sum(ant) / len(ant), "erro_heldout_medio_depois": sum(dep) / len(dep),
              "segundos": round(time.time() - t0, 1), "por_camada": relatorio}
    conv.write_sidecar(resumo)
    print(json.dumps({k: v for k, v in resumo.items() if k != "por_camada"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
