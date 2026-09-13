"""Quanto de um LoRA sobrevive quando ele e fundido num peso de 4 bits -- medido no caminho real.

O MECANISMO, LIDO NO CODIGO (traçado, 2026-09-13)
-------------------------------------------------
`LoraLoaderModelOnly` sobre um checkpoint quantizado NAO mantem o LoRA como ramo separado.
`ModelPatcher.patch_weight_to_device` (`comfy/model_patcher.py:899`) faz, para cada chave com
patch:

    convert_weight  ->  W.dequantize()                                   (comfy/ops.py:1449)
    calculate_weight -> W + delta, em `lora_compute_dtype` (fp16 nesta placa)
    set_weight      ->  W.requantize_from_float(W', scale="recalculate",
                                                stochastic_rounding=seed)  (comfy/ops.py:1455-1457)

Ou seja: dequantiza, soma, e REQUANTIZA para os mesmos 4 bits, com escalas recalculadas e
arredondamento estocastico. O kernel nativo continua sendo o mesmo (o peso volta a ser
`QuantizedTensor` do mesmo layout) -- isto ja tinha sido medido em 2026-08-19: 680/680 despachos
no kernel com LoRA aplicado. O que NUNCA foi medido e o que este arquivo mede: **o delta do LoRA
e tipicamente MENOR que o passo de quantizacao**, entao ao requantizar ele so sobrevive em media
(arredondamento estocastico e nao-enviesado) e cada camada com patch ganha um ruido de
requantizacao NOVO, independente do erro de quantizacao original.

O QUE E MEDIDO, POR CAMADA, NO CAMINHO REAL
-------------------------------------------
    W0      = dequantize(peso de 4 bits)                          o que o modelo tinha
    delta   = calculate_weight(patches, W0) - W0                   o LoRA exato, em fp32
    W1q     = patch_weight_to_device(chave, return_weight=True)    o caminho REAL do ComfyUI
    d_eff   = dequantize(W1q) - W0                                 o que de fato mudou no peso

    sobrevivencia = <d_eff, delta> / <delta, delta>   1.0 = o delta esta la, em media
    cosseno       = cos(d_eff, delta)                 1.0 = e SO o delta; baixo = delta + ruido
    ruido_extra   = |d_eff - delta| / |delta|         quanto ruido a requantizacao acrescentou,
                                                      em unidades do proprio LoRA
    magnitude     = |delta| / |W0|                    quao pequeno o LoRA e frente ao peso

Com `--source` (o BF16 original) entram dois erros contra a verdade:
    err_antes  = |W0 - Ws| / |Ws|                     erro de quantizacao original da camada
    err_depois = |deq(W1q) - (Ws + delta)| / |Ws + delta|   erro da camada quantizada+LoRA contra
                                                      o BF16+LoRA, que e o que o usuario queria

PREVISOES ESCRITAS ANTES DE RODAR (criterio previo)
---------------------------------------------------
    P1  sobrevivencia mediana entre 0,9 e 1,1: arredondamento estocastico nao tem vies.
        Refuta: mediana < 0,8 (o LoRA esta sendo perdido) ou > 1,2.
    P2  err_depois / err_antes ~ sqrt(2) = 1,41: o ruido novo e independente do velho e do
        mesmo tamanho. Refuta: razao < 1,1 (a requantizacao e quase gratis) ou > 2,0.
    P3  ruido_extra > 1 na maioria das camadas: o ruido acrescentado e MAIOR que o LoRA.
        Refuta: mediana < 0,5.

NAO COBERTO
-----------
Peso, nao saida: isto nao diz o que a imagem faz com o ruido -- para isso `ltx_video.py --lora`
e `qwen_edit_test.py --lora` renderizam os bracos com e sem. Nao mede o bypass
(`LoraLoaderBypassModelOnly`), que por construcao nao toca o peso e nao tem este ruido; ele e
o braco de comparacao nos renders. Amostra `--n` camadas espacadas, nao todas, salvo `--n 0`.
Um LoRA e um modelo por chamada.

    python_embeded\\python.exe -s tools\\probe_lora_requant.py --ckpt X.safetensors --lora Y.safetensors
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="nome em diffusion_models/ (ou checkpoints/ com --checkpoint-mode)")
    p.add_argument("--checkpoint-mode", action="store_true",
                   help="o arquivo e checkpoint unico (LTX 2.3): load_checkpoint_guess_config")
    p.add_argument("--lora", required=True, help="nome em loras/")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--source", default=None, help="caminho do BF16 original, para err_antes/err_depois")
    p.add_argument("--n", type=int, default=32, help="camadas amostradas, espacadas; 0 = todas")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    # Antes de importar torch: e assim que a placa e escolhida sem tocar no resto do ComfyUI.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "ComfyUI"))
    sys.path.insert(0, str(ROOT / "tools"))
    sys.argv = ["main.py"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    # `main.py` e quem le `extra_model_paths.yaml`; um processo que importa `folder_paths` sozinho
    # so enxerga `ComfyUI/models`. Sem esta linha, tudo que mora em P:, W: ou D: "nao existe".
    from utils.extra_config import load_extra_path_config
    _yaml = ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if _yaml.exists():
        load_extra_path_config(str(_yaml))
    import comfy.sd
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils
    import comfy.model_management as mm
    from comfy.model_patcher import get_key_weight
    from comfy_kitchen.tensor.base import QuantizedTensor
    from _conversion import data_start, read_header, read_tensor

    dev = torch.device("cuda:0")
    t0 = time.time()
    if a.checkpoint_mode:
        path = folder_paths.get_full_path_or_raise("checkpoints", a.ckpt)
        model = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=False)[0]
    else:
        path = folder_paths.get_full_path_or_raise("diffusion_models", a.ckpt)
        model = comfy.sd.load_diffusion_model(path)
    print(f"modelo carregado em {time.time() - t0:.0f}s: {a.ckpt}", flush=True)

    # ---- casamento de chaves, contado e nao deduzido ----
    lora_path = folder_paths.get_full_path_or_raise("loras", a.lora)
    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    n_lora_tensors = len(lora_sd)
    avisos: list[str] = []

    class Captura(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if "lora key not loaded" in msg or "NOT LOADED" in msg:
                avisos.append(msg)
    h = Captura()
    logging.getLogger().addHandler(h)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_c = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_c, key_map)
    patcher = model.clone()
    applied = set(patcher.add_patches(loaded, a.strength))
    logging.getLogger().removeHandler(h)
    # `add_patches` devolve as chaves DO LORA que casaram: strings, ou tuplas
    # `(chave_do_modelo, offset)` quando o ComfyUI funde q/k/v num unico `qkv` e o LoRA mira uma
    # fatia. `patcher.patches` e indexado pela chave do modelo, e `calculate_weight` aplica o
    # offset (narrow). Medir por chave do modelo, entao, e uma medicao por peso fisico.
    def chave_modelo(k):
        return k if isinstance(k, str) else k[0]
    nao_no_modelo = sorted({chave_modelo(k) for k in set(loaded) - applied})
    chaves = sorted({chave_modelo(k) for k in applied})
    print(f"LoRA {a.lora}: {n_lora_tensors} tensores -> {len(loaded)} alvos -> "
          f"{len(applied)} aplicados em {len(chaves)} peso(s); {len(nao_no_modelo)} alvo(s) sem peso no modelo; "
          f"{len(avisos)} chave(s) do LoRA que o ComfyUI nao soube mapear", flush=True)
    if not chaves:
        raise SystemExit("nenhuma chave do LoRA casou com o modelo: nada a medir. "
                         "Isso e um resultado (LoRA de outra arquitetura), nao um erro do probe.")

    # ---- fonte BF16, opcional ----
    src_header = src_start = None
    if a.source:
        src_header, _ = read_header(Path(a.source))
        src_start = data_start(Path(a.source))

    def acha_na_fonte(model_key: str):
        sem = model_key[len("diffusion_model."):] if model_key.startswith("diffusion_model.") else model_key
        for cand in (model_key, "model." + model_key, sem, "model.diffusion_model." + sem):
            if cand in src_header:
                return cand
        fins = [k for k in src_header if k.endswith("." + sem)]
        return fins[0] if len(fins) == 1 else None

    def le_fonte(k: str) -> torch.Tensor:
        info = src_header[k]
        s, e = info["data_offsets"]
        with open(a.source, "rb") as fh:
            return read_tensor(fh, src_start + s, e - s, info["dtype"], info["shape"])

    # ---- amostra ----
    # Embaralhado com semente fixa, nao espacado: com ~11 pesos por bloco, um passo de ~11
    # sobre a lista ordenada caia sempre na MESMA familia (o primeiro run mediu 8x `w3` e nada
    # mais). Embaralhar cobre familias e profundidades; a semente fixa mantem a amostra
    # reproduzivel entre modelos e entre LoRAs.
    if a.n and a.n < len(chaves):
        import random
        amostra = sorted(random.Random(20260913).sample(chaves, a.n))
    else:
        amostra = chaves

    seed_de = comfy.utils.string_to_seed
    linhas = []
    n_quant = n_nao_quant = 0
    falhas_calculo: list[str] = []

    class CapturaErro(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if "ERROR lora" in msg or "Calculate Weight Failed" in msg:
                falhas_calculo.append(msg)
    he = CapturaErro()
    logging.getLogger().addHandler(he)
    temp_dtype = mm.lora_compute_dtype(dev)
    for key in amostra:
        mod_name = key.rsplit(".", 1)[0]
        mod = comfy.utils.get_attr(patcher.model, mod_name)
        mod.to(dev)
        weight, set_func, convert_func = get_key_weight(patcher.model, key)
        if not isinstance(weight, QuantizedTensor):
            n_nao_quant += 1
            mod.to(mm.unet_offload_device())
            continue
        n_quant += 1
        W0 = weight.dequantize().float()
        delta = comfy.lora.calculate_weight(patcher.patches[key], W0.clone(), key,
                                            intermediate_dtype=torch.float32) - W0
        # O caminho real: dequantiza em lora_compute_dtype, soma, requantiza com semente da chave.
        W1q = patcher.patch_weight_to_device(key, device_to=dev, return_weight=True)
        mesmo_layout = isinstance(W1q, QuantizedTensor) and W1q._layout_cls == weight._layout_cls
        d_eff = W1q.dequantize().float() - W0
        dd = float((delta * delta).sum())
        # CONTROLE: requantizar W0 sem delta nenhum, pelo mesmo set_weight (mesmo dtype
        # intermediario, mesma semente). Se isto ja custa erro, a requantizacao nao e idempotente
        # e parte do "custo do LoRA" e custo de qualquer patch -- inclusive um que falhou.
        R0 = set_func(W0.to(temp_dtype).clone(), seed=seed_de(key), return_weight=True).dequantize().float()
        ruido_zero = float((R0 - W0).norm() / W0.norm())
        lin = {"chave": key, "shape": list(W0.shape), "layout": weight._layout_cls,
               "mesmo_layout_apos_patch": bool(mesmo_layout),
               "magnitude": float(delta.norm() / W0.norm()),
               "delta_zero": dd == 0.0,
               "ruido_zero": ruido_zero,
               "sobrevivencia": float((d_eff * delta).sum() / dd) if dd > 0 else None,
               "cosseno": float(torch.nn.functional.cosine_similarity(d_eff.flatten(), delta.flatten(), dim=0)) if dd > 0 else None,
               "ruido_extra": float((d_eff - delta).norm() / delta.norm()) if dd > 0 else None}
        if src_header is not None:
            sk = acha_na_fonte(key)
            if sk is not None:
                Ws = le_fonte(sk).to(dev).float()
                if Ws.shape == W0.shape:
                    alvo = Ws + delta
                    lin["err_antes"] = float((W0 - Ws).norm() / Ws.norm())
                    lin["err_depois"] = float((W0 + d_eff - alvo).norm() / alvo.norm())
                    lin["err_zero"] = float((R0 - Ws).norm() / Ws.norm())
                    lin["fonte_chave"] = sk
                del Ws
        linhas.append(lin)
        mod.to(mm.unet_offload_device())
        del W0, delta, W1q, d_eff, R0
        torch.cuda.empty_cache()
        print(f"  {key[-60:]:<60} magn {lin['magnitude']:.2e}  sobrev {lin['sobrevivencia'] or 0:.3f}  "
              f"cos {lin['cosseno'] or 0:.3f}  ruido/lora {lin['ruido_extra'] or 0:.2f}"
              + f"  ruido0 {ruido_zero:.4f}"
              + (f"  err {lin['err_antes']:.4f}->{lin['err_depois']:.4f} (zero {lin['err_zero']:.4f})" if 'err_antes' in lin else ""),
              flush=True)

    logging.getLogger().removeHandler(he)

    def mediana(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None

    def resumo(campo):
        xs = [l[campo] for l in linhas if l.get(campo) is not None]
        return {"mediana": mediana(xs), "min": min(xs) if xs else None, "max": max(xs) if xs else None, "n": len(xs)}

    rep = {"ckpt": a.ckpt, "lora": a.lora, "strength": a.strength, "source": a.source,
           "lora_tensores": n_lora_tensors, "alvos": len(loaded), "aplicados": len(applied),
           "pesos_com_patch": len(chaves),
           "alvos_sem_peso_no_modelo": nao_no_modelo[:20], "chaves_nao_mapeadas": len(avisos),
           "amostra_quantizadas": n_quant, "amostra_nao_quantizadas": n_nao_quant,
           "todas_mantem_layout": all(l["mesmo_layout_apos_patch"] for l in linhas),
           "sobrevivencia": resumo("sobrevivencia"), "cosseno": resumo("cosseno"),
           "ruido_extra": resumo("ruido_extra"), "magnitude": resumo("magnitude"),
           "err_antes": resumo("err_antes"), "err_depois": resumo("err_depois"),
           "err_zero": resumo("err_zero"), "ruido_zero": resumo("ruido_zero"),
           "falhas_calculate_weight": len(falhas_calculo), "exemplo_falha": falhas_calculo[:1],
           "camadas_com_delta_zero": sum(1 for l in linhas if l["delta_zero"]),
           "camadas": linhas}
    razoes = [l["err_depois"] / l["err_antes"] for l in linhas if l.get("err_antes")]
    rep["razao_err_depois_antes"] = {"mediana": mediana(razoes), "min": min(razoes) if razoes else None,
                                     "max": max(razoes) if razoes else None}

    print("\n== resumo ==", flush=True)
    print(f"casamento : {n_lora_tensors} tensores do LoRA -> {len(applied)} alvos em {len(chaves)} pesos do modelo; "
          f"{len(avisos)} chaves nao mapeadas; {len(nao_no_modelo)} alvos sem peso", flush=True)
    print(f"amostra   : {n_quant} quantizadas, {n_nao_quant} nao quantizadas; layout preservado em todas: "
          f"{rep['todas_mantem_layout']}", flush=True)
    if falhas_calculo:
        print(f"ATENCAO   : {len(falhas_calculo)} falha(s) em calculate_weight -- o ComfyUI logou e SEGUIU; "
              f"{rep['camadas_com_delta_zero']} camada(s) da amostra ficaram com delta ZERO e mesmo assim "
              f"foram requantizadas. Ex: {falhas_calculo[0][:110]}", flush=True)
    for c in ("magnitude", "sobrevivencia", "cosseno", "ruido_extra", "ruido_zero", "err_antes", "err_zero", "err_depois"):
        r = rep[c]
        if r["n"]:
            print(f"{c:<14} mediana {r['mediana']:.4f}   min {r['min']:.4f}   max {r['max']:.4f}   (n={r['n']})", flush=True)
    if razoes:
        r = rep["razao_err_depois_antes"]
        print(f"err_depois/err_antes  mediana {r['mediana']:.3f}   min {r['min']:.3f}   max {r['max']:.3f}", flush=True)
    s = rep["sobrevivencia"]["mediana"]; x = rep["ruido_extra"]["mediana"]; q = rep["razao_err_depois_antes"]["mediana"]
    print("P1 (sobrevivencia 0,9-1,1):", "CONFIRMADA" if s is not None and 0.9 <= s <= 1.1 else ("REFUTADA" if s is not None else "nao medida"), flush=True)
    print("P2 (err_depois/err_antes ~1,41, refuta <1,1 ou >2,0):",
          ("CONFIRMADA" if 1.1 <= q <= 2.0 else "REFUTADA") if q is not None else "nao medida (sem --source)", flush=True)
    print("P3 (ruido_extra > 1 na maioria):", "CONFIRMADA" if x is not None and x > 1 else ("REFUTADA" if x is not None else "nao medida"), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"json em {a.json}", flush=True)
    print("\nNAO COBERTO: peso, nao saida -- o que a imagem faz com este ruido e medido pelos renders "
          "com e sem LoRA. Bypass nao entra aqui. Amostra de camadas, nao todas, salvo --n 0. "
          "Um LoRA, um modelo, uma forca.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
