"""Um checkpoint quantizado emite matematica quantizada, ou so ocupa menos VRAM?

DE ONDE VEM. Em 2026-08-31 mediu-se, contando chamadas, que o encoder ConvRot INT4 do
Winnougan carrega e roda no ComfyUI de estoque e faz **zero** chamadas ao caminho de 4
bits: 350 dequantize e nada mais. A trava e `comfy_force_cast_weights`, ligada para TODO
objeto CLIP em `comfy/sd.py:269` via `set_model_compute_dtype(torch.float32)`.

Isso levanta duas perguntas que valem mais que aquele arquivo:

  1. O `gemma` deste projeto e carregado como text encoder. Sofre o mesmo?
  2. O `zimage-v2-w4a4` deste projeto e modelo de DIFUSAO, outro caminho
     (`comfy/ops.py:1667`, que passa `disabled=` e nao `full_precision_mm`). Toda a
     medicao de epsilon-por-passo desta bancada assumiu que ele executa quantizado. Se
     nao executar, aquelas medicoes compararam duas dequantizacoes.

COMO MEDE. Nao re-deriva `_use_quantized`: le o kwarg que o proprio `ops.py` calcula a
partir dele e passa adiante (`weight_only_quant`, `ops.py:1414-1419`), embrulhando
`forward_comfy_cast_weights`. Em paralelo conta `QuantizedTensor.dequantize`, que e o
sinal contrario. Instrumenta **depois** do load, porque o load dequantiza de forma
legitima.

  weight_only_quant True em toda Linear  -> a matematica e quantizada
  dequantize > 0 e weight_only_quant 0   -> peso 4/8 bits na VRAM, matematica BF16

NAO COBERTO: nao inspeciona SASS. Nao afirma fidelidade -- nao ha referencia BF16 casada
aqui. Uma placa. Para difusao roda poucos passos num lado pequeno, o suficiente para o
dispatch acontecer, nao para julgar imagem.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC = r'''
import json, sys, traceback
from collections import Counter
sys.path.insert(0, "ComfyUI"); sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd, comfy.sample
from comfy_kitchen.tensor.base import QuantizedTensor

MODE = %(MODE)r; CKPT = %(CKPT)r; CLIP = %(CLIP)r; CLIP_TYPE = %(CLIP_TYPE)r
PROMPT = %(PROMPT)r; STEPS = %(STEPS)d; SIDE = %(SIDE)d; SEED = %(SEED)d
rep = {"mode": MODE, "ckpt": CKPT}

if MODE == "te":
    obj = comfy.sd.load_clip(
        ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CKPT)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, CLIP_TYPE))
    root = obj.cond_stage_model
else:
    obj = comfy.sd.load_diffusion_model(
        folder_paths.get_full_path_or_raise("diffusion_models", CKPT))
    root = obj.model
    clip = comfy.sd.load_clip(
        ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, CLIP_TYPE))
    positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0])]
    negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(""))[0])]
    del clip
    import comfy.model_management as mm; mm.soft_empty_cache()

# estado dos modulos quantizados, antes de qualquer forward
quantized_mods = []
for name, mod in root.named_modules():
    if getattr(mod, "layout_type", None) is not None:
        quantized_mods.append(mod)
        if len(quantized_mods) == 1:
            rep["exemplo_modulo"] = {
                "name": name, "class_qualname": type(mod).__qualname__,
                "quant_format": getattr(mod, "quant_format", None),
                "layout_type": getattr(mod, "layout_type", None),
                "full_precision_mm": getattr(mod, "_full_precision_mm", None),
                "full_precision_mm_config": getattr(mod, "_full_precision_mm_config", None),
                "comfy_force_cast_weights": bool(getattr(mod, "comfy_force_cast_weights", False)),
                "weight_is_qt": isinstance(getattr(mod, "weight", None), QuantizedTensor),
            }
rep["n_modulos_quantizados"] = len(quantized_mods)
rep["quant_format"] = dict(Counter(str(getattr(m, "quant_format", None)) for m in quantized_mods))
rep["full_precision_mm"] = dict(Counter(str(getattr(m, "_full_precision_mm", None)) for m in quantized_mods))
rep["comfy_force_cast_weights"] = dict(Counter(str(bool(getattr(m, "comfy_force_cast_weights", False))) for m in quantized_mods))

if not quantized_mods:
    rep["erro"] = "nenhum modulo com layout_type: o checkpoint nao chegou quantizado"
    print("@@JSON@@" + json.dumps(rep)); sys.exit(0)

# ---- instrumentacao, SO depois do load
counts = Counter()
classes = {type(m) for m in quantized_mods}
originais = {}
for cls in classes:
    originais[cls] = cls.forward_comfy_cast_weights
    def make(orig):
        def spy(self, input, *a, **k):
            # A classe MixedPrecisionOps.Linear serve TODA Linear do modelo, quantizada ou
            # nao. Contar todas mistura camadas que nunca foram quantizadas com camadas
            # que foram e cairam no caminho BF16 -- foi o que a primeira versao fez, e o
            # resultado saiu "MISTO 340 contra 76" quando as 76 nem tinham peso quantizado.
            if getattr(self, "layout_type", None) is None:
                counts["fora_de_escopo_sem_layout"] += 1
                return orig(self, input, *a, **k)
            if k.get("weight_only_quant"):
                counts["quantizado"] += 1
            elif isinstance(input, QuantizedTensor):
                counts["quantizado_com_entrada_qt"] += 1
            else:
                counts["nao_quantizado"] += 1
            return orig(self, input, *a, **k)
        return spy
    cls.forward_comfy_cast_weights = make(originais[cls])

_orig_dq = QuantizedTensor.dequantize
def counted_dq(self, *a, **k):
    counts["dequantize"] += 1
    return _orig_dq(self, *a, **k)
QuantizedTensor.dequantize = counted_dq

try:
    if MODE == "te":
        out = obj.encode_from_tokens_scheduled(obj.tokenize(PROMPT))
        cond = out[0][0].float()
        rep["saida"] = {"shape": list(cond.shape), "norm": float(cond.norm())}
    else:
        lf = obj.model.latent_format
        latent = torch.zeros([1, lf.latent_channels, SIDE // 8, SIDE // 8], device="cpu")
        noise = comfy.sample.prepare_noise(latent, SEED, None)
        s = comfy.sample.sample(obj, noise, STEPS, 1.0, "euler", "simple",
                                positive, negative, latent, denoise=1.0,
                                disable_pbar=True, seed=SEED)
        rep["saida"] = {"shape": list(s.shape), "norm": float(s.float().norm())}
    rep["rodou"] = True
except Exception:
    rep["rodou"] = False
    rep["traceback"] = traceback.format_exc()[-2500:]

rep["counts"] = dict(counts)
print("@@JSON@@" + json.dumps(rep))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", help="nome do arquivo dentro de text_encoders/ ou diffusion_models/")
    p.add_argument("--mode", choices=["te", "diffusion"], required=True)
    p.add_argument("--clip-type", default="LUMINA2",
                   help="para --mode te e o tipo do proprio arquivo; para diffusion e o do --clip")
    p.add_argument("--clip", default="qwen_3_4b.safetensors",
                   help="so em --mode diffusion: text encoder que produz o condicionamento")
    p.add_argument("--prompt", default="a red apple on a weathered wooden table, soft window light")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.device)
    src = SRC % {"MODE": a.mode, "CKPT": a.ckpt, "CLIP": a.clip, "CLIP_TYPE": a.clip_type,
                 "PROMPT": a.prompt, "STEPS": a.steps, "SIDE": a.size, "SEED": a.seed}
    proc = subprocess.run([str(ROOT / "python_embeded/python.exe"), "-s", "-c", src],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    rep = None
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            rep = json.loads(line[len("@@JSON@@"):])
    if rep is None:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"nao devolveu JSON (rc={proc.returncode})")

    print("=" * 78)
    print(f"{rep['ckpt']}   modo {rep['mode']}   device cuda:{a.device}")
    print("=" * 78)
    print(f"modulos quantizados        {rep['n_modulos_quantizados']}")
    if rep.get("erro"):
        print(f"ERRO  {rep['erro']}")
        return
    print(f"quant_format               {rep['quant_format']}")
    print(f"_full_precision_mm         {rep['full_precision_mm']}")
    print(f"comfy_force_cast_weights   {rep['comfy_force_cast_weights']}")
    print(f"exemplo                    {rep['exemplo_modulo']}")
    print(f"rodou                      {rep['rodou']}")
    if rep["rodou"]:
        print(f"saida                      {rep['saida']}")
    else:
        print(rep.get("traceback"))

    c = rep["counts"]
    q = c.get("quantizado", 0) + c.get("quantizado_com_entrada_qt", 0)
    nq = c.get("nao_quantizado", 0)
    dq = c.get("dequantize", 0)
    print()
    print("-" * 78)
    print("CONTAGEM durante o forward (instrumentado APOS o load)")
    print("-" * 78)
    print(f"forwards com matematica quantizada   {q}")
    print(f"forwards SEM                         {nq}")
    print(f"QuantizedTensor.dequantize           {dq}")
    print(f"(fora de escopo: Linear sem layout   {c.get('fora_de_escopo_sem_layout', 0)})")
    print()
    if q > 0 and nq == 0:
        print("VEREDITO  toda Linear quantizada faz matematica quantizada.")
    elif q > 0:
        print(f"VEREDITO  MISTO: {q} quantizados contra {nq} nao. Ler camada a camada antes")
        print("          de afirmar qualquer coisa sobre o modelo inteiro.")
    else:
        print("VEREDITO  ZERO forwards quantizados. O peso ocupa menos VRAM e a matematica e")
        print("          dequantizada. E economia de memoria, nao de tempo.")

    print()
    print("NAO COBERTO: sem SASS; sem referencia BF16 casada, entao nada aqui afirma")
    print("  fidelidade; uma placa; em difusao, poucos passos num lado pequeno -- o bastante")
    print("  para o dispatch acontecer, nao para julgar imagem.")


if __name__ == "__main__":
    main()
