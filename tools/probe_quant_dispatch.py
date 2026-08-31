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

DYNVRAM = %(DYNVRAM)s
if DYNVRAM:
    # Sem isto, `load_torch_file` usa `safetensors.safe_open`, que RECUSA um arquivo com
    # bytes depois do ultimo tensor: "incomplete metadata, file not fully covered". Os dois
    # checkpoints publicos da Abiray tem exatamente isso -- 83 e 64 bytes de marcador colados
    # no fim, presentes tambem no arquivo do servidor (Content-Length bate byte a byte), entao
    # nao e download quebrado. O caminho do dynamic-VRAM usa outro leitor e os aceita.
    import comfy_aimdo.control, comfy.model_management, comfy.memory_management
    comfy_aimdo.control.init()
    _ok = comfy_aimdo.control.init_devices(
        d.index for d in comfy.model_management.get_all_torch_devices())
    comfy.memory_management.aimdo_enabled = bool(_ok)

MODE = %(MODE)r; CKPT = %(CKPT)r; CLIP = %(CLIP)r; CLIP_TYPE = %(CLIP_TYPE)r
PROMPT = %(PROMPT)r; STEPS = %(STEPS)d; SIDE = %(SIDE)d; SEED = %(SEED)d
FRAMES = %(FRAMES)d
FORWARD_ONLY = %(FORWARD_ONLY)s; FORWARD_N = %(FORWARD_N)d; FORWARD_M = %(FORWARD_M)d
import os as _os
from comfy_kitchen.backends import cuda as _ckc
rep = {"mode": MODE, "ckpt": CKPT, "dynamic_vram": DYNVRAM,
       "force_int8_fallback_env": _os.environ.get("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK", "0"),
       "force_int8_fallback_visto": bool(_ckc._FORCE_INT4_INT8_FALLBACK)}

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

# Contar "quantizado" pelo kwarg do ComfyUI diz que a matematica NAO e BF16 dequantizada.
# Nao diz QUAL kernel rodou -- e o defeito que a AUDITORIA_2026-08-18 item 18 aponta em
# `diffusion_smoke.py`: contar o despachante nao distingue os ramos. Aqui se conta o
# `linear_dtype` que chega em `convrot_w4a4_linear`, que e o que decide o ramo.
from comfy_kitchen.tensor import convrot_w4a4 as _ckt
_orig_convrot = _ckt.convrot_w4a4_linear
def _spy_convrot(*a, **k):
    # Chama o DESPACHANTE original (que consulta o registry), nunca o backend direto: chamar
    # o backend na mao foi o que transformou "roda na CPU" num ValueError em vez de num
    # numero -- util como denuncia, inutil como medicao.
    counts[f"convrot_linear_dtype={k.get('linear_dtype', 'int4')}"] += 1
    return _orig_convrot(*a, **k)
_ckt.convrot_w4a4_linear = _spy_convrot

# E registra qual implementacao o registry escolhe de fato, que e o que separa CUDA de eager
# -- a distincao que a primeira versao deste modo nao fazia e por isso mediu o backend errado.
# Patch NO OBJETO que o tensor layer segura (`from comfy_kitchen.registry import registry`),
# nao num modulo homonimo: `comfy_kitchen.registry` e a propria instancia, e `.registry` nela
# nao existe.
_reg_obj = _ckt.registry
_orig_get = _reg_obj.get_implementation
def _spy_get(nome, *a, **k):
    impl = _orig_get(nome, *a, **k)
    counts[f"impl:{nome}={getattr(impl, '__module__', '?')}"] += 1
    return impl
_reg_obj.get_implementation = _spy_get

_orig_dq = QuantizedTensor.dequantize
def counted_dq(self, *a, **k):
    counts["dequantize"] += 1
    return _orig_dq(self, *a, **k)
QuantizedTensor.dequantize = counted_dq

try:
    if FORWARD_ONLY:
        # SOBE O MODELO PARA A GPU ANTES DE CHAMAR. Sem isto os modulos ficam no device de
        # offload (CPU), o registry do comfy-kitchen resolve para o backend EAGER, e tudo
        # roda -- com contadores dizendo "quantizado" e resultado que nao depende de
        # COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK, porque essa flag so existe no backend CUDA.
        # Foi assim que a primeira versao deste modo deu "toda Linear quantizada faz
        # matematica quantizada" para quatro checkpoints medindo o backend errado. O que
        # denunciou foi um `ValueError: Expected a cuda device, but got: cpu` ao instrumentar
        # a chamada -- ate ali o probe estava alegre.
        import comfy.model_management as _mm
        _mm.load_models_gpu([obj] if MODE != "te" else [obj.patcher])
        devs = Counter(str(getattr(getattr(m, "weight", None), "device", "?")) for m in quantized_mods)
        rep["device_dos_pesos"] = dict(devs)
        # Modo independente de arquitetura: chama DIRETO algumas Linear quantizadas do modelo
        # ja carregado, com entrada sintetica da forma certa. Nao prova que uma geracao inteira
        # dispara -- prova que ESTES modulos, como o loader os deixou, despacham quantizado.
        # Existe porque samplers nao sao intercambiaveis: o MiniMax H3 quer uma lista de
        # latentes (video e audio) e nenhum probe generico adivinha isso.
        alvos = quantized_mods[:FORWARD_N]
        dev = next(iter(alvos[0].parameters()), None)
        dev = dev.device if dev is not None else torch.device("cuda:0")
        feitos = []
        for m in alvos:
            k = getattr(m, "in_features", None)
            if k is None:
                continue
            w = getattr(m, "weight", None)
            dt = getattr(getattr(w, "_params", None), "orig_dtype", torch.bfloat16)
            g = torch.Generator(device=m.weight.device).manual_seed(1234 + k)
            x = torch.randn(FORWARD_M, k, device=m.weight.device, dtype=dt, generator=g)
            y = m(x)
            # Impressao digital deterministica: com ela, rodar de novo sob
            # COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1 diz se a camada JA estava no ramo INT8
            # (identica) ou no de 4 bits (diferente). Sem ela, "quantizado" nao distingue qual
            # dos dois kernels quantizados rodou -- que e exatamente o que separa os
            # checkpoints publicos com linear_dtype "int8" dos que omitem o campo.
            yf = y.float()
            # Soma e norma do tensor INTEIRO, nao os 4 primeiros elementos. A primeira versao
            # usava os 4 primeiros e deu identico para o Z-Image nos dois ramos -- um modelo
            # que sabidamente muda de kernel com a flag. Quatro valores em bf16 coincidem com
            # facilidade; a soma sobre milhoes nao.
            feitos.append({"in": k, "out": int(y.shape[-1]), "dtype": str(y.dtype),
                           "fp": [round(float(yf.sum()), 4), round(float(yf.norm()), 4),
                                  round(float(yf.abs().max()), 4)]})
        rep["saida"] = {"forwards_diretos": len(feitos), "exemplos": feitos[:3],
                        "fingerprints": [f["fp"] for f in feitos]}
    elif MODE == "te":
        out = obj.encode_from_tokens_scheduled(obj.tokenize(PROMPT))
        cond = out[0][0].float()
        rep["saida"] = {"shape": list(cond.shape), "norm": float(cond.norm())}
    else:
        lf = obj.model.latent_format
        # `latent_dimensions` e 2 para modelo de imagem e 3 para video, e um modelo de video
        # que recebe latente 4-D nao falha limpo: falha dentro do transformer com um erro de
        # forma que nao diz nada sobre o latente. Ler o formato, nao supor imagem -- e o que
        # `calibrate_activations.py` ja fazia e este probe nao fazia.
        dims = getattr(lf, "latent_dimensions", 2)
        side = max(SIDE // 8, 8)
        if dims == 3:
            ratio = getattr(lf, "temporal_downscale_ratio", 4)
            frames = max(1, (FRAMES - 1) // ratio + 1)
            shape = [1, lf.latent_channels, frames, side, side]
        else:
            shape = [1, lf.latent_channels, side, side]
        rep["latent_shape"] = shape
        latent = torch.zeros(shape, device="cpu")
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
    p.add_argument("--forward-only", action="store_true",
                   help="nao amostra: chama direto N Linear quantizadas do modelo carregado "
                        "com entrada sintetica. Independe do sampler, que nao e intercambiavel "
                        "entre arquiteturas.")
    p.add_argument("--forward-n", type=int, default=8, help="quantas Linear no --forward-only")
    p.add_argument("--forward-m", type=int, default=256, help="linhas da entrada sintetica")
    p.add_argument("--dynamic-vram", action="store_true",
                   help="inicializa o comfy-aimdo antes de carregar. Necessario para os dois "
                        "checkpoints publicos da Abiray, que tem bytes depois do ultimo tensor "
                        "e sao recusados pelo safetensors estrito.")
    p.add_argument("--frames", type=int, default=9,
                   help="so vale para modelo de video (latent_dimensions == 3)")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.device)
    src = SRC % {"MODE": a.mode, "CKPT": a.ckpt, "CLIP": a.clip, "CLIP_TYPE": a.clip_type,
                 "PROMPT": a.prompt, "STEPS": a.steps, "SIDE": a.size, "SEED": a.seed,
                 "FRAMES": a.frames,
                 "DYNVRAM": "True" if a.dynamic_vram else "False",
                 "FORWARD_ONLY": "True" if a.forward_only else "False",
                 "FORWARD_N": a.forward_n, "FORWARD_M": a.forward_m}
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
    print(f"{rep['ckpt']}   modo {rep['mode']}   device cuda:{a.device}"
          f"   dynamic-vram {rep.get('dynamic_vram')}")
    print(f"FORCE_INT4_INT8_FALLBACK   env={rep.get('force_int8_fallback_env')}  "
          f"visto pelo modulo={rep.get('force_int8_fallback_visto')}")
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
        saida = dict(rep["saida"])
        fps = saida.pop("fingerprints", None)
        print(f"saida                      {saida}")
        if fps:
            # Linha propria e em JSON, para poder ser extraida por grep entre execucoes: e
            # assim que se compara o mesmo checkpoint com e sem FORCE_INT4_INT8_FALLBACK.
            print("FINGERPRINTS " + json.dumps(fps))
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
    # A implementacao que o registry escolheu, e o linear_dtype que chegou no despachante.
    # Sem estas duas linhas, "quantizado" nao distingue CUDA de eager nem int4 de int8 -- e
    # foi exatamente assim que a primeira versao deste modo mediu o backend errado.
    for k in sorted(c):
        if k.startswith("impl:") or k.startswith("convrot_linear_dtype"):
            print(f"  {k:<58} {c[k]}")
    if rep.get("device_dos_pesos"):
        print(f"  device dos pesos quantizados: {rep['device_dos_pesos']}")
    print()
    if not rep.get("rodou"):
        # O forward falhou. Um veredito aqui seria lido como resultado: a primeira versao
        # imprimiu "ZERO forwards quantizados" para um MiniMax H3 que nem chegou a rodar --
        # ele quer uma LISTA de latentes (video e audio) e morreu em `audio_src = x[1]`.
        # Contador zerado por nao ter executado e contador zerado por ter executado
        # dequantizado dao o mesmo numero e significam coisas opostas.
        print("SEM VEREDITO  o forward nao completou, entao os contadores acima nao dizem")
        print("              nada sobre o dispatch. Ler o traceback.")
    elif q > 0 and nq == 0:
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
