"""Quanto custa, e quanto rende, destravar a matematica quantizada num text encoder.

DE ONDE VEM. Medido em 2026-08-31: o ComfyUI liga `comfy_force_cast_weights` em TODO objeto
CLIP (`comfy/sd.py:269`, `set_model_compute_dtype(torch.float32)`), e `sd1_clip.py:114` fixa
`full_precision_mm=True` por cima. Com as duas travas, um text encoder quantizado guarda o
peso em 4 ou 8 bits na VRAM e faz a conta em BF16 dequantizado: **zero** chamadas ao kernel.
Soltando as duas, o kernel dispara: 311,6 -> 117,7 ms num encoder de 13,2 GiB.

COMO SE DESTRAVA, e a armadilha. `comfy_force_cast_weights` nao pode ser escrito no modulo:
`model_patcher.py:1016` reescreve o atributo em cada modulo toda vez que o modelo sobe para a
GPU. Escrever nele funciona so se o modelo ja estava residente, o que depende do estado da
VRAM -- a mesma linha de comando deu 350 chamadas ao kernel numa execucao e 0 na seguinte.
Aqui se desliga na FONTE, `clip.patcher.force_cast_weights = False`, e o proprio `patch_model`
propaga -- SO QUE ISSO SO VALE SE `load` RODAR DE NOVO DEPOIS DA ESCRITA. Medido em 2026-09-03:
num arquivo pequeno (qwen_3_4b, 2,4 GiB) o modelo sobe inteiro ja dentro do `load_clip`,
`ModelPatcher.load` nao e chamado outra vez, e soltar na fonte nao chega a modulo nenhum -- 0
chamadas de kernel, e este probe disparou o proprio aviso. Escrever no modulo funciona nesse caso;
forcar `unload_all_models()` tambem. Por isso agora se escreve nos DOIS lugares.
`_full_precision_mm` continua sendo escrito no modulo, mas DEPOIS do primeiro load,
para ficar do mesmo lado da barreira -- e o estado reportado tambem e lido depois, senao o
relatorio descreve uma execucao diferente da que foi medida.

O que faltava era a outra metade: **quanto isso muda a saida.** Aquele modelo nao tem BF16
nesta bancada, entao a comparacao possivel era contra o proprio braco dequantizado. Este
arquivo usa um caso em que o BF16 existe -- `qwen_3_4b.safetensors`, que e o encoder do
proprio Z-Image desta bancada, quantizado aqui com `tools/quant_w4a4.py --profile qwen`.

TRES BRACOS, um eixo de cada vez (dois com `--sem-bf16`, quando a referencia nao existe):

  A  bf16              o arquivo original, sem quantizacao nenhuma. A referencia.
  B  quant_travado     o quantizado como o ComfyUI carrega hoje: peso 4 bits, conta BF16.
  C  quant_destravado  o MESMO arquivo, as duas travas soltas: conta de 4 bits.

  B contra A  -> o que o PESO de 4 bits custa, sem tocar na politica do ComfyUI
  C contra B  -> o que soltar a trava adiciona (a quantizacao da ATIVACAO)
  C contra A  -> o custo total de quem destravar

Sem os tres, "destravar custa X" e ambiguo: parte do erro ja estava la por causa do peso.

E O EIXO QUE MAIS IMPORTA E O PROMPT. Medido em 2026-08-31 neste mesmo arquivo, 3080 Ti: o
cruzamento fica entre 75 e 199 tokens -- abaixo dele destravar PERDE (1,49x mais lento com 22
tokens), acima ganha ate 3,67x. Um numero de "quanto rende destravar" sem dizer o comprimento
do prompt nao significa nada, e por isso este probe cronometra CADA prompt e imprime a forma
da saida junto.

NAO COBERTO: um modelo, os prompts que forem passados, uma placa sm86, sem metrica
perceptual e sem gerar imagem -- mede o CONDICIONAMENTO, nao o resultado. Nao diz se a
imagem final muda de forma perceptivel; a bancada ja mediu que imagem livre nao discrimina
quantizacoes. E as travas sao soltas por monkeypatch pos-load, que nao e um caminho que o
ComfyUI ofereca a um usuario.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "a red apple on a weathered wooden table, soft window light",
    "portrait of an elderly watchmaker, 85mm lens, shallow depth of field",
    "wide shot of a rain-soaked neon alley at night, reflections on asphalt",
]

SRC = r'''
import json, sys, time, traceback
from collections import Counter
sys.path.insert(0, "ComfyUI"); sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor import convrot_w4a4 as ckt
from comfy_kitchen.tensor import w4a8_int8 as ckw8

CKPT = %(CKPT)r; CLIP_TYPE = %(CLIP_TYPE)r; DESTRAVA = %(DESTRAVA)s
PROMPTS = json.loads(%(PROMPTS)r); REPEATS = %(REPEATS)d
OUTDIR = %(OUTDIR)r; ROTULO = %(ROTULO)r
rep = {"ckpt": CKPT, "destrava": DESTRAVA}

clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CKPT)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=getattr(comfy.sd.CLIPType, CLIP_TYPE))
root = clip.cond_stage_model

quant = [m for _, m in root.named_modules() if getattr(m, "layout_type", None) is not None]
rep["n_quantizadas"] = len(quant)

# COMO SE DESTRAVA, e por que a forma obvia nao funciona de maneira confiavel.
#
# `model_patcher.py:1016` faz `m.comfy_force_cast_weights = self.force_cast_weights` para cada
# modulo TODA vez que o modelo sobe para a GPU. Escrever o atributo no modulo antes disso e
# apagado no primeiro encode. E o pior: se o modelo ja tiver subido durante o `load_clip` --
# o que depende do estado da VRAM naquele instante -- a escrita SOBREVIVE. Ou seja, a versao
# anterior deste probe funcionava ou nao conforme o ComfyUI resolvesse carregar cedo ou tarde,
# e as duas primeiras medicoes que ele produziu vieram de execucoes em que pegou por acaso.
#
# A fonte e `patcher.force_cast_weights`, ligado por `set_model_compute_dtype(torch.float32)`
# em `comfy/sd.py:269`. Desligar la faz o proprio `patch_model` propagar False. `manual_cast_dtype`
# tambem sai, senao o upcast para float32 continua acontecendo por outro caminho.
if DESTRAVA:
    clip.patcher.force_cast_weights = False
    clip.patcher.object_patches.pop("manual_cast_dtype", None)
    clip.patcher.model_options.get("transformer_options", {}).pop("manual_cast_dtype", None)

# conta o dispatch, instrumentado DEPOIS do load (o load dequantiza legitimamente)
counts = Counter()
_fw = ckt._convrot_w4a4_forward
def counted_fw(*a, **k):
    counts["convrot"] += 1
    return _fw(*a, **k)
ckt._convrot_w4a4_forward = counted_fw
# Contar so o convrot cegaria este probe para um encoder asym_w4a8_int8 -- que e justamente
# o formato do Gemma deste projeto. Um contador que nao enxerga o formato medido devolveria
# "zero chamadas quantizadas" e pareceria uma conclusao.
_fw8 = ckw8._w4a8_int8_forward
def counted_fw8(*a, **k):
    counts["w4a8"] += 1
    return _fw8(*a, **k)
ckw8._w4a8_int8_forward = counted_fw8
_dq = QuantizedTensor.dequantize
def counted_dq(self, *a, **k):
    counts["dequantize"] += 1
    return _dq(self, *a, **k)
QuantizedTensor.dequantize = counted_dq

def encode(p):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = clip.encode_from_tokens_scheduled(clip.tokenize(p))
    torch.cuda.synchronize()
    return out[0][0].float().cpu(), (time.perf_counter() - t0) * 1e3

try:
    encode(PROMPTS[0])  # aquecimento: a primeira chamada paga a subida do modelo para a VRAM
    # `_full_precision_mm` vem de `sd1_clip.py:114` e NAO e reescrito por `patch_model`, mas e
    # setado apos o load para ficar do mesmo lado da barreira que o outro atributo: o estado
    # que este probe reporta tem de ser o estado em que os encodes medidos rodaram.
    if DESTRAVA:
        for m in quant:
            m._full_precision_mm = False
            # Ver o bloco "A DESTRAVA E POR DOIS LADOS" no cabecalho: soltar so na fonte deixou
            # este proprio probe com 0 chamadas de kernel no qwen_3_4b, e ele disparou o proprio
            # aviso "o braco DESTRAVADO nao chamou o caminho de 4 bits".
            m.comfy_force_cast_weights = False
    # Lido AGORA, depois do primeiro carregamento na GPU. A versao anterior lia antes e
    # reportava um estado que `patch_model` ainda ia sobrescrever -- um relatorio que descrevia
    # uma execucao diferente da que foi medida.
    rep["full_precision_mm"] = dict(Counter(str(getattr(m, "_full_precision_mm", None)) for m in quant))
    rep["force_cast"] = dict(Counter(str(bool(getattr(m, "comfy_force_cast_weights", False))) for m in quant))
    encode(PROMPTS[0])  # segundo aquecimento, ja com o estado final
    counts.clear()
    saidas = []
    for i, p in enumerate(PROMPTS):
        cond, _ = encode(p)
        # Gravado em disco, NAO serializado em JSON. A primeira versao devolvia
        # `cond.reshape(-1).tolist()`, o que num Gemma -- condicionamento [1, 49, 1024, 3840],
        # 192 milhoes de floats -- levou um processo a 32 GiB de RAM antes de ser morto. O
        # tamanho da saida de um text encoder depende do modelo, entao o probe nao pode supor
        # que cabe numa lista Python.
        alvo = f"{OUTDIR}/{ROTULO}_{i}.pt"
        torch.save(cond, alvo)
        # Tempo POR PROMPT, nao so do primeiro: e o que transforma "curto perde, longo ganha"
        # em uma curva. A forma da saida vai junto porque e ela que diz quantos tokens sao --
        # "prompt longo" em caracteres nao e uma medida.
        ts_p = sorted(round(encode(p)[1], 1) for _ in range(REPEATS))
        saidas.append({"prompt": p, "shape": list(cond.shape),
                       "norm": float(cond.norm()), "arquivo": alvo,
                       "ms_mediana": ts_p[len(ts_p) // 2], "ms_todas": ts_p})
    rep["counts_apos_aquecimento"] = dict(counts)
    ts = []
    for _ in range(REPEATS):
        _, ms = encode(PROMPTS[0])
        ts.append(round(ms, 1))
    rep["ms_mediana"] = sorted(ts)[len(ts) // 2]
    rep["ms_todas"] = ts
    rep["saidas"] = saidas
    rep["ok"] = True
except Exception:
    rep["ok"] = False
    rep["traceback"] = traceback.format_exc()[-2500:]

print("@@JSON@@" + json.dumps(rep))
'''


def run(ckpt, clip_type, destrava, prompts, repeats, device, outdir, rotulo, force_int8=False):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    # A env var e definida por este probe, nunca herdada: herdar faria o resultado depender do
    # ambiente de quem chamou, que e como uma medicao vira loteria.
    env["COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK"] = "1" if force_int8 else "0"
    src = SRC % {"CKPT": ckpt, "CLIP_TYPE": clip_type,
                 "DESTRAVA": "True" if destrava else "False",
                 "PROMPTS": json.dumps(prompts), "REPEATS": repeats,
                 "OUTDIR": outdir.replace("\\", "/"), "ROTULO": rotulo}
    proc = subprocess.run([str(ROOT / "python_embeded/python.exe"), "-s", "-c", src],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    print(proc.stdout[-2500:])
    print(proc.stderr[-2500:], file=sys.stderr)
    raise SystemExit(f"braco {ckpt} destrava={destrava} nao devolveu JSON (rc={proc.returncode})")


import torch  # noqa: E402  -- so para ler os tensores gravados pelos bracos


def carregar(caminho):
    return torch.load(caminho, map_location="cpu", weights_only=True).float().reshape(-1)


def rel_rmse(a, b):
    d = a - b
    den = a.norm()
    return float(d.norm() / den) if float(den) else float("nan")


def cosseno(a, b):
    da, db = a.norm(), b.norm()
    if not (float(da) and float(db)):
        return float("nan")
    return float((a @ b) / (da * db))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16", default="qwen_3_4b.safetensors",
                   help="referencia sem quantizacao. Passe --sem-bf16 quando ela nao existir "
                        "mais no disco -- e o caso do Gemma deste projeto, cujo fonte de "
                        "23,5 GiB foi apagado.")
    p.add_argument("--sem-bf16", action="store_true", dest="sem_bf16",
                   help="so mede C contra B: o que soltar as travas ADICIONA, e o tempo. Nao "
                        "diz o custo total contra o modelo original, porque sem referencia "
                        "esse numero nao existe -- e inventar um zero no lugar dele seria pior "
                        "que nao ter.")
    p.add_argument("--quant", default="qwen_3_4b_w4a4_convrot.safetensors")
    p.add_argument("--clip-type", default="LUMINA2")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--force-int8", action="store_true", dest="force_int8",
                   help="roda todos os bracos com COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK=1. "
                        "Rodar duas vezes, com e sem, e o que diz se o braco destravado toma o "
                        "ramo nativo int4 ou ja estava no INT8.")
    p.add_argument("--out-dir", default=str(ROOT / "bench" / "te_lock_cost"),
                   help="onde os condicionamentos sao gravados. Sao grandes: o Gemma escreve "
                        "770 MiB por prompt e por braco. Limpo no inicio de cada execucao.")
    p.add_argument("--manter", action="store_true",
                   help="nao apaga os .pt ao final")
    a = p.parse_args()
    prompts = a.prompt or PROMPTS
    outdir = Path(a.out_dir)
    if outdir.exists():
        for velho in outdir.glob("*.pt"):
            velho.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"bf16   {a.bf16}")
    print(f"quant  {a.quant}")
    print(f"{len(prompts)} prompts, mediana de {a.repeats} para o tempo, cuda:{a.device}")
    print(f"FORCE_INT4_INT8_FALLBACK = {'1' if a.force_int8 else '0'}")
    print("=" * 78)

    plano = [("quant_travado", a.quant, False), ("quant_destravado", a.quant, True)]
    if not a.sem_bf16:
        plano.insert(0, ("bf16", a.bf16, False))
    arms = {}
    for rotulo, ckpt, destrava in plano:
        print(f"\n--- {rotulo} ---", flush=True)
        r = run(ckpt, a.clip_type, destrava, prompts, a.repeats, a.device,
                str(outdir), rotulo, a.force_int8)
        arms[rotulo] = r
        if not r.get("ok"):
            print(r.get("traceback"))
            return 1
        print(f"  Linear quantizadas {r['n_quantizadas']}  "
              f"full_precision_mm {r.get('full_precision_mm')}  "
              f"force_cast {r.get('force_cast')}   (lido APOS o primeiro load na GPU)")
        print(f"  dispatch apos aquecimento {r['counts_apos_aquecimento']}")
        print(f"  encode {r['ms_mediana']:.1f} ms  {r['ms_todas']}")

    tra, des = arms["quant_travado"], arms["quant_destravado"]
    def quantizados(r):
        c = r["counts_apos_aquecimento"]
        return c.get("convrot", 0) + c.get("w4a8", 0)
    if quantizados(tra):
        print("\nAVISO: o braco TRAVADO chamou o caminho de 4 bits. As travas nao estao onde")
        print("       este probe supoe, e as comparacoes abaixo nao significam o que dizem.")
    if not quantizados(des):
        print("\nAVISO: o braco DESTRAVADO nao chamou o caminho de 4 bits. Soltar os dois")
        print("       atributos nao bastou; nao ha nada para comparar.")

    print()
    print("-" * 78)
    print("erro do condicionamento, por prompt (rel-RMSE contra o bf16)")
    print("-" * 78)
    tem_ref = "bf16" in arms
    if tem_ref:
        print(f"{'prompt':<44}{'B travado':>12}{'C destrav':>12}{'C-vs-B':>12}")
    else:
        print("sem referencia bf16: so C contra B, que e o que soltar as travas ADICIONA")
        print(f"{'prompt':<44}{'C-vs-B':>12}")
    acc = {"B": [], "C": [], "CB": []}
    for i, pr in enumerate(prompts):
        b = carregar(tra["saidas"][i]["arquivo"])
        c = carregar(des["saidas"][i]["arquivo"])
        ref = carregar(arms["bf16"]["saidas"][i]["arquivo"]) if tem_ref else None
        tamanhos = [b.numel(), c.numel()] + ([ref.numel()] if tem_ref else [])
        if len(set(tamanhos)) != 1:
            print(f"{pr[:42]:<44}  formas diferentes: {tamanhos} -- pulado")
            continue
        ecb = rel_rmse(b, c)
        acc["CB"].append(ecb)
        if tem_ref:
            eb, ec = rel_rmse(ref, b), rel_rmse(ref, c)
            acc["B"].append(eb); acc["C"].append(ec)
            print(f"{pr[:42]:<44}{eb:>12.4e}{ec:>12.4e}{ecb:>12.4e}")
        else:
            print(f"{pr[:42]:<44}{ecb:>12.4e}")
    if acc["CB"]:
        m = {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}
        print()
        print("LEITURA")
        if tem_ref:
            print(f"{'media':<44}{m['B']:>12.4e}{m['C']:>12.4e}{m['CB']:>12.4e}")
            r0 = carregar(arms['bf16']['saidas'][0]['arquivo'])
            print(f"  cosseno com o bf16, prompt 0:  travado "
                  f"{cosseno(r0, carregar(tra['saidas'][0]['arquivo'])):.6f}"
                  f"   destravado "
                  f"{cosseno(r0, carregar(des['saidas'][0]['arquivo'])):.6f}")
            print(f"  o PESO quantizado ja custa           {m['B']:.4e}  (B contra A)")
            print(f"  destravar ADICIONA                   {m['CB']:.4e}  (C contra B)")
            print(f"  custo total de quem destrava         {m['C']:.4e}  (C contra A)")
            if m["B"]:
                print(f"  destravar multiplica o erro por      {m['C'] / m['B']:.2f}x")
        else:
            print(f"{'media':<44}{m['CB']:>12.4e}")
            print(f"  cosseno travado x destravado, prompt 0: "
                  f"{cosseno(carregar(tra['saidas'][0]['arquivo']), carregar(des['saidas'][0]['arquivo'])):.6f}")
            print(f"  destravar ADICIONA                   {m['CB']:.4e}  (C contra B)")
            print("  SEM referencia: o custo total contra o modelo original nao e medido aqui.")

    print()
    print("-" * 78)
    print("tempo POR PROMPT (mediana), e a forma da saida que diz quantos tokens sao")
    print("-" * 78)
    print(f"{'tokens':>8} {'forma':>22}" + "".join(f"{r:>14}" for r in arms) + "   destravar")
    for i in range(len(prompts)):
        sh = tra["saidas"][i]["shape"]
        tok = sh[-2] if len(sh) >= 2 else "?"
        linha = f"{tok:>8} {str(sh):>22}"
        for r in arms:
            linha += f"{arms[r]['saidas'][i].get('ms_mediana', float('nan')):>14.1f}"
        t, d = tra["saidas"][i].get("ms_mediana"), des["saidas"][i].get("ms_mediana")
        if t and d:
            linha += (f"   {t / d:.2f}x mais rapido" if d <= t
                      else f"   {d / t:.2f}x MAIS LENTO")
        print(linha)

    print()
    print("-" * 78)
    print("tempo do encode no prompt 0, mediana (razao sempre >= 1, com a direcao dita)")
    print("-" * 78)
    for rotulo in [r for r in ("bf16", "quant_travado", "quant_destravado") if r in arms]:
        print(f"  {rotulo:<20}{arms[rotulo]['ms_mediana']:>9.1f} ms   {arms[rotulo]['ms_todas']}")
    t, d = tra["ms_mediana"], des["ms_mediana"]
    if t and d:
        if d <= t:
            print(f"  -> destravar e {t / d:.2f}x mais rapido que o travado")
        else:
            print(f"  -> destravar e {d / t:.2f}x MAIS LENTO que o travado")

    print()
    if not a.manter:
        for velho in outdir.glob("*.pt"):
            velho.unlink()
    else:
        print()
        print(f"condicionamentos mantidos em {outdir}")

    print("NAO COBERTO: um modelo, uma placa sm86, sem metrica perceptual e sem gerar imagem --")
    print("  isto mede o CONDICIONAMENTO, nao o resultado. As travas sao soltas por monkeypatch")
    print("  pos-load, que nao e um caminho que o ComfyUI ofereca a um usuario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
