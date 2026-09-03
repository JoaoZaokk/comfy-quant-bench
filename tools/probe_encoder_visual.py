r"""A foto que falta no card do Qwen3-4B W4A4 -- e, no mesmo run, o que custa DESTRAVAR o encoder.

POR QUE ESTA FERRAMENTA EXISTE
------------------------------
O card publicado de `JoaoZaokk/Qwen3-4B-W4A4-ConvRot` faz alegacao de fidelidade -- rel-RMSE 0,1442,
cosseno 0,98957 -- e **nao tem uma unica imagem**. Cosseno nao e foto. Um text encoder nao desenha
nada sozinho, mas ele decide o que o modelo de difusao desenha, entao a pergunta "isso se ve?" tem
resposta e ela nunca foi buscada. Apontado pelo dono em 2026-09-01.

TRES BRACOS, E O TERCEIRO E O MOTIVO DE REESCREVER ISTO
--------------------------------------------------------
A versao anterior tinha dois bracos, `bf16` e `w4a4`, e o segundo estava mal rotulado. Esta bancada
mediu em 2026-08-31 que um text encoder quantizado carregado pelo caminho normal do ComfyUI roda
**dequantizado**: `comfy_force_cast_weights` (de `comfy/sd.py:269`) e `full_precision_mm`
(fixo em `comfy/sd1_clip.py:114`) trocam a matematica por BF16 e o kernel de 4 bits nunca e
chamado. Entao aquele braco media *peso de 4 bits com matematica de 16*, que e o que o usuario
recebe hoje, e **nao e o que o card sugere**.

    bf16    7,49 GiB   precisao cheia                       a verdade
    w4a4t   2,42 GiB   peso 4 bits, matematica BF16         o que se recebe hoje
    w4a4s   2,42 GiB   kernel 4 bits de verdade             o que o card implica

Com o terceiro braco o mesmo run responde a segunda pergunta que estava em aberto e escrita como
nao coberta no `CLAUDE.md`: **soltar as travas do TE e seguro para a qualidade?** Ninguem mediu.

O EIXO E UM SO
--------------
Mesmo modelo de difusao (Z-Image v2 BF16, **nao quantizado**), mesma semente, mesmo prompt, mesmo
sampler, mesmos passos. So o encoder muda. Qualquer diferenca na imagem vem do encoder, porque nao
ha outro lugar de onde possa vir.

DOIS COMPRIMENTOS DE PROMPT, DE PROPOSITO
------------------------------------------
O proprio card mede que o SINAL do ganho de tempo vira com o numero de tokens (1,49x mais lento em
22 tokens, 3,67x mais rapido em 850; o cruzamento fica entre 75 e 199). E o erro do encoder W4A4 e
maior no prompt longo que no curto (2,55e-1 contra 1,44e-1). Medir num comprimento so responderia
metade da pergunta, e provavelmente a metade errada.

O CONDITIONING E CALCULADO UMA VEZ POR BRACO
---------------------------------------------
A versao anterior recarregava o encoder E o modelo de difusao para cada combinacao de prompt e
semente -- 15,5 GiB de disco por imagem, para amostrar 15 s. Aqui cada braco e UM processo: carrega
o encoder, codifica os dois prompts, libera o encoder, carrega a difusao uma vez e amostra todas as
combinacoes. Doze carregamentos viram tres.

NAO COBERTO
-----------
Duas sementes e dois prompts sao amostra, nao avaliacao. Sem metrica perceptual. Um modelo de
difusao, uma placa. E a comparacao e LIVRE: tres encoders levam o amostrador por trajetorias
diferentes, e esta bancada ja mediu que a imagem livre de dois formatos do mesmo modelo mede caos
tanto quanto fidelidade. Por isso a folha existe para o olho, e a distancia no pixel vai junto
apenas como ordenacao entre bracos que compartilham semente e prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "bench" / "encoder_visual"

BRACO = r'''
import json, sys, time
from collections import Counter
sys.path.insert(0, "ComfyUI")
sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd, comfy.sample
import comfy.model_management as mm
import comfy_kitchen.tensor.convrot_w4a4 as ckt
from comfy_kitchen.tensor.base import QuantizedTensor

UNET=%(UNET)r; CLIP=%(CLIP)r; DESTRAVA=%(DESTRAVA)s; ROTULO=%(ROTULO)r
PROMPTS=%(PROMPTS)s; SEEDS=%(SEEDS)s
STEPS=%(STEPS)d; CFG=%(CFG)s; SIDE=%(SIDE)d; DIR=%(DIR)r

clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.LUMINA2)

# A DESTRAVA E POR DOIS LADOS, e medir isso corrigiu a regra escrita no CLAUDE.md.
# Ela mandava soltar SO na fonte (`patcher.force_cast_weights`) e avisava que escrever no modulo
# "sobrevive so se o modelo ja estiver residente". Medido em 2026-09-03 no qwen_3_4b W4A4, um eixo
# por vez, contando chamadas de kernel:
#
#     modo            convrot   deq   force_cast
#     travado               0   252   True
#     fonte                 0   252   True     <- a prescricao do CLAUDE.md: NAO propaga
#     modulo              252     0   False    <- o que ele desaconselha: funciona
#     fonte_reload        252     0   False    <- fonte + unload_all_models() forcado
#
# `ModelPatcher.load` foi instrumentado e NAO E CHAMADO durante o encode: para este arquivo de
# 2,4 GiB o modelo ja subiu inteiro dentro do `load_clip`, entao a linha 1016 rodou uma unica vez,
# antes da escrita, com True. Escrever na fonte depois disso nao chega a modulo nenhum. O
# `fonte_reload` prova o mecanismo: forcar o descarregamento faz a linha 1016 rodar de novo e
# propagar. Entao a instabilidade que o CLAUDE.md descreve ("350 chamadas numa run e 0 na
# seguinte") nao e sobre a escrita no modulo sobreviver -- e sobre `load` rodar ou nao entre a
# escrita e o forward, o que depende do estado da VRAM. Escrever nos dois lugares e imune a ordem.
if DESTRAVA:
    clip.patcher.force_cast_weights = False
    clip.patcher.object_patches.pop("manual_cast_dtype", None)
    clip.patcher.model_options.get("transformer_options", {}).pop("manual_cast_dtype", None)

# `layout_type` mora no MODULO, nao no `.weight`. A primeira versao deste probe perguntou ao peso
# e achou lista VAZIA num arquivo com 350 camadas quantizadas -- e como a destrava do segundo
# cadeado e um `for m in quant`, ela varreu nada e o braco "solto" saiu byte a byte igual ao
# travado. O contador de despacho foi o que denunciou: 0 chamadas de kernel onde tinha de haver.
quant = [m for _, m in clip.cond_stage_model.named_modules()
         if getattr(m, "layout_type", None) is not None]
if not quant:
    # Nao segue em frente fingindo. Um braco "destravado" sobre lista vazia e um braco travado
    # com outro nome, e essa e a comparacao que nao mede nada.
    print("AVISO: nenhuma camada com layout_type em " + CLIP, file=sys.stderr)

# Contadores instalados DEPOIS do load: o proprio carregamento dequantiza de forma legitima e
# contar aquilo poluiria o numero que interessa, que e o do forward.
contagem = Counter()
_fw = ckt._convrot_w4a4_forward
def _fw_contado(*a, **k):
    contagem["convrot_w4a4"] += 1
    return _fw(*a, **k)
ckt._convrot_w4a4_forward = _fw_contado
_dq = QuantizedTensor.dequantize
def _dq_contado(self, *a, **k):
    contagem["dequantize"] += 1
    return _dq(self, *a, **k)
QuantizedTensor.dequantize = _dq_contado

def codifica(p):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = clip.encode_from_tokens_scheduled(clip.tokenize(p))
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3

codifica(PROMPTS["curto"])          # aquecimento: a primeira chamada paga a subida para a VRAM
if DESTRAVA:
    # `_full_precision_mm` vem de `sd1_clip.py:114` e NAO e reescrito por `patch_model`, mas e
    # setado aqui para ficar do mesmo lado da barreira que o outro atributo: o estado reportado
    # tem de ser o estado em que os encodes medidos rodaram.
    for m in quant:
        m._full_precision_mm = False
        m.comfy_force_cast_weights = False   # o outro lado; ver o bloco acima
codifica(PROMPTS["curto"])          # segundo aquecimento, ja com o estado final
contagem.clear()

estado = {
 "camadas_quantizadas": len(quant),
 "full_precision_mm": dict(Counter(str(getattr(m, "_full_precision_mm", None)) for m in quant)),
 "force_cast":        dict(Counter(str(bool(getattr(m, "comfy_force_cast_weights", False))) for m in quant)),
}

conds, tempos = {}, {}
for chave, texto in PROMPTS.items():
    out, ms = codifica(texto)
    conds[chave] = out
    tempos[chave] = ms
    torch.save(out[0][0].float().cpu(), DIR + "/cond_" + chave + "_" + ROTULO + ".pt")
desp = dict(contagem)

del clip
mm.soft_empty_cache()

model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", UNET))
lf = model.model.latent_format
side = SIDE // 8
vazio = [[torch.zeros_like(conds["curto"][0][0][:, :1]), {}]]   # negativo nulo: cfg 1,0 nao o usa
lat_info = {}
for chave in PROMPTS:
    for semente in SEEDS:
        latent = torch.zeros([1, lf.latent_channels, side, side], device="cpu")
        noise = comfy.sample.prepare_noise(latent, semente, None)
        t0 = time.perf_counter()
        out = comfy.sample.sample(model, noise, STEPS, CFG, "euler", "simple",
                                  [list(conds[chave][0])], vazio, latent, denoise=1.0,
                                  disable_noise=False, start_step=None, last_step=None,
                                  force_full_denoise=False, noise_mask=None, callback=None,
                                  disable_pbar=True, seed=semente)
        el = time.perf_counter() - t0
        out = out.cpu().float()
        nome = chave + "-" + ROTULO + "_s" + str(semente)
        torch.save(out, DIR + "/latent_" + nome + ".pt")
        lat_info[nome] = {"seconds": el, "std": float(out.std()), "absmax": float(out.abs().max())}

print("RESULT " + json.dumps({"rotulo": ROTULO, "clip": CLIP, "destrava": bool(DESTRAVA),
                              "estado": estado, "despacho": desp,
                              "ms_encode": tempos, "latentes": lat_info}))
'''

PROMPTS = {
    "curto": "a red apple on a weathered wooden table, soft window light",
    "longo": ("a weathered fisherman mending a net on a stone quay at dawn, thick wool sweater "
              "stiff with salt, hands cracked from cold water, coiled rope and a rusted iron "
              "cleat beside him, low mist over the harbour, a wooden boat with peeling blue "
              "paint moored behind, gulls on the far breakwater, pale gold light raking across "
              "the wet stone, shallow depth of field, photographic, fine grain"),
}


def roda(a, rotulo, clip, destrava):
    src = BRACO % {"UNET": a.unet, "CLIP": clip, "DESTRAVA": repr(bool(destrava)),
                   "ROTULO": rotulo, "PROMPTS": repr(PROMPTS), "SEEDS": repr(list(a.seeds)),
                   "STEPS": a.steps, "CFG": repr(a.cfg), "SIDE": a.size,
                   "DIR": str(SAIDA).replace("\\", "/")}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    r = subprocess.run([str(RAIZ / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(RAIZ), timeout=7200)
    for linha in r.stdout.splitlines():
        if linha.startswith("RESULT "):
            return json.loads(linha[7:])
    print(f"  BRACO {rotulo} FALHOU\n  stdout: {r.stdout.strip()[-1500:]}")
    print(f"  stderr: {r.stderr.strip()[-3000:]}")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--te-bf16", default="qwen_3_4b.safetensors")
    p.add_argument("--te-w4a4", default="qwen_3_4b_w4a4_convrot.safetensors")
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 7])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    a = p.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    bracos = [("bf16", a.te_bf16, False),
              ("w4a4t", a.te_w4a4, False),
              ("w4a4s", a.te_w4a4, True)]
    res = {}
    for rotulo, clip, destrava in bracos:
        print(f"--- {rotulo}  ({clip}{'  DESTRAVADO' if destrava else ''}) ---", flush=True)
        r = roda(a, rotulo, clip, destrava)
        if r is None:
            continue
        res[rotulo] = r
        e, d = r["estado"], r["despacho"]
        print(f"  {e['camadas_quantizadas']} camadas quantizadas   "
              f"force_cast {e['force_cast']}   fpmm {e['full_precision_mm']}")
        print(f"  despacho: {d}   encode curto {r['ms_encode']['curto']:.1f}ms "
              f"longo {r['ms_encode']['longo']:.1f}ms")

    import torch
    # PRIMEIRO o controle de PRESENCA. Sem ele, um probe cujos tres bracos morrem imprime
    # "PASSOU", que foi exatamente o que aconteceu na primeira execucao desta ferramenta.
    faltando = [r for r in ("bf16", "w4a4t", "w4a4s") if r not in res]
    controle_presenca = not faltando
    if faltando:
        print()
        print(f"CONTROLE DE PRESENCA FALHOU: bracos ausentes {faltando}")

    # O CONTROLE PAREADO: os tres conditionings tem de diferir DOIS A DOIS. Se `bf16` e
    # `w4a4t` coincidissem, o arquivo quantizado nao estaria em uso; se `w4a4t` e `w4a4s`
    # coincidissem, a destrava nao teria pegado -- e este segundo caso e o que ja aconteceu nesta
    # bancada, silenciosamente, por escrever o atributo no modulo em vez de na fonte.
    print()
    print(f"{'par':>26s} {'rel-RMSE':>10s} {'cosseno':>10s}")
    print("-" * 50)
    controle_ok = controle_presenca
    n_comparacoes = 0
    pares = [("bf16", "w4a4t"), ("bf16", "w4a4s"), ("w4a4t", "w4a4s")]
    conds = {}
    for chave in PROMPTS:
        for x, y in pares:
            f1, f2 = SAIDA / f"cond_{chave}_{x}.pt", SAIDA / f"cond_{chave}_{y}.pt"
            if not (f1.exists() and f2.exists()):
                continue
            u, v = torch.load(f1).float(), torch.load(f2).float()
            rel = ((v - u).norm() / u.norm()).item()
            cos = torch.nn.functional.cosine_similarity(u.reshape(1, -1), v.reshape(1, -1)).item()
            conds[f"{chave}:{x}-{y}"] = {"rel_rmse": rel, "cosseno": cos}
            n_comparacoes += 1
            marca = ""
            if rel <= 1e-6:
                marca = "   CONTROLE FALHOU: identicos"
                controle_ok = False
            print(f"{chave + ' ' + x + '-' + y:>26s} {rel:10.4f} {cos:10.5f}{marca}")

    esperadas = len(PROMPTS) * len(pares)
    if n_comparacoes != esperadas:
        print(f"CONTROLE DE COBERTURA FALHOU: {n_comparacoes} comparacoes de {esperadas}")
        controle_ok = False

    # O terceiro controle e o contador de despacho, e ele e o que ja pegou um erro real: um probe
    # que so comparasse saidas teria reportado "soltar as travas nao muda nada" e seria acreditado.
    print()
    for rotulo in ("bf16", "w4a4t", "w4a4s"):
        if rotulo not in res:
            continue
        d = res[rotulo]["despacho"]
        k = d.get("convrot_w4a4", 0)
        esperado = {"bf16": "0 (nao ha peso quantizado)", "w4a4t": "0 (as travas do TE)",
                    "w4a4s": ">0 (o kernel de 4 bits)"}[rotulo]
        estado = "OK" if ((k > 0) == (rotulo == "w4a4s")) else "INESPERADO"
        if estado != "OK":
            controle_ok = False
        print(f"{rotulo:>8s}  convrot_w4a4={k:5d}  dequantize={d.get('dequantize', 0):5d}   "
              f"esperado {esperado:28s} {estado}")

    (SAIDA / "resumo.json").write_text(
        json.dumps({"bracos": res, "conditioning": conds, "controle_ok": controle_ok}, indent=2),
        encoding="utf-8")
    print(f"\ncontrole geral: {'PASSOU' if controle_ok else 'FALHOU -- nao leia a folha'}")
    print(f"latentes em {SAIDA}. Decodifique com tools/decode_encoder_visual.py")
    print("\nNAO COBERTO: duas sementes e dois prompts sao amostra, nao avaliacao. Sem metrica")
    print("  perceptual. Um modelo de difusao, uma placa. E a comparacao e LIVRE -- tres encoders")
    print("  levam o amostrador por trajetorias diferentes, entao a distancia no pixel mede caos")
    print("  tanto quanto fidelidade. A folha existe para o olho; o numero so ordena.")
    return 0 if controle_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
