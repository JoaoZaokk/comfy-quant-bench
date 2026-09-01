r"""A foto que falta no card do Qwen3-4B W4A4: o que 0,1442 de rel-RMSE no conditioning faz na imagem.

POR QUE ESTA FERRAMENTA EXISTE
------------------------------
O card publicado de `JoaoZaokk/Qwen3-4B-W4A4-ConvRot` faz alegacao de fidelidade -- rel-RMSE 0,1442,
cosseno 0,98957 -- e **nao tem uma unica imagem**. Cosseno nao e foto. Um text encoder nao desenha
nada sozinho, mas ele decide o que o modelo de difusao desenha, entao a pergunta "isso se ve?" tem
resposta e ela nunca foi buscada.

Apontado pelo dono em 2026-09-01, junto com um defeito pior no card do Wan.

O DESENHO: UM EIXO, E SO UM
----------------------------
Mesmo modelo de difusao (Z-Image v2 BF16, **nao quantizado**), mesma semente, mesmo prompt, mesmo
sampler, mesmos passos. **So o arquivo do encoder muda.** Qualquer diferenca na imagem e do encoder,
porque nao ha outro lugar de onde ela possa vir.

    te_bf16   qwen_3_4b.safetensors               7,49 GiB, o original
    te_w4a4   qwen_3_4b_w4a4_convrot.safetensors  2,42 GiB, 3,09x mais leve

DOIS COMPRIMENTOS DE PROMPT, DE PROPOSITO
------------------------------------------
O proprio card mede que o SINAL do ganho de tempo vira com o numero de tokens (1,49x mais lento em
22 tokens, 3,67x mais rapido em 850). Se o tempo depende do comprimento, a fidelidade tambem pode
depender -- e esta bancada ja mediu que o erro de um encoder W4A4 e maior no prompt longo que no
curto (2,55e-1 contra 1,44e-1). Medir num comprimento so responderia metade.

NAO COBERTO
-----------
Duas sementes e dois prompts sao uma amostra, nao uma avaliacao. Sem metrica perceptual. Um modelo
de difusao. E a comparacao e livre: dois encoders levam o amostrador por trajetorias diferentes, e
esta bancada ja mediu que a imagem livre de dois formatos mede caos tanto quanto fidelidade -- por
isso a folha existe para o olho, e a distancia no pixel vai junto apenas como ordenacao.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "bench" / "encoder_visual"

BRACO = r'''
import json, sys, time
sys.path.insert(0, "ComfyUI")
sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, folder_paths, comfy.sd, comfy.sample

UNET=%(UNET)r; CLIP=%(CLIP)r; PROMPT=%(PROMPT)r; NEG=%(NEG)r
SEED=%(SEED)d; STEPS=%(STEPS)d; CFG=%(CFG)s; SIDE=%(SIDE)d; OUTPT=%(OUTPT)r

clip = comfy.sd.load_clip(
    ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", CLIP)],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.LUMINA2)
toks = clip.tokenize(PROMPT)
cond = clip.encode_from_tokens_scheduled(toks)
positive = [list(cond[0])]
negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(NEG))[0])]
# O conditioning cru vai no relatorio: se dois bracos produzissem o MESMO tensor, a imagem tambem
# seria a mesma e a folha nao mediria nada. E o controle de que o eixo variou.
c = positive[0][0]
info_cond = {"shape": list(c.shape), "mean": float(c.mean()), "std": float(c.std()),
             "absmax": float(c.abs().max())}
torch.save(c.cpu(), OUTPT.replace("latent_", "cond_"))
del clip
import comfy.model_management as mm; mm.soft_empty_cache()

model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", UNET))
lf = model.model.latent_format
side = SIDE // 8
latent = torch.zeros([1, lf.latent_channels, side, side], device="cpu")
noise = comfy.sample.prepare_noise(latent, SEED, None)
t0 = time.perf_counter()
out = comfy.sample.sample(model, noise, STEPS, CFG, "euler", "simple",
                          positive, negative, latent, denoise=1.0, disable_noise=False,
                          start_step=None, last_step=None, force_full_denoise=False,
                          noise_mask=None, callback=None, disable_pbar=True, seed=SEED)
el = time.perf_counter() - t0
out = out.cpu().float()
torch.save(out, OUTPT)
print("RESULT " + json.dumps({
    "clip": CLIP, "seconds": el, "cond": info_cond,
    "mean": float(out.mean()), "std": float(out.std()), "absmax": float(out.abs().max()),
}))
'''

PROMPTS = {
    "curto": "a red apple on a weathered wooden table, soft window light",
    "longo": ("a weathered fisherman mending a net on a stone quay at dawn, thick wool sweater "
              "stiff with salt, hands cracked from cold water, coiled rope and a rusted iron "
              "cleat beside him, low mist over the harbour, a wooden boat with peeling blue "
              "paint moored behind, gulls on the far breakwater, pale gold light raking across "
              "the wet stone, shallow depth of field, photographic, fine grain"),
}


def roda(a, clip, nome, prompt, semente):
    saida = SAIDA / f"latent_{nome}.pt"
    src = BRACO % {"UNET": a.unet, "CLIP": clip, "PROMPT": prompt, "NEG": a.negative,
                   "SEED": semente, "STEPS": a.steps, "CFG": repr(a.cfg), "SIDE": a.size,
                   "OUTPT": str(saida)}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    r = subprocess.run([str(RAIZ / "python_embeded" / "python.exe"), "-s", "-c", src],
                       capture_output=True, text=True, env=env, cwd=str(RAIZ), timeout=5400)
    for linha in r.stdout.splitlines():
        if linha.startswith("RESULT "):
            return json.loads(linha[7:])
    print(f"  BRACO {nome} FALHOU\n  stdout: {r.stdout.strip()[-800:]}")
    print(f"  stderr: {r.stderr.strip()[-2500:]}")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet", default="beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--te-bf16", default="qwen_3_4b.safetensors")
    p.add_argument("--te-w4a4", default="qwen_3_4b_w4a4_convrot.safetensors")
    p.add_argument("--negative", default="")
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 7])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--size", type=int, default=1024)
    a = p.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    res = {}
    for chave, prompt in PROMPTS.items():
        for semente in a.seeds:
            for rot, clip in (("bf16", a.te_bf16), ("w4a4", a.te_w4a4)):
                nome = f"{chave}_s{semente}_{rot}"
                print(f"--- {nome}  ({clip}) ---", flush=True)
                r = roda(a, clip, nome, prompt, semente)
                if r is None:
                    continue
                res[nome] = r
                c = r["cond"]
                print(f"  {r['seconds']:.1f}s   cond {c['shape']} std {c['std']:.4f}   "
                      f"|latente| std {r['std']:.4f}")

    # O CONTROLE: os conditionings dos dois bracos TEM de diferir. Se fossem iguais, o arquivo
    # quantizado nao estaria sendo usado e a folha inteira compararia o encoder consigo mesmo.
    import torch
    print()
    print(f"{'par':>22s} {'rel-RMSE do conditioning':>26s} {'cosseno':>9s}")
    print("-" * 62)
    for chave in PROMPTS:
        for semente in a.seeds:
            f1 = SAIDA / f"cond_{chave}_s{semente}_bf16.pt"
            f2 = SAIDA / f"cond_{chave}_s{semente}_w4a4.pt"
            if not (f1.exists() and f2.exists()):
                continue
            x, y = torch.load(f1).float(), torch.load(f2).float()
            rel = ((y - x).norm() / x.norm()).item()
            cos = torch.nn.functional.cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)).item()
            marca = "" if rel > 1e-6 else "   CONTROLE FALHOU: identicos"
            print(f"{chave + ' s' + str(semente):>22s} {rel:26.4f} {cos:9.5f}{marca}")

    (SAIDA / "resumo.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nlatentes em {SAIDA}. Decodifique com tools/decode_esparso_visual.py apontado para ca.")
    print("\nNAO COBERTO: duas sementes e dois prompts sao amostra, nao avaliacao. Sem metrica")
    print("  perceptual. Um modelo de difusao. E a comparacao e LIVRE -- dois encoders levam o")
    print("  amostrador por trajetorias diferentes, entao a distancia no pixel mede caos tanto")
    print("  quanto fidelidade. A folha existe para o olho; o numero so ordena.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
