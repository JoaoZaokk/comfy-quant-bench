"""Braco 1 da fila: congela o corpo ternario e ajusta SO o conjunto denso. (EXECUTADO, GPU)

A hipotese, escrita em `bench/criterio_fila_gpu_compensacao_2026-09-22.md` antes de qualquer numero:
o que o Bonsai compra pode nao exigir retreinar o corpo, e sim deixar poucas camadas em alta precisao
e **ajusta-las contra a saida do modelo denso**. No klein-4B o conjunto denso e 69 tensores,
195.042.816 parametros, 5,03% do modelo.

Ja medido antes de rodar isto, e e o que da sentido ao experimento:

    espaco de PESO    o braco 0 esta MAIS PERTO do original que o Bonsai  (0,46626 contra 0,53474)
    espaco de SAIDA   o braco 0 esta 3,5620x PIOR, perdendo 8 de 8 passos

O professor e o proprio BF16: grava-se `(x, t, c)` e a saida dele, e o aluno ve exatamente as mesmas
entradas. Sem isso a comparacao mediria caos de trajetoria, o que esta bancada ja registrou.

OS DOIS CONTROLES SAO OBRIGATORIOS e rodam com `--controles`:

  zero      o mesmo arnes com ZERO passos de otimizacao tem de produzir arquivo **byte a byte
            identico** ao braco 0. Se sair diferente, o arnes toca peso que nao devia e nenhum
            numero vale.
  ruido     perturbar o denso com ruido do MESMO tamanho do ajuste, em vez de ajustar, tem de
            **piorar**. Se melhorar, o que se mede e regularizacao por acidente.

NAO COBRE: nenhuma imagem. Isto otimiza e mede a PREVISAO do modelo em entradas casadas, que e o
instrumento validado nesta bancada -- imagem livre mede caos e divergencia de latente e enviesada
para falha macia. Nao mede velocidade: o corpo ternario roda desempacotado em bf16, sem kernel de
1,58 bit, entao nenhum tempo daqui vale como ganho de inferencia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ComfyUI"))
sys.path.insert(0, str(ROOT / "tools"))

# O `comfy.options` le `sys.argv` no import e nao entende as MINHAS flags, entao ele ve um argv
# limpo. A primeira versao disto nao guardava o argv de volta e o script destruia os proprios
# argumentos: o argparse abaixo recebia `["main.py"]` e recusava tudo como faltando.
ARGV_REAL = list(sys.argv)
sys.argv = ["main.py"]

import comfy.options

comfy.options.enable_args_parsing()
# `sys.argv` fica limpo para SEMPRE: o `comfy.cli_args` e parseado tarde, dentro de
# `model_management`, e rejeita flag que nao seja dele. Devolver o argv aqui fez exatamente isso na
# segunda tentativa. Quem le os MEUS argumentos e o `parse_args(ARGV_REAL[1:])` la embaixo.

import os.path as _op

import torch
import utils.extra_config

_y = _op.join("ComfyUI", "extra_model_paths.yaml")
if _op.isfile(_y):
    utils.extra_config.load_extra_path_config(_y)
else:
    print("AVISO: extra_model_paths.yaml ausente; roots montados invisiveis", file=sys.stderr)

import comfy.model_management as mm
import comfy.sample
import comfy.sd
import folder_paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowbit_canon import BLOCO, pilhas_reais  # fonte canonica

PROMPTS = [
    "a red apple on a weathered wooden table, soft window light",
    "a portrait of an elderly fisherman, deep wrinkles, overcast light",
    "a neon-lit night market in the rain, puddles reflecting signs",
    "a single ice crystal on dark slate, macro, fine internal structure",
]


def nomes_densos(sd_chaves) -> list[str]:
    """O conjunto denso: 2-D fora de qualquer pilha de blocos, mais tudo 1-D. Regra MEDIDA."""
    pil = pilhas_reais(sd_chaves)
    fora = []
    for k in sd_chaves:
        m = BLOCO.match(k)
        if m is None or m.group("pilha") not in pil:
            fora.append(k)
    return sorted(fora)


def liga_checkpointing(nucleo) -> int:
    """Recomputa as ativacoes dos blocos no backward, em vez de guardar todas.

    Sem isto o backward estoura: MEDIDO, `OutOfMemory` tentando alocar 4,50 GiB com 19,74 GiB ja
    alocados numa placa de 24. O gradiente precisa atravessar os 25 blocos congelados para chegar aos
    densos da ENTRADA (`x_embedder`, `context_embedder`, `time_*`), entao congelar peso nao reduz
    ativacao guardada -- sao coisas diferentes, e confundir as duas foi o meu erro.

    `use_reentrant=False` porque os blocos recebem kwargs e a versao reentrante nao os aceita. O
    ComfyUI nao tem checkpointing proprio (lido em `comfy/ldm/flux/model.py`), entao o embrulho e
    feito aqui e so nesta instancia do modelo.
    """
    from torch.utils.checkpoint import checkpoint
    n = 0
    for nome in ("double_blocks", "single_blocks"):
        blocos = getattr(nucleo, nome, None)
        if blocos is None:
            continue
        for b in blocos:
            orig = b.forward

            def novo(*args, _o=orig, **kw):
                return checkpoint(_o, *args, use_reentrant=False, **kw)

            b.forward = novo
            n += 1
    return n


def grava_professor(ref_unet, clip_nome, clip_tipo, passos, lado, cfg, sementes, n_prompts, dev):
    """Carrega o BF16 UMA vez e grava N trajetorias. Devolve lista de exemplos."""
    print(f"--- professor BF16: {ref_unet} ---", flush=True)

    # O encoder vem PRIMEIRO e sai antes do difusor entrar. O Qwen3-4B tem 8,04 GiB e o klein-4B
    # 7,2: os dois na placa nao caem em 24 GiB com ativacao. A primeira versao carregava o difusor
    # primeiro e morria com "trying to allocate 9663676416 bytes" e 2,46 GiB livres. Nao era peso
    # congelado nem backward, era ORDEM DE CARGA.
    clip = comfy.sd.load_clip(
        ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", clip_nome)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, clip_tipo.upper()))
    conds = {}
    for p in PROMPTS[:n_prompts]:
        conds[p] = ([list(clip.encode_from_tokens_scheduled(clip.tokenize(p))[0])],
                    [list(clip.encode_from_tokens_scheduled(clip.tokenize(""))[0])])
    del clip
    mm.unload_all_models()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    print(f"  encoder fora; alocado {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    model = comfy.sd.load_diffusion_model(
        folder_paths.get_full_path_or_raise("diffusion_models", ref_unet))

    for _pos, _neg in conds.values():
        for _lst in (_pos, _neg):
            for _c in _lst:
                for _k, _v in (_c[1] or {}).items():
                    if torch.is_tensor(_v):
                        print(f"  cond {_k}: {tuple(_v.shape)} {_v.dtype} em {_v.device}")
                if torch.is_tensor(_c[0]):
                    print(f"  cond tensor: {tuple(_c[0].shape)} {_c[0].dtype} em {_c[0].device}")
        break
    print(f"  alocado apos encodar e soltar o clip: "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    exemplos = []
    lf = model.model.latent_format
    for p in PROMPTS[:n_prompts]:
        pos, neg = conds[p]
        for s in sementes:
            capt = []

            def wrapper(apply_model, args, _capt=capt):
                out = apply_model(args["input"], args["timestep"], **args["c"])
                _capt.append({"x": args["input"].detach().to("cpu", torch.float32).clone(),
                              "t": args["timestep"].detach().to("cpu", torch.float32).clone(),
                              "c": {k: (v.detach().to("cpu", torch.float32).clone()
                                        if torch.is_tensor(v) else v)
                                    for k, v in args["c"].items()},
                              "out": out.detach().to("cpu", torch.float32).clone()})
                return out

            model.model_options = dict(model.model_options)
            model.model_options["model_function_wrapper"] = wrapper
            aloc = torch.cuda.memory_allocated() / 2**30
            res = torch.cuda.memory_reserved() / 2**30
            print(f"      antes de amostrar: alocado {aloc:.2f} GiB, reservado {res:.2f} GiB",
                  flush=True)
            # `lado` vem em PIXELS; o latente e 8x menor. Passar o valor cru dava um latente
            # 512x512 em vez de 64x64 -- 64x mais tokens -- e era a causa REAL dos tres OOMs
            # que eu atribui primeiro ao backward e depois a ordem de carga.
            lat = max(1, lado // 8)
            latent = torch.zeros([1, lf.latent_channels, lat, lat], device="cpu")
            noise = comfy.sample.prepare_noise(latent, s, None)
            comfy.sample.sample(model, noise, passos, cfg, "euler", "simple", pos, neg,
                                latent, denoise=1.0, disable_pbar=True, seed=s)
            exemplos.extend(capt)
            print(f"    prompt {PROMPTS[:n_prompts].index(p)}  semente {s}: "
                  f"{len(capt)} passos", flush=True)

    del model
    mm.unload_all_models()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    print(f"  professor: {len(exemplos)} exemplos de treino\n", flush=True)
    return exemplos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref-unet", required=True)
    p.add_argument("--aluno-unet", required=True, help="o braco 0, em nomenclatura ComfyUI")
    p.add_argument("--saida", required=True)
    p.add_argument("--clip", default="qwen_3_4b.safetensors")
    p.add_argument("--clip-type", default="flux2")
    p.add_argument("--passos", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sementes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--modo", choices=["ajuste", "zero", "ruido"], default="ajuste",
                   help="zero e ruido sao os controles que TEM de falhar de um jeito especifico")
    p.add_argument("--checkpointing", action="store_true",
                   help="recomputa ativacao dos blocos; QUEBRA no flux por soma in-place")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args(ARGV_REAL[1:])

    dev = torch.device(f"cuda:{a.device}")
    prof = grava_professor(a.ref_unet, a.clip, a.clip_type, a.passos, a.size, a.cfg,
                           a.sementes, a.prompts, dev)

    # A chave e do PROPRIO ComfyUI, nao um remendo meu: `comfy/ldm/flux/math.py:47-51` escolhe
    # entre a versao pura em PyTorch e o op fundido do comfy_kitchen olhando este sinalizador. Sem
    # ele o backward morre com "Trying to backward through comfy_kitchen.apply_rope.default but no
    # autograd formula was registered" -- e o `apply_rope` nao e o unico op fundido no caminho, entao
    # ligar isto e a forma suportada de fazer o modelo ser diferenciavel.
    if a.modo == "ajuste":
        mm.in_training = True
        print("  comfy.model_management.in_training = True (RoPE puro, diferenciavel)")

    print(f"--- aluno: {a.aluno_unet}  modo {a.modo} ---", flush=True)
    aluno = comfy.sd.load_diffusion_model(
        folder_paths.get_full_path_or_raise("diffusion_models", a.aluno_unet))
    nucleo = aluno.model.diffusion_model
    mm.load_models_gpu([aluno], force_full_load=True)

    por_nome = dict(nucleo.named_parameters())
    densos = [k for k in nomes_densos(por_nome.keys())]
    treinaveis = []
    for k, v in por_nome.items():
        alvo = k in set(densos)
        v.requires_grad_(alvo)
        if alvo:
            treinaveis.append((k, v))
    n_par = sum(v.numel() for _, v in treinaveis)
    print(f"  {len(por_nome)} parametros no nucleo; {len(treinaveis)} treinaveis, {n_par:,} valores")
    if not treinaveis:
        print("RECUSADO: nenhum parametro treinavel selecionado.", file=sys.stderr)
        return 2

    antes = {k: v.detach().to("cpu", torch.float32).clone() for k, v in treinaveis}

    if a.modo == "ruido":
        # mesma MAGNITUDE que o ajuste produziria, em posicao aleatoria: controle de orcamento casado
        g = torch.Generator(device="cpu").manual_seed(20260922)
        for k, v in treinaveis:
            r = torch.randn(v.shape, generator=g, dtype=torch.float32) * (a.lr * a.epocas)
            with torch.no_grad():
                v.add_(r.to(v.device, v.dtype))
        print(f"  ruido aplicado, escala {a.lr * a.epocas:.3e}")
    elif a.modo == "ajuste":
        # Checkpointing e OPT-IN e por padrao FICA DESLIGADO. Ele foi adicionado para um OOM que
        # nao era de ativacao -- era latente 64x maior por um `//8` que faltava -- e com o latente
        # certo o grafo cabe. E ele NAO funciona aqui de graca: `comfy/ldm/flux/layers.py:248` faz
        # `img += ...`, soma in-place, e a recomputacao do checkpoint levanta
        # "variable needed for gradient computation has been modified by an inplace operation".
        if a.checkpointing:
            n_ck = liga_checkpointing(nucleo)
            print(f"  gradient checkpointing em {n_ck} blocos (cuidado: in-place no flux)")
        opt = torch.optim.Adam([v for _, v in treinaveis], lr=a.lr)
        hist = []
        for ep in range(a.epocas):
            perdas = []
            for ex in prof:
                x = ex["x"].to(dev, torch.bfloat16)
                t = ex["t"].to(dev, torch.bfloat16)
                c = {k: (v.to(dev, torch.bfloat16) if torch.is_tensor(v) else v)
                     for k, v in ex["c"].items()}
                alvo = ex["out"].to(dev, torch.float32)
                out = nucleo(x, t, **c) if not hasattr(aluno.model, "apply_model") else \
                    aluno.model.apply_model(x, t, **c)
                perda = torch.nn.functional.mse_loss(out.float(), alvo)
                opt.zero_grad(set_to_none=True)
                perda.backward()
                opt.step()
                perdas.append(float(perda.detach()))
            m = sum(perdas) / len(perdas)
            hist.append(m)
            if ep == 0 or (ep + 1) % 5 == 0 or ep == a.epocas - 1:
                print(f"    epoca {ep + 1:3d}/{a.epocas}  perda media {m:.6e}", flush=True)
        print(f"  perda {hist[0]:.6e} -> {hist[-1]:.6e}  "
              f"({hist[0] / hist[-1] if hist[-1] else float('inf'):.4f}x)")
    else:
        print("  modo zero: nenhum passo de otimizacao, por construcao")

    depois = {k: v.detach().to("cpu", torch.float32).clone() for k, v in treinaveis}
    mudou = sum(1 for k in antes if not torch.equal(antes[k], depois[k]))
    desvio = max((float((depois[k] - antes[k]).norm() / antes[k].norm().clamp(min=1e-30))
                  for k in antes), default=0.0)
    print(f"  tensores densos que mudaram: {mudou}/{len(antes)}   maior desvio rel-L2 {desvio:.3e}")

    rel = {"modo": a.modo, "n_exemplos": len(prof), "epocas": a.epocas, "lr": a.lr,
           "densos_treinaveis": len(treinaveis), "params_treinaveis": n_par,
           "densos_mudados": mudou, "maior_desvio_rel_l2": desvio}
    Path(a.saida).with_suffix(".json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(f"\n  relatorio em {Path(a.saida).with_suffix('.json')}")

    print("\n=== NAO COBERTO ===")
    print("  Nenhuma imagem. Isto otimiza e mede PREVISAO em entradas casadas.")
    print("  Nenhum tempo daqui vale: o corpo ternario roda desempacotado, sem kernel de 1,58 bit.")
    print(f"  {len(prof)} exemplos para {n_par:,} parametros treinaveis -- fortemente")
    print("  subdeterminado, e isso limita o que um ganho aqui significa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
