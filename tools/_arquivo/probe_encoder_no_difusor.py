"""O que o text encoder DEVOLVE ao difusor, quantizado contra o gemeo BF16.

A PERGUNTA, e ela veio do dono da bancada: nao gere imagem, nao amostre nada -- so imite a chamada
que o ComfyUI faz e olhe a resposta. E o teste certo, porque um text encoder num pipeline de
difusao **nao gera texto**: ele devolve um tensor de condicionamento, e e so isso que o difusor ve.

POR QUE ISTO SUBSTITUI O TESTE DE GERACAO. O encoder quantizado foi julgado antes fazendo ele
gerar texto, e sai lixo grudado em toda resposta. Mas geracao e outro caminho e mais severo:
autorregressiva, entao erro de um token vira entrada do proximo e compoe por centenas de passos;
e passa por um argmax sobre 150 mil palavras, onde erro pequeno troca a palavra inteira.
Condicionar e UMA passada, sem realimentacao, usando o estado interno direto. A geracao mostra o
dano de forma dramatica e SUPERESTIMA o impacto no uso real.

    python_embeded\\python.exe -s tools/probe_encoder_no_difusor.py

Carrega os dois pelo `comfy.sd.load_clip` de verdade e chama `encode_from_tokens_scheduled`, que e
o que o no de prompt do ComfyUI chama. Nada e reimplementado aqui.

NAO COBERTO: mede distancia ate o gemeo BF16, nunca qualidade absoluta -- o BF16 e o alvo, nao a
verdade. Nao diz que erro de condicionamento e tolerado pelo difusor: para isso e preciso gerar, e
esta bancada ja mediu que imagem de trajetoria livre responde "quebrou ou nao" e nao "qual e
melhor". Um prompt por execucao, salvo `--prompt` repetido.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

_ARGV = sys.argv[:]
sys.argv = ["main.py", "--disable-dynamic-vram"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
sys.argv = _ARGV

import torch  # noqa: E402

PROMPTS_PADRAO = [
    "a red apple on a weathered wooden table, soft window light",
    "portrait of an elderly fisherman, weathered face, golden hour",
    "cyberpunk street market at night, neon reflections on wet asphalt, crowds",
]


def condiciona(nome: str, tipo_clip: str, prompts: list[str]) -> dict:
    """Exatamente o que o no de prompt do ComfyUI faz: tokenize + encode_from_tokens_scheduled."""
    import comfy.sd
    import folder_paths

    caminho = folder_paths.get_full_path_or_raise("text_encoders", nome)
    clip = comfy.sd.load_clip(
        ckpt_paths=[caminho],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, tipo_clip),
        model_options={"disable_dynamic": True})

    fmts: dict[str, int] = {}
    for m in clip.cond_stage_model.modules():
        f = getattr(m, "quant_format", None)
        if f:
            fmts[f] = fmts.get(f, 0) + 1

    saida = {"arquivo": nome, "formatos": fmts, "conds": [], "segundos": []}
    for p in prompts:
        t0 = time.perf_counter()
        c = clip.encode_from_tokens_scheduled(clip.tokenize(p))
        saida["segundos"].append(round(time.perf_counter() - t0, 3))
        saida["conds"].append(c[0][0].detach().float().cpu())
    del clip
    torch.cuda.empty_cache()
    return saida


def compara(r: torch.Tensor, q: torch.Tensor) -> dict:
    if r.shape != q.shape:
        return {"erro": f"formas diferentes: {tuple(r.shape)} contra {tuple(q.shape)}"}
    d = q - r
    plano_r, plano_q = r.reshape(1, -1), q.reshape(1, -1)
    # por token: quantos tokens mudaram muito, e qual mudou mais
    if r.ndim >= 2:
        eixo = tuple(range(1, r.ndim))
        por_token = (torch.linalg.vector_norm(d.reshape(r.shape[0], -1), dim=1)
                     / torch.linalg.vector_norm(r.reshape(r.shape[0], -1), dim=1).clamp_min(1e-12))
    else:
        por_token = torch.zeros(1)
    return {
        "forma": list(r.shape),
        "rel_rmse": float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(r)),
        "cosseno": float(torch.nn.functional.cosine_similarity(plano_q, plano_r)),
        "absmax_ref": float(r.abs().max()), "absmax_quant": float(q.abs().max()),
        "media_ref": float(r.mean()), "media_quant": float(q.mean()),
        "desvio_ref": float(r.std()), "desvio_quant": float(q.std()),
        "pior_token": float(por_token.max()), "mediana_token": float(por_token.median()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default="qwen_3_4b.safetensors")
    p.add_argument("--quant", default="qwen_3_4b_w4a4_convrot.safetensors")
    p.add_argument("--clip-type", default="LUMINA2")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--json", type=Path, default=RAIZ / "bench" / "encoder_condicionamento.json")
    a = p.parse_args()
    prompts = a.prompt or PROMPTS_PADRAO

    print("=" * 76)
    print("O que o encoder DEVOLVE ao difusor. Sem gerar imagem, sem amostrar.")
    print("=" * 76)

    saidas = {}
    for rot, nome in (("ref", a.ref), ("quant", a.quant)):
        saidas[rot] = condiciona(nome, a.clip_type, prompts)
        s = saidas[rot]
        print(f"  {rot:5} {nome:44} {s['formatos'] or 'BF16'}")
        print(f"        forma {tuple(s['conds'][0].shape)}   {s['segundos']} s por prompt")

    linhas = []
    for i, prompt in enumerate(prompts):
        c = compara(saidas["ref"]["conds"][i], saidas["quant"]["conds"][i])
        c["prompt"] = prompt
        linhas.append(c)

    print()
    print(f"{'prompt':46} {'rel-RMSE':>10} {'cosseno':>10} {'pior token':>11}")
    print("-" * 80)
    for c in linhas:
        if "erro" in c:
            print(f"{c['prompt'][:46]:46}   {c['erro']}")
            continue
        print(f"{c['prompt'][:46]:46} {c['rel_rmse']:10.4e} {c['cosseno']:10.6f} "
              f"{c['pior_token']:11.4f}")

    ok = [c for c in linhas if "erro" not in c]
    if ok:
        c = ok[0]
        print()
        print("Estatistica do tensor, primeiro prompt -- muda a ESCALA ou so a direcao?")
        print(f"{'':14} {'referencia':>14} {'quantizado':>14}")
        for rot, kr, kq in (("absmax", "absmax_ref", "absmax_quant"),
                            ("media", "media_ref", "media_quant"),
                            ("desvio", "desvio_ref", "desvio_quant")):
            print(f"  {rot:12} {c[kr]:14.5f} {c[kq]:14.5f}")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(
        {"ref": a.ref, "quant": a.quant, "formatos": saidas["quant"]["formatos"],
         "segundos": {k: saidas[k]["segundos"] for k in saidas}, "prompts": linhas},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {a.json}")
    print("\nNAO COBERTO: mede distancia ate o gemeo BF16, nunca qualidade absoluta -- o BF16 e o")
    print("alvo, nao a verdade. E NAO diz se o difusor tolera esse erro: para isso e preciso gerar,")
    print("e esta bancada ja mediu cinco vezes que imagem de trajetoria livre responde 'quebrou ou")
    print("nao' e nao 'qual e melhor'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
