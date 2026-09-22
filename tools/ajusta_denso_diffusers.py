"""Braco 1 pelo DIFFUSERS: congela o corpo ternario e ajusta so o conjunto denso. (EXECUTADO, GPU)

Substitui `tools/ajusta_denso_compensacao.py`, que morreu contra o ComfyUI e nao contra o problema.

**POR QUE DIFFUSERS, e o credito e dele:** ele perguntou por que eu nao tinha olhado o RUNTIME do
Bonsai. Os repos de peso publicam so README e NOTICE, mas o README aponta
`github.com/PrismML-Eng/Bonsai-Image-Demo`, cujo `pyproject.toml` depende de `prism-image-studio`, e
esse repo -- `github.com/PrismML-Eng/image-studio`, hoje PUBLICO -- carrega o modelo assim:

    backend_gpu/pipeline_gpu.py:214   model = Flux2Transformer2DModel.from_config(cfg)

**O runtime deles usa diffusers, nao ComfyUI.** E o `Flux2Transformer2DModel` do diffusers 0.38.0 tem
**zero somas in-place** (contado na fonte), enquanto o `comfy/ldm/flux/layers.py` tem cinco e por isso
nao e diferenciavel -- no nosso 0.33 e tambem no upstream v0.37, lido nos dois. Ou seja: **nao era
preciso patch nenhum, nem no core nem por monkeypatch.** Eu passei cinco tentativas brigando com a
implementacao errada porque nao fui ler o runtime de quem publicou o modelo.

Conferido antes de escrever isto: `from_config` no config deles instancia **3.875.544.576**
parametros e **169 tensores**, com **zero** nome faltando, zero sobrando e zero shape divergente
contra o nosso checkpoint em nomenclatura diffusers. O remap para BFL nao entra aqui.

E em nomenclatura diffusers o conjunto denso volta a ser **69 tensores** (as 60 normas ficam fora dos
blocos aqui), o que casa com o criterio -- a correcao para 9 valia so para a nomenclatura BFL.

AS ENTRADAS DO PROFESSOR sao capturadas embrulhando `pipe.transformer.forward`, nao reconstruidas: o
`img_ids`/`txt_ids`/`guidance` do Flux2 saem do proprio pipeline, e reconstrui-los a mao seria
exatamente o tipo de reimplementacao que produz diferenca silenciosa.

OS DOIS CONTROLES, do criterio:
  zero    ajuste com ZERO passos tem de dar peso byte a byte identico ao braco 0
  ruido   perturbar o denso com ruido do mesmo tamanho tem de PIORAR

NAO COBRE: nenhuma imagem. Mede e otimiza PREVISAO em entradas casadas. Nenhum tempo daqui vale: o
corpo ternario roda desempacotado em bf16/fp16, sem kernel de 1,58 bit.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")
PROMPTS = [
    "a red apple on a weathered wooden table, soft window light",
    "a portrait of an elderly fisherman, deep wrinkles, overcast light",
    "a neon-lit night market in the rain, puddles reflecting signs",
    "a single ice crystal on dark slate, macro, fine internal structure",
]


def pilhas_reais(nomes) -> set[str]:
    ind: dict[str, set[str]] = {}
    for k in nomes:
        m = BLOCO.match(k)
        if m:
            ind.setdefault(m.group("pilha"), set()).add(m.group("i"))
    return {n for n, i in ind.items() if len(i) >= 2}


def nomes_densos(chaves) -> list[str]:
    """2-D fora de qualquer pilha de blocos, mais tudo 1-D. Regra MEDIDA no klein-4B."""
    pil = pilhas_reais(chaves)
    return sorted(k for k in chaves
                  if (m := BLOCO.match(k)) is None or m.group("pilha") not in pil)


def carrega_transformer(caminho: Path, config: Path, dtype, dev):
    from diffusers import Flux2Transformer2DModel
    from safetensors.torch import load_file
    cfg = json.loads(config.read_text(encoding="utf-8"))
    m = Flux2Transformer2DModel.from_config(cfg)
    sd = load_file(str(caminho))
    falta, sobra = m.load_state_dict(sd, strict=False)
    if falta or sobra:
        raise RuntimeError(f"state_dict nao casa: {len(falta)} faltando, {len(sobra)} sobrando; "
                           f"primeiros {list(falta)[:3]} / {list(sobra)[:3]}")
    return m.to(device=dev, dtype=dtype).eval()


def grava_professor(raiz: Path, ref_transformer: Path, n_prompts, sementes, passos, lado, dev):
    """Roda o pipeline do diffusers e CAPTURA as entradas e a saida do transformer."""
    from diffusers import Flux2Pipeline
    print(f"--- professor: pipeline do diffusers em {raiz} ---", flush=True)
    tr = carrega_transformer(ref_transformer, raiz / "transformer" / "config.json",
                             torch.bfloat16, dev)
    pipe = Flux2Pipeline.from_pretrained(str(raiz), transformer=tr, vae=None,
                                         torch_dtype=torch.bfloat16)
    pipe.to(dev)

    capt: list[dict] = []
    orig = tr.forward

    def espia(*args, **kw):
        out = orig(*args, **kw)
        saida = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
        capt.append({"kw": {k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v)
                            for k, v in kw.items()},
                     "out": saida.detach().to("cpu", torch.float32).clone()})
        return out

    tr.forward = espia
    for i, p in enumerate(PROMPTS[:n_prompts]):
        for s in sementes:
            antes = len(capt)
            pipe(prompt=p, height=lado, width=lado, num_inference_steps=passos,
                 generator=torch.Generator(device="cpu").manual_seed(s),
                 output_type="latent")
            print(f"    prompt {i} semente {s}: {len(capt) - antes} chamadas", flush=True)
    tr.forward = orig

    del pipe, tr
    torch.cuda.empty_cache()
    print(f"  professor: {len(capt)} exemplos\n", flush=True)
    return capt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raiz", required=True, help="pasta do pipeline (model_index.json etc)")
    p.add_argument("--ref-transformer", required=True)
    p.add_argument("--aluno-transformer", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--passos", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--sementes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--modo", choices=["ajuste", "zero", "ruido"], default="ajuste")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()

    dev = torch.device(f"cuda:{a.device}")
    raiz = Path(a.raiz)
    prof = grava_professor(raiz, Path(a.ref_transformer), a.prompts, a.sementes,
                           a.passos, a.size, dev)
    if not prof:
        print("RECUSADO: nenhum exemplo capturado.", file=sys.stderr)
        return 2

    print(f"--- aluno: {Path(a.aluno_transformer).name}  modo {a.modo} ---", flush=True)
    aluno = carrega_transformer(Path(a.aluno_transformer), raiz / "transformer" / "config.json",
                                torch.bfloat16, dev)
    por_nome = dict(aluno.named_parameters())
    densos = set(nomes_densos(por_nome.keys()))
    treinaveis = []
    for k, v in por_nome.items():
        v.requires_grad_(k in densos)
        if k in densos:
            treinaveis.append((k, v))
    n_par = sum(v.numel() for _, v in treinaveis)
    print(f"  {len(por_nome)} parametros; {len(treinaveis)} treinaveis, {n_par:,} valores")

    antes = {k: v.detach().to("cpu", torch.float32).clone() for k, v in treinaveis}
    hist: list[float] = []

    if a.modo == "ruido":
        g = torch.Generator(device="cpu").manual_seed(20260922)
        escala = a.lr * a.epocas
        for _k, v in treinaveis:
            r = torch.randn(v.shape, generator=g, dtype=torch.float32) * escala
            with torch.no_grad():
                v.add_(r.to(v.device, v.dtype))
        print(f"  ruido aplicado, escala {escala:.3e}")
    elif a.modo == "ajuste":
        opt = torch.optim.Adam([v for _, v in treinaveis], lr=a.lr)
        for ep in range(a.epocas):
            perdas = []
            for ex in prof:
                kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ex["kw"].items()}
                alvo = ex["out"].to(dev, torch.float32)
                out = aluno(**kw)
                saida = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
                perda = torch.nn.functional.mse_loss(saida.float(), alvo)
                opt.zero_grad(set_to_none=True)
                perda.backward()
                opt.step()
                perdas.append(float(perda.detach()))
            hist.append(sum(perdas) / len(perdas))
            if ep == 0 or (ep + 1) % 5 == 0 or ep == a.epocas - 1:
                print(f"    epoca {ep + 1:3d}/{a.epocas}  perda {hist[-1]:.6e}", flush=True)
        print(f"  perda {hist[0]:.6e} -> {hist[-1]:.6e}")
    else:
        print("  modo zero: nenhum passo de otimizacao, por construcao")

    depois = {k: v.detach().to("cpu", torch.float32).clone() for k, v in treinaveis}
    mudou = sum(1 for k in antes if not torch.equal(antes[k], depois[k]))
    desvio = max((float((depois[k] - antes[k]).norm() / antes[k].norm().clamp(min=1e-30))
                  for k in antes), default=0.0)
    print(f"  densos mudados: {mudou}/{len(antes)}   maior desvio rel-L2 {desvio:.3e}")

    from safetensors.torch import load_file, save_file
    sai = Path(a.saida)
    if sai.exists():
        print(f"RECUSADO: {sai} ja existe.", file=sys.stderr)
        return 2
    base = load_file(str(a.aluno_transformer))
    for k, v in treinaveis:
        base[k] = v.detach().to("cpu", base[k].dtype).clone()
    parcial = sai.with_suffix(sai.suffix + ".partial")
    save_file(base, str(parcial))
    parcial.replace(sai)
    print(f"  escrito {sai}  {sai.stat().st_size:,} B")

    rel = {"modo": a.modo, "n_exemplos": len(prof), "epocas": a.epocas, "lr": a.lr,
           "treinaveis": len(treinaveis), "params_treinaveis": n_par,
           "densos_mudados": mudou, "maior_desvio_rel_l2": desvio,
           "perda_inicial": hist[0] if hist else None, "perda_final": hist[-1] if hist else None}
    sai.with_suffix(".json").write_text(json.dumps(rel, indent=2), encoding="utf-8")

    print("\n=== NAO COBERTO ===")
    print("  Nenhuma imagem. Isto otimiza e mede PREVISAO em entradas casadas pelo pipeline.")
    print(f"  {len(prof)} exemplos para {n_par:,} parametros -- fortemente subdeterminado, e isso")
    print("  limita o que um ganho aqui significa.")
    print("  Nenhum tempo vale: corpo ternario desempacotado, sem kernel de 1,58 bit.")
    print("  O pipeline do diffusers nao e o caminho do ComfyUI: o mesmo peso pode render diferente")
    print("  la, e isso NAO foi conferido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
