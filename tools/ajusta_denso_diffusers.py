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

**O conjunto denso sao 9 tensores, 195.035.136 valores, NAS DUAS NOMENCLATURAS.** Esta linha dizia
69 em diffusers, com o argumento de que as 60 normas cairiam fora dos blocos aqui; a propria
execucao desta ferramenta imprimiu `9 treinaveis` e a medicao direta confirmou: as 60 `norm_q`/
`norm_k` sao `single_transformer_blocks.N.attn.norm_k.weight` -- DENTRO de bloco -- em diffusers
tambem. Entao a correcao de 69 para 9 vale para as duas, e o criterio fica corrigido aqui.

AS ENTRADAS DO PROFESSOR sao capturadas embrulhando `pipe.transformer.forward`, nao reconstruidas: o
`img_ids`/`txt_ids`/`guidance` do Flux2 saem do proprio pipeline, e reconstrui-los a mao seria
exatamente o tipo de reimplementacao que produz diferenca silenciosa.

OS DOIS CONTROLES, do criterio:
  zero    ajuste com ZERO passos tem de dar peso byte a byte identico ao braco 0
  ruido   perturbar o denso com ruido do mesmo tamanho tem de PIORAR

DE PASSAGEM, LIDO NO PIPELINE DELES e nao suposto: `Flux2KleinPipeline.encode_prompt` tem
`text_encoder_out_layers: tuple[int] = (9, 18, 27)` (`pipeline_flux2_klein.py:434`). O
condicionamento do klein e um TAP DE TRES CAMADAS do Qwen3, nao o ultimo hidden state -- o que
significa que qualquer tentativa de reproduzir esse condicionamento fora do diffusers tem de
reproduzir o tap tambem. Aqui nao importa, porque o professor e o aluno recebem o MESMO
`encoder_hidden_states` capturado do pipeline; fica registrado porque estava na lista de aberto.

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


def ler_metadata(p: Path) -> dict[str, str] | None:
    """`__metadata__` do safetensors, lendo SO o header. Devolve None se nao houver."""
    import struct
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__")


def classe_do_pipeline(raiz: Path):
    """A classe sai do `_class_name` do `model_index.json`, nunca de um chute meu.

    MEDIDO, e custou uma corrida: eu tinha escrito `Flux2Pipeline` e o klein declara
    `Flux2KleinPipeline`. As duas existem no diffusers 0.38.0 e NAO usam o mesmo text encoder --
    a generica chama `_get_mistral_3_small_prompt_embeds`, a do klein chama
    `_get_qwen3_prompt_embeds`, e o `model_index.json` do klein diz `Qwen3ForCausalLM`. O erro
    que isso produz nao fala de pipeline nenhum: morre em `apply_chat_template` dizendo que o
    tokenizer nao tem `chat_template`, o que manda o leitor consertar o tokenizer.
    """
    import diffusers
    nome = json.loads((raiz / "model_index.json").read_text(encoding="utf-8"))["_class_name"]
    cls = getattr(diffusers, nome, None)
    if cls is None:
        raise RuntimeError(f"diffusers {diffusers.__version__} nao tem {nome}")
    return nome, cls


def grava_professor(raiz: Path, ref_transformer: Path, n_prompts, sementes, passos, lado, dev):
    """Roda o pipeline do diffusers e CAPTURA as entradas e a saida do transformer."""
    nome, Pipe = classe_do_pipeline(raiz)
    print(f"--- professor: {nome} do diffusers em {raiz} ---", flush=True)
    tr = carrega_transformer(ref_transformer, raiz / "transformer" / "config.json",
                             torch.bfloat16, dev)
    # A VAE ENTRA, e nao e desperdicio. Com `vae=None` o `vae_scale_factor` cai no fallback 8 --
    # que por sorte e o valor certo do klein (`block_out_channels` tem 4 entradas, 2**3 = 8, lido
    # no config) -- mas `pipeline_flux2_klein.py:907` le `self.vae.bn.running_mean`
    # INCONDICIONALMENTE depois do loop, inclusive com `output_type="latent"`. Morreria de
    # AttributeError no fim, com todo o trabalho feito e nenhuma captura devolvida.
    pipe = Pipe.from_pretrained(str(raiz), transformer=tr, torch_dtype=torch.bfloat16)
    pipe.to(dev)
    # ... e sai da placa em seguida: `decode` nunca e chamado, e `bn.running_mean` faz `.to(...)`
    # para o device do latente. 168 MB que nao competem com o encoder de 8,04 GiB.
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to("cpu")

    capt: list[dict] = []
    orig = tr.forward

    def espia(*args, **kw):
        out = orig(*args, **kw)
        saida = out[0] if isinstance(out, tuple) else getattr(out, "sample", out)
        # O ALVO FICA NO DTYPE EM QUE O PROFESSOR O PRODUZIU. A primeira versao gravava
        # `.to("cpu", torch.float32)`, o que dobra a RAM sem ganhar informacao nenhuma: o
        # transformer roda em bf16, entao promover a fp32 e exato-e-inutil e voltar depois tambem.
        # O custo real vai IMPRESSO em `grava_professor` em vez de estimado aqui -- a primeira
        # versao deste comentario citava 3,62 GiB para 64 exemplos, numero que eu DEDUZI de um
        # shape errado ([1, 4608, 3072]); o medido no smoke foi 0,02 GiB para 2 exemplos.
        capt.append({"kw": {k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v)
                            for k, v in kw.items()},
                     "out": saida.detach().to("cpu").clone()})
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
    custo = sum(t.numel() * t.element_size()
                for ex in capt
                for t in [ex["out"], *[v for v in ex["kw"].values() if torch.is_tensor(v)]])
    print(f"  professor: {len(capt)} exemplos, {custo / 2**30:.2f} GiB de RAM retidos "
          f"(alvo em {capt[0]['out'].dtype})\n", flush=True)
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
    p.add_argument("--casar-com",
                   help="json de um ajuste, obrigatorio no modo ruido: a escala da perturbacao "
                        "sai do desvio rel-L2 MEDIDO ali, tensor a tensor")
    p.add_argument("--mestre", choices=["bf16", "fp32", "bf16-sr"], default="bf16",
                   help="dtype do peso MESTRE dos treinaveis. bf16 e o que o braco 1 usou: com lr "
                        "1e-5 o passo do Adam fica abaixo de meio-ulp do bf16 e 63,4%% dos "
                        "elementos NUNCA mudaram (medido 2026-09-22). fp32 = mestre fp32 com "
                        "autocast bf16 no forward. bf16-sr = peso bf16 com arredondamento "
                        "estocastico (`torchao.optim._AdamW`, weight_decay 0 = Adam).")
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
        # O CONTROLE EXIGE RUIDO DO MESMO TAMANHO DO AJUSTE, e `lr * epocas` nao e esse tamanho --
        # e uma proxy que o Adam nao respeita (passo normalizado, momento, 30 epocas de direcao
        # coerente). Com `--casar-com` a escala vem do desvio rel-L2 que o ajuste MEDIU, tensor a
        # tensor, e o controle passa a responder "a mesma perturbacao, sem direcao, piora?".
        # Sem `--casar-com` o tool RECUSA em vez de silenciosamente medir outra coisa.
        if not a.casar_com:
            print("RECUSADO: modo ruido exige --casar-com <json do ajuste>, para que a perturbacao "
                  "tenha o tamanho MEDIDO do ajuste e nao uma proxy.", file=sys.stderr)
            return 2
        ref = json.loads(Path(a.casar_com).read_text(encoding="utf-8"))
        alvo_rel = ref.get("desvios_por_tensor")
        if not alvo_rel:
            print(f"RECUSADO: {a.casar_com} nao tem `desvios_por_tensor`; foi escrito por uma "
                  "versao anterior desta ferramenta.", file=sys.stderr)
            return 2
        faltando = [k for k, _ in treinaveis if k not in alvo_rel]
        if faltando:
            print(f"RECUSADO: {len(faltando)} treinaveis sem desvio de referencia, p.ex. "
                  f"{faltando[:3]}", file=sys.stderr)
            return 2
        g = torch.Generator(device="cpu").manual_seed(20260922)
        for k, v in treinaveis:
            r = torch.randn(v.shape, generator=g, dtype=torch.float32)
            n = v.detach().to("cpu", torch.float32).norm()
            r *= alvo_rel[k] * n / r.norm().clamp(min=1e-30)   # ||r|| = rel * ||W||, exato
            with torch.no_grad():
                v.add_(r.to(v.device, v.dtype))
        print(f"  ruido casado com {Path(a.casar_com).name}: rel-L2 alvo "
              f"{min(alvo_rel.values()):.3e} a {max(alvo_rel.values()):.3e}")
    elif a.modo == "ajuste":
        from contextlib import nullcontext
        # O EIXO `--mestre`. So muda ONDE a atualizacao e acumulada; dado, epocas, lr e ordem ficam
        # os do braco 1. Os 9 treinaveis sao todos `nn.Linear`, entao sob autocast o matmul roda em
        # bf16 igual ao braco 1 -- o que muda e so o peso guardado entre um passo e o proximo.
        params = [v for _, v in treinaveis]
        contexto = nullcontext()
        if a.mestre == "fp32":
            for v in params:
                v.data = v.data.float()
            opt = torch.optim.Adam(params, lr=a.lr)
            contexto = torch.autocast("cuda", dtype=torch.bfloat16)
        elif a.mestre == "bf16-sr":
            from torchao.optim import _AdamW
            opt = _AdamW(params, lr=a.lr, weight_decay=0.0, bf16_stochastic_round=True)
        else:
            opt = torch.optim.Adam(params, lr=a.lr)
        print(f"  mestre {a.mestre}: {type(opt).__module__}.{type(opt).__name__}, "
              f"peso {params[0].dtype}", flush=True)
        for ep in range(a.epocas):
            perdas = []
            for ex in prof:
                kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ex["kw"].items()}
                alvo = ex["out"].to(dev, torch.float32)
                with contexto:
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

    # Medido no dtype em que o arquivo GRAVA (bf16): um mestre fp32 que mudou menos que meio-ulp
    # volta ao mesmo bf16 na escrita, e contar isso como mudanca mentiria sobre o arquivo.
    depois = {k: v.detach().to("cpu", torch.bfloat16).float() for k, v in treinaveis}
    mudou = sum(1 for k in antes if not torch.equal(antes[k], depois[k]))
    n_el = sum(t.numel() for t in antes.values())
    el_mud = sum(int((antes[k] != depois[k]).sum()) for k in antes)
    print(f"  ELEMENTOS mudados (no bf16 gravado): {el_mud / n_el * 100:.2f}% de {n_el:,}")
    # POR TENSOR, nao so o maior: o modo `ruido` precisa casar o tamanho da perturbacao camada a
    # camada, e um unico maximo obrigaria a aplicar o desvio do pior tensor em todos.
    por_tensor = {k: float((depois[k] - antes[k]).norm() / antes[k].norm().clamp(min=1e-30))
                  for k in antes}
    desvio = max(por_tensor.values(), default=0.0)
    print(f"  densos mudados: {mudou}/{len(antes)}   maior desvio rel-L2 {desvio:.3e}")

    from safetensors.torch import load_file, save_file
    sai = Path(a.saida)
    if sai.exists():
        print(f"RECUSADO: {sai} ja existe.", file=sys.stderr)
        return 2
    base = load_file(str(a.aluno_transformer))
    for k, v in treinaveis:
        base[k] = v.detach().to("cpu", base[k].dtype).clone()
    # O `__metadata__` DA ORIGEM VIAJA. `save_file` sem `metadata=` grava None, e foi assim que o
    # controle `zero` saiu com os 169 tensores byte a byte identicos e o ARQUIVO diferente do braco
    # 0 por 32 bytes -- exatamente o `{"format": "pt"}` que o escritor de la grava. Perder isso e
    # perder proveniencia num artefato que pode ser publicado.
    meta = ler_metadata(Path(a.aluno_transformer))
    parcial = sai.with_suffix(sai.suffix + ".partial")
    save_file(base, str(parcial), metadata=meta)
    parcial.replace(sai)
    print(f"  escrito {sai}  {sai.stat().st_size:,} B")

    rel = {"modo": a.modo, "mestre": a.mestre, "elementos_mudados_frac": el_mud / n_el,
           "n_exemplos": len(prof), "epocas": a.epocas, "lr": a.lr,
           "treinaveis": len(treinaveis), "params_treinaveis": n_par,
           "densos_mudados": mudou, "maior_desvio_rel_l2": desvio,
           "desvios_por_tensor": por_tensor,
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
