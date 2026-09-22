"""Renderiza braços do FLUX.2-klein pelo `Flux2KleinPipeline` do diffusers, um pipeline, N transformers.

POR QUE ESTE ARQUIVO EXISTE. O braço 1 foi AJUSTADO no diffusers (`ajusta_denso_diffusers.py`) e
MEDIDO no ComfyUI (`probe_epsilon_ckpt_ab.py`). Ninguém tinha conferido que o mesmo peso se comporta
igual nos dois runtimes. Este tool é a metade diffusers do cruzamento: mesmos prompts, sementes,
passos e lado que o `quality_ladder` usa do lado ComfyUI.

O RUÍDO É O MESMO NOS DOIS RUNTIMES, e esta docstring já disse o contrário. A primeira versão
afirmava que cada runtime gerava o seu ruído e proibia comparar imagem do diffusers com imagem do
ComfyUI -- DEDUZIDO, não medido. Medido 2026-09-22 no klein, 1024 px, sementes 11 e 12, 5 prompts:
o mesmo braço nos dois runtimes dá SSIM 0,983 (b1) e 0,988 (b0), mesma composição a olho. O BF16
dá 0,862: imagem nítida diverge mais com a diferença numérica mínima entre as implementações. Então
a comparação DIRETA entre runtimes vale, e é o cruzamento mais forte (`cruza_runtimes.py` mede os
dois jeitos).

UM PIPELINE, TROCANDO SÓ O TRANSFORMER. Text encoder, VAE e scheduler são carregados uma vez; cada
braço entra por `carrega_transformer` do próprio ajuste (mesmo `from_config` + `load_state_dict`
estrito), então nenhum braço passa por um carregador diferente dos outros.

A classe sai do `model_index.json` (`classe_do_pipeline`), nunca de um chute: `Flux2Pipeline` e
`Flux2KleinPipeline` existem os dois e usam text encoders diferentes.

NÃO COBRE: métrica nenhuma — grava só PNG; `metricas_imagem.py` mede depois. Uma placa.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ajusta_denso_diffusers import carrega_transformer, classe_do_pipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raiz", required=True, help="pasta do pipeline (model_index.json etc)")
    p.add_argument("--braco", action="append", required=True, metavar="ROTULO=TRANSFORMER",
                   help="safetensors em nomenclatura DIFFUSERS; repetir por braço")
    p.add_argument("--prompt-file", type=Path, required=True,
                   help="um prompt por linha; '#' e linha vazia ignorados (mesma regra do ladder)")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    linhas = [ln.strip() for ln in a.prompt_file.read_text(encoding="utf-8").splitlines()]
    prompts = [ln for ln in linhas if ln and not ln.startswith("#")]
    for i, t in enumerate(prompts):
        print(f"  prompt {i}: {t[:96]}", flush=True)
    bracos = []
    for spec in a.braco:
        rot, _, arq = spec.partition("=")
        if not arq or not Path(arq).is_file():
            print(f"RECUSADO: braço {spec!r} sem arquivo existente.", file=sys.stderr)
            return 2
        bracos.append((rot, Path(arq)))

    raiz = Path(a.raiz)
    nome, Pipe = classe_do_pipeline(raiz)
    dev = torch.device("cuda:0")
    cfg = raiz / "transformer" / "config.json"
    a.out.mkdir(parents=True, exist_ok=True)

    pipe = None
    for rot, arq in bracos:
        t0 = time.perf_counter()
        # O VELHO SAI DA PLACA ANTES DE O NOVO ENTRAR. A primeira versao carregava o novo direto
        # na GPU e so depois fazia `del antigo`; na terceira troca a 3090 ficou em 24.266/24.576
        # MiB e o WDDM passou a paginar: 46-75 s/passo contra 0,6 s. `del` nao devolve memoria
        # se sobra QUALQUER referencia ao modulo, e `.to("cpu")` devolve mesmo assim.
        if pipe is not None:
            pipe.transformer.to("cpu")
            torch.cuda.empty_cache()
        tr = carrega_transformer(arq, cfg, torch.bfloat16, torch.device("cpu")).to(dev)
        if pipe is None:
            pipe = Pipe.from_pretrained(str(raiz), transformer=tr, torch_dtype=torch.bfloat16)
            pipe.to(dev)
            print(f"--- {nome} em {raiz} ---", flush=True)
        else:
            pipe.transformer = tr
            torch.cuda.empty_cache()
        print(f"  VRAM alocada {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
        print(f"\n--- {rot}  ({arq.name})  carregado em {time.perf_counter() - t0:.1f} s ---",
              flush=True)
        for i, texto in enumerate(prompts):
            for s in a.seeds:
                t1 = time.perf_counter()
                # guidance_scale 1.0: o klein declara `is_distilled: true`, o pipeline ja ignora
                # CFG (`pipeline_flux2_klein.py:593`); passar 1.0 so evita o aviso da linha 584.
                # decode pelo proprio pipeline: e o caminho que um usuario do diffusers teria.
                # Sem latente gravado: o cruzamento (X) mede imagem, e decodificar um latente do
                # klein fora do pipeline exige refazer o unpatchify e a desnormalizacao `bn` --
                # outra implementacao para errar.
                img = pipe(prompt=texto, height=a.size, width=a.size,
                           num_inference_steps=a.steps, guidance_scale=1.0,
                           generator=torch.Generator(device="cpu").manual_seed(s),
                           output_type="pil").images[0]
                img.save(a.out / f"{rot}__p{i}_s{s}.png")
                print(f"  p{i} s{s}  {time.perf_counter() - t1:.1f} s", flush=True)

    print(f"\nescrito em {a.out}")
    print("NAO COBERTO: nenhuma metrica aqui. O ruido inicial bateu com o do ComfyUI no klein "
          "(medido 2026-09-22); em outro modelo, conferir antes de comparar entre runtimes.")
    return 0


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("bench:render_klein_diffusers") as _g:
        if _g.refused:
            print(_g.refused)
            raise SystemExit(1)
        raise SystemExit(main())
