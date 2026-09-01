"""Camada 3 do avaliador: a guarda do braco de REFERENCIA. Este modelo esta ouvindo?

O QUE ELA PERGUNTA, E O QUE ELA SE RECUSA A PERGUNTAR
-----------------------------------------------------
Ela **nao** pergunta se a imagem esta boa. Essa pergunta esta proibida nesta bancada: nenhum corte
medido aqui separa usavel de inutilizavel -- 0,1837 correta contra 0,2147 destruida, 0,7173 boa
contra 0,8255 destruida.

Ela pergunta **"este modelo esta respondendo ao proprio condicionamento?"**, que tem resposta.

POR QUE ELA EXISTE
------------------
Em 2026-09-01 o Wan 2.1 VACE custou **quatro renderizacoes** porque o braco de referencia estava
quebrado e ninguem olhou. O FP16 sem quantizacao nenhuma saia como trama tecida em toda
configuracao. A primeira leitura foi "a quantizacao destruiu o modelo, divergencia 1,2365" -- um
numero que nao media nada, porque os dois lados estavam quebrados.

Causa: `WAN21_Vace.extra_conds` preenche `vace_frames` com zeros quando nao ha no de controle,
passa por `process_latent_in` (zero vira NAO nulo), concatena mascara de uns e aplica em
`vace_strength=1.0`. Nao e "sem controle": e um controle constante em forca total, sem erro e sem
aviso.

A camada 1 (`avaliar.py`) ja marca estaticamente qualquer checkpoint com `vace_blocks`. Isso pega
ESTA armadilha, porque ela ja e conhecida. Esta camada tem que pegar a PROXIMA, que ninguem viu.

A MEDICAO
---------
Dois prompts bem diferentes por duas sementes, no modelo NAO quantizado, tudo o mais igual:

    d_prompt   quanto o PROMPT move o latente          (duas estimativas, uma por semente)
    d_semente  quanto a SEMENTE move o latente          (duas estimativas, um por prompt)
    resposta   d_prompt / d_semente

`d_semente` e o **controle negativo**, e e ele que torna a medida interpretavel: um `d_prompt`
pequeno sozinho pode so significar que o modelo e estavel. Pequeno *em relacao ao que a semente
move* significa que o condicionamento nao esta chegando.

Duas estimativas de cada, nao uma, porque uma execucao nao e uma medicao nesta bancada.

CRITERIO E RESULTADO
--------------------
O criterio foi escrito em `bench/criterio_guarda_referencia.md` **antes** de qualquer medicao,
junto com as tres condicoes que a refutariam. **EXECUTADO** na 3090 em 2026-09-01, quatro bracos,
tres familias:

    braco                                        resposta   veredito       previsao   acertou
    Wan VACE 1.0  destruido, verdade conhecida     0,2477    REPROVADO      < 0,5      sim
    Wan VACE 0.0  bom, verdade conhecida           0,7911    OLHAR          > 0,8      NAO, por 1,1%
    Z-Image v2 BF16               bom              1,3459    SEM VEREDITO   > 0,8      sim
    HunyuanVideo 1.5 FP16         bom              2,8603    SEM VEREDITO   > 0,8      sim

**As tres condicoes de refutacao ficaram em silencio**: 3,19x separam o quebrado do bom mais
proximo, o limiar de reprova nao rejeitou nenhum dos tres sadios, e o destruido nao passou.
Ela teria abortado a rodada do Wan na primeira imagem em vez da quarta.

O mecanismo aparece cru: no braco destruido o prompt move o latente **0,075** e a semente move
**0,250** -- o modelo gera a partir do ruido e ignora o pedido. Nos sadios o prompt move 0,76 a
1,58.

O limite SAO de 0,8 errou por 1,1% e **nao foi mexido**. Mudar um limiar depois de ver o numero
que ele deveria classificar nao e calibrar, e descrever.

USO
---
    python.exe -s tools\\avaliar_referencia.py --modelo wan2.1_vace_1.3B_fp16.safetensors \\
        --clip umt5_xxl_fp8_e4m3fn_scaled.safetensors --clip-type wan \\
        --steps 25 --size 480 --frames 33 --vace-strength 0.0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importar `quality_ladder` executa o preambulo de argv que o ComfyUI exige de quem o importa como
# biblioteca -- neutraliza, liga o parse, forca, restaura. Reusar aquele modulo em vez de copiar
# `encode`/`sample_all` e deliberado: as duas funcoes carregam armadilhas ja pagas (o encoder sai
# da placa antes do transformer entrar, `disable_dynamic=True` evita o HostBuffer do aimdo que
# derruba o processo sem traceback, a forma do latente sai do `latent_format` do proprio modelo).
import quality_ladder as ql  # noqa: E402
import torch  # noqa: E402

# Seguros de importar aqui: o preambulo do `quality_ladder` acima ja rodou.
import comfy.model_management as comfy_mm  # noqa: E402
import comfy.sample as comfy_sample  # noqa: E402
import comfy.sd as comfy_sd  # noqa: E402
import folder_paths  # noqa: E402

# Dois prompts que nao compartilham assunto, iluminacao, enquadramento nem paleta. Se o modelo
# esta ouvindo, estes dois nao podem produzir o mesmo latente.
PROMPT_A = ("a cinematic shot of an elderly watchmaker in a small workshop, sitting at a walnut "
            "workbench, warm window light, brass gears on the table, photorealistic")
PROMPT_B = ("an empty snow-covered parking lot at night under a single sodium streetlight, "
            "falling snow, no people, wide static shot, cold blue tones")

LIMIAR_REPROVA = 0.5
"""Pre-registrado em `bench/criterio_guarda_referencia.md`, antes de medir. **Validado**: cai entre
0,2477 (destruido) e 0,7911 (o sadio mais proximo), com 3,19x de folga, e nao rejeitou nenhum dos
tres bracos sadios medidos. Nao mexer neste numero depois de ver um resultado: um limiar escolhido
depois de ver os numeros nao e uma guarda, e uma descricao."""

LIMIAR_SAUDAVEL = 0.8
"""Acima disto a previsao escrita antes dizia "braco sao". **A previsao errou por 1,1%** -- o Wan
bom deu 0,7911 e caiu em `OLHAR` -- e o limiar continua 0,8 exatamente por isso. Um limite superior
defensavel precisa de bracos sadios NOVOS, medidos depois, nao dos mesmos que testaram este. O
custo do erro e pequeno: um braco sao ouve "olhe a imagem", que nesta bancada e sempre verdade."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True, help="checkpoint NAO quantizado, em diffusion_models")
    p.add_argument("--clip", nargs="+", required=True)
    p.add_argument("--clip-type", default="lumina2")
    p.add_argument("--clip-device", choices=["default", "cpu"], default="default")
    p.add_argument("--negative", default="")
    p.add_argument("--seeds", type=int, nargs=2, default=[1, 2])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--vace-strength", type=float, default=None)
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--distorch", default=None)
    p.add_argument("--distorch-compute", default="cuda:0")
    p.add_argument("--rotulo", default=None, help="nome deste braco no relatorio")
    p.add_argument("--json", type=Path, default=None, help="acumula o laudo neste arquivo")
    return p.parse_args()


def distancia(a: torch.Tensor, b: torch.Tensor) -> float:
    """Distancia relativa, na direcao de `a`. Assimetrica de proposito e sempre com o mesmo
    denominador nas duas pontas de uma comparacao, senao a razao mede a escolha do denominador."""
    return float((b - a).norm() / a.norm())


def medir(args) -> dict:
    conditioning = ql.encode(args, folder_paths, comfy_sd, comfy_mm, [PROMPT_A, PROMPT_B])
    latentes, por_passo, gib = ql.sample_all(
        args, args.modelo, conditioning, comfy_sample, comfy_sd, comfy_mm, folder_paths)

    s1, s2 = args.seeds
    # (indice_do_prompt, semente). sample_all devolve exatamente estas quatro.
    a, b, c, d = (0, s1), (1, s1), (0, s2), (1, s2)

    d_prompt = [distancia(latentes[a], latentes[b]), distancia(latentes[c], latentes[d])]
    d_semente = [distancia(latentes[a], latentes[c]), distancia(latentes[b], latentes[d])]
    # Uma razao por semente, emparelhando o numerador e o denominador que compartilham a semente
    # s1 -- e nao a media de uma sobre a media da outra, que perderia o pareamento.
    razoes = [d_prompt[0] / d_semente[0], d_prompt[1] / d_semente[1]]
    resposta = statistics.mean(razoes)

    return {
        "modelo": args.modelo,
        "rotulo": args.rotulo or args.modelo,
        "gib": round(gib, 2),
        "vace_strength": args.vace_strength,
        "resposta": round(resposta, 4),
        "resposta_por_semente": [round(r, 4) for r in razoes],
        "espalhamento_da_resposta": round(max(razoes) / min(razoes), 3) if min(razoes) else None,
        "d_prompt": [round(x, 4) for x in d_prompt],
        "d_semente": [round(x, 4) for x in d_semente],
        "norma_latente": {f"prompt{i}_semente{s}": round(float(latentes[(i, s)].norm()), 1)
                          for i in (0, 1) for s in (s1, s2)},
        "s_por_passo": round(statistics.mean(por_passo), 4),
        "condicoes": {"steps": args.steps, "cfg": args.cfg, "size": args.size,
                      "frames": args.frames, "sampler": args.sampler,
                      "scheduler": args.scheduler, "seeds": args.seeds},
    }


def veredito(resposta: float) -> tuple[str, str]:
    if resposta < LIMIAR_REPROVA:
        return ("REPROVADO", "o prompt quase nao move a saida em relacao ao que a semente move: "
                             "o condicionamento nao esta chegando ao modelo. Nao acredite em "
                             "nenhum numero medido contra esta referencia")
    if resposta >= LIMIAR_SAUDAVEL:
        return ("SEM VEREDITO", "o braco responde ao prompt. Isto NAO diz que a imagem esta boa -- "
                                "so que esta classe de quebra nao esta presente")
    return ("OLHAR", f"entre os dois limiares pre-registrados ({LIMIAR_REPROVA} e "
                     f"{LIMIAR_SAUDAVEL}); a guarda nao tem o que dizer aqui. Olhe a imagem")


NAO_COBERTO = [
    "Detecta UMA classe de quebra -- condicionamento que nao chega. Um modelo pode ouvir o prompt",
    "  perfeitamente e ainda assim gerar lixo; isto nao substitui olhar a imagem.",
    "Mede a REFERENCIA, nao o braco quantizado. E sobre o arquivo que ninguem confere.",
    "Uma placa, um scheduler, um tamanho, um numero de passos, dois prompts, duas sementes.",
    "Os dois limiares valem para o par de verdade conhecida do Wan e para os bracos sao medidos",
    "  ate agora. Um modelo cuja saida legitimamente varia pouco com o prompt (um refinador, um",
    "  upscaler, um modelo de controle) reprovaria aqui sem estar quebrado.",
    "Nenhuma imagem e decodificada: a medida e no latente.",
]


def main() -> int:
    args = parse_args()
    from _bench_guard import BenchGuard
    with BenchGuard("bench:guarda_referencia") as guarda:
        if guarda.refused:
            # Imprimir o motivo nao e cosmetico. Na primeira execucao esta ferramenta saiu com
            # codigo 1 e DUAS linhas de saida ("lock adquirido", "lock solto") -- o mesmo defeito
            # de cegueira silenciosa que a camada 1 tinha com o segundo dialeto. A recusa era
            # legitima (o 3080 Ti tinha 2,8 GiB de outro trabalho) e a saida documentada e
            # `CUDA_VISIBLE_DEVICES`, que tira a placa da run e da guarda ao mesmo tempo.
            print(guarda.refused)
            print("\nSe a placa que ESTE trabalho vai usar esta ociosa, rode com "
                  "CUDA_VISIBLE_DEVICES apontando so para ela.")
            return 1

        laudo = medir(args)

    laudo["veredito"], laudo["motivo"] = veredito(laudo["resposta"])
    laudo["limiares"] = {"reprova_abaixo": LIMIAR_REPROVA, "sao_acima": LIMIAR_SAUDAVEL,
                         "pre_registrados_em": "bench/criterio_guarda_referencia.md"}
    laudo["nao_coberto"] = NAO_COBERTO

    print(f"\n=== {laudo['rotulo']} ===")
    print(f"  d_prompt   {laudo['d_prompt']}   (quanto o PROMPT move)")
    print(f"  d_semente  {laudo['d_semente']}   (quanto a SEMENTE move -- controle negativo)")
    print(f"  resposta   {laudo['resposta']}   por semente {laudo['resposta_por_semente']}, "
          f"espalhamento {laudo['espalhamento_da_resposta']}x")
    print(f"  |latente|  {laudo['norma_latente']}")
    print(f"  {laudo['veredito']}: {laudo['motivo']}")

    if args.json:
        anterior = []
        if args.json.is_file():
            try:
                anterior = json.loads(args.json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                anterior = []
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([*anterior, laudo], indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"  acumulado em {args.json}")

    print("\nNAO COBERTO POR ESTA EXECUCAO:")
    for linha in NAO_COBERTO:
        print(f"  - {linha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
