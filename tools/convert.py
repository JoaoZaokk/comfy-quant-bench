"""Uma porta so para os conversores de checkpoint desta bancada.

    python_embeded\\python.exe -s tools/convert.py w4a4  --input modelo.safetensors
    python_embeded\\python.exe -s tools/convert.py w4a8  --input modelo.safetensors
    python_embeded\\python.exe -s tools/convert.py int8  --input modelo.safetensors --no-convrot
    python_embeded\\python.exe -s tools/convert.py mixed --input modelo.safetensors --calibration x.calib.pt
    python_embeded\\python.exe -s tools/convert.py smooth --source g.safetensors --calibrate-with c.safetensors
    python_embeded\\python.exe -s tools/convert.py native --input z.safetensors --output n.safetensors --arch zimage
    python_embeded\\python.exe -s tools/convert.py from-svdq --input svdq.safetensors --output bf16.safetensors
    python_embeded\\python.exe -s tools/convert.py awq --input modelo.safetensors
    python_embeded\\python.exe -s tools/convert.py weight-only --input modelo_w4a8.safetensors
    python_embeded\\python.exe -s tools/convert.py te-residuos --input gemma_w4a8.safetensors --format int8

    python_embeded\\python.exe -s tools/convert.py --list          # o que cada um faz
    python_embeded\\python.exe -s tools/convert.py w4a4 --help     # as flags daquele

POR QUE SUBCOMANDO E NAO FLAG. O pedido original era `--w4a4` escolhendo a funcao. Nao da, e o
motivo e medido, nao estetico: os namespaces de `--profile` sao DISJUNTOS entre as ferramentas.
EXECUTADO em 2026-08-31:

    convert.py mixed --profile gemma    -> invalid choice (so zimage, ltx_2_5, hunyuan_video_15, wan_2_1)
    convert.py w4a8  --profile zimage   -> invalid choice (so auto, ltx_2_5, hunyuan_video_15, gemma, qwen)

Um argparse plano nao expressa "esta flag aceita estes valores QUANDO aquela outra flag esta
presente". Viraria validacao pos-parse, que e o mesmo defeito com outro nome. Ha tambem quatro
grafias do mesmo conceito na arvore (`--profile`, `--arch`, `--auto-detect`, e o `--profile` sem
`auto` do mixed), e `--output` e obrigatorio em dois dos sete e opcional em cinco.

POR QUE OS SETE ARQUIVOS CONTINUAM EXISTINDO. Nao e conservadorismo: EXECUTADO, oito arquivos
importam esses modulos pelo nome, e quebram se sumirem --

    plot_weight_balance.py, svdquant_probe.py, weight_balance.py, test_quant_mixed_sigma.py,
    test_quant_mixed_provenance.py, test_svdq_write_contract.py, test_native_probe.py,
    test_svdq_verify.py

e `test_svdq_write_contract.py:229` lista os SETE nomes de arquivo literalmente. Este despachante
e a porta da frente; os modulos seguem sendo os modulos.

O CONTRATO DE ESCRITA E UM SO, E ADOTADO. Esta frase ja disse duas coisas erradas: primeiro que o
contrato "virou um so, em `_conversion.py`" quando nenhum conversor o usava; depois, em 2026-09-01,
que "o unico modulo que o importa e o proprio teste dele" -- e isso tambem envelheceu. Contado em
2026-09-29: todo conversor listado abaixo escreve por `_conversion.Conversion.commit` (sidecar
incluso, no mesmo commit atomico), seleciona camadas por `_profiles` e, os que quantizam, montam as
camadas por `_formats`. Nenhum carrega mais `.partial`, `os.replace` ou `fsync` proprios.

NAO COBERTO: este arquivo nao valida nada e nao converte nada -- so encaminha. Toda recusa,
guarda e verificacao continua onde sempre esteve, na ferramenta escolhida. Ele tambem nao
uniformiza `--device`: `int8` roda em CPU por padrao e de proposito.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

# subcomando -> (modulo, uma linha do que faz)
COMANDOS: dict[str, tuple[str, str]] = {
    "w4a4": ("quant_w4a4",
             "ConvRot W4A4: peso 4 bits, ativacao 4 bits, MMA nativa na sm86/89. "
             "Rapido em lote grande, e o menos fiel dos dois ramos."),
    "w4a8": ("quant_w4a8",
             "asym_w4a8_int8: peso 4 bits, matmul pelo caminho INT8, com codebook Lloyd-Max. "
             "O padrao recomendado."),
    "int8": ("quant_int8",
             "int8_tensorwise, com ou sem rotacao Hadamard. Unico que roda em CPU por padrao."),
    "mixed": ("quant_mixed",
              "Escolhe w4a4 / w4a8 / sem quantizar POR CAMADA, medindo os kernels reais nas "
              "ativacoes reais de um .calib.pt."),
    "smooth": ("quant_w4a4_smooth",
               "W4A4 com SmoothQuant dobrado nas RMSNorm. So Gemma hoje."),
    "native": ("to_native",
               "Renomeia um checkpoint com nomes diffusers para os nomes de modulo do ComfyUI. "
               "NAO quantiza."),
    "from-svdq": ("svdq_to_bf16",
                  "Sentido inverso: le um checkpoint SVDQuant INT4 e recupera BF16 sondando o "
                  "kernel. Exige CUDA."),
    "awq": ("quant_awq_w4a16",
            "awq_w4a16: codigos Q4_1 do gguf-py no layout AWQ W4A16, ativacao BF16. Roda em CPU."),
    "weight-only": ("quant_weight_only",
                    "Marca as camadas de um checkpoint ja quantizado como W8A16/W4A16 "
                    "(full_precision_matrix_mult). Copia os tensores byte a byte."),
    "te-residuos": ("quant_te_residuos",
                    "Quantiza so a embedding e a projecao que ficaram BF16 num text encoder LTX "
                    "(int8 ou fp8). Roda em CPU."),
}


def listar() -> int:
    largura = max(len(c) for c in COMANDOS)
    recuo = " " * (largura + 4)
    print("conversores desta bancada:\n")
    for nome, (modulo, o_que) in COMANDOS.items():
        palavras, linha, linhas = o_que.split(), "", []
        for w in palavras:
            if len(linha) + len(w) + 1 > 74:
                linhas.append(linha)
                linha = w
            else:
                linha = f"{linha} {w}".strip()
        linhas.append(linha)
        print(f"  {nome:<{largura}}  {linhas[0]}")
        for extra in linhas[1:]:
            print(f"{recuo}{extra}")
        print(f"{recuo}({modulo}.py)\n")
    print("As flags de cada um: convert.py <subcomando> --help")
    print("\nNAO COBERTO: esta lista e escrita a mao neste arquivo. Se um conversor novo entrar "
          "em tools/ e ninguem o adicionar aqui, ele nao aparece -- o despachante nao varre o "
          "diretorio.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        p = argparse.ArgumentParser(prog="convert.py", description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("subcomando", choices=list(COMANDOS), help="qual conversor")
        p.add_argument("--list", action="store_true", help="o que cada conversor faz")
        p.print_help()
        return 0
    if argv[0] == "--list":
        return listar()

    nome = argv[0]
    if nome not in COMANDOS:
        # aceita as grafias com traco que alguem tentaria primeiro
        alt = nome.lstrip("-")
        if alt in COMANDOS:
            nome = alt
        else:
            print(f"subcomando desconhecido: {nome!r}", file=sys.stderr)
            print(f"conhecidos: {', '.join(COMANDOS)}", file=sys.stderr)
            print("convert.py --list  para o que cada um faz", file=sys.stderr)
            return 2

    modulo_nome, _ = COMANDOS[nome]
    modulo = importlib.import_module(modulo_nome)
    if not hasattr(modulo, "main"):
        print(f"{modulo_nome}.py nao expoe main()", file=sys.stderr)
        return 2

    # Cada conversor le sys.argv no proprio parse_args(). Encaminhar e trocar argv pelo tempo da
    # chamada -- explicitamente, e devolvendo no finally, porque um SystemExit no meio nao pode
    # deixar o argv do processo alterado para quem importar este modulo.
    salvo = sys.argv
    sys.argv = [f"{modulo_nome}.py", *argv[1:]]
    try:
        return int(modulo.main() or 0)
    finally:
        sys.argv = salvo


if __name__ == "__main__":
    raise SystemExit(main())
