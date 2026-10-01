from pathlib import Path

p = Path("F:/COMFY_PORTABLE/tools/convert.py")
s = p.read_text(encoding="utf-8")


def troca(velho: str, novo: str) -> None:
    global s
    assert s.count(velho) == 1, velho[:80]
    s = s.replace(velho, novo)


troca("""    python_embeded\\\\python.exe -s tools/convert.py from-svdq --input svdq.safetensors --output bf16.safetensors
""", """    python_embeded\\\\python.exe -s tools/convert.py from-svdq --input svdq.safetensors --output bf16.safetensors
    python_embeded\\\\python.exe -s tools/convert.py awq --input modelo.safetensors
    python_embeded\\\\python.exe -s tools/convert.py weight-only --input modelo_w4a8.safetensors
    python_embeded\\\\python.exe -s tools/convert.py te-residuos --input gemma_w4a8.safetensors --format int8
""")
i = s.index("CUIDADO: esta frase dizia")
j = s.index("NAO COBERTO: este arquivo nao valida nada")
s = s[:i] + """O CONTRATO DE ESCRITA E UM SO, E ADOTADO. Esta frase ja disse duas coisas erradas: primeiro que o
contrato "virou um so, em `_conversion.py`" quando nenhum conversor o usava; depois, em 2026-09-01,
que "o unico modulo que o importa e o proprio teste dele" -- e isso tambem envelheceu. Contado em
2026-09-29: todo conversor listado abaixo escreve por `_conversion.Conversion.commit` (sidecar
incluso, no mesmo commit atomico), seleciona camadas por `_profiles` e, os que quantizam, montam as
camadas por `_formats`. Nenhum carrega mais `.partial`, `os.replace` ou `fsync` proprios.

""" + s[j:]
troca('''    "from-svdq": ("svdq_to_bf16",
                  "Sentido inverso: le um checkpoint SVDQuant INT4 e recupera BF16 sondando o "
                  "kernel. Exige CUDA."),
}''', '''    "from-svdq": ("svdq_to_bf16",
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
}''')
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
