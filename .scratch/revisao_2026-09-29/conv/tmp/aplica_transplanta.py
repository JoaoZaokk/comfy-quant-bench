from pathlib import Path

p = Path("F:/COMFY_PORTABLE/tools/transplanta_klein.py")
s = p.read_text(encoding="utf-8")
i = s.index('    ordem = sorted(B.h, key=lambda k: B.h[k]["data_offsets"][0])')
j = s.rindex("\n", 0, s.index("=== NAO COBERTO ===")) + 1
novo = Path("F:/COMFY_PORTABLE/.scratch/revisao_2026-09-29/conv/tmp/transplanta_fim.py").read_text(encoding="utf-8")
s = s[:i] + novo + s[j:]
s = s.replace("import argparse\nimport json\nimport os\nimport struct\nimport sys\nfrom pathlib import Path\n",
              "import argparse\nimport sys\nfrom pathlib import Path\n")
s = s.replace("from compara_codigos_bonsai import BLOCO, Leitor, pilhas_reais  # noqa: E402\n",
              "import _conversion as C  # noqa: E402\nfrom compara_codigos_bonsai import BLOCO, Leitor, pilhas_reais  # noqa: E402\n")
s = s.replace("Escrita em streaming (.partial + fsync + os.replace), um tensor por vez, sem safe_open (2x commit aqui).",
              "Escrita em streaming pelo nucleo `_conversion` (.partial exclusivo + fsync + os.replace), um tensor por\nvez, sem safe_open (2x commit aqui); uma passada por variante.")
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
