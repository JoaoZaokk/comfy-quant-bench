from pathlib import Path

p = Path("F:/COMFY_PORTABLE/tools/grava_pesos_recuperados.py")
s = p.read_text(encoding="utf-8")


def troca(velho: str, novo: str) -> None:
    global s
    assert s.count(velho) == 1, velho
    s = s.replace(velho, novo)


troca("import argparse\nimport json\nimport struct\nimport sys\nfrom pathlib import Path\n",
      "import argparse\nimport sys\nfrom pathlib import Path\n")
troca('sys.path.insert(0, str(RAIZ / "ComfyUI"))\n\nimport torch\nfrom safetensors.torch import save_file\n',
      'sys.path.insert(0, str(RAIZ / "ComfyUI"))\nsys.path.insert(0, str(RAIZ / "tools"))\n\n'
      'import torch  # noqa: E402\n\nimport _conversion as C  # noqa: E402\n')
i = s.index("def cabecalho(modelo: Path)")
j = s.index("def mascara_elemento(")
s = s[:i] + s[j:]
troca('''    if a.saida.exists():
        raise SystemExit(f"{a.saida} ja existe; este script nao sobrescreve")
''', '''    # Pelo nucleo desde 2026-09-29 (revisao, achado 2): gravava com `save_file` DIRETO no nome
    # final -- sem `.partial`, sem fsync, e uma queda no meio deixava um arquivo truncado com o nome
    # definitivo. Recusas (saida existente, parcial antigo, saida == modelo) agora antes do calculo.
    conv = C.Conversion(a.modelo, a.saida)
    conv.refuse_unsafe(allow_quantized_source=True)
''')
troca("    head, base = cabecalho(a.modelo)\n", "    head = conv.header\n    fonte = conv.tensors()\n")
troca("        w = ler_peso(a.modelo, head, base, chave)\n", "        w = fonte[chave].cuda()\n")
troca('''    a.saida.parent.mkdir(parents=True, exist_ok=True)
    save_file(saida, str(a.saida), metadata=meta)
''', '''    a.saida.parent.mkdir(parents=True, exist_ok=True)
    # Mesma ordem e mesmo `__metadata__` ordenado que o `save_file` usava, entao os bytes nao mudam.
    entradas = C.save_file_order(C.plan_write(k, t) for k, t in saida.items())
    meta = C.save_file_metadata(meta)
    conv.guard(conv.planned_size(entradas, meta))
    conv.commit(entradas, meta)
''')
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
