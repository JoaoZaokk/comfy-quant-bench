from pathlib import Path

p = Path("F:/COMFY_PORTABLE/tools/ajusta_denso_diffusers.py")
s = p.read_text(encoding="utf-8")


def troca(velho: str, novo: str) -> None:
    global s
    assert s.count(velho) == 1, velho[:80]
    s = s.replace(velho, novo)


troca("import torch\n", "import torch\n\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n\n"
      "import _conversion as C  # noqa: E402\n")

# carrega_transformer: sem load_file (safe_open/mmap); cada tensor lido por faixa de bytes.
troca("""    from diffusers import Flux2Transformer2DModel
    from safetensors.torch import load_file
    cfg = json.loads(config.read_text(encoding="utf-8"))
    m = Flux2Transformer2DModel.from_config(cfg)
    sd = load_file(str(caminho))
""", """    from diffusers import Flux2Transformer2DModel
    cfg = json.loads(config.read_text(encoding="utf-8"))
    m = Flux2Transformer2DModel.from_config(cfg)
    # Era `load_file` (safe_open + mmap), contra a regra de streaming (revisao 2026-09-29, achado
    # 1). O modelo inteiro vai para a memoria de qualquer jeito -- e treinado --, mas lido faixa a
    # faixa, sem o mapeamento que neste host compromete ate 2x o arquivo.
    with C.LazyTensors(caminho) as fonte:
        sd = {k: fonte[k] for k in fonte}
""")

# ler_metadata: duplicava a leitura de header do nucleo.
i = s.index("def ler_metadata(p: Path)")
j = s.index("\n\n\n", i) + 3
s = s[:i] + s[j:]

# recusas ANTES do professor e do treino (antes: so no fim, depois de horas de GPU).
troca("""    a = p.parse_args()

    dev = torch.device(f"cuda:{a.device}")
""", """    a = p.parse_args()

    # Pelo nucleo desde 2026-09-29 (revisao, achado 2). A saida era recusada so DEPOIS do
    # professor e do treino, gravada com `save_file` num `.partial` sem recusa de parcial antigo e
    # sem fsync, e o relatorio `.json` com `write_text` (sobrescrevendo). Agora as recusas vem antes
    # de qualquer trabalho, e modelo + relatorio entram juntos no commit atomico.
    sai = Path(a.saida)
    conv = C.Conversion(Path(a.aluno_transformer), sai, sai.with_suffix(".json"))
    conv.refuse_unsafe(allow_quantized_source=True)

    dev = torch.device(f"cuda:{a.device}")
""")

i = s.index("    from safetensors.torch import load_file, save_file\n")
j = s.index('    sai.with_suffix(".json").write_text(json.dumps(rel, indent=2), encoding="utf-8")\n')
j_fim = j + len('    sai.with_suffix(".json").write_text(json.dumps(rel, indent=2), encoding="utf-8")\n')
trecho = s[i:j_fim]
# o dicionario `rel` fica; a escrita vira `grava_saida`.
k0 = trecho.index("    rel = {")
k1 = trecho.index('    sai.with_suffix(".json")')
rel_bloco = trecho[k0:k1]
novo = rel_bloco + ("    grava_saida(conv, treinaveis, rel)\n"
                    "    print(f\"  escrito {sai}  {sai.stat().st_size:,} B\")\n")
s = s[:i] + novo + s[j_fim:]

troca("def main() -> int:\n", '''def grava_saida(conv: C.Conversion, treinaveis, relatorio: dict) -> None:
    """O aluno com os densos treinados, mais o relatorio `.json`, num commit atomico so.

    Tudo que nao foi treinado e copiado verbatim da fonte (streaming, sem materializar o modelo);
    os treinados vao no dtype que a fonte tem para eles. A ordem dos tensores e a do `save_file`
    que este script usava (dtype, depois nome), entao o layout nao muda.

    O `__metadata__` DA ORIGEM VIAJA. `save_file` sem `metadata=` grava None, e foi assim que o
    controle `zero` saiu com os 169 tensores byte a byte identicos e o ARQUIVO diferente do braco
    0 por 32 bytes -- exatamente o `{"format": "pt"}` que o escritor de la grava. Perder isso e
    perder proveniencia num artefato que pode ser publicado. As chaves saem ordenadas: o
    `save_file` as ordenava por um HashMap de semente aleatoria (`_conversion.save_file_metadata`).
    """
    header = conv.header
    treinados = {k: v.detach().to("cpu", C.TORCH_DTYPES[header[k]["dtype"]]).clone() for k, v in treinaveis}
    entradas = [C.plan_write(k, treinados[k]) if k in treinados else C.plan_copy(k, info)
                for k, info in header.items()]
    entradas = C.save_file_order(entradas)
    meta = C.save_file_metadata(conv.metadata or None)
    conv.guard(conv.planned_size(entradas, meta))
    conv.commit(entradas, meta, sidecar=relatorio)


def main() -> int:
''')
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
