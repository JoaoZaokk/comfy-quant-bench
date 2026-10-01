from pathlib import Path

p = Path("F:/COMFY_PORTABLE/tools/verify_w4a4.py")
s = p.read_text(encoding="utf-8")


def troca(velho: str, novo: str) -> None:
    global s
    assert s.count(velho) == 1, velho[:90]
    s = s.replace(velho, novo)


troca("""Three output formats produced by this bench are understood:

    convrot_w4a4     quant_w4a4.py, quant_w4a4_smooth.py, quant_mixed.py
    asym_w4a8_int8   quant_w4a8.py, quant_mixed.py
    int8_tensorwise  quant_int8.py  (both --convrot and --no-convrot)
""", """Five output formats produced by this bench are understood:

    convrot_w4a4     quant_w4a4.py, quant_w4a4_smooth.py, quant_mixed.py
    asym_w4a8_int8   quant_w4a8.py, quant_mixed.py
    int8_tensorwise  quant_int8.py  (both --convrot and --no-convrot), quant_te_residuos.py --format int8
    awq_w4a16        quant_awq_w4a16.py            (structure and source bytes only; no smoke)
    float8_e4m3fn    quant_te_residuos.py --format fp8   (structure and source bytes only; no smoke)

The format NAMES come from `_formats.py`, the same module the writers build their layers with, and
`tools/test_formats_e2e.py` checks that every tensor a writer plans passes the spec below. The specs
here stay deliberately wider than what our writers emit (other producers' dtypes and scale ranks are
accepted), so the relation is writer-output SUBSET-OF verifier-acceptance, not equality.
""")

troca("""from _native_probe import (  # noqa: E402
    LOADER_LINEAR_DTYPE,
    LOADER_QUANT_GROUP_SIZE,
    native_backend_ready,
)
""", """import _conversion as C  # noqa: E402
import _formats as QF  # noqa: E402
from _native_probe import (  # noqa: E402
    LOADER_LINEAR_DTYPE,
    LOADER_QUANT_GROUP_SIZE,
    native_backend_ready,
)
""")

troca("""    # comfy-kitchen op that executes this format, used both by the backend probe and the smoke.
    linear_op: str
""", """    # comfy-kitchen op that executes this format, used both by the backend probe and the smoke.
    # None for the formats this file verifies structurally only (awq_w4a16, float8_e4m3fn): the
    # backend probe and --kernel-smoke skip them and say so in the coverage block.
    linear_op: str | None
""")
troca("""    # (layer, config, load, x, options) -> kwargs for `linear_op`
    smoke_kwargs: Callable[..., dict]
""", """    # (layer, config, load, x, options) -> kwargs for `linear_op`
    smoke_kwargs: Callable[..., dict] | None
""")

novos = Path("F:/COMFY_PORTABLE/.scratch/revisao_2026-09-29/conv/tmp/verify_novos_formatos.py").read_text(encoding="utf-8")
troca("FORMATS: dict[str, Format] = {\n", novos + "FORMATS: dict[str, Format] = {\n")
troca('''    "convrot_w4a4": Format(
        name="convrot_w4a4",''', '''    QF.ConvrotW4A4.name: Format(
        name=QF.ConvrotW4A4.name,''')
troca('''    "asym_w4a8_int8": Format(
        name="asym_w4a8_int8",''', '''    QF.AsymW4A8.name: Format(
        name=QF.AsymW4A8.name,''')
troca('''    "int8_tensorwise": Format(
        name="int8_tensorwise",''', '''    QF.Int8Tensorwise.name: Format(
        name=QF.Int8Tensorwise.name,''')
troca('''        probe_ops=_int8_probe_ops,
        smoke_kwargs=_int8_smoke,
    ),
}
''', '''        probe_ops=_int8_probe_ops,
        smoke_kwargs=_int8_smoke,
    ),
    QF.AwqW4A16.name: Format(
        name=QF.AwqW4A16.name,
        packed_shape=lambda shape: [shape[0], shape[1] // 2],
        tensors=_awq_tensors,
        linear_op=None,
        probe_ops=_sem_probe,
        smoke_kwargs=None,
    ),
    "float8_e4m3fn": Format(
        name="float8_e4m3fn",
        packed_shape=lambda shape: list(shape),
        tensors=_fp8_tensors,
        linear_op=None,
        probe_ops=_sem_probe,
        smoke_kwargs=None,
    ),
}
''')

troca('''def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(header_size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    return raw, metadata
''', '''read_header = C.read_header  # erro_por_camada.py e refina_escalas.py importam daqui
''')
troca('''def data_start(path: Path) -> int:
    with path.open("rb") as handle:
        return 8 + struct.unpack("<Q", handle.read(8))[0]
''', '''data_start = C.data_start
''')

# SmoothQuant: o gatilho passa a ser o metadado do PROPRIO arquivo, nao texto livre do sidecar.
troca('''    O gatilho e o campo `quantization` do sidecar. Um arquivo sem sidecar, ou com sidecar que nao
    diz SmoothQuant, cai no conjunto vazio e a regra estrita continua valendo para tudo -- que e o
    lado seguro: na duvida, cobra byte-identidade.
    """
    sidecar = model.with_suffix(".quant.json")
    if not sidecar.is_file() or source_header is None:
        return frozenset()
    try:
        manifesto = json.loads(sidecar.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return frozenset()
    if "smoothquant" not in str(manifesto.get("quantization", "")).lower():
        return frozenset()
''', '''    O gatilho e a chave `smoothquant_alpha` no `__metadata__` do PROPRIO arquivo, que o
    `quant_w4a4_smooth` sempre gravou. Ate 2026-09-29 era a substring "smoothquant" no campo de
    texto livre `quantization` do sidecar (revisao, achado 7): sem sidecar -- o caso que o sidecar
    fora do commit atomico produzia -- a regra estrita reprovava um arquivo bom. O sidecar continua
    valendo como segundo sinal. Nenhum dos dois: conjunto vazio e regra estrita para tudo -- o lado
    seguro, na duvida cobra byte-identidade.
    """
    if source_header is None:
        return frozenset()
    _, metadata = read_header(model)
    suavizado = "smoothquant_alpha" in metadata
    sidecar = model.with_suffix(".quant.json")
    if not suavizado and sidecar.is_file():
        try:
            manifesto = json.loads(sidecar.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            manifesto = {}
        suavizado = "smoothquant" in str(manifesto.get("quantization", "")).lower()
    if not suavizado:
        return frozenset()
''')

# probe e smoke pulam os formatos so estruturais
troca('''    ops: dict = {}
    for fmt_name, layer_name in sorted(first_layer.items()):
        present = frozenset(key[len(layer_name):] for key in header
                            if key.startswith(layer_name + "."))
        ops.update(FORMATS[fmt_name].probe_ops(layers[layer_name], present))
    return ops
''', '''    ops: dict = {}
    for fmt_name, layer_name in sorted(first_layer.items()):
        present = frozenset(key[len(layer_name):] for key in header
                            if key.startswith(layer_name + "."))
        ops.update(FORMATS[fmt_name].probe_ops(layers[layer_name], present))
    return ops


def structural_only_formats(counts: dict[str, int]) -> list[str]:
    """Formatos presentes que este verificador so confere na estrutura (sem probe e sem smoke)."""
    return sorted(name for name in counts if FORMATS[name].linear_op is None)
''')

troca('''    backend = native_backend_ready(portable_root,
                                   probe_ops_for(layers, first_layer, model_header))
    for op, module in sorted(backend["resolved"].items()):
        print(f"Normal ComfyUI backend: {op} -> {module}")
    if not backend["native_ready"]:
        print("ERROR: normal ComfyUI is not selecting the CUDA backend for every format present")
        return 1
''', '''    so_estrutura = structural_only_formats(counts)
    if so_estrutura:
        coverage.note(f"formats verified structurally only (no backend probe, no smoke): "
                      f"{', '.join(so_estrutura)}.")
    ops = probe_ops_for(layers, first_layer, model_header)
    if not ops:
        coverage.note("no format present has a backend probe here, so backend resolution did not "
                      "run at all.")
        return 0
    backend = native_backend_ready(portable_root, ops)
    for op, module in sorted(backend["resolved"].items()):
        print(f"Normal ComfyUI backend: {op} -> {module}")
    if not backend["native_ready"]:
        print("ERROR: normal ComfyUI is not selecting the CUDA backend for every format present")
        return 1
''')
troca('''    for fmt_name in sorted(counts):
        layer_name = first_layer[fmt_name]
''', '''    for fmt_name in sorted(counts):
        if FORMATS[fmt_name].linear_op is None:
            continue
        layer_name = first_layer[fmt_name]
''')
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
