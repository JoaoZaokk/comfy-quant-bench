# ---- awq_w4a16 ------------------------------------------------------------------------------
# Escrito por quant_awq_w4a16.py (`_formats.AwqW4A16`): codigos Q4_1 reempacotados.
#   <l>.weight        I8   [N, K/2]
#   <l>.weight_scale  BF16 [K/G, N]    d
#   <l>.weight_zeros  BF16 [K/G, N]    m + 8 d
# Ate 2026-09-29 este verificador nao conhecia o formato e recusava todo arquivo do awq como
# "unexpected format" (revisao 2026-09-29, achado 7). So a estrutura: o smoke e o preflight de
# backend nao cobrem este formato (ver `linear_op=None` abaixo).
def _awq_tensors(layer: str, config: dict) -> dict[str, TensorSpec]:
    group = config.get("group_size")
    if not group:
        raise SystemExit(
            f"{layer}: awq_w4a16 metadata has no 'group_size', so the expected shape of its "
            f"scales cannot be computed. Metadata present: {sorted(config)}.")

    def por_grupo(packed: list[int], cfg: dict) -> list[list[int]]:
        return [[(packed[1] * 2) // int(group), packed[0]]]

    return {
        f"{layer}.weight": TensorSpec(frozenset({"I8"}), 2),
        f"{layer}.weight_scale": TensorSpec(frozenset({"BF16", "F16", "F32"}), 2, shapes=por_grupo),
        f"{layer}.weight_zeros": TensorSpec(frozenset({"BF16", "F16", "F32"}), 2, shapes=por_grupo),
    }


# ---- float8_e4m3fn --------------------------------------------------------------------------
# Escrito por quant_te_residuos.py --format fp8 (embedding e projecao de texto): peso F8_E4M3 na
# forma da fonte e UMA escala F32 por tensor. So a estrutura, como o awq.
def _fp8_tensors(layer: str, config: dict) -> dict[str, TensorSpec]:  # noqa: ARG001
    return {
        f"{layer}.weight": TensorSpec(frozenset({"F8_E4M3"}), 2),
        f"{layer}.weight_scale": TensorSpec(frozenset({"F32", "F16", "BF16"}),
                                            shapes=lambda packed, cfg: [[], [1]]),
    }


def _sem_probe(config: dict, present: frozenset[str]) -> dict:  # noqa: ARG001
    return {}


