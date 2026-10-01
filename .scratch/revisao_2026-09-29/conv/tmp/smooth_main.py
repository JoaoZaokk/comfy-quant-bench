def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"No such source: {source}")
    output = (args.output or source.with_name(f"{source.stem}_w4a4_smooth.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")

    # A checagem de CUDA vinha AQUI, antes de toda recusa: sem placa visivel nenhuma guarda de
    # entrada rodava, e `test_smooth_guards.py` (seis recusas e um controle, todos de cabecalho)
    # falhava inteiro por ambiente. Ela desceu para logo antes do preflight de backend, que e o
    # primeiro passo que de fato precisa da placa (revisao 2026-09-29, achado 10).
    conv = C.Conversion(source, output, sidecar)
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata

    layer_ids = sorted({int(LAYER_RE.match(k).group(1)) for k in header if LAYER_RE.match(k)})
    selected, norm_keys = [], []
    for layer in layer_ids:
        for norm, members in GROUPS.items():
            norm_keys.append(f"model.layers.{layer}.{norm}.weight")
            selected += [f"model.layers.{layer}.{m}.weight" for m in members]
        selected += [f"model.layers.{layer}.{m}.weight" for m in UNSMOOTHED]
    missing = [k for k in selected + norm_keys if k not in header]
    if missing:
        raise SystemExit(f"Source lacks {len(missing)} expected tensors, e.g. {missing[0]}")
    # `missing` nao pega o caso vazio: numa fonte sem nenhuma chave `model.layers.N.`, `layer_ids`
    # sai vazia, `selected` e `norm_keys` saem vazias e `missing` tambem -- entao a checagem acima
    # aprova. Medido em 2026-09-01 apontando este conversor para um Wan 2.1: imprimiu
    # `Layers: 0   quantized: 0` e saiu com rc=0, ou seja, um `--dry-run` responde SUCESSO para
    # uma conversao que nao tem o que converter. Achado pelo caso de CONTROLE de um teste de
    # recusas, nao por um caso que procurava o defeito.
    if not selected:
        raise SystemExit(
            f"Source has no `model.layers.N.` tensors: this converter is Gemma-3-shaped "
            f"(esperava {'/'.join(GROUPS)} alimentando "
            f"{', '.join(m for members in GROUPS.values() for m in members)}). "
            f"Nada a converter em {source.name}.")
    # A checagem acima confere NOMES; nada aqui conferia o DTYPE. Medido em 2026-09-01 apontando
    # este conversor para o gemma_3_12B_it_heretic_fp8_e4m3fn: passou por toda a `refuse_unsafe`,
    # passou pelo preflight de backend, rodou os seis prompts de calibragem ate o fim (96 normas,
    # ~3 min de GPU) e so entao morreu com `KeyError: 'F8_E4M3'` dentro do leitor. O custo do erro
    # tardio e o trabalho jogado fora; a informacao para recusar ja estava no cabecalho.
    ilegiveis = sorted({header[k]["dtype"] for k in selected + norm_keys
                        if header[k]["dtype"] not in HIGH_PRECISION_DTYPES})
    if ilegiveis:
        raise SystemExit(
            f"Source carries dtypes this converter cannot read: {', '.join(ilegiveis)}. "
            f"Aceita {', '.join(sorted(HIGH_PRECISION_DTYPES))} -- passe o checkpoint de alta "
            f"precisao, nao uma versao ja comprimida.")

    print(f"Source: {source}")
    print(f"Layers: {len(layer_ids)}   quantized: {len(selected)}   "
          f"smoothed: {len(selected) - 2 * len(layer_ids)}   alpha: {args.alpha}")
    print(f"Output: {output}")
    if args.dry_run:
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    # This writes the same `convrot_w4a4` format that quant_w4a4.py hard-refuses to produce
    # without the CUDA backend. The eager backend's output is structurally identical -- same
    # packed int4 container, same per-row f32 scales -- so neither verify_w4a4.py nor
    # inspect_quant.py can tell the two apart afterwards. A checkpoint quantized by eager and
    # believed to be CUDA is exactly the failure the whole preflight exists to prevent.
    from _native_probe import native_backend_ready
    from quant_w4a4 import w4a4_probe_ops

    backend = native_backend_ready(PORTABLE_ROOT, w4a4_probe_ops(args.convrot_groupsize))
    if not backend.get("native_ready"):
        raise SystemExit(
            "Refusing: normal ComfyUI resolves the ConvRot ops to "
            + ", ".join(f"{op}={module}"
                        for op, module in sorted(backend["resolved"].items()))
            + ", not a CUDA backend. Producing convrot_w4a4 from the eager path yields a file "
              "nothing downstream can distinguish from a real one.")
    print(f"Backend: {backend['resolved']['quantize_convrot_w4a4_weight']}")

    fmt = F.ConvrotW4A4(args.convrot_groupsize)
    formats = {key: fmt for key in selected}
    out_meta = F.quant_metadata(metadata, F.layer_configs(formats), "ConvRot W4A4",
                                {"smoothquant_alpha": str(args.alpha)})
    norm_set = set(norm_keys)
    # Membro suavizado -> (camada, norma); a norma aponta para o proprio grupo.
    grupo_de: dict[str, tuple[int, str]] = {}
    for layer in layer_ids:
        for norm, members in GROUPS.items():
            grupo_de[f"model.layers.{layer}.{norm}.weight"] = (layer, norm)
            for m in members:
                grupo_de[f"model.layers.{layer}.{m}.weight"] = (layer, norm)

    # Transmite desde 2026-09-29. Antes esta era a ferramenta que MAIS acumulava -- pesos
    # quantizados, escalas e as normas reescritas de todas as camadas -- e a guarda de RAM rodava
    # DEPOIS do acumulo (e da calibragem), onde so podia jogar o trabalho fora. Agora o pico e UM
    # grupo de norma (o maior: q/k/v ou gate/up) mais o que ja saiu dele e ainda espera a vez no
    # header, e as guardas rodam antes da calibragem.
    grupo_bytes = max(sum(header[f"model.layers.{layer}.{m}.weight"]["data_offsets"][1]
                          - header[f"model.layers.{layer}.{m}.weight"]["data_offsets"][0]
                          for m in members)
                      for layer in layer_ids for members in GROUPS.values())
    estado: dict = {"stats": None}
    prontos: dict[str, list[torch.Tensor]] = {}
    lambda_report: list[dict] = []

    import comfy_kitchen as ck

    with conv.tensors() as fonte:
        def calcula_grupo(layer: int, norm: str) -> None:
            stats = estado["stats"]
            norm_key = f"model.layers.{layer}.{norm}.weight"
            member_keys = [f"model.layers.{layer}.{m}.weight" for m in GROUPS[norm]]
            weights = {k: fonte[k].cuda().float() for k in member_keys}
            act_max = stats[norm_key].cuda().clamp(min=1e-5)
            # One lambda per norm: weight_max spans every consumer of that norm.
            weight_max = torch.stack([w.abs().amax(dim=0) for w in weights.values()]).amax(0)
            lam = (act_max.pow(args.alpha) / weight_max.clamp(min=1e-5).pow(1 - args.alpha)
                   ).clamp(min=1e-5)
            for key, weight in weights.items():
                # FP32 na entrada do quantizador desde 2026-09-29 (decisao do dono), como os
                # outros conversores desde 26/09; ate ali este era o unico que ainda passava
                # `.to(torch.bfloat16)` antes da rotacao. Muda os bytes de conversoes FUTURAS.
                prontos[key], _ = fmt.quantize(weight * lam, ck)
            norm_weight = fonte[norm_key].cuda().float()
            # Gemma's RMSNorm applies (1 + w), so the fold has to go through that offset.
            prontos[norm_key] = [((norm_weight + 1.0) / lam - 1.0).to(torch.bfloat16).cpu()]
            lambda_report.append({
                "norm": norm_key,
                "act_channel_ratio_before": (act_max.max() / act_max.median()).item(),
                "act_channel_ratio_after": ((act_max / lam).max() / (act_max / lam).median()).item(),
            })
            del weights, act_max, weight_max, lam, norm_weight
            torch.cuda.empty_cache()

        def produtor(key: str, i: int):
            def produz() -> torch.Tensor:
                if key not in prontos:
                    if key in grupo_de:
                        calcula_grupo(*grupo_de[key])
                    else:  # o_proj / down_proj: quantizados, nunca suavizados
                        prontos[key], _ = fmt.quantize(fonte[key].cuda().float(), ck)
                tensor = prontos[key][i]
                prontos[key][i] = None
                if all(t is None for t in prontos[key]):
                    del prontos[key]
                return tensor
            return produz

        # Tres casos, na mesma ordem de antes: camada quantizada (peso + escala), norma reescrita,
        # ou copia verbatim.
        entradas = []
        for name, info in header.items():
            if name in formats:
                for i, (key, dtype, shape) in enumerate(fmt.tensors(name, info["shape"])):
                    entradas.append(C.plan_lazy(key, dtype, shape, C.nbytes_of(dtype, shape),
                                                produtor(name, i)))
            elif name in norm_set:
                entradas.append(C.plan_lazy(name, "BF16", info["shape"],
                                            C.nbytes_of("BF16", info["shape"]), produtor(name, 0)))
            else:
                entradas.append(C.plan_copy(name, info))
        conv.guard(conv.planned_size(entradas, out_meta), accumulated=3 * grupo_bytes,
                   label="W4A4 SmoothQuant (streaming)")

        estado["stats"] = stats = calibrate(args)
        if len(stats) != len(norm_keys):
            raise SystemExit(f"Calibration saw {len(stats)} norms, expected {len(norm_keys)}")

        started = time.perf_counter()

        def progresso(indice: int, total: int, chave: str) -> None:
            if chave in formats and (indice % 64 == 0 or indice == total):
                print(f"[{indice}/{total}] tensors written", flush=True)

        def manifesto() -> dict:
            before = sum(r["act_channel_ratio_before"] for r in lambda_report) / len(lambda_report)
            after = sum(r["act_channel_ratio_after"] for r in lambda_report) / len(lambda_report)
            estado["ratios"] = (before, after)
            return {
                "source": str(source), "output": str(output), "output_size": conv.output_size,
                "quantization": "ConvRot W4A4 + SmoothQuant", "smoothquant_alpha": args.alpha,
                "calibrated_with": args.calibrate_with,
                "calibration_note": "activations captured from the W4A4 checkpoint with weights "
                                    "retyped so ComfyUI dequantizes them: 4-bit weights, "
                                    "full-precision activations",
                "calibration_prompts": len(CALIBRATION_PROMPTS),
                "smoothed_projections": sorted({m for members in GROUPS.values() for m in members}),
                "quantized_unsmoothed": list(UNSMOOTHED),
                "quantized_tensors": len(selected),
                "act_channel_ratio_before": round(before, 2), "act_channel_ratio_after": round(after, 2),
                "convrot_groupsize": args.convrot_groupsize,
                # Registrado desde 2026-09-29: a entrada do quantizador mudou de BF16 para FP32,
                # entao dois arquivos deste conversor so sao comparaveis se este campo bater.
                "quantizer_input": F.QUANTIZER_INPUT,
                # quant_w4a4.py records this and this file did not, so its outputs were the only
                # convrot_w4a4 checkpoints in the project with no record of which backend produced them.
                "backend": backend["resolved"]["quantize_convrot_w4a4_weight"],
                "backend_linear": backend["resolved"]["convrot_w4a4_linear"],
                "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "conversion_seconds": round(time.perf_counter() - started, 3),
            }

        conv.commit(entradas, out_meta, progress=progresso, sidecar=manifesto)

    before, after = estado["ratios"]
    print(f"activation channel outlier ratio, mean over {len(lambda_report)} norms: "
          f"{before:.0f} -> {after:.1f}")
    elapsed = time.perf_counter() - started
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}) in {elapsed:.1f} s")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
