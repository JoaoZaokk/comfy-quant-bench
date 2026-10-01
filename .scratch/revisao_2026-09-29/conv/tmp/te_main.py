def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("--input precisa ser um .safetensors existente")

    header, metadata = C.read_header(source)
    selecionados = [n for n, i in header.items()
                    if ALVO.fullmatch(n) and i["dtype"] in HIGH_PRECISION_DTYPES and len(i["shape"]) == 2]
    if not selecionados:
        raise SystemExit("nenhuma embedding/projecao em alta precisao nesta fonte")
    quant_meta = json.loads(metadata.get("_quantization_metadata", '{"format_version": "1.0", "layers": {}}'))
    ja = [n for n in selecionados if n.removesuffix(".weight") in quant_meta["layers"]]
    if ja:
        raise SystemExit(f"recusado: ja declarados quantizados no metadata: {ja}")

    output = (args.output or saida_padrao(source, selecionados, args.format)).resolve()
    sidecar = output.with_suffix(".quant.json")
    conv = C.Conversion(source, output, sidecar)
    # a fonte PODE ser quantizada (o Gemma W4A8); os tensores tocados nao, conferido acima
    conv.refuse_unsafe(allow_quantized_source=True)

    print(f"fonte: {source}\nsaida: {output}\nformato: {FORMATO[args.format]}")
    for n in selecionados:
        print(f"  {n}: {header[n]['shape']} {header[n]['dtype']}")
    if args.dry_run:
        return 0

    for n in selecionados:
        quant_meta["layers"][n.removesuffix(".weight")] = {"format": FORMATO[args.format]}
    out_meta = dict(metadata)
    out_meta["_quantization_metadata"] = json.dumps(quant_meta, separators=(",", ":"))
    out_meta["te_residuos"] = FORMATO[args.format]

    inicio = time.perf_counter()
    medidas: dict[str, dict] = {}
    escalas: dict[str, torch.Tensor] = {}
    selecionados_set = set(selecionados)

    with conv.tensors() as fonte:
        def produz_peso(n: str):
            def f() -> torch.Tensor:
                peso = fonte[n]
                q, escala = quantiza(peso, args.format)
                medidas[n] = confere(n, peso, q, escala, args.format)
                print(f"  {n}: {medidas[n]}", flush=True)
                escalas[n] = escala.contiguous()
                return q.contiguous()
            return f

        def produz_escala(n: str):
            def f() -> torch.Tensor:
                if n not in escalas:
                    raise RuntimeError(f"{n}_scale: produtor chamado antes do peso")
                return escalas.pop(n)
            return f

        # Transmite desde 2026-09-29: um tensor quantizado por vez, dentro do laco de escrita. As
        # formas sao analiticas -- int8: I8 [N, K] + escala por linha F32 [N, 1]; fp8: peso F8_E4M3
        # [N, K] + UMA escala F32 escalar [] -- e o nucleo confere forma e dtype de cada uma.
        entradas = []
        for n, info in header.items():
            if n not in selecionados_set:
                entradas.append(C.plan_copy(n, info))
                continue
            linhas, colunas = info["shape"]
            # PESO fp8 com dtype fp8 no header (ver ARMADILHA no topo); plan_write o gravaria como U8
            dt_peso = "F8_E4M3" if args.format == "fp8" else "I8"
            forma_escala = [] if args.format == "fp8" else [linhas, 1]
            entradas.append(C.plan_lazy(n, dt_peso, [linhas, colunas], linhas * colunas, produz_peso(n)))
            entradas.append(C.plan_lazy(f"{n}_scale", "F32", forma_escala,
                                        C.nbytes_of("F32", forma_escala), produz_escala(n)))

        maior = max(header[n]["shape"][0] * header[n]["shape"][1] for n in selecionados)
        # fonte bf16 + saida 1 byte do maior tensor, mais um bloco fp32 de trabalho
        conv.guard(conv.planned_size(entradas, out_meta), accumulated=maior * 3 + LINHAS_POR_BLOCO * 16,
                   label="quant residuos TE")

        def manifesto() -> str:
            return json.dumps({
                "source": str(source), "source_size": source.stat().st_size,
                "output": str(output), "output_size": conv.output_size,
                "quantization": FORMATO[args.format],
                "escala": "por linha" if args.format == "int8" else "por tensor",
                "quantized_tensors": selecionados,
                "erro_contra_fonte": medidas,
                "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
                "torch_version": torch.__version__,
                "conversion_seconds": round(time.perf_counter() - inicio, 3),
            }, indent=2, ensure_ascii=False)

        conv.commit(entradas, out_meta, sidecar=manifesto)

    segundos = time.perf_counter() - inicio
    print(f"gravado {output} ({C.human_size(output.stat().st_size)}) em {segundos:.0f} s\ngravado {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
