def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--doador", type=Path, required=True)
    a.add_argument("--regex", required=True)
    a.add_argument("--saida", type=Path, required=True)
    a.add_argument("--esperadas", type=int, help="recusa se o regex casar outro numero de tensores")
    a.add_argument("--rtn4", type=int, metavar="GRUPO",
                   help="quantiza os tensores do doador em 4 bits RTN simulado (absmax simetrico, -7..7, "
                        "grupo no eixo K) e grava dequantizado no dtype original")
    a.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 5, 6, 8),
                   help="bits do RTN simulado de --rtn4 (niveis simetricos +-(2^(b-1)-1))")
    args = a.parse_args()

    hb, meta_b = C.read_header(args.base)
    hd, _ = C.read_header(args.doador)
    meta = meta_b or None  # sem __metadata__ na base, a saida tambem sai sem
    if set(hb) != set(hd) or any(hb[k]["shape"] != hd[k]["shape"] or hb[k]["dtype"] != hd[k]["dtype"] for k in hb):
        raise SystemExit("recusado: base e doador nao tem as mesmas chaves/shapes/dtypes")
    rx = re.compile(args.regex)
    do_doador = sorted(k for k in hb if rx.fullmatch(k))
    if args.esperadas is not None and len(do_doador) != args.esperadas:
        raise SystemExit(f"recusado: regex casou {len(do_doador)} tensores, esperado {args.esperadas}")
    # Pelo nucleo desde 2026-09-29 (revisao, achado 2): recusa saida existente, parcial antigo e
    # fonte == saida; `.partial` exclusivo, fsync, bytes conferidos, os.replace. Era uma copia a
    # mao correta do mesmo contrato.
    conv = C.Conversion(args.base, args.saida)
    conv.refuse_unsafe(allow_quantized_source=True)
    doador = C.LazyTensors(args.doador, hd)

    def rtn_do_doador(k: str):
        def produz():
            ini, fim = hd[k]["data_offsets"]
            with args.doador.open("rb") as f:
                f.seek(doador.start + ini)
                dados = rtn4(f.read(fim - ini), hd[k]["dtype"], hd[k]["shape"], args.rtn4, args.bits)
            return torch.frombuffer(bytearray(dados), dtype=C.TORCH_DTYPES[hd[k]["dtype"]]).reshape(hd[k]["shape"])
        return produz

    ordem = sorted(hb, key=lambda k: hb[k]["data_offsets"][0])
    entradas = []
    for k in ordem:
        if k not in do_doador:
            entradas.append(C.plan_copy(k, hb[k]))
        elif args.rtn4:
            entradas.append(C.plan_lazy(k, hd[k]["dtype"], hd[k]["shape"],
                                        hd[k]["data_offsets"][1] - hd[k]["data_offsets"][0], rtn_do_doador(k)))
        else:
            entradas.append(C.plan_copy_from(k, hd[k], args.doador))
    novo_meta = None
    if meta is not None:
        novo_meta = dict(meta, mistura_doador=args.doador.name, mistura_regex=args.regex,
                         **({"mistura_rtn_grupo": str(args.rtn4), "mistura_rtn_bits": str(args.bits)} if args.rtn4 else {}))
    conv.guard(conv.planned_size(entradas, novo_meta))
    conv.commit(entradas, novo_meta)
    print(f"{args.saida.name}: {len(do_doador)} tensores do doador, {len(ordem) - len(do_doador)} da base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
