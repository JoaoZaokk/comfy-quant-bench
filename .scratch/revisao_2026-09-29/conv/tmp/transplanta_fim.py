    ordem = sorted(B.h, key=lambda k: B.h[k]["data_offsets"][0])
    # Pelo nucleo desde 2026-09-29 (revisao, achado 2). A escrita a mao abria os tres `.partial` em
    # "wb" depois de um `exists()` (janela de corrida), conferia tamanho com `assert` (que some com
    # `-O`) e deixava os parciais no disco se falhasse no meio. Custo da troca: cada variante e uma
    # passada propria sobre A e B, em vez de uma passada escrevendo as tres ao mesmo tempo.
    convs = {}
    for t in saidas:
        dest = a.dir / f"klein4b_transp_{t}_{a.rotulo_b}_bfl.safetensors"
        conv = C.Conversion(Path(a.b), dest)
        try:
            conv.refuse_unsafe(allow_quantized_source=True)
        except SystemExit as exc:
            print(f"RECUSADO: {exc}", file=sys.stderr)
            return 2
        convs[t] = conv

    def produtor(fn, k: str):
        def produz() -> torch.Tensor:
            xa = A.get(k) if k in corpo else None
            xb = B.get(k) if k in corpo else None
            by = fn(k, xa, xb)
            return torch.frombuffer(bytearray(by), dtype=C.TORCH_DTYPES[B.h[k]["dtype"]]).reshape(B.h[k]["shape"])
        return produz

    for t, (desc, fn) in saidas.items():
        conv = convs[t]
        entradas = [C.plan_lazy(k, B.h[k]["dtype"], B.h[k]["shape"],
                                B.h[k]["data_offsets"][1] - B.h[k]["data_offsets"][0], produtor(fn, k))
                    for k in ordem]
        meta = {"transplante": desc, "a": str(a.a), "b": str(a.b)}

        def progresso(i: int, _n: int, _k: str) -> None:
            if (i - 1) % 30 == 0:
                print(f"  {t} {i - 1}/{len(ordem)}", flush=True)

        conv.guard(conv.planned_size(entradas, meta))
        # O cabecalho e o do braco B (mesmos nomes, dtypes, formas e offsets); este script sempre
        # gravou o `__metadata__` DEPOIS dos tensores e com o `json.dumps` padrao -- mantido, para
        # a saida sair byte a byte igual a de antes.
        conv.commit(entradas, meta, progress=progresso, metadata_last=True, ensure_ascii=True)
        print(f"escrito {conv.output}  {conv.output.stat().st_size:,} B  ({desc})")
