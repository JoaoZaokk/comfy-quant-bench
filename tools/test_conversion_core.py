"""Exercita as oito partes do contrato de `_conversion.py` contra arquivos de verdade.

EXECUTADO, sem GPU: escreve safetensors reais num diretorio temporario e le de volta. Nao ha
mock -- o ponto de um contrato de escrita e o byte no disco, e um mock de sistema de arquivos
testaria o mock.

    python_embeded\\python.exe -s tools/test_conversion_core.py

NAO COBERTO: nao testa quantizacao nenhuma (o nucleo nao quantiza). Nao testa concorrencia --
dois processos escrevendo o mesmo `.partial` nao e um caso que este contrato trate, e o
`os.replace` do Windows nao e o mesmo do POSIX sob contencao. Nao testa arquivo maior que a RAM.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import _conversion as C  # noqa: E402

OK = 0
FALHOU = 0


def checa(condicao: bool, rotulo: str, detalhe: str = "") -> None:
    global OK, FALHOU
    if condicao:
        OK += 1
        print(f"  PASS  {rotulo}")
    else:
        FALHOU += 1
        print(f"  FAIL  {rotulo}" + (f"\n          {detalhe}" if detalhe else ""))


def levanta(fn, esperado: str, rotulo: str) -> None:
    try:
        fn()
    except SystemExit as exc:
        msg = str(exc)
        checa(esperado.lower() in msg.lower(), rotulo, f"mensagem foi: {msg!r}")
        return
    except Exception as exc:  # noqa: BLE001
        checa(False, rotulo, f"levantou {type(exc).__name__}: {exc}")
        return
    checa(False, rotulo, "nao levantou nada")


def escreve_safetensors(path: Path, tensores: dict[str, torch.Tensor],
                        metadata: dict | None = None) -> None:
    header, offset, blobs = {}, 0, []
    for k, t in tensores.items():
        b = C.as_bytes(t)
        header[k] = {"dtype": C.header_dtype(t), "shape": list(t.shape),
                     "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for b in blobs:
            f.write(b)


def le_tudo(path: Path) -> dict[str, torch.Tensor]:
    header, _ = C.read_header(path)
    d0 = C.data_start(path)
    out = {}
    with open(path, "rb") as f:
        for k, info in header.items():
            a, b = info["data_offsets"]
            out[k] = C.read_tensor(f, d0 + a, b - a, info["dtype"], info["shape"])
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="conv_core_"))
    print(f"temporario: {tmp}\n")

    fonte = tmp / "fonte.safetensors"
    original = {
        "a.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "b.weight": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
        "c.bias": torch.arange(8, dtype=torch.float16),
    }
    escreve_safetensors(fonte, original)

    print("-- leitura e ida-e-volta de dtype")
    lido = le_tudo(fonte)
    checa(set(lido) == set(original), "header traz todas as chaves")
    checa(torch.equal(lido["a.weight"], original["a.weight"]), "F32 ida-e-volta")
    checa(torch.equal(lido["b.weight"], original["b.weight"]),
          "BF16 ida-e-volta (caminho .view(int16), onde .numpy() cru levanta)")
    checa(C.header_dtype(torch.zeros(2, dtype=torch.float8_e4m3fn)) == "U8",
          "fp8_e4m3fn vira U8 no header")
    checa(C.header_dtype(torch.zeros(2, dtype=torch.float8_e5m2)) == "U8",
          "fp8_e5m2 vira U8 no header (o KeyError do ticket 24)")
    checa(len(C.as_bytes(torch.zeros(4, dtype=torch.float8_e5m2))) == 4,
          "as_bytes de fp8_e5m2 nao levanta")
    checa(C.SAFETENSORS_DTYPE.get(torch.int16) == "I16",
          "I16 esta na tabela (quant_mixed nao tinha)")

    print("\n-- parte 1: recusar saida insegura")
    levanta(lambda: C.Conversion(fonte, fonte).refuse_unsafe(),
            "overwrite the source", "recusa source == output")
    ja = tmp / "ja_existe.safetensors"
    escreve_safetensors(ja, {"x": torch.zeros(2)})
    levanta(lambda: C.Conversion(fonte, ja).refuse_unsafe(),
            "existing output", "recusa output existente")
    side = tmp / "com_sidecar.quant.json"
    side.write_text("{}", encoding="utf-8")
    levanta(lambda: C.Conversion(fonte, tmp / "com_sidecar.safetensors", side).refuse_unsafe(),
            "existing sidecar", "recusa sidecar existente")

    print("\n-- parte 2: recusar source ja quantizado, DUAS causas separadas")
    q_meta = tmp / "q_meta.safetensors"
    escreve_safetensors(q_meta, {"a.weight": torch.zeros(4, 4)},
                        {"_quantization_metadata": json.dumps({"layers": {}})})
    levanta(lambda: C.Conversion(q_meta, tmp / "o1.safetensors").refuse_unsafe(),
            "already has quantization metadata", "recusa por __metadata__")

    q_inline = tmp / "q_inline.safetensors"
    escreve_safetensors(q_inline, {"a.weight": torch.zeros(4, 4),
                                   "a.comfy_quant": torch.zeros(1, dtype=torch.uint8),
                                   "b.comfy_quant": torch.zeros(1, dtype=torch.uint8)})
    levanta(lambda: C.Conversion(q_inline, tmp / "o2.safetensors").refuse_unsafe(),
            "carries 2 inline", "recusa por marcador embutido, com a CONTAGEM")

    try:
        C.Conversion(q_inline, tmp / "o3.safetensors").refuse_unsafe(allow_quantized_source=True)
        checa(True, "allow_quantized_source deixa passar (para svdq_to_bf16)")
    except SystemExit as exc:
        checa(False, "allow_quantized_source deixa passar", str(exc))

    print("\n-- parte 3: recusar .partial obsoleto")
    alvo = tmp / "com_partial.safetensors"
    parcial = alvo.with_suffix(alvo.suffix + ".partial")
    parcial.write_bytes(b"lixo")
    levanta(lambda: C.Conversion(fonte, alvo).refuse_unsafe(),
            "stale partial", "recusa partial obsoleto")
    parcial.unlink()

    print("\n-- partes 4, 5, 6, 7: planejar, transmitir, conferir, trocar")
    saida = tmp / "saida.safetensors"
    conv = C.Conversion(fonte, saida)
    hdr = conv.header
    chamou = {"n": 0}

    def preguicoso() -> torch.Tensor:
        chamou["n"] += 1
        return torch.full((2, 2), 7.0, dtype=torch.float32)

    entradas = [
        C.plan_copy("a.weight", hdr["a.weight"]),
        C.plan_write("novo.weight", torch.arange(16, dtype=torch.int8).reshape(4, 4)),
        C.plan_lazy("preguicoso.weight", "F32", [2, 2], 16, preguicoso),
        C.plan_copy("c.bias", hdr["c.bias"]),
    ]
    checa(chamou["n"] == 0, "produtor preguicoso NAO foi chamado ao planejar")
    escritos = conv.commit(entradas, {"marca": "teste"})
    checa(chamou["n"] == 1, "produtor preguicoso chamado exatamente uma vez ao escrever")
    checa(escritos == sum(e.nbytes for e in entradas),
          "bytes escritos == bytes planejados", f"{escritos}")
    checa(saida.is_file(), "saida existe")
    checa(not conv.partial.exists(), "partial removido no sucesso")

    de_volta = le_tudo(saida)
    checa(list(de_volta) == ["a.weight", "novo.weight", "preguicoso.weight", "c.bias"],
          "ordem do header preservada")
    checa(torch.equal(de_volta["a.weight"], original["a.weight"]),
          "faixa copiada e byte-identica a fonte")
    checa(torch.equal(de_volta["c.bias"], original["c.bias"]), "segunda faixa copiada intacta")
    checa(float(de_volta["preguicoso.weight"][0, 0]) == 7.0, "tensor preguicoso chegou ao disco")
    _, meta = C.read_header(saida)
    checa(meta.get("marca") == "teste", "metadata escrito")

    offs = [json.loads(saida.read_bytes()[8:8 + struct.unpack('<Q', saida.read_bytes()[:8])[0]])[k]
            ["data_offsets"] for k in de_volta]
    contiguo = all(offs[i][1] == offs[i + 1][0] for i in range(len(offs) - 1))
    checa(offs[0][0] == 0 and contiguo, "data_offsets contiguos e comecando em zero")

    print("\n-- parte 6: a mensagem de erro carrega OS DOIS numeros")
    ruim = tmp / "ruim.safetensors"
    conv2 = C.Conversion(fonte, ruim)
    mentiroso = C.plan_lazy("x", "F32", [2, 2], 999, lambda: torch.zeros(2, 2))
    try:
        conv2.commit([mentiroso])
        checa(False, "commit com tamanho mentido levanta")
    except RuntimeError as exc:
        m = str(exc)
        checa("999" in m and "16" in m, "mensagem diz planejado E real", m)
    checa(not ruim.exists(), "saida NAO criada quando a escrita falha")
    checa(not conv2.partial.exists(), "partial removido tambem no fracasso")

    print("\n-- parte 8: guardas dizem quanto falta")
    try:
        C.guard_disk(tmp, 10 ** 15)
        checa(False, "guarda de disco recusa pedido impossivel")
    except SystemExit as exc:
        m = str(exc)
        checa("short by" in m.lower() and "free" in m.lower(),
              "recusa de disco diz o livre e o que falta", m)
    try:
        C.guard_disk(tmp, 1024)
        checa(True, "guarda de disco deixa passar pedido pequeno")
    except SystemExit as exc:
        checa(False, "guarda de disco deixa passar pedido pequeno", str(exc))

    print("\n-- parte 7: fsync ANTES do replace, medido na execucao")
    # A primeira versao disto procurava 'os.fsync' e 'os.replace' no TEXTO de commit() e falhava,
    # porque o docstring do metodo cita os.replace antes de o codigo chamar os.fsync. Procurar
    # texto testa a prosa; o que importa e a ordem das chamadas. Entao: instrumenta e observa.
    ordem: list[str] = []
    fsync_real, replace_real = os.fsync, os.replace
    os.fsync = lambda fd: (ordem.append("fsync"), fsync_real(fd))[1]          # noqa: E731
    os.replace = lambda a, b: (ordem.append("replace"), replace_real(a, b))[1]  # noqa: E731
    try:
        C.Conversion(fonte, tmp / "ordem.safetensors").commit(
            [C.plan_write("x", torch.zeros(2, 2))])
    finally:
        os.fsync, os.replace = fsync_real, replace_real
    checa(ordem == ["fsync", "replace"],
          "fsync executou antes de os.replace", f"ordem observada: {ordem}")

    src = Path(C.__file__).read_text(encoding="utf-8")
    corpo = src[src.index("def commit"):src.index("def write_sidecar")]
    checa("finally" in corpo and "unlink" in corpo, "commit() tem finally com unlink")

    print(f"\n{'=' * 62}\n{OK} passaram, {FALHOU} falharam")
    print("NAO COBERTO: nao testa quantizacao (o nucleo nao quantiza), nem concorrencia entre "
          "dois processos no mesmo .partial, nem arquivo maior que a RAM.")
    return 1 if FALHOU else 0


if __name__ == "__main__":
    raise SystemExit(main())
