"""O contrato de escrita de checkpoint, escrito UMA vez.

POR QUE ISTO EXISTE. Sete ferramentas em `tools/` escrevem um `.safetensors` a partir de outro, e
ate 2026-08-31 cada uma escrevia o mesmo contrato de oito partes a mao. O ticket
`.scratch/varredura-2026-08-22/issues/08-converter-core.md` mede a divergencia que isso ja produziu;
o dono decidiu pelo nucleo em 2026-08-31. As partes:

    1. recusar source==output, output existente, sidecar existente
    2. recusar um source ja quantizado (`__metadata__` E marcadores `.comfy_quant` embutidos)
    3. recusar um `.partial` obsoleto
    4. planejar o header inteiro, com todo `data_offsets` calculado antes de escrever um byte
    5. transmitir ("write", tensor) / ("copy", (inicio, tamanho))
    6. conferir bytes-escritos == bytes-planejados
    7. flush + fsync + os.replace, com `finally: partial.unlink()`
    8. guardar RAM livre e disco livre antes de comecar

ONDE AS COPIAS DIVERGIRAM, e qual regra este modulo adota (TRACADO nos sete arquivos, 2026-08-31):

  - **Disco: tres regras.** `quant_w4a4` checava `estimativa + 1 GiB` e dizia quanto havia livre;
    `quant_w4a8`, `quant_int8` e `quant_mixed` checavam `< tamanho do source` e levantavam
    `SystemExit` sem numero nenhum; `quant_w4a4_smooth`, `to_native` e `svdq_to_bf16` nao checavam.
    Aqui vale a primeira, porque uma recusa sem numero nao diz ao usuario o que liberar.
  - **RAM: 4 de 7.** `quant_w4a4_smooth` era o pior caso justamente sem guarda -- acumula
    `quantized`, `scales` e `new_norms` de TODAS as camadas antes de escrever. Aqui a guarda e
    obrigatoria para quem declara acumulo.
  - **Mensagem de erro sem numero.** `quant_w4a4_smooth` levantava `RuntimeError("length
    mismatch")` sem dizer de quanto errou. Aqui a mensagem carrega escrito e planejado.
  - **`SAFETENSORS_DTYPE` divergente.** `quant_mixed` nao tinha `torch.int16`. Aqui a tabela e uma.
  - **fp8 e bfloat16 na escrita.** `header_dtype()`/`as_bytes()` existiam so em `quant_w4a8`; um
    `float8_e5m2` dava `KeyError` e um `bfloat16` estourava dentro de `.numpy()` -- **depois de
    quantizar o modelo inteiro**. Aqui sao o unico caminho de escrita.

O QUE ESTE MODULO NAO UNIFICA, de proposito:

  - **`--device`.** `quant_int8` roda em CPU por padrao e de proposito. Unificar moveria um
    conversor deliberadamente sem GPU para disputar a placa.
  - **A escolha por formato.** O unico ponto que muda de formato para formato e o que uma camada
    vira: 2 tensores no w4a4 e no int8, 4 no w4a8, 2 ou 4 no mixed. Isso fica no chamador.
  - **`to_native` e `svdq_to_bf16`.** O primeiro nao tem camada quantizada nenhuma (o que varia
    nele e o MAPA DE NOMES, fusao N-para-1); o segundo consome quantizado e corre no sentido
    inverso, entao a parte 2 do contrato nao se aplica a ele. Os dois usam a metade de ESCRITA
    daqui e pulam as recusas que nao lhes servem, o que este modulo permite explicitamente em vez
    de forcar.

PROVENIENCIA: as regras acima sao TRACADAS -- lidas nos sete arquivos, nao medidas. O que e
EXECUTADO e `tools/test_conversion_core.py`, que exercita as oito partes contra arquivos de
verdade em disco temporario.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

GIB = 1024 ** 3
CHUNK = 16 * 1024 ** 2

SAFETENSORS_DTYPE = {
    torch.int8: "I8", torch.uint8: "U8", torch.int16: "I16",
    torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
}
TORCH_DTYPES = {
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64, "BOOL": torch.bool,
}


def human_size(n: int) -> str:
    return f"{n / GIB:.2f} GiB" if n >= GIB else f"{n / 1024**2:.1f} MiB"


def header_dtype(tensor: torch.Tensor) -> str:
    """Nome de dtype do header safetensors para `tensor`.

    Os dois formatos fp8 viajam como U8 cru, casando com como `comfy/ops.py` le `weight_s_rel` e
    `weight_codebook` de volta. Antes de o ticket 24 ser consertado, so o `float8_e4m3fn` de
    `weight_s_rel` recebia esse tratamento; um `float8_e5m2` batia em
    `SAFETENSORS_DTYPE[torch.float8_e5m2]` -> KeyError.
    """
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return "U8"
    return SAFETENSORS_DTYPE[tensor.dtype]


def as_bytes(tensor: torch.Tensor) -> memoryview:
    """Visao de bytes crus de `tensor`, pronta para `.write()`.

    `torch.Tensor.numpy()` levanta TypeError para bfloat16 e para os dois fp8 -- EXECUTADO neste
    torch embutido: `torch.zeros(4, dtype=torch.bfloat16).numpy()` diz "Got unsupported ScalarType
    BFloat16". Ver por int16 (bf16) ou uint8 (fp8) antes faz o `numpy()` passar.
    """
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.view(torch.uint8)
    elif tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return memoryview(tensor.numpy()).cast("B")


def read_header(path: Path) -> tuple[dict, dict]:
    """Cabecalho e `__metadata__` de um safetensors, sem mmap.

    Nunca mmap: este host quebrou `torch_cpu.dll` com `0xc0000005` mapeando a fonte Gemma de
    21,93 GiB, e falhou com `os error 1455` antes disso.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", {}) or {}
    return header, metadata


def data_start(path: Path) -> int:
    with open(path, "rb") as f:
        return 8 + struct.unpack("<Q", f.read(8))[0]


def read_tensor(handle, start: int, size: int, dtype: str, shape: list[int]) -> torch.Tensor:
    handle.seek(start)
    raw = bytearray(size)
    view = memoryview(raw)
    position = 0
    while position < size:
        count = handle.readinto(view[position:])
        if not count:
            raise EOFError(f"unexpected end of source, {size - position} bytes short")
        position += count
    return torch.frombuffer(raw, dtype=TORCH_DTYPES[dtype]).reshape(shape)


def copy_range(source_handle, output_handle, start: int, size: int) -> None:
    source_handle.seek(start)
    remaining = size
    while remaining:
        chunk = source_handle.read(min(CHUNK, remaining))
        if not chunk:
            raise EOFError(f"unexpected end of source, {remaining} bytes short")
        output_handle.write(chunk)
        remaining -= len(chunk)


# ---------------------------------------------------------------- recusas (partes 1, 2, 3)

def refuse_unsafe_output(source: Path, output: Path, sidecar: Path | None) -> None:
    """Parte 1. Incondicional -- e `quant_mixed` era o unico a condicionar isto a `--dry-run`.

    Condicionar significa que uma execucao seca passa por cima da checagem que a execucao real
    faria, entao o dry-run deixava de ser um ensaio do caminho real, que e a unica coisa que ele
    serve para ser.
    """
    if output == source:
        raise SystemExit("Refusing to overwrite the source model")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    if sidecar is not None and sidecar.exists():
        raise SystemExit(f"Refusing to overwrite existing sidecar: {sidecar}")


def refuse_already_quantized(header: dict, metadata: dict) -> None:
    """Parte 2, com as DUAS causas separadas e contadas.

    Comfy-Org e Lightricks publicam tensores `.comfy_quant` por camada com `__metadata__` vazio,
    entao o teste de metadados sozinho passa reto por eles.
    """
    if metadata.get("_quantization_metadata"):
        raise SystemExit(
            "Refusing to requantize a checkpoint that already has quantization metadata")
    inline = sum(1 for name in header if name.endswith(".comfy_quant"))
    if inline:
        raise SystemExit(f"Refusing to requantize: source carries {inline} inline "
                         "'.comfy_quant' markers, so it is already quantized")


def refuse_stale_partial(partial: Path) -> None:
    """Parte 3. A unica parte que os sete ja tinham -- e ainda assim com dois comportamentos."""
    if partial.exists():
        raise SystemExit(f"Refusing to overwrite stale partial output: {partial}")


# ---------------------------------------------------------------- guardas (parte 8)

def guard_disk(destination: Path, needed_bytes: int, *, headroom_gib: float = 1.0) -> None:
    """Parte 8, metade do disco. Diz quanto falta -- tres das copias so levantavam sem numero."""
    free = shutil.disk_usage(destination).free
    need = needed_bytes + int(headroom_gib * GIB)
    if free < need:
        raise SystemExit(
            f"Insufficient disk space at {destination}: need {human_size(need)} "
            f"({human_size(needed_bytes)} of output plus {headroom_gib:.0f} GiB headroom), "
            f"free {human_size(free)}. Short by {human_size(need - free)}.")


def guard_ram(accumulated_bytes: int, *, label: str, headroom_gib: float = 2.0) -> None:
    """Parte 8, metade da RAM. Obrigatoria para quem ACUMULA antes de escrever.

    `quant_w4a4` transmite e por isso usa a formula de streaming (3x o maior tensor + 2 GiB);
    quem acumula passa a soma de tudo que vai segurar. A copia que mais precisava disto --
    `quant_w4a4_smooth` -- era exatamente a que nao tinha.
    """
    import psutil

    from _ram_guard import check

    refusal = check(psutil.virtual_memory().available, accumulated_bytes, headroom_gib, label)
    if refusal:
        raise SystemExit(refusal)


# ---------------------------------------------------------------- plano e escrita (4-7)

@dataclass
class Entry:
    """Uma saida planejada. `payload` e o que a parte 5 vai transmitir.

    - ("copy", (inicio_no_source, tamanho)) -- faixa de bytes copiada verbatim
    - ("write", tensor)                     -- tensor ja em memoria
    - ("write", callable() -> tensor)       -- tensor produzido na hora de escrever

    O terceiro caso e o que permite a este nucleo servir tanto quem transmite (quant_w4a4, que
    quantiza dentro do laco de escrita) quanto quem acumula (quant_w4a8, quant_mixed), sem obrigar
    o primeiro a segurar o modelo inteiro em RAM so para caber na costura.
    """
    key: str
    dtype: str
    shape: list[int]
    payload: tuple[str, object]
    nbytes: int


def plan_write(key: str, tensor: torch.Tensor) -> Entry:
    tensor = tensor.detach().cpu().contiguous()
    return Entry(key, header_dtype(tensor), list(tensor.shape), ("write", tensor),
                 tensor.numel() * tensor.element_size())


def plan_lazy(key: str, dtype: str, shape: list[int], nbytes: int,
              producer: Callable[[], torch.Tensor]) -> Entry:
    return Entry(key, dtype, list(shape), ("write", producer), nbytes)


def plan_copy(key: str, info: dict) -> Entry:
    start, end = info["data_offsets"]
    return Entry(key, info["dtype"], list(info["shape"]), ("copy", (start, end - start)),
                 end - start)


def plan_copy_many(key: str, dtype: str, shape: list[int],
                   ranges: Iterable[tuple[int, int]]) -> Entry:
    """UM tensor de saida montado por N faixas do source, concatenadas NA ORDEM DADA.

    Existe por causa do `to_native.py`, que funde `attention.to_{q,k,v}` num `attention.qkv` -- o
    unico dos seis conversores cujo destino nao e um-para-um com a origem. Sem isto ele so caberia
    no nucleo via `plan_lazy`, que puxaria o tensor fundido inteiro para a RAM e perderia a
    propriedade que faz este nucleo servir: a copia continua sendo transmitida em blocos, sem
    materializar nada.

    A ordem da lista E o layout do tensor de saida. Trocar duas faixas produz um arquivo com o
    tamanho certo, o header certo e os pesos embaralhados -- e nenhuma das partes 6, 7 ou 8 pega
    isso, porque todas as tres olham para bytes. Quem chama e responsavel pela ordem.
    """
    ranges = [(int(inicio), int(tamanho)) for inicio, tamanho in ranges]
    if not ranges:
        raise ValueError(f"{key}: plan_copy_many sem nenhuma faixa")
    return Entry(key, dtype, list(shape), ("copy_many", ranges),
                 sum(tamanho for _, tamanho in ranges))


@dataclass
class Conversion:
    """Escreve um safetensors a partir de outro, cumprindo as oito partes.

    Uso:

        conv = Conversion(source, output)
        conv.refuse_unsafe()                      # partes 1, 2, 3
        conv.guard(output_bytes, accumulated=...)  # parte 8
        conv.commit(entries, metadata)             # partes 4, 5, 6, 7

    `refuse_unsafe(allow_quantized_source=True)` existe para `svdq_to_bf16`, cuja ENTRADA e
    quantizada por construcao. Deixar isso explicito e melhor do que aquela ferramenta nao ter
    recusa nenhuma, que e o estado de hoje.
    """
    source: Path
    output: Path
    sidecar: Path | None = None
    _header: dict = field(default_factory=dict, repr=False)
    _metadata: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.source = Path(self.source).resolve()
        self.output = Path(self.output).resolve()
        if self.sidecar is not None:
            self.sidecar = Path(self.sidecar).resolve()
        self._header, self._metadata = read_header(self.source)

    @property
    def header(self) -> dict:
        return self._header

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def partial(self) -> Path:
        return self.output.with_suffix(self.output.suffix + ".partial")

    def refuse_unsafe(self, *, allow_quantized_source: bool = False) -> None:
        refuse_unsafe_output(self.source, self.output, self.sidecar)
        if not allow_quantized_source:
            refuse_already_quantized(self._header, self._metadata)
        refuse_stale_partial(self.partial)

    def guard(self, output_bytes: int, *, accumulated: int = 0, label: str = "conversion") -> None:
        if accumulated:
            guard_ram(accumulated, label=label)
        guard_disk(self.output.parent, output_bytes)

    def commit(self, entries: Iterable[Entry], metadata: dict | None = None,
               *, progress: Callable[[int, int, str], None] | None = None) -> int:
        """Partes 4 a 7. Devolve os bytes de dado escritos.

        Ordem que importa e que so era visivel lendo dois lugares em `quant_w4a4`:
        flush -> fsync -> **fechar** -> os.replace, com `finally: partial.unlink()`. Trocar fsync
        e replace de lugar deixa um arquivo com nome definitivo e conteudo nao durado.
        """
        entries = list(entries)

        # parte 4: o header inteiro antes de um byte de dado
        offset = 0
        out_header: dict = {}
        for e in entries:
            out_header[e.key] = {"dtype": e.dtype, "shape": e.shape,
                                 "data_offsets": [offset, offset + e.nbytes]}
            offset += e.nbytes
        planned = offset
        if metadata:
            out_header["__metadata__"] = metadata
        blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((8 - len(blob) % 8) % 8)

        partial = self.partial
        written = 0
        # `"xb"`, nunca `"wb"`: criacao EXCLUSIVA. `refuse_stale_partial` ja checa antes, mas entre
        # a checagem e a abertura cabe outro processo; so o modo exclusivo fecha essa janela.
        #
        # Aberto FORA do `try` de proposito. Se o arquivo ja existe, o FileExistsError sobe sem
        # passar pelo `finally`, que apagaria o `.partial` de QUEM ESTA ESCREVENDO AGORA.
        #
        # Esta linha era `"wb"` ate 2026-09-01, e nisso o nucleo era mais FRACO que os seis
        # conversores que ele substitui -- todos abrem `"xb"`. Pego por
        # `test_svdq_write_contract.py` na migracao do primeiro deles, que e exatamente o risco de
        # extrair um contrato: a extracao pode perder uma garantia sem ninguem notar.
        out = open(partial, "xb")
        try:
            with open(self.source, "rb") as src, out:
                src_data = 8 + struct.unpack("<Q", src.read(8))[0]
                out.write(struct.pack("<Q", len(blob)))
                out.write(blob)
                for i, e in enumerate(entries, 1):
                    kind, payload = e.payload
                    if kind == "copy":
                        start, size = payload
                        copy_range(src, out, src_data + start, size)
                        written += size
                    elif kind == "copy_many":
                        for start, size in payload:
                            copy_range(src, out, src_data + start, size)
                            written += size
                    else:
                        tensor = payload() if callable(payload) else payload
                        buf = as_bytes(tensor)
                        if len(buf) != e.nbytes:
                            raise RuntimeError(
                                f"{e.key}: planned {e.nbytes} bytes, tensor has {len(buf)}")
                        out.write(buf)
                        written += len(buf)
                    if progress:
                        progress(i, len(entries), e.key)
                # parte 6, com os dois numeros na mensagem
                if written != planned:
                    raise RuntimeError(
                        f"length mismatch: wrote {written} bytes, planned {planned} "
                        f"(short by {planned - written})")
                # parte 7
                out.flush()
                os.fsync(out.fileno())
            os.replace(partial, self.output)
        finally:
            if partial.exists():
                partial.unlink()
        return written

    def write_sidecar(self, payload: dict) -> None:
        if self.sidecar is None:
            raise ValueError("this Conversion has no sidecar path")
        self.sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def nao_coberto() -> str:
    """O que este nucleo NAO garante, para as ferramentas imprimirem no fim de cada execucao."""
    return (
        "NAO COBERTO por este nucleo: ele garante que a ESCRITA e atomica, planejada e conferida "
        "em bytes -- nunca que os numeros escritos estao certos. Nao valida a quantizacao, nao "
        "confere o resultado contra a fonte, e nao sabe se o kernel vai rodar. Para isso: "
        "verify_w4a4.py para a estrutura, e um render pareado para a qualidade."
    )
