
O SIDECAR ENTROU NO CONTRATO (2026-09-29, achado 3 da revisao). Ate aqui todo conversor gravava o
`.quant.json` com `write_text` DEPOIS do `commit()`: modo "w", sem `.partial`, sem fsync, e a recusa
de "sidecar existente" so acontecia horas antes. Uma queda entre o `os.replace` do modelo e o
sidecar deixava o modelo sem `.quant.json` -- rerun recusado (saida existe) e `verify_w4a4` pulando
em silencio a comparacao byte a byte com a fonte, que ele acha pelo sidecar. Agora
`commit(..., sidecar=...)` grava o sidecar num `.partial` exclusivo, faz fsync, poe o SIDECAR no
lugar primeiro e o MODELO depois; se a troca do modelo falhar, o sidecar recem-posto e removido.
Uma queda de energia entre as duas trocas deixa um sidecar orfao (que descreve um arquivo que nao
existe e nao engana ninguem), nunca um modelo sem sidecar.

LEITURA TAMBEM MORA AQUI. `read_header`/`read_tensor`/`LazyTensors` sao o caminho de leitura sem
mmap de todo conversor -- antes havia cinco `read_header` e tres `read_tensor` nos conversores, e o
`read_tensor` do `quant_w4a8` (o que meia arvore importava) so lia BF16/F16/F32.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

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
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "F16": 2,
               "BF16": 2, "I32": 4, "F32": 4, "I64": 8, "F64": 8}
FP8_TORCH = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_HEADER = frozenset({"U8", "F8_E4M3", "F8_E5M2"})


def human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError


def nbytes_of(dtype: str, shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * DTYPE_BYTES[dtype]


def header_dtype(tensor: torch.Tensor) -> str:
    """Nome de dtype do header safetensors para `tensor`.

    Os dois formatos fp8 viajam como U8 cru, casando com como `comfy/ops.py` le `weight_s_rel` e
    `weight_codebook` de volta. Antes de o ticket 24 ser consertado, so o `float8_e4m3fn` de
    `weight_s_rel` recebia esse tratamento; um `float8_e5m2` batia em
    `SAFETENSORS_DTYPE[torch.float8_e5m2]` -> KeyError.
    """
    if tensor.dtype in FP8_TORCH:
        return "U8"
    return SAFETENSORS_DTYPE[tensor.dtype]


def as_bytes(tensor: torch.Tensor) -> memoryview:
    """Visao de bytes crus de `tensor`, pronta para `.write()`.

    `torch.Tensor.numpy()` levanta TypeError para bfloat16 e para os dois fp8 -- EXECUTADO neste
    torch embutido: `torch.zeros(4, dtype=torch.bfloat16).numpy()` diz "Got unsupported ScalarType
    BFloat16". Ver por int16 (bf16) ou uint8 (fp8) antes faz o `numpy()` passar.
    """
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype in FP8_TORCH:
        tensor = tensor.view(torch.uint8)
    elif tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return memoryview(tensor.numpy()).cast("B")


# ---------------------------------------------------------------- leitura (sem mmap)

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


class LazyTensors(Mapping):
    """Um safetensors visto como `{nome: tensor}`, lendo UM tensor por acesso, por faixa de bytes.

    Existe para os lugares que queriam um dict do modelo inteiro e por isso chamavam
    `safetensors.torch.load_file` (safe_open + mmap): `svdq_to_bf16` na fonte e na referencia, e
    `ajusta_denso_diffusers`. Quem itera ou testa `in` so toca o header; o dado so e lido em
    `[]`. Como gerenciador de contexto mantem um handle aberto; fora dele abre um por leitura.
    """

    def __init__(self, path: Path, header: dict | None = None, start: int | None = None):
        self.path = Path(path)
        if header is None:
            header, _ = read_header(self.path)
        self.header = header
        self.start = data_start(self.path) if start is None else start
        self._handle = None

    def __enter__(self) -> "LazyTensors":
        self._handle = open(self.path, "rb")
        return self

    def __exit__(self, *exc) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False

    def __getitem__(self, key: str) -> torch.Tensor:
        info = self.header[key]
        begin, end = info["data_offsets"]
        if self._handle is not None:
            return read_tensor(self._handle, self.start + begin, end - begin, info["dtype"],
                               info["shape"])
        with open(self.path, "rb") as handle:
            return read_tensor(handle, self.start + begin, end - begin, info["dtype"], info["shape"])

    def __iter__(self) -> Iterator[str]:
        return iter(self.header)

    def __len__(self) -> int:
        return len(self.header)

    def __contains__(self, key: object) -> bool:
        return key in self.header


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
    """Parte 8, metade da memoria: RAM fisica disponivel E commit livre do Windows.

    `accumulated_bytes` e o pico que a ferramenta segura. Desde 2026-09-29 todo conversor daqui
    transmite, e o pico e o da formula de streaming (3x o maior tensor tocado). O commit livre e
    conferido junto porque e a regra do projeto e porque ele falta antes da RAM fisica neste host
    (`_ram_guard.py`).
    """
    import psutil

    from _ram_guard import check, check_commit

    refusal = check(psutil.virtual_memory().available, accumulated_bytes, headroom_gib, label)
    if refusal:
        raise SystemExit(refusal)
    refusal = check_commit(accumulated_bytes, headroom_gib, label)
    if refusal:
        raise SystemExit(refusal)


# ---------------------------------------------------------------- plano e escrita (4-7)

@dataclass
class Entry:
    """Uma saida planejada. `payload` e o que a parte 5 vai transmitir.

    - ("copy", (inicio_no_source, tamanho)) -- faixa de bytes copiada verbatim
    - ("copy_many", [(inicio, tamanho), ...]) -- N faixas concatenadas num tensor
    - ("write", tensor)                     -- tensor ja em memoria
    - ("write", callable() -> tensor)       -- tensor produzido na hora de escrever

    O quarto caso e o que deixa todo conversor TRANSMITIR: a quantizacao roda dentro do laco de
    escrita, um tensor por vez, e o plano (e portanto o header inteiro) sai so das formas.
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


def header_bytes(entries: Iterable[Entry], metadata: dict | None = None, *,
                 metadata_last: bool = False, ensure_ascii: bool = False) -> tuple[bytes, int]:
    """Parte 4: o header serializado e o total de bytes de dado planejados.

    Funcao PURA e publica de proposito. Ela nao le a fonte, nao chama nenhum produtor e nao toca
    GPU -- so precisa das formas, que `plan_lazy` ja carrega explicitamente. Isso e o que torna
    possivel verificar uma migracao sem placa: monta-se as entradas do conversor migrado a partir
    do header da FONTE, chama-se isto, e compara-se com os bytes de header do arquivo que aquele
    conversor ja escreveu. Header identico prova que o plano e o layout nao mudaram; so o dado
    quantizado fica para a janela de GPU. Ver `tools/verificar_migracao.py`.

    `__metadata__` PRIMEIRO e `ensure_ascii=False`, e as duas escolhas sao sobre BYTES, nao sobre
    estilo. O dict do Python preserva ordem de insercao e o JSON sai nessa ordem, entao mover
    `__metadata__` para o fim muda os bytes do header de todo arquivo escrito por aqui. Contado em
    2026-09-01: **cinco de cinco** conversores quantizadores desta bancada escrevem
    `{"__metadata__": ...}` primeiro e serializam com `ensure_ascii=False`.

    `metadata_last` e `ensure_ascii=True` existem so para os dois scripts legados que gravavam
    assim a mao (`extrai_transformer`, `transplanta_klein`) continuarem produzindo os MESMOS bytes
    ao passar para o nucleo. O nucleo segue a convencao existente de cada um; nao impoe uma nova.
    """
    offset = 0
    out_header: dict = {}
    if metadata and not metadata_last:
        out_header["__metadata__"] = metadata
    for e in entries:
        out_header[e.key] = {"dtype": e.dtype, "shape": e.shape,
                             "data_offsets": [offset, offset + e.nbytes]}
        offset += e.nbytes
    if metadata and metadata_last:
        out_header["__metadata__"] = metadata
    blob = json.dumps(out_header, separators=(",", ":"), ensure_ascii=ensure_ascii).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    return blob, offset


# A ordem em que `safetensors.torch.save_file` 0.8 poe os tensores: dtype DESCENDENTE nesta
# enumeracao (a do crate `safetensors`), depois nome. Medido neste torch embutido em 2026-09-29:
# {F32, BF16, F16, I8} saem nessa ordem e, dentro de um dtype, em ordem alfabetica; o
# `__metadata__` sai primeiro com as chaves ordenadas. Serve aos dois scripts que gravavam com
# `save_file` (`grava_pesos_recuperados`, `ajusta_denso_diffusers`) produzirem os MESMOS bytes
# pelo nucleo, em streaming.
_SAVE_FILE_DTYPE_ORDER = ("BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8", "F8_E5M2", "F8_E4M3",
                          "F8_E8M0", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "C64",
                          "F64", "I64", "U64")


def save_file_order(entries: Iterable[Entry]) -> list[Entry]:
    rank = {name: i for i, name in enumerate(_SAVE_FILE_DTYPE_ORDER)}
    return sorted(entries, key=lambda e: (-rank[e.dtype], e.key))


def save_file_metadata(metadata: dict | None) -> dict | None:
    return None if metadata is None else dict(sorted(metadata.items()))


def _check_written(e: Entry, tensor: torch.Tensor, buf: memoryview) -> None:
    if len(buf) != e.nbytes:
        raise RuntimeError(f"{e.key}: planned {e.nbytes} bytes, tensor has {len(buf)}")
    # Mesmo numero de bytes nao basta: um produtor que devolve outra forma ou outro dtype gravaria
    # um header mentindo sobre o dado. Com o plano saindo so das formas (streaming), esta e a unica
    # conferencia entre o que o header promete e o que o kernel entregou.
    declared_ok = (e.dtype == header_dtype(tensor)
                   or (tensor.dtype in FP8_TORCH and e.dtype in FP8_HEADER))
    if not declared_ok or list(tensor.shape) != list(e.shape):
        raise RuntimeError(f"{e.key}: planned {e.dtype} {list(e.shape)}, producer returned "
                           f"{header_dtype(tensor)} {list(tensor.shape)}")


def _sidecar_text(payload) -> str:
    if callable(payload):
        payload = payload()
    return payload if isinstance(payload, str) else json.dumps(payload, indent=2)


def _write_exclusive(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class Conversion:
    """Escreve um safetensors a partir de outro, cumprindo as oito partes.

    Uso:

        conv = Conversion(source, output, sidecar)
        conv.refuse_unsafe()                               # partes 1, 2, 3
        conv.guard(output_bytes, accumulated=...)          # parte 8
        conv.commit(entries, metadata, sidecar=manifesto)  # partes 4, 5, 6, 7 + sidecar

    `refuse_unsafe(allow_quantized_source=True)` existe para `svdq_to_bf16`, cuja ENTRADA e
    quantizada por construcao.
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
        self.data_start = data_start(self.source)
        self.output_size: int | None = None

    @property
    def header(self) -> dict:
        return self._header

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def partial(self) -> Path:
        return self.output.with_suffix(self.output.suffix + ".partial")

    @property
    def sidecar_partial(self) -> Path | None:
        return None if self.sidecar is None else self.sidecar.with_suffix(
            self.sidecar.suffix + ".partial")

    def tensors(self) -> LazyTensors:
        """A fonte como `{nome: tensor}` preguicoso, para os produtores de `plan_lazy`."""
        return LazyTensors(self.source, self._header, self.data_start)

    def refuse_unsafe(self, *, allow_quantized_source: bool = False) -> None:
        refuse_unsafe_output(self.source, self.output, self.sidecar)
        if not allow_quantized_source:
            refuse_already_quantized(self._header, self._metadata)
        refuse_stale_partial(self.partial)
        if self.sidecar_partial is not None:
            refuse_stale_partial(self.sidecar_partial)

    def guard(self, output_bytes: int, *, accumulated: int = 0, label: str = "conversion") -> None:
        if accumulated:
            guard_ram(accumulated, label=label)
        guard_disk(self.output.parent, output_bytes)

    @staticmethod
    def planned_size(entries: Iterable[Entry], metadata: dict | None = None) -> int:
        """Tamanho EXATO do arquivo que `commit(entries, metadata)` vai escrever. Nao le dado."""
        blob, planned = header_bytes(entries, metadata)
        return 8 + len(blob) + planned

    def commit(self, entries: Iterable[Entry], metadata: dict | None = None,
               *, progress: Callable[[int, int, str], None] | None = None,
               sidecar: dict | str | Callable[[], dict | str] | None = None,
               metadata_last: bool = False, ensure_ascii: bool = False) -> int:
        """Partes 4 a 7, e o sidecar dentro delas. Devolve os bytes de dado escritos.

        Ordem que importa: flush -> fsync -> **fechar** -> os.replace, com `finally:
        partial.unlink()`. Trocar fsync e replace de lugar deixa um arquivo com nome definitivo e
        conteudo nao durado.

        `sidecar`: o manifesto (dict, texto JSON pronto, ou callable que devolve um dos dois,
        chamado DEPOIS de todo dado escrito -- pode ler `self.output_size` e contagens feitas
        durante a escrita). Vai para `<sidecar>.partial` exclusivo com fsync, e e posto no lugar
        ANTES do modelo; se a troca do modelo falhar, o sidecar recem-posto e removido.
        """
        entries = list(entries)
        if sidecar is not None and self.sidecar is None:
            raise ValueError("commit(sidecar=...) on a Conversion with no sidecar path")

        blob, planned = header_bytes(entries, metadata, metadata_last=metadata_last,
                                     ensure_ascii=ensure_ascii)
        self.output_size = 8 + len(blob) + planned

        partial = self.partial
        side_partial = self.sidecar_partial if sidecar is not None else None
        written = 0
        side_created = side_placed = False
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
                        _check_written(e, tensor, buf)
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
            if side_partial is not None:
                text = _sidecar_text(sidecar)
                _write_exclusive(side_partial, text.encode("utf-8"))
                side_created = True
                if self.sidecar.exists():
                    raise SystemExit(f"Refusing to overwrite existing sidecar: {self.sidecar}")
                os.replace(side_partial, self.sidecar)
                side_placed = True
            # Rechecado AQUI, e nao so em `refuse_unsafe` horas antes: `os.replace` sobrescreve.
            if self.output.exists():
                raise SystemExit(f"Refusing to overwrite existing output: {self.output}")
            os.replace(partial, self.output)
            side_placed = False
        finally:
            if partial.exists():
                partial.unlink()
            if side_created and side_partial.exists():
                side_partial.unlink()
            if side_placed and self.sidecar.exists():
                self.sidecar.unlink()
        return written

    def write_sidecar(self, payload: dict) -> None:
        """Sidecar avulso, para quem ainda grava depois do commit. Exclusivo e atomico.

        Prefira `commit(..., sidecar=...)`, que poe sidecar e modelo no lugar juntos. Este metodo
        continua para os chamadores de fora deste pacote (`quant_misto_w4a8_int8`,
        `refina_escalas`): era `write_text` em modo "w" -- sem `.partial`, sem fsync, e
        sobrescrevia um sidecar criado depois da recusa inicial.
        """
        if self.sidecar is None:
            raise ValueError("this Conversion has no sidecar path")
        write_json_exclusive(self.sidecar, payload)


def write_json_exclusive(path: Path, payload: dict | str) -> None:
    """JSON em `path` via `.partial` exclusivo + fsync + os.replace. Recusa se `path` existir."""
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing sidecar: {path}")
    _write_exclusive(partial, _sidecar_text(payload).encode("utf-8"))
    try:
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing sidecar: {path}")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def nao_coberto() -> str:
    """O que este nucleo NAO garante, para as ferramentas imprimirem no fim de cada execucao."""
    return (
        "NAO COBERTO por este nucleo: ele garante que a ESCRITA e atomica, planejada e conferida "
        "em bytes, forma e dtype -- nunca que os numeros escritos estao certos. Nao valida a "
        "quantizacao, nao confere o resultado contra a fonte, e nao sabe se o kernel vai rodar. "
        "Para isso: verify_w4a4.py para a estrutura, e um render pareado para a qualidade."
    )
