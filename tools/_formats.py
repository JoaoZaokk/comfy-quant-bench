"""Formatos quantizados: o que uma camada VIRA, escrito uma vez. Modelo explicito, nao dict solto.

Uma instancia e um formato COM os seus parametros -- `ConvrotW4A4(convrot_groupsize=256)` -- e
responde as quatro perguntas que antes cada conversor respondia a mao (revisao de 2026-09-29,
achados 7 e 8):

    accepts(shape)         a camada cabe neste formato? (divisibilidade de K)
    tensors(name, shape)   quais tensores de saida, com dtype e forma, NA ORDEM de escrita
    layer_config()         a entrada da camada em `_quantization_metadata["layers"]`
    quantize(weight, ck)   o kernel, devolvendo os tensores na mesma ordem de `tensors()`

Antes, `quant_w4a4` (`planejar`), `quant_w4a4_smooth` (`store`), `quant_w4a8`, `quant_int8`,
`quant_mixed` (que copiava os ramos do w4a4 e do w4a8) e `quant_awq_w4a16` escreviam cada um a
sua versao disto, e a string do formato aparecia crua em seis lugares mais o `verify_w4a4`. Uma
consequencia concreta: a correcao FP32 de 26/09 ("a rotacao em BF16 perde precisao") entrou em
quatro conversores e nao no smooth.

POR QUE `tensors()` IMPORTA MAIS QUE OS OUTROS. Com as formas de saida conhecidas antes de
quantizar, todo conversor planeja o header inteiro so das formas e quantiza DENTRO do laco de
escrita (`_conversion.plan_lazy`), um tensor por vez. Antes, w4a8/int8/mixed/awq/smooth
quantizavam o modelo inteiro num dict e so depois escreviam -- porque "as formas saem do kernel".
Nao saem: sao analiticas, e o `verify_w4a4` ja as calculava. As formas abaixo conferem com o que o
backend eager do comfy-kitchen devolve em CPU (medido 2026-09-29) e com os headers dos arquivos
reais escritos pelo backend CUDA que `verify_w4a4` cita; e `_conversion.commit` confere forma e
dtype de cada tensor produzido contra o plano, entao um backend que divergir levanta em vez de
gravar um header mentiroso.

O que NAO mora aqui: a preferencia de dispositivo e o dtype de ENTRADA do kernel sao do chamador
(`quant_int8` roda em CPU por padrao de proposito). Todos os chamadores passam FP32 desde
2026-09-29 -- o smooth incluso, por decisao do dono (registrada em `QUANTIZER_INPUT`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import torch

import _conversion as C

# Registrado no sidecar de cada conversao. Mudou em 2026-09-29 para o smooth (BF16 -> FP32); os
# demais ja quantizavam do FP32 desde 26/09.
QUANTIZER_INPUT = "float32"

FORMAT_VERSION = "1.0"


def _base(name: str) -> str:
    return name.removesuffix(".weight")


@dataclass(frozen=True)
class ConvrotW4A4:
    """ConvRot W4A4: int4 empacotado em I8 [N, K/2] + escala F32 por linha [N]."""
    convrot_groupsize: int = 256
    # O loader le 64 como constante (`comfy/ops.py`), e o kernel int4 MMA recusa outro valor;
    # nao e eixo utilizavel (`.agent-reference/comfy/05-hard-rules-quantizacao.md`).
    quant_group_size: int = 64
    name = "convrot_w4a4"

    def accepts(self, shape: list[int]) -> bool:
        return len(shape) == 2 and shape[1] % self.convrot_groupsize == 0

    def tensors(self, name: str, shape: list[int]) -> list[tuple[str, str, list[int]]]:
        rows, cols = shape
        return [(name, "I8", [rows, cols // 2]), (f"{_base(name)}.weight_scale", "F32", [rows])]

    def layer_config(self) -> dict:
        return {"format": self.name, "convrot_groupsize": self.convrot_groupsize}

    def quantize(self, weight: torch.Tensor, ck) -> tuple[list[torch.Tensor], dict]:
        qdata, scales = ck.quantize_convrot_w4a4_weight(
            weight, convrot_groupsize=self.convrot_groupsize,
            quant_group_size=self.quant_group_size, stochastic_rounding=0)
        return [qdata.cpu().contiguous(), scales.cpu().contiguous()], {}


@dataclass(frozen=True)
class AsymW4A8:
    """asym_w4a8_int8 simetrico: int4 em I8 [N, K/2], s_rel fp8 (gravado U8) [N, K/gs],
    s_channel F32 [N] e, com codebook Lloyd-Max, codebook F32 [16]."""
    group_size: int = 16
    convrot_groupsize: int = 256
    codebook: bool = True
    name = "asym_w4a8_int8"

    def accepts(self, shape: list[int]) -> bool:
        if len(shape) != 2:
            return False
        k = shape[1]
        # mirrors AsymW4A8Int8Layout.Params._validate_tensor_fields
        if k % 16 or k % self.group_size or k % self.convrot_groupsize:
            return False
        if self.group_size < 4 or (16 % self.group_size and self.group_size % 16):
            return False
        return True

    def tensors(self, name: str, shape: list[int]) -> list[tuple[str, str, list[int]]]:
        rows, cols = shape
        base = _base(name)
        out = [(name, "I8", [rows, cols // 2]),
               (f"{base}.weight_s_rel", "U8", [rows, cols // self.group_size]),
               (f"{base}.weight_s_channel", "F32", [rows])]
        if self.codebook:
            out.append((f"{base}.weight_codebook", "F32", [16]))
        return out

    def layer_config(self) -> dict:
        return {"format": self.name, "group_size": self.group_size,
                "convrot_groupsize": self.convrot_groupsize}

    def quantize(self, weight: torch.Tensor, ck) -> tuple[list[torch.Tensor], dict]:
        qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
            weight, group_size=self.group_size, convrot_groupsize=self.convrot_groupsize,
            symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=self.codebook,
            codebook_tensor=None, stochastic_rounding=0)
        if correction is not None:
            raise SystemExit("symmetric=True returned a correction tensor; ComfyUI would drop it")
        out = [qdata.cpu().contiguous(), s_rel.cpu().contiguous(), s_channel.cpu().contiguous()]
        if self.codebook:
            if codebook is None:
                raise RuntimeError("codebook=True returned no codebook tensor")
            out.append(codebook.cpu().contiguous())
        return out, {}


@dataclass(frozen=True)
class Int8Tensorwise:
    """int8_tensorwise por linha, com ou sem ConvRot: I8 [N, K] + escala F32 [N, 1].

    A chave da escala e `<camada>.weight_scale`, a MESMA string do w4a4 (`name + "_scale"` com
    `name` terminando em `.weight`). O comentario antigo do quant_int8 dizia que era diferente; nao
    era.
    """
    convrot: bool = True
    convrot_groupsize: int = 256
    name = "int8_tensorwise"

    def accepts(self, shape: list[int]) -> bool:
        # Rotation needs K divisible by the Hadamard size; plain int8 has no such constraint.
        return len(shape) == 2 and (not self.convrot or shape[1] % self.convrot_groupsize == 0)

    def tensors(self, name: str, shape: list[int]) -> list[tuple[str, str, list[int]]]:
        rows, cols = shape
        return [(name, "I8", [rows, cols]), (f"{name}_scale", "F32", [rows, 1])]

    def layer_config(self) -> dict:
        conf = {"format": self.name}
        if self.convrot:
            conf.update({"convrot": True, "convrot_groupsize": self.convrot_groupsize})
        return conf

    def quantize(self, weight: torch.Tensor, ck=None) -> tuple[list[torch.Tensor], dict]:
        from comfy_kitchen.backends.eager import quantization as eager
        import comfy_kitchen

        if self.convrot:
            implementation = comfy_kitchen.registry.get_implementation(
                "quantize_int8_convrot_weight",
                kwargs={"weight": weight, "group_size": self.convrot_groupsize})
            qdata, scale = implementation(weight, group_size=self.convrot_groupsize)
        else:
            # --no-convrot always calls comfy-kitchen's eager backend directly, on any device:
            # it bypasses ck.registry, so there is no native backend for it to resolve to.
            qdata, scale = eager.quantize_int8_rowwise(weight)
        return [qdata.cpu().contiguous(), scale.cpu().contiguous().float()], {}


def q4_1_awq(weight: torch.Tensor, group: int = 32):
    """Codigos Q4_1 do gguf-py reempacotados no layout awq_w4a16. Devolve (q, scale, zeros, rel)."""
    import gguf
    import numpy as np

    n, k = weight.shape
    x = weight.to(torch.float32).numpy()
    blocos = gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q4_1).reshape(n, k // group, 20)
    d = blocos[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    m = blocos[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    qs = blocos[..., 4:20]
    vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(n, k)  # ggml: low nibbles hold j, high hold j + 16
    packed = (vals[:, 0::2] | (vals[:, 1::2] << 4)).astype(np.uint8)
    scale = torch.from_numpy(d.T.copy()).to(torch.bfloat16)
    zeros = torch.from_numpy((m + 8.0 * d).T.copy()).to(torch.bfloat16)
    deq = ((torch.from_numpy(vals).view(n, k // group, group).float() - 8.0) * scale.float().t().unsqueeze(-1)
           + zeros.float().t().unsqueeze(-1)).view(n, k)
    rel = float((deq - weight.float()).norm() / weight.float().norm())
    return torch.from_numpy(packed.view(np.int8)), scale, zeros, rel


@dataclass(frozen=True)
class AwqW4A16:
    """awq_w4a16 a partir de codigos Q4_1: I8 [N, K/2] + escala e zeros BF16 [K/G, N]."""
    group_size: int = 32
    name = "awq_w4a16"

    def accepts(self, shape: list[int]) -> bool:
        return len(shape) == 2 and shape[1] % self.group_size == 0

    def tensors(self, name: str, shape: list[int]) -> list[tuple[str, str, list[int]]]:
        rows, cols = shape
        base = _base(name)
        return [(name, "I8", [rows, cols // 2]),
                (f"{base}.weight_scale", "BF16", [cols // self.group_size, rows]),
                (f"{base}.weight_zeros", "BF16", [cols // self.group_size, rows])]

    def layer_config(self) -> dict:
        return {"format": self.name, "group_size": self.group_size}

    def quantize(self, weight: torch.Tensor, ck=None) -> tuple[list[torch.Tensor], dict]:
        q, scale, zeros, rel = q4_1_awq(weight, self.group_size)
        return [q, scale, zeros], {"rel": rel}


QuantFormat = ConvrotW4A4 | AsymW4A8 | Int8Tensorwise | AwqW4A16
FORMAT_NAMES = (ConvrotW4A4.name, AsymW4A8.name, Int8Tensorwise.name, AwqW4A16.name)


def plan_layer(fmt, name: str, shape: list[int], produce_weight: Callable[[], torch.Tensor], ck,
               on_quantized: Callable[[str, dict], None] | None = None) -> list[C.Entry]:
    """As entradas preguicosas de UMA camada: o kernel roda uma vez, no primeiro tensor.

    O primeiro produtor quantiza e guarda os demais tensores; cada produtor seguinte consome o
    seu. Isso amarra os produtores a ordem em que `commit()` os chama, e um produtor chamado fora
    de ordem levanta com mensagem explicita em vez de escrever lixo.
    """
    specs = fmt.tensors(name, shape)
    pendente: dict[int, torch.Tensor] = {}

    def primeiro() -> torch.Tensor:
        tensores, extra = fmt.quantize(produce_weight(), ck)
        if len(tensores) != len(specs):
            raise RuntimeError(f"{name}: {fmt.name} produced {len(tensores)} tensors, "
                               f"planned {len(specs)}")
        for i, tensor in enumerate(tensores[1:], 1):
            pendente[i] = tensor
        if on_quantized is not None:
            on_quantized(name, extra)
        return tensores[0]

    def seguinte(i: int) -> Callable[[], torch.Tensor]:
        def produz() -> torch.Tensor:
            if i not in pendente:
                raise RuntimeError(f"{specs[i][0]}: produtor chamado antes do peso da camada {name}. "
                                   "As entradas de uma camada tem que ser escritas em ordem.")
            return pendente.pop(i)
        return produz

    entradas = []
    for i, (key, dtype, out_shape) in enumerate(specs):
        producer = primeiro if i == 0 else seguinte(i)
        entradas.append(C.plan_lazy(key, dtype, out_shape, C.nbytes_of(dtype, out_shape), producer))
    return entradas


def plan_model(header: dict, formats: dict, produce_weight: Callable[[str], torch.Tensor], ck,
               on_quantized: Callable[[str, dict], None] | None = None) -> list[C.Entry]:
    """Todas as entradas, na ordem do header: copia verbatim, ou as N entradas do formato da camada.

    `formats`: {nome do peso: instancia de formato}. `produce_weight(nome)` le o peso da fonte e o
    prepara para o kernel (dispositivo, dtype) -- decisao do conversor, nao do formato.
    """
    entradas = []
    for name, info in header.items():
        fmt = formats.get(name)
        if fmt is None:
            entradas.append(C.plan_copy(name, info))
        else:
            entradas.extend(plan_layer(fmt, name, info["shape"], lambda n=name: produce_weight(n),
                                       ck, on_quantized))
    return entradas


def layer_configs(formats: dict) -> dict:
    """{nome do peso: formato} -> `_quantization_metadata["layers"]` ({camada: config})."""
    return {_base(name): fmt.layer_config() for name, fmt in formats.items()}


def quant_metadata(metadata: dict, layers: dict, quantization: str,
                   extra: dict | None = None) -> dict:
    """O `__metadata__` de saida: o da fonte, mais `_quantization_metadata` e `quantization`.

    `layers` e o dict de `layer_configs()`. Mesma ordem de chaves que todos os conversores
    escreviam a mao: as da fonte, depois `_quantization_metadata`, depois `quantization`, depois
    as extras.
    """
    out = dict(metadata)
    out["_quantization_metadata"] = json.dumps({"format_version": FORMAT_VERSION, "layers": layers},
                                               separators=(",", ":"))
    out["quantization"] = quantization
    if extra:
        out.update(extra)
    return out


def streaming_peak(header: dict, names) -> int:
    """Pico de RAM de um conversor em streaming: 3x o maior tensor tocado (bytes lidos, tensor de
    origem, resultado) -- a formula que o quant_w4a4 sempre usou, agora valida para todos."""
    return 3 * max((header[n]["data_offsets"][1] - header[n]["data_offsets"][0] for n in names),
                   default=0)
