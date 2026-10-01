"""Fonte canônica da quantização por grupo dos tools do klein (ternário/int4) e do formato `lowbit_affine`.

Antes daqui a mesma conta vivia copiada em `qat_ternario_klein.py`, `constroi_ternario_ingenuo.py`,
`ajusta_denso_*.py`, `compara_codigos_bonsai.py` e `mistura_klein.py`. Este módulo não importa ComfyUI nem
diffusers: roda no Colab e em casa. O empacotamento usa o `kernel.pack_codes` do próprio loader
(`custom_nodes/comfy-lowbit-loader/kernel.py`, importado por caminho, sem editá-lo), então o arquivo gerado é
exatamente o que o loader lê.

    Quantizador(niveis=1, grupo=128)   ternário absmean (braço 0, Bonsai): código round(clamp(w/d, -1, 1)),
                                       d = média |w| do grupo; escala ótima L2 dado o código
    Quantizador(niveis=7, grupo=32)    int4 simétrico: d = max |w| / 7, código em -7..7, escala ótima L2
    ternariza(w, grupo)                o PTQ ingênuo do `constroi_ternario_ingenuo` (preserva -0.0); o QAT
                                       usa `Quantizador.quantiza` (zero +0.0) -- ver `quantiza`
    rtn_simetrico(w, grupo, bits)      RTN puro com escala absmax (a receita do `mistura_klein --rtn4`);
                                       NÃO é o int4 acima: aqui a escala é a absmax, lá a ótima L2

Empacotado (`lowbit_affine`): código sem sinal u = t + niveis, W = u * scale + zero, zero = -niveis * scale.
Para o ternário a desquantização (fp32, um arredondamento) reproduz bit a bit o valor desempacotado
`bf16(t * s)`; `confere_exato` prova isso por camada antes de gravar.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch

FORMAT = "lowbit_affine"
BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")
FORMATOS = {"ternario": 1, "int4": 7}


def pilhas_reais(nomes) -> set[str]:
    """Uma pilha precisa de >= 2 índices distintos: `adaLN_modulation.1` não é pilha."""
    ind: dict[str, set[str]] = {}
    for k in nomes:
        m = BLOCO.match(k)
        if m:
            ind.setdefault(m.group("pilha"), set()).add(m.group("i"))
    return {n for n, i in ind.items() if len(i) >= 2}


def corpo(params) -> set[str]:
    """Nomes dos pesos 2-D dentro das pilhas de blocos. `params`: iterável de (nome, tensor)."""
    params = list(params)
    pil = pilhas_reais(k for k, _ in params)
    return {k for k, v in params if v.ndim == 2 and (m := BLOCO.match(k)) and m.group("pilha") in pil}


@dataclass(frozen=True)
class Quantizador:
    niveis: int = 1
    grupo: int = 128

    def __post_init__(self):
        if self.niveis < 1 or self.grupo < 1:
            raise ValueError(f"Quantizador inválido: niveis={self.niveis}, grupo={self.grupo}")
        if 2 * self.niveis + 1 > 16:
            raise ValueError(f"{2 * self.niveis + 1} níveis não cabem em 4 bits (lowbit_affine vai até 4)")

    @classmethod
    def do_formato(cls, formato: str, grupo: int) -> Quantizador:
        return cls(FORMATOS[formato], grupo)

    @property
    def bits(self) -> int:
        """Largura do código sem sinal no `lowbit_affine`: 3 níveis -> 2 bits, 15 níveis -> 4 bits."""
        return 2 if 2 * self.niveis + 1 <= 4 else 4

    def _codigo_float(self, w: torch.Tensor):
        n, k = w.shape
        g = w.float().reshape(n, k // self.grupo, self.grupo)
        if self.niveis == 1:
            d = g.abs().mean(dim=2, keepdim=True).clamp(min=1e-30)
        else:
            d = (g.abs().amax(dim=2, keepdim=True) / self.niveis).clamp(min=1e-30)
        t = (g / d).clamp(-self.niveis, self.niveis).round()
        num = (g * t).sum(dim=2, keepdim=True)
        den = (t * t).sum(dim=2, keepdim=True)
        s = torch.where(den > 0, num / den, torch.zeros_like(num))
        return t, s

    def codigo_e_escala(self, w: torch.Tensor):
        """(código int8 [n, k], escala fp32 [n, k // grupo, 1])."""
        t, s = self._codigo_float(w)
        return t.reshape(w.shape).to(torch.int8), s

    def reconstroi(self, codigo: torch.Tensor, escala: torch.Tensor, dtype) -> torch.Tensor:
        n, k = codigo.shape
        return (codigo.float().reshape(n, k // self.grupo, self.grupo) * escala).reshape(n, k).to(dtype)

    def quantiza(self, w: torch.Tensor, zero_com_sinal: bool = False) -> torch.Tensor:
        """Valor desempacotado na dtype de `w`.

        Duas receitas antigas diferiam SÓ no sinal do zero: o QAT passava o código por int8 (zero = +0.0) e o
        `constroi_ternario_ingenuo` multiplicava o código float, onde round(-0,3) = -0,0 e o produto grava
        -0.0 (~metade dos zeros do PTQ ingênuo; o C0 do `transplanta_klein` depende disso byte a byte).
        `zero_com_sinal=True` reproduz a segunda."""
        if zero_com_sinal:
            t, s = self._codigo_float(w)
            n, k = w.shape
            return (t * s).reshape(n, k).to(w.dtype)
        t, s = self.codigo_e_escala(w)
        return self.reconstroi(t, s, w.dtype)

    def empacota(self, codigo: torch.Tensor, escala: torch.Tensor, dtype_escala=torch.float32) -> Empacotado:
        n, k = codigo.shape
        u = (codigo.to(torch.int16) + self.niveis).to(torch.uint8)
        s = escala.reshape(n, k // self.grupo).float()
        return Empacotado(qdata=kernel().pack_codes(u, self.bits), scale=s.to(dtype_escala),
                          zero=(-self.niveis * s).to(dtype_escala), bits=self.bits, group_size=self.grupo)


def ternariza(w: torch.Tensor, grupo: int = 128) -> torch.Tensor:
    """O PTQ ingênuo do `constroi_ternario_ingenuo` (zero com sinal, byte a byte o de sempre). O QAT usa
    `Quantizador(1, grupo).quantiza(w)` (zero +0.0), a receita dele de sempre."""
    return Quantizador(1, grupo).quantiza(w, zero_com_sinal=True)


def rtn_simetrico(w: torch.Tensor, grupo: int, bits: int = 4) -> torch.Tensor:
    """RTN absmax por grupo no eixo K, devolvido em fp32 (a conta do `mistura_klein.rtn4` antes do cast)."""
    t = w.float()
    n, k = t.shape
    if k % grupo:
        raise ValueError(f"K={k} não divisível pelo grupo {grupo}")
    g = t.reshape(n, k // grupo, grupo)
    lv = 2 ** (bits - 1) - 1
    s = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / lv
    return ((g / s).round().clamp(-lv, lv) * s).reshape(n, k)


# ------------------------------------------------------------------------------------ lowbit_affine

@dataclass
class Empacotado:
    qdata: torch.Tensor  # uint8 (N, K * bits / 8), LSB primeiro ao longo de K
    scale: torch.Tensor  # (N, K / G)
    zero: torch.Tensor   # (N, K / G)
    bits: int
    group_size: int

    def desquantiza(self, dtype) -> torch.Tensor:
        return kernel().dequantize_torch(self.qdata, self.scale, self.zero, self.bits, self.group_size, dtype)


_KERNEL = None


def kernel():
    """O `kernel.py` do comfy-lowbit-loader, carregado por caminho (a pasta tem hífen e o pacote puxa o
    ComfyUI; o kernel sozinho só precisa do torch). Procura: $LOWBIT_KERNEL_PY, o loader desta bancada e um
    `kernel.py` do lado deste arquivo (como sobe para o Colab)."""
    global _KERNEL
    if _KERNEL is None:
        aqui = Path(__file__).resolve().parent
        candidatos = [os.environ.get("LOWBIT_KERNEL_PY"),
                      aqui.parent / "custom_nodes" / "comfy-lowbit-loader" / "kernel.py",
                      aqui / "lowbit_kernel.py", aqui / "kernel.py"]
        for c in candidatos:
            if c and Path(c).is_file():
                spec = importlib.util.spec_from_file_location("lowbit_canon_kernel", str(c))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _KERNEL = mod
                break
        else:
            raise FileNotFoundError("kernel.py do comfy-lowbit-loader não encontrado (defina LOWBIT_KERNEL_PY ou "
                                    "suba custom_nodes/comfy-lowbit-loader/kernel.py como lowbit_kernel.py)")
    return _KERNEL


def conf_tensor() -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": FORMAT}).encode("utf-8")), dtype=torch.uint8)


def confere_exato(referencia: torch.Tensor, pack: Empacotado) -> tuple[int, int]:
    """(elementos com bits diferentes, dos quais só sinal do zero). Referência na dtype final (bf16)."""
    got = pack.desquantiza(referencia.dtype)
    ib = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[referencia.element_size()]
    dif = got.view(ib) != referencia.view(ib)
    n = int(dif.sum())
    zeros = int((dif & (got == 0) & (referencia == 0)).sum()) if n else 0
    return n, zeros


def state_dict_lowbit(dense: dict, lowbit: dict) -> dict:
    """ComfyUI state dict: o mesmo contrato do `formats.to_comfy_state_dict` do loader (testado contra ele)."""
    sd = dict(dense)
    for nome, p in lowbit.items():
        sd[f"{nome}.weight"] = p.qdata
        sd[f"{nome}.weight_scale"] = p.scale
        sd[f"{nome}.weight_zeros"] = p.zero
        sd[f"{nome}.comfy_quant"] = conf_tensor()
    return sd
