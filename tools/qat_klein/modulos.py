"""Lineares quantizadas do aluno, otimizador e operações no peso mestre.

Antes: dois globais (`NIVEIS`, `STE_LIGADO`) e o `forward` de cada nn.Linear trocado por uma closure, com
`instala_escalas` sobrescrevendo o que `instala_ste` tinha posto. Agora cada Linear do corpo vira uma
`LinearQuant` (subclasse de nn.Linear, os MESMOS tensores `weight`/`bias` e o mesmo caminho no modelo,
então state_dict e checkpoints não mudam de nome), com um `Quantizador` imutável e uma `ChaveSTE`
compartilhada que liga/desliga a quantização (desligada só para a referência BF16 da guarda `sens`).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from lowbit_canon import Quantizador


class ChaveSTE:
    """Liga/desliga a quantização no forward. Desligada SÓ para medir a referência BF16 antes do 1o passo."""

    def __init__(self):
        self.ligado = True

    @contextmanager
    def desligada(self):
        antes, self.ligado = self.ligado, False
        try:
            yield
        finally:
            self.ligado = antes


class _STE(torch.autograd.Function):
    """Devolve q(w) EXATO no forward (w + (q(w) - w).detach() em bf16 arredondaria duas vezes) e passa o
    gradiente inteiro no backward."""

    @staticmethod
    def forward(ctx, w, quant):
        return quant.quantiza(w)

    @staticmethod
    def backward(ctx, g):
        return g, None


class LinearQuant(torch.nn.Linear):
    """nn.Linear do corpo. modo "ste": peso efetivo = q(mestre) por STE. modo "escalas": código congelado no
    arredondamento do BF16 (buffer `_cod`, não persistente) x escala treinável por grupo (parâmetro `_esc`)."""

    @classmethod
    def de(cls, lin: torch.nn.Linear, quant: Quantizador, chave: ChaveSTE, dtype_calc, modo: str = "ste"):
        m = cls.__new__(cls)
        torch.nn.Module.__init__(m)
        m.in_features, m.out_features = lin.in_features, lin.out_features
        m.weight = lin.weight
        m.bias = lin.bias
        m.quant, m.chave, m.dtype_calc, m.modo = quant, chave, dtype_calc, modo
        if modo == "escalas":
            with torch.no_grad():
                cod, esc = quant.codigo_e_escala(m.weight)
            m.register_buffer("_cod", cod, persistent=False)
            m._esc = torch.nn.Parameter(esc.float())
            m.weight.requires_grad_(False)
        elif modo != "ste":
            raise ValueError(modo)
        return m

    def peso_efetivo(self) -> torch.Tensor:
        if not self.chave.ligado:
            return self.weight
        if self.modo == "escalas":
            return self.quant.reconstroi(self._cod, self._esc, torch.float32)
        return _STE.apply(self.weight, self.quant)

    def codigo_e_escala(self):
        """(código, escala) que o exportado grava."""
        if self.modo == "escalas":
            return self._cod, self._esc.detach().float()
        return self.quant.codigo_e_escala(self.weight.detach())

    def forward(self, x):
        w = self.peso_efetivo().to(self.dtype_calc)
        b = None if self.bias is None else self.bias.to(self.dtype_calc)
        return F.linear(x.to(self.dtype_calc), w, b)


def instala(modelo: torch.nn.Module, corpo: set[str], quant: Quantizador, chave: ChaveSTE, dtype_calc,
            modo: str = "ste") -> list[tuple[str, LinearQuant]]:
    """Troca cada nn.Linear cujo peso está no corpo por uma LinearQuant. Recusa se algum peso do corpo não
    for de nn.Linear. Devolve [(nome, módulo)] na ordem de named_modules."""
    alvos = [(n, m) for n, m in modelo.named_modules()
             if isinstance(m, torch.nn.Linear) and f"{n}.weight" in corpo]
    if len(alvos) != len(corpo):
        raise SystemExit(f"RECUSADO: {len(corpo)} pesos de corpo e {len(alvos)} Linear trocados -- algum peso de "
                         f"corpo nao e nn.Linear.")
    novos = []
    for nome, lin in alvos:
        pai_nome, _, filho = nome.rpartition(".")
        pai = modelo.get_submodule(pai_nome) if pai_nome else modelo
        novo = LinearQuant.de(lin, quant, chave, dtype_calc, modo)
        setattr(pai, filho, novo)
        novos.append((nome, novo))
    return novos


def cria_otimizador(nome: str, params, lr: float):
    """`params`: lista de tensores OU de grupos {"params", "lr"} (lr por grupo: corpo x denso)."""
    if nome == "adamw8bit-sr":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True)
    if nome == "adamw4bit-sr":
        from torchao.optim import AdamW4bit
        return AdamW4bit(params, lr=lr, weight_decay=0.0, bf16_stochastic_round=True)
    if nome == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=0.0)
    if nome == "adam-fp32":
        return torch.optim.Adam(params, lr=lr)
    raise SystemExit(f"otimizador desconhecido: {nome}")


def _arredonda_sr_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 com arredondamento ESTOCASTICO (soma ruido uniforme nos 16 bits que o bf16 descarta
    e trunca). Sem isso um passo de L1 de ~1e-6 num peso de ~1e-2 some inteiro no arredondamento."""
    i = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, i.shape, device=x.device, dtype=torch.int32)
    return ((i + r) & -65536).view(torch.float32).to(torch.bfloat16)


@torch.no_grad()
def prox_l1_relativo(corpo_params, passo_l1: float, grupo: int) -> None:
    """w <- sign(w) * max(|w| - passo_l1 * d_g, 0), d_g = media|w| do grupo de `grupo` no eixo K.

    POR QUE L1 E NAO WEIGHT DECAY: o codigo ternario depende de r = |w|/d_g. Weight decay multiplica o
    grupo inteiro pelo mesmo fator e r nao muda -- nenhum codigo troca. L1 subtrai uma constante: um peso
    em r = 0,5 vai para (0,5 - delta)/(1 - delta) < 0,5 e e' rebaixado; o gradiente segura os que importam.
    Relativo a d_g para o mesmo lambda valer em camadas de escala diferente. [TRACADO, nao medido ainda]"""
    for w in corpo_params:
        n, k = w.shape
        g = w.float().reshape(n, k // grupo, grupo)
        d = g.abs().mean(dim=2, keepdim=True)
        novo = torch.sign(g) * (g.abs() - passo_l1 * d).clamp(min=0)
        novo = novo.reshape(n, k)
        w.copy_(_arredonda_sr_bf16(novo) if w.dtype == torch.bfloat16 else novo.to(w.dtype))


def estatistica_codigos(corpo_mods, cod0: dict, cod_ant: dict) -> dict:
    """Fração de códigos != inicial e != última medição. Guarda o código atual em `cod_ant`.
    No modo so-escalas o código é congelado por construção: as frações são 0."""
    tot = dif0 = difa = 0
    with torch.no_grad():
        for nome, m in corpo_mods:
            c = m.codigo_e_escala()[0].cpu()
            tot += c.numel()
            dif0 += int((c != cod0[nome]).sum())
            if nome in cod_ant:
                difa += int((c != cod_ant[nome]).sum())
            cod_ant[nome] = c
    return {"codigos_mudados_vs_inicio": dif0 / tot, "codigos_trocados_desde_ultimo": difa / tot,
            "codigos_total": tot}
