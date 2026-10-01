"""CLI do QAT (a MESMA de antes, mais flags opcionais) e a `Config` imutável validada uma vez.

A impressão digital cobre só o que define a trajetória (dados, código, otimizador, lr, lote...). Mudar
`--max-passos`, intervalos de log/holdout/ckpt, repo de envio ou a guarda de parada numa retomada é
legítimo; mudar o resto é outra corrida e a retomada recusa (salvo `--retoma-config-diferente`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lowbit_canon import Quantizador


class ErroConfig(SystemExit):
    """Recusa antes de carregar qualquer coisa. Sai com código 2, como os RECUSADO de antes."""

    def __init__(self, msg: str):
        super().__init__(2)
        self.msg = msg


def parser(descricao: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=descricao, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raiz", required=True, help="pasta do pipeline (model_index.json etc)")
    p.add_argument("--professor", required=True, help="transformer BF16, nomes diffusers")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--prompts-holdout", type=Path, required=True)
    p.add_argument("--sementes", type=int, nargs="+", default=[1, 2])
    p.add_argument("--passos", type=int, default=8)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--dir", type=Path, required=True, help="pasta PERSISTENTE: shards, ckpt, journal")
    p.add_argument("--otim", choices=["adamw8bit-sr", "adamw4bit-sr", "adamw8bit", "adam-fp32"], required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-passos", type=int, required=True)
    p.add_argument("--lote", type=int, default=1,
                   help="exemplos por passo, concatenados (nao acumulacao): mesmo ruido de gradiente "
                        "de acumular, mas usa a placa melhor se couber")
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--so-escalas", action="store_true",
                   help="congela o codigo do corpo no arredondamento do BF16 e treina SO uma escala por grupo "
                        "(implica --congela-resto)")
    p.add_argument("--professor-hf", default=None, metavar="REPO",
                   help="repo de DADOS no HF que guarda os shards do professor entre VMs (baixa antes, sobe depois)")
    p.add_argument("--formato", choices=["ternario", "int4"], default="ternario",
                   help="codigo do corpo: ternario (absmean, -1..1) ou int4 (absmax/7, -7..7; use --grupo 32)")
    p.add_argument("--ckpt-min", type=float, default=25.0)
    p.add_argument("--log-cada", type=int, default=50)
    p.add_argument("--holdout-cada", type=int, default=500)
    p.add_argument("--sem-grad-ckpt", action="store_true")
    p.add_argument("--so-professor", action="store_true", help="grava os shards e sai")
    p.add_argument("--push-hf", default=None,
                   help="repo PRIVADO do HF para o checkpoint (Colab). O token de escrita tem de "
                        "estar no ambiente, posto pelo dono; a escrita e provada antes de treinar")
    p.add_argument("--hf-publico", action="store_true",
                   help="cria o repo do --push-hf como PUBLICO (nao conta na cota privada)")
    p.add_argument("--inicia-de", type=Path, default=None,
                   help="checkpoint de OUTRA corrida: carrega mestre + estado do otimizador, mas zera o "
                        "contador de passos e a ordem dos exemplos (mesma semente de embaralhamento). "
                        "Serve para REPETIR a mesma sequencia de exemplos a partir de pesos ja treinados.")
    p.add_argument("--congela-resto", action="store_true",
                   help="treina SO o corpo ternario; os tensores fora dele (densos, norms) ficam os do professor")
    p.add_argument("--lr-denso", type=float, default=None,
                   help="lr dos tensores FORA do corpo ternario (modulacao, embedders, normas); padrao = --lr")
    p.add_argument("--l1-corpo", type=float, default=0.0,
                   help="lambda do L1 proximal relativo ao grupo no corpo (esparsifica; ver prox_l1_relativo). "
                        "Passo por iteracao = lr * lambda * d_g")
    p.add_argument("--sens-passos", default="0,3,6",
                   help="passos do sampler usados na guarda de condicionamento (sens)")
    p.add_argument("--sens-min", type=float, default=0.0,
                   help="piso ABSOLUTO de sens para `melhor` (sens = d_aluno/d_ref); 0 = so' o relativo")
    p.add_argument("--sens-tol", type=float, default=0.1,
                   help="piso RELATIVO: sens >= (1 - tol) * maior sens ja' visto no braco")
    p.add_argument("--parada-paciencia", type=int, default=0,
                   help="para o braco se holdout_rel piorar N avaliacoes seguidas (0 = nunca)")
    p.add_argument("--cruzado", action="store_true",
                   help="grava a saida do professor com a condicao do prompt vizinho (fase 2; ver grava_cruzado)")
    p.add_argument("--cruzado-frac", type=float, default=0.0,
                   help="fracao dos passos que treina num exemplo cruzado em vez de um normal")
    p.add_argument("--avalia-ckpt", type=Path, nargs="+", default=None,
                   help="so' mede holdout, holdout_rel e sens destes checkpoints e sai (calibracao)")
    p.add_argument("--device", default="0", help="indice da GPU (0, 1, ...) ou 'cpu' (testes)")
    # novas, todas opcionais
    p.add_argument("--exporta", choices=["desempacotado", "lowbit", "ambos"], default="desempacotado",
                   help="formato do exportado: desempacotado bf16 (o de sempre, que a cadeia de render le), "
                        "lowbit_affine empacotado (identico bit a bit na desquantizacao) ou os dois")
    p.add_argument("--retoma-config-diferente", action="store_true",
                   help="aceita retomar um checkpoint cuja impressao digital da configuracao e' outra")
    p.add_argument("--para-no-passo", type=int, default=0,
                   help="(teste) age como SIGTERM ao fim deste passo: checkpoint e saida")
    return p


# campos que definem a trajetória; mudar qualquer um numa retomada é outra corrida
TRAJETORIA = ("formato", "grupo", "otim", "lr", "lr_denso", "lote", "so_escalas", "congela_resto",
              "l1_corpo", "cruzado_frac", "sementes", "passos", "size")


@dataclass(frozen=True)
class Config:
    raiz: Path
    professor: str
    prompts: Path
    prompts_holdout: Path
    sementes: tuple
    passos: int
    size: int
    dir: Path
    otim: str
    lr: float
    max_passos: int
    lote: int
    grupo: int
    so_escalas: bool
    professor_hf: str | None
    formato: str
    ckpt_min: float
    log_cada: int
    holdout_cada: int
    sem_grad_ckpt: bool
    so_professor: bool
    push_hf: str | None
    hf_publico: bool
    inicia_de: Path | None
    congela_resto: bool
    lr_denso: float
    l1_corpo: float
    sens_passos: tuple
    sens_min: float
    sens_tol: float
    parada_paciencia: int
    cruzado: bool
    cruzado_frac: float
    avalia_ckpt: tuple | None
    device: str
    exporta: str = "desempacotado"
    retoma_config_diferente: bool = False
    para_no_passo: int = 0
    extras: dict = field(default_factory=dict, compare=False)

    @classmethod
    def de_args(cls, a: argparse.Namespace) -> Config:
        d = dict(vars(a))
        d["raiz"] = Path(d["raiz"])
        d["sementes"] = tuple(d["sementes"])
        d["sens_passos"] = tuple(int(x) for x in str(d["sens_passos"]).split(",") if x.strip())
        d["lr_denso"] = d["lr"] if d["lr_denso"] is None else d["lr_denso"]
        d["congela_resto"] = bool(d["congela_resto"] or d["so_escalas"])  # --so-escalas implica
        d["avalia_ckpt"] = tuple(d["avalia_ckpt"]) if d["avalia_ckpt"] else None
        d["device"] = str(d["device"])
        c = cls(**d)
        c.valida()
        return c

    def valida(self) -> None:
        if self.so_escalas and self.l1_corpo > 0:
            raise ErroConfig("--so-escalas com --l1-corpo: o L1 proximal age no peso mestre 2-D, e no modo "
                             "so-escalas o treinavel e' a escala 3-D por grupo (quebraria no primeiro passo)")
        if not 0.0 <= self.cruzado_frac < 1.0:
            raise ErroConfig(f"--cruzado-frac {self.cruzado_frac} fora de [0, 1)")
        if self.lote < 1 or self.grupo < 1 or self.passos < 1 or self.max_passos < 0:
            raise ErroConfig("--lote, --grupo, --passos >= 1 e --max-passos >= 0")
        if self.so_professor and self.avalia_ckpt:
            raise ErroConfig("--so-professor e --avalia-ckpt sao modos exclusivos")
        if not self.sens_passos or any(p < 0 or p >= self.passos for p in self.sens_passos):
            raise ErroConfig(f"--sens-passos {self.sens_passos} fora de 0..{self.passos - 1}")

    @property
    def quantizador(self) -> Quantizador:
        return Quantizador.do_formato(self.formato, self.grupo)

    def dispositivo(self):
        import torch
        return torch.device("cpu" if self.device == "cpu" else f"cuda:{int(self.device)}")

    def impressao(self) -> dict:
        """{campo: valor} da trajetória + sha dos arquivos de prompts, e o hash de tudo."""
        d = {k: getattr(self, k) for k in TRAJETORIA}
        d["sementes"] = list(d["sementes"])
        for nome in ("prompts", "prompts_holdout"):
            d[nome + "_sha"] = hashlib.sha256(Path(getattr(self, nome)).read_bytes()).hexdigest()[:16]
        d["hash"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
        return d

    def como_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items() if k != "extras"}
