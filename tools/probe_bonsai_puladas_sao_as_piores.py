"""As 9 camadas que o Bonsai deixou em FP16 sao as MAIS SENSIVEIS a quantizacao? (EXECUTADO)

Este probe existe para tentar **derrubar** a conclusao central de
`bench/bonsai_image_engenharia_reversa.md`, que diz: o conjunto em FP16 e capacidade de ADAPTACAO,
nao o conjunto de camadas sensiveis.

Se as 9 puladas forem justamente as de maior erro de quantizacao ternaria entre as 109 candidatas,
a lista `skip_patterns` deles coincide com um criterio mensuravel de sensibilidade, e a leitura de
"capacidade de adaptacao" fica muito mais fraca -- passa a ser indistinguivel de "eles mediram (ou
adivinharam bem) quais camadas nao aguentam".

Mede sobre o **original**, nao sobre o Bonsai: a pergunta e o que um quantizador veria ANTES de
decidir. Erro por camada = rel-L2 de uma quantizacao ternaria g128 no eixo K com escala otima por
minimo erro quadratico (o melhor caso do formato), exatamente o mesmo procedimento aplicado as
quantizadas e as puladas, para que a comparacao seja entre iguais.

DOIS CONTROLES que podem explicar a lista sem nenhuma sensibilidade:

  A. **Divisibilidade.** Se alguma pulada tem K nao divisivel por 128, ela e mecanicamente
     inquantizavel naquele grupo e a exclusao nao diz nada sobre fragilidade.
  B. **Tamanho.** Se as puladas forem as camadas pequenas, pular custa quase nada em bytes, e a
     lista se explica por economia, nao por erro. Mede a fracao de parametros que elas somam.

Nao cobre: erro em espaco de PESO, que esta bancada registra que ordena formatos e nao decide
qualidade. A pergunta certa seria erro de SAIDA por camada em ativacao real, que exige calibracao na
GPU e nao foi feita aqui. Entao um resultado "as puladas nao sao as piores em peso" NAO fecha a
possibilidade de elas serem as piores em ativacao -- e isso esta dito no veredito.
"""
from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path

import torch

DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
DOUBLE = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
          "attn.add_v_proj", "attn.to_add_out", "attn.to_out.0",
          "ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out"]
SINGLE = ["attn.to_qkv_mlp_proj", "attn.to_out"]
QUANTIZADAS = ([f"transformer_blocks.{b}.{s}" for b in range(5) for s in DOUBLE]
               + [f"single_transformer_blocks.{b}.{s}" for b in range(20) for s in SINGLE])
PULADAS = ["x_embedder", "context_embedder", "proj_out",
           "time_guidance_embed.timestep_embedder.linear_1",
           "time_guidance_embed.timestep_embedder.linear_2",
           "double_stream_modulation_img.linear", "double_stream_modulation_txt.linear",
           "single_stream_modulation.linear", "norm_out.linear"]


class Arquivo:
    def __init__(self, caminho: Path):
        self.f = open(caminho, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.mm[:8], "little")
        self.header = json.loads(self.mm[8:8 + n])
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def tensor(self, chave: str) -> torch.Tensor:
        e = self.header[chave]
        a, b = e["data_offsets"]
        raw = bytearray(memoryview(self.mm)[self.base + a:self.base + b])
        return torch.frombuffer(raw, dtype=DT[e["dtype"]]).view(*e["shape"])

    def fecha(self):
        self.mm.close()
        self.f.close()


def erro_ternario(w: torch.Tensor, g: int) -> tuple[float, bool]:
    """rel-L2 do melhor ternario g128 no eixo K. Devolve tambem se K e divisivel por g."""
    K = w.shape[-1]
    divisivel = (K % g) == 0
    gg = g if divisivel else K  # sem divisibilidade, um grupo por linha -- o melhor caso possivel
    G = w.numel() // gg
    wg = w.reshape(G, gg)
    d = wg.abs().mean(dim=1, keepdim=True).clamp(min=1e-30)
    t = (wg / d).clamp(-1, 1).round()
    num = (wg * t).sum(dim=1)
    den = (t * t).sum(dim=1)
    s = torch.where(den > 0, num / den, torch.zeros_like(num)).unsqueeze(1)
    return float((wg - t * s).norm() / wg.norm()), divisivel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--saida", default=None)
    a = p.parse_args()

    o = Arquivo(Path(a.original))
    linhas = []
    for grupo, fqns in (("QUANTIZADA", QUANTIZADAS), ("PULADA", PULADAS)):
        for fqn in fqns:
            chave = f"{fqn}.weight"
            if chave not in o.header:
                print(f"  AUSENTE no original: {chave}")
                continue
            w = o.tensor(chave).float()
            err, div = erro_ternario(w, a.group_size)
            linhas.append({"fqn": fqn, "grupo": grupo, "shape": list(w.shape),
                           "params": w.numel(), "err_ternario": err, "K_div_128": div})

    linhas.sort(key=lambda x: -x["err_ternario"])
    total = sum(x["params"] for x in linhas)

    print(f"{'#':>4s} {'grupo':11s} {'err':>7s} {'K%128':>6s} {'params':>14s} {'shape':22s} camada")
    for i, x in enumerate(linhas, 1):
        marca = "  <<<" if x["grupo"] == "PULADA" else ""
        print(f"{i:4d} {x['grupo']:11s} {x['err_ternario']:7.4f} {str(x['K_div_128']):>6s} "
              f"{x['params']:14,} {str(x['shape']):22s} {x['fqn']}{marca}")

    pul = [x for x in linhas if x["grupo"] == "PULADA"]
    qua = [x for x in linhas if x["grupo"] == "QUANTIZADA"]
    postos = [i for i, x in enumerate(linhas, 1) if x["grupo"] == "PULADA"]

    def med(v):
        v = sorted(v)
        return v[len(v) // 2]

    print(f"\n=== O TESTE ===")
    print(f"  {len(pul)} puladas, {len(qua)} quantizadas, {len(linhas)} no total")
    print(f"  postos das puladas no ranking de erro (1 = pior): {postos}")
    print(f"  se fossem as mais sensiveis, seriam {list(range(1, len(pul) + 1))}")
    print(f"  erro mediano  pulada {med(x['err_ternario'] for x in pul):.4f}"
          f"   quantizada {med(x['err_ternario'] for x in qua):.4f}")
    print(f"  pior pulada   {max(x['err_ternario'] for x in pul):.4f}"
          f"   pior quantizada {max(x['err_ternario'] for x in qua):.4f}")
    dentro_top = sum(1 for r in postos if r <= len(pul))
    print(f"  puladas dentro do top {len(pul)} de erro: {dentro_top}/{len(pul)}")

    print(f"\n=== CONTROLE A: divisibilidade ===")
    nd = [x for x in pul if not x["K_div_128"]]
    print(f"  puladas com K NAO divisivel por {a.group_size}: {len(nd)}/{len(pul)}"
          f"  {[x['fqn'] for x in nd]}")
    ndq = [x for x in qua if not x["K_div_128"]]
    print(f"  quantizadas com K NAO divisivel: {len(ndq)}/{len(qua)}")
    if nd:
        print(f"  ATENCAO: para essas, a exclusao pode ser MECANICA e nao dizer nada sobre fragilidade.")

    print(f"\n=== CONTROLE B: tamanho ===")
    pp = sum(x["params"] for x in pul)
    print(f"  puladas somam {pp:,} params = {100 * pp / total:.2f}% dos {total:,} candidatos")
    print(f"  maior pulada    {max(x['params'] for x in pul):,}"
          f"   maior quantizada {max(x['params'] for x in qua):,}")
    print(f"  (o README deles diz 'menos de 5% dos parametros' em FP16)")

    print(f"\n=== VEREDITO ===")
    if dentro_top >= len(pul) - 1:
        print(f"  A CONCLUSAO DO RELATORIO ESTA AMEACADA: as puladas sao praticamente as de maior")
        print(f"  erro, entao a lista coincide com um criterio de sensibilidade e a leitura de")
        print(f"  'capacidade de adaptacao' nao se distingue de 'eles escolheram as fragis'.")
    elif dentro_top == 0:
        print(f"  A conclusao do relatorio SOBREVIVE: nenhuma das {len(pul)} puladas esta no top")
        print(f"  {len(pul)} de erro de quantizacao. A lista nao e um ranking de sensibilidade.")
    else:
        print(f"  PARCIAL: {dentro_top} das {len(pul)} puladas estao no top {len(pul)} de erro. A lista")
        print(f"  tem alguma correlacao com sensibilidade, mas nao e um ranking dela. Reportar assim,")
        print(f"  sem arredondar para nenhum dos dois lados.")

    print(f"\n=== NAO COBERTO ===")
    print("  Erro em espaco de PESO. Esta bancada registra que peso ordena formatos e nao decide")
    print("  qualidade, entao 'nao sao as piores em peso' NAO fecha a possibilidade de serem as")
    print("  piores em ATIVACAO -- isso exigiria calibracao na GPU, que nao foi feita.")
    print("  Uma unica receita ternaria (absmean com escala otima). Outro limiar reordenaria a lista.")
    print("  Nao mede o efeito de pular cada camada na saida, que e a pergunta que decidiria de vez.")

    if a.saida:
        Path(a.saida).write_text(json.dumps({"camadas": linhas, "postos_das_puladas": postos},
                                           indent=2), encoding="utf-8")
        print(f"\n  JSON em {a.saida}")
    o.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
