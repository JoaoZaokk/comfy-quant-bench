"""O treino do Bonsai compra precisao nos canais IMPORTANTES a custa dos pouco usados? (EXECUTADO)

Liga dois achados de 2026-09-21 que sairam de lados opostos:

  `bench/bonsai_image_engenharia_reversa.md` -- o Bonsai TREINOU, e o resultado fica **mais longe** do
  original que um PTQ absmean ingenuo, em **100 de 100** camadas. Um quantizador nao pode perder de um
  quantizador mais simples na metrica que ele minimiza, entao os codigos deles minimizam outra coisa.

  `bench/10eros_fp8_terceiro_promocao_medida.md` -- um terceiro quantizando o 10Eros escolhe, por
  camada, um candidato com erro de peso PIOR (0,028137 contra 0,026502) porque o erro numa projecao
  dos 40% de canais mais importantes e menor. Aceitam erro onde ele nao e usado.

A pergunta que fecha o circuito: **a "outra coisa" que o Bonsai minimiza e isso?** Se o treino deles
concentrou a melhora nos canais de entrada importantes e pagou nos irrelevantes, entao treinar na grade
e um objetivo ponderado por canal, obtido por gradiente em vez de por escolha de candidato -- e as duas
escolas convergem.

COMO MEDE. Por camada `[N, K]`, para cada canal de entrada `j`:

    importancia(j) = ||W_original[:, j]||_2          <- proxy SO DE PESO, ver ressalva
    erro_bonsai(j) = ||W_bonsai[:, j] - W_orig[:, j]|| / ||W_orig[:, j]||
    erro_ptq(j)    = ||W_ptq[:, j]    - W_orig[:, j]|| / ||W_orig[:, j]||
    vantagem(j)    = erro_ptq(j) - erro_bonsai(j)     <- positivo = Bonsai melhor NAQUELE canal

`W_ptq` e o mesmo PTQ absmean g128 no eixo K com escala otima por minimo L2 que venceu no outro probe,
e a escala do Bonsai tambem e recalculada otima, para que a comparacao seja entre CODIGOS.

O teste: a vantagem do Bonsai por quartil de importancia. Ja se sabe que na media ela e negativa
(ele perde em 100/100). Se ela for **positiva no quartil mais importante e negativa no menos**, o
mecanismo e ponderacao por canal. Se for uniformemente negativa nos quatro quartis, nao e -- o treino
so piorou o peso, e o ganho dele esta em outro lugar (composicao entre camadas, por exemplo), que este
probe nao alcanca.

CONTROLE que impede a leitura facil: mede a mesma vantagem **do PTQ contra si mesmo com escala
reescalada uniformemente**. Um reescalonamento uniforme nao pode ter assimetria por quartil, entao se
o controle mostrar assimetria, o instrumento esta medindo artefato da propria decomposicao por canal e
nao o efeito.

RESSALVA QUE NAO SAI DAQUI: `importancia` sai do PESO, nao de ativacao real. Esta bancada ja mediu que
crest factor de ativacao nao prediz erro W4A4 (Spearman +0,096) e que o erro por camada ordena formatos
sem localizar penhascos -- entao "canal importante" aqui e uma hipotese de importancia, nao importancia
medida. A versao com ativacao exigiria calibrar o Klein 4B na GPU.
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


def escala_otima(w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    num = (w * t).sum(dim=1)
    den = (t * t).sum(dim=1)
    return torch.where(den > 0, num / den, torch.zeros_like(num)).unsqueeze(1)


def recompoe(o: torch.Tensor, t: torch.Tensor, g: int) -> torch.Tensor:
    """t sao codigos em forma [G, g]; devolve t*escala_otima no shape original de o."""
    G = o.numel() // g
    return (t * escala_otima(o.reshape(G, g), t)).reshape(o.shape)


def por_canal(o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """rel-L2 por canal de ENTRADA (coluna). o, a no shape [N, K]."""
    num = (a - o).pow(2).sum(dim=0).sqrt()
    den = o.pow(2).sum(dim=0).sqrt().clamp(min=1e-30)
    return num / den


def mede(o: torch.Tensor, b: torch.Tensor, g: int) -> dict:
    G = o.numel() // g
    og, bg = o.reshape(G, g), b.reshape(G, g)

    # codigo do Bonsai: dividir pela unica magnitude nao-nula do grupo
    mag = bg.abs().amax(dim=1, keepdim=True)
    t_bons = torch.where(mag > 0, (bg / mag.clamp(min=1e-30)).round(), torch.zeros_like(bg))
    # PTQ absmean estilo BitNet b1.58
    d = og.abs().mean(dim=1, keepdim=True).clamp(min=1e-30)
    t_ptq = (og / d).clamp(-1, 1).round()

    w_bons = recompoe(o, t_bons, g)
    w_ptq = recompoe(o, t_ptq, g)

    # CONTROLE DE ORCAMENTO CASADO. A primeira versao deste probe usava `w_ptq * 1,02` como controle
    # "uniforme", e ele produziu 66% da assimetria que eu queria atribuir ao Bonsai -- porque um erro
    # de escala de 2% entra proporcional ao sinal, e o sinal varia por canal. Nao era controle.
    # O controle certo troca a MESMA QUANTIDADE de codigos que o Bonsai trocou, em posicoes
    # SORTEADAS, entao ele tem o mesmo orcamento de erro e nenhuma preferencia por canal por
    # construcao. E a licao que este repo ja registra: casar o orcamento antes de comparar criterios.
    dif = (t_bons != t_ptq)
    n_dif = int(dif.sum())
    gerador = torch.Generator().manual_seed(20260921)
    t_rand = t_ptq.clone()
    if n_dif:
        plano = t_rand.reshape(-1)
        alvo = torch.randperm(plano.numel(), generator=gerador)[:n_dif]
        # troca cada codigo sorteado por um dos outros dois valores ternarios
        passo = torch.randint(1, 3, (n_dif,), generator=gerador, dtype=plano.dtype)
        plano[alvo] = ((plano[alvo] + 1 + passo) % 3) - 1
    w_rand = recompoe(o, t_rand, g)

    imp = o.pow(2).sum(dim=0).sqrt()                     # importancia por canal de entrada
    e_b, e_p, e_r = por_canal(o, w_bons), por_canal(o, w_ptq), por_canal(o, w_rand)
    vant = e_p - e_b                                     # > 0 = Bonsai melhor naquele canal
    vant_ctrl = e_p - e_r

    K = imp.numel()
    ordem = imp.argsort()                                 # crescente em importancia
    quartis = [ordem[i * K // 4:(i + 1) * K // 4] for i in range(4)]
    return {
        "K": K,
        "codigos_trocados": n_dif, "fracao_trocada": n_dif / t_bons.numel(),
        "rel_l2_bonsai": float((w_bons - o).norm() / o.norm()),
        "rel_l2_aleatorio_casado": float((w_rand - o).norm() / o.norm()),
        "rel_l2_ptq": float((w_ptq - o).norm() / o.norm()),
        "vantagem_media": float(vant.mean()),
        "vantagem_por_quartil": [float(vant[q].mean()) for q in quartis],
        "controle_por_quartil": [float(vant_ctrl[q].mean()) for q in quartis],
        "erro_bonsai_por_quartil": [float(e_b[q].mean()) for q in quartis],
        "erro_ptq_por_quartil": [float(e_p[q].mean()) for q in quartis],
        "importancia_por_quartil": [float(imp[q].mean()) for q in quartis],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--bonsai", required=True)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--saida", default=None)
    a = p.parse_args()

    orig, bons = Arquivo(Path(a.original)), Arquivo(Path(a.bonsai))
    linhas = []
    print(f"{'camada':50s} {'vant med':>9s} | vantagem por quartil de importancia (Q1 fraco -> Q4 forte)")
    for fqn in QUANTIZADAS:
        chave = f"{fqn}.weight"
        if chave not in orig.header:
            continue
        r = mede(orig.tensor(chave).float(), bons.tensor(chave).float(), a.group_size)
        r["fqn"] = fqn
        linhas.append(r)
        q = r["vantagem_por_quartil"]
        print(f"{fqn:50s} {r['vantagem_media']:+9.5f} | "
              f"{q[0]:+8.5f} {q[1]:+8.5f} {q[2]:+8.5f} {q[3]:+8.5f}", flush=True)

    n = len(linhas)
    def medq(k, i):
        v = sorted(x[k][i] for x in linhas)
        return v[len(v) // 2]

    print(f"\n=== MEDIANAS sobre {n} camadas ===")
    print(f"  vantagem media do Bonsai sobre o PTQ:        {sorted(x['vantagem_media'] for x in linhas)[n//2]:+.6f}"
          f"   (negativo = perde, ja sabido)")
    print(f"{'':4s} {'quartil':10s} {'importancia':>12s} {'erro bonsai':>12s} {'erro ptq':>12s} "
          f"{'vantagem':>10s} {'controle':>10s}")
    for i, nome in enumerate(("Q1 fraco", "Q2", "Q3", "Q4 forte")):
        print(f"{'':4s} {nome:10s} {medq('importancia_por_quartil', i):12.4f} "
              f"{medq('erro_bonsai_por_quartil', i):12.5f} {medq('erro_ptq_por_quartil', i):12.5f} "
              f"{medq('vantagem_por_quartil', i):+10.5f} {medq('controle_por_quartil', i):+10.5f}")

    v1, v4 = medq("vantagem_por_quartil", 0), medq("vantagem_por_quartil", 3)
    c1, c4 = medq("controle_por_quartil", 0), medq("controle_por_quartil", 3)
    ganha_q4 = sum(1 for x in linhas if x["vantagem_por_quartil"][3] > x["vantagem_por_quartil"][0])
    ganha_ctrl = sum(1 for x in linhas if x["controle_por_quartil"][3] > x["controle_por_quartil"][0])

    # o orcamento esta casado? o aleatorio tem de produzir rel-L2 parecido com o do Bonsai
    rb = sorted(x["rel_l2_bonsai"] for x in linhas)[n // 2]
    rr = sorted(x["rel_l2_aleatorio_casado"] for x in linhas)[n // 2]
    fr = sorted(x["fracao_trocada"] for x in linhas)[n // 2]
    print(f"\n  ORCAMENTO CASADO: {100 * fr:.2f}% dos codigos trocados nos dois bracos")
    print(f"    rel-L2 mediano   Bonsai {rb:.5f}   aleatorio casado {rr:.5f}   razao {rr / rb:.4f}")
    print(f"  camadas em que a vantagem no Q4 e maior que no Q1:  Bonsai {ganha_q4}/{n}"
          f"   controle {ganha_ctrl}/{n}")
    print(f"  assimetria Q4 - Q1:  Bonsai {v4 - v1:+.6f}   CONTROLE casado {c4 - c1:+.6f}")

    print()
    if abs(c4 - c1) > abs(v4 - v1) / 3:
        print("  LEITURA: o CONTROLE tem assimetria da mesma ordem do efeito. A decomposicao por canal")
        print("  ja produz assimetria sozinha, entao este instrumento NAO separa as hipoteses. Nao")
        print("  concluir nada sobre ponderacao por canal a partir daqui.")
    elif v4 > 0 > v1:
        print("  LEITURA: o Bonsai GANHA do PTQ nos canais importantes e PERDE nos fracos, com o")
        print("  controle plano. Treinar na grade compra precisao onde o canal e usado e paga onde nao")
        print("  e -- o mesmo objetivo que o fp8 de terceiro persegue escolhendo candidato, obtido por")
        print("  gradiente. As duas escolas convergem no criterio.")
    elif v4 - v1 > 0:
        print(f"  LEITURA: ha assimetria na direcao esperada (Q4 - Q1 = {v4 - v1:+.6f}) mas o Bonsai")
        print("  nao chega a GANHAR no quartil forte. Compativel com ponderacao por canal parcial;")
        print("  nao prova, porque perder menos nao e ganhar.")
    else:
        print("  LEITURA: nao ha assimetria na direcao esperada. O treino deles NAO comprou precisao")
        print("  nos canais importantes em espaco de peso -- o ganho esta em outro lugar (composicao")
        print("  entre camadas, ou importancia que este proxy de peso nao ve). A hipotese morre aqui.")

    print("\n=== NAO COBERTO ===")
    print("  `importancia` sai do PESO (norma da coluna), nao de ativacao real. Esta bancada mediu que")
    print("  crest de ativacao nao prediz erro W4A4 (Spearman +0,096), entao este proxy e HIPOTESE de")
    print("  importancia. A versao com ativacao exige calibrar o Klein 4B na GPU.")
    print("  So o braco ternario, so um PTQ de comparacao, so espaco de peso, nenhuma imagem.")
    print("  Quartis por camada: um canal forte numa camada nao e comparavel a um de outra.")

    if a.saida:
        Path(a.saida).write_text(json.dumps({"camadas": linhas}, indent=2), encoding="utf-8")
        print(f"\n  JSON em {a.saida}")
    orig.fecha()
    bons.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
