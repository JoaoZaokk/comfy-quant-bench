"""O ternario do Bonsai aproxima o original PIOR que um PTQ ingenuo faria? (EXECUTADO)

Fecha o argumento da secao 3 de `bench/bonsai_image_engenharia_reversa.md` por um angulo diferente
do teste de sinal. A pergunta e de OTIMIZACAO, nao de formato:

  Se eles tivessem feito PTQ, os pesos deles seriam, por definicao, a MELHOR aproximacao ternaria do
  original que a receita escolhida alcanca. Entao um PTQ ingenuo meu nao pode ganhar deles em
  distancia ao original -- no maximo empata, se eu acertar a receita.

  Se eles TREINARAM para a saida do modelo, os pesos nao estao minimizando distancia ao original, e
  um PTQ ingenuo pode ficar MAIS PERTO. Isso e o contrario do que se espera de um quantizador, e e
  a assinatura de quem otimizou outra funcao.

Dois PTQ ingenuos como bracos de comparacao, ambos g128 no eixo K (o eixo que ja foi medido):
  absmax   escala = max|w| no grupo, limiar em 0,5*escala -- o mais simples
  absmean  escala/limiar no estilo BitNet b1.58: d = mean|w|, t = round(clip(w/d,-1,1)),
           escala escolhida por minimo erro quadratico dado o codigo

E o CONTROLE que impede a leitura facil: a **escala** deles tambem entra na conta. Se o Bonsai
estiver mais longe so porque escolheu uma escala pior, isso nao e treino, e receita ruim. Entao mede
tambem `rel_l2_com_codigo_deles_e_escala_otima`: mantem o CODIGO ternario deles e recalcula a escala
por minimo erro quadratico. Se o Bonsai continuar mais longe que os PTQ mesmo com escala otima, a
diferenca esta no CODIGO, que e onde o treino agiria.

Nao cobre: nada de ativacao, nada de imagem. E distancia em espaco de PESO, que esta bancada ja
registrou que ordena formatos e nao decide qualidade.
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
    """s* = argmin ||w - s*t||^2 por grupo = <w,t>/<t,t>. Zero onde o grupo e todo zero."""
    num = (w * t).sum(dim=1)
    den = (t * t).sum(dim=1)
    return torch.where(den > 0, num / den, torch.zeros_like(num)).unsqueeze(1)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def mede(o: torch.Tensor, b: torch.Tensor, g: int) -> dict:
    n = o.numel()
    G = n // g
    og = o.reshape(G, g)
    bg = b.reshape(G, g)

    # o codigo ternario DELES: dividir pela unica magnitude nao-nula do grupo
    mag = bg.abs().amax(dim=1, keepdim=True)
    t_deles = torch.where(mag > 0, (bg / mag.clamp(min=1e-30)).round(), torch.zeros_like(bg))
    s_deles_otima = escala_otima(og, t_deles)

    # PTQ ingenuo A: absmax, limiar em metade da escala
    smax = og.abs().amax(dim=1, keepdim=True)
    t_absmax = torch.where(og.abs() > 0.5 * smax, og.sign(), torch.zeros_like(og))
    s_absmax = escala_otima(og, t_absmax)

    # PTQ ingenuo B: estilo BitNet b1.58 -- d = mean|w|, t = round(clip(w/d, -1, 1))
    d = og.abs().mean(dim=1, keepdim=True).clamp(min=1e-30)
    t_bitnet = (og / d).clamp(-1, 1).round()
    s_bitnet = escala_otima(og, t_bitnet)

    return {
        "params": n,
        "rel_l2_bonsai": rel(bg, og),
        "rel_l2_bonsai_escala_otima": rel(t_deles * s_deles_otima, og),
        "rel_l2_ptq_absmax": rel(t_absmax * s_absmax, og),
        "rel_l2_ptq_bitnet": rel(t_bitnet * s_bitnet, og),
        "frac_zero_bonsai": float((bg == 0).double().mean()),
        "frac_zero_absmax": float((t_absmax == 0).double().mean()),
        "frac_zero_bitnet": float((t_bitnet == 0).double().mean()),
        "codigo_igual_ao_absmax": float((t_deles == t_absmax).double().mean()),
        "codigo_igual_ao_bitnet": float((t_deles == t_bitnet).double().mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--bonsai", required=True)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--saida", default=None)
    a = p.parse_args()

    orig, bons = Arquivo(Path(a.original)), Arquivo(Path(a.bonsai))
    print(f"{'camada':50s} {'bonsai':>8s} {'b+s*':>8s} {'absmax':>8s} {'bitnet':>8s} "
          f"{'z_b':>6s} {'z_am':>6s} {'z_bn':>6s} {'=am':>6s} {'=bn':>6s}")
    linhas = []
    for fqn in QUANTIZADAS:
        chave = f"{fqn}.weight"
        if chave not in orig.header:
            continue
        r = mede(orig.tensor(chave).float(), bons.tensor(chave).float(), a.group_size)
        r["fqn"] = fqn
        linhas.append(r)
        print(f"{fqn:50s} {r['rel_l2_bonsai']:8.4f} {r['rel_l2_bonsai_escala_otima']:8.4f} "
              f"{r['rel_l2_ptq_absmax']:8.4f} {r['rel_l2_ptq_bitnet']:8.4f} "
              f"{r['frac_zero_bonsai']:6.3f} {r['frac_zero_absmax']:6.3f} {r['frac_zero_bitnet']:6.3f} "
              f"{r['codigo_igual_ao_absmax']:6.3f} {r['codigo_igual_ao_bitnet']:6.3f}", flush=True)

    def med(k):
        v = sorted(x[k] for x in linhas)
        return v[len(v) // 2]

    print(f"\n=== MEDIANAS sobre {len(linhas)} camadas ===")
    print(f"  rel-L2 do Bonsai contra o original                 {med('rel_l2_bonsai'):.4f}")
    print(f"  rel-L2 do Bonsai com a escala OTIMA recalculada     {med('rel_l2_bonsai_escala_otima'):.4f}")
    print(f"  rel-L2 de um PTQ absmax ingenuo                    {med('rel_l2_ptq_absmax'):.4f}")
    print(f"  rel-L2 de um PTQ estilo BitNet b1.58               {med('rel_l2_ptq_bitnet'):.4f}")
    print(f"  fracao de zeros: bonsai {med('frac_zero_bonsai'):.3f}  absmax {med('frac_zero_absmax'):.3f}"
          f"  bitnet {med('frac_zero_bitnet'):.3f}")
    print(f"  codigo do Bonsai igual ao do absmax: {med('codigo_igual_ao_absmax'):.3f}"
          f"   ao do bitnet: {med('codigo_igual_ao_bitnet'):.3f}")

    mbo = med("rel_l2_bonsai_escala_otima")
    mam, mbn = med("rel_l2_ptq_absmax"), med("rel_l2_ptq_bitnet")
    melhor_ptq = min(mam, mbn)
    perde_am = sum(1 for x in linhas if x["rel_l2_bonsai_escala_otima"] > x["rel_l2_ptq_absmax"])
    perde_bn = sum(1 for x in linhas if x["rel_l2_bonsai_escala_otima"] > x["rel_l2_ptq_bitnet"])
    print(f"\n  camadas em que o Bonsai (escala otima) fica MAIS LONGE que o absmax: {perde_am}/{len(linhas)}")
    print(f"  camadas em que fica MAIS LONGE que o bitnet:                          {perde_bn}/{len(linhas)}")

    print()
    if mbo > melhor_ptq * 1.01:
        print(f"  LEITURA: o Bonsai fica MAIS LONGE do original ({mbo:.4f}) que um PTQ ingenuo "
              f"({melhor_ptq:.4f}), {mbo/melhor_ptq:.3f}x, MESMO com a escala recalculada por minimo\n"
              f"  erro quadratico. Um quantizador nao pode perder de um quantizador mais simples na\n"
              f"  metrica que ele minimiza. Logo os codigos deles nao minimizam distancia ao peso --\n"
              f"  foram escolhidos por outra funcao objetivo. Confirma o veredito de treino por um\n"
              f"  caminho independente do teste de inversao de sinal.")
    elif mbo < melhor_ptq * 0.99:
        print(f"  LEITURA: o Bonsai fica MAIS PERTO ({mbo:.4f}) que os dois PTQ ingenuos "
              f"({melhor_ptq:.4f}).\n  Isso NAO distingue treino de um PTQ melhor que os meus dois "
              f"bracos -- e compativel com as duas\n  coisas, e o teste de inversao de sinal e que "
              f"decide.")
    else:
        print(f"  LEITURA: empate dentro de 1% ({mbo:.4f} contra {melhor_ptq:.4f}). Este angulo nao "
              f"separa as\n  hipoteses; vale o teste de inversao de sinal.")

    print(f"\n=== NAO COBERTO ===")
    print("  Distancia em espaco de PESO. Esta bancada ja registrou que peso ordena formatos e nao")
    print("  decide qualidade -- entao 'mais longe' aqui NAO quer dizer pior imagem, e o ponto e")
    print("  justamente esse: eles trocaram fidelidade de peso por outra coisa.")
    print("  Dois PTQ ingenuos so. Nao tentei HQQ, Lloyd-Max, GPTQ, nem busca de limiar por camada:")
    print("  um PTQ melhor que os meus dois pode existir e reduziria a folga medida.")
    print("  A escala 'otima' aqui e otima por grupo para o codigo dado, em L2 -- nao e a escala que")
    print("  eles usam de fato no deploy (que vem do pack, em fp32, com zero-point).")

    if a.saida:
        Path(a.saida).write_text(json.dumps({"camadas": linhas}, indent=2), encoding="utf-8")
        print(f"\n  JSON em {a.saida}")
    orig.fecha()
    bons.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
