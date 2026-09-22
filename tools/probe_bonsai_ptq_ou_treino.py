"""Bonsai Image contra o FLUX.2-klein-4B original: foi PTQ do original, ou foi TREINADO?

EXECUTADO, nao tracado. Le os dois arquivos por **mmap somente-leitura** e nao por
`safetensors.safe_open`, porque nesta maquina `safe_open` cobra 2x o arquivo em commit charge
(medido, `CLAUDE.md`) e dois arquivos de 7,22 GiB estourariam o limite de 98,72 GiB.

O teste decisivo e RECEITA-AGNOSTICO, e essa e a razao de ele valer:

  Qualquer PTQ ternario -- nao importa o limiar, absmean, absmax, Lloyd-Max, HQQ -- decide o codigo
  de um peso pela MAGNITUDE dele dentro do grupo. Entao, se for PTQ, dentro de cada grupo de 128
  **todo peso com codigo 0 tem |original| menor que todo peso com codigo +-1**. Separabilidade
  perfeita, AUC = 1,000, sem precisar adivinhar o limiar deles.

  Se foi TREINADO (peso latente/QAT), o codigo final nao guarda essa ordem: o treino move o sinal e
  a magnitude livremente. AUC cai para perto de 0,5.

Controle que TEM de passar, senao o script nao esta medindo o que diz: as 9 camadas que o
`quantization_config.json` deles declara PULADAS precisam sair **byte a byte identicas** ao original.
Se elas tambem mudaram, a leitura "PTQ com allowlist" morre antes de qualquer AUC.

Nao cobre: ativacao, imagem, velocidade, o pack gemlite, e o MLX. So peso.
"""
from __future__ import annotations

import argparse
import json
import mmap
import math
from pathlib import Path

import torch

# Do proprio quantization_config.json deles (transformer-gemlite-int2/), 2026-05-21.
SKIP_PATTERNS = [
    "proj_out", "x_embedder", "context_embedder", "time_text_embed", "time_guidance_embed",
    "norm_out", "double_stream_modulation_img", "double_stream_modulation_txt",
    "single_stream_modulation",
]
PULADAS_DECLARADAS = [
    "x_embedder", "context_embedder", "proj_out",
    "time_guidance_embed.timestep_embedder.linear_1",
    "time_guidance_embed.timestep_embedder.linear_2",
    "double_stream_modulation_img.linear", "double_stream_modulation_txt.linear",
    "single_stream_modulation.linear", "norm_out.linear",
]
DOUBLE = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
          "attn.add_v_proj", "attn.to_add_out", "attn.to_out.0",
          "ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out"]
SINGLE = ["attn.to_qkv_mlp_proj", "attn.to_out"]
QUANTIZADAS = ([f"transformer_blocks.{b}.{s}" for b in range(5) for s in DOUBLE]
               + [f"single_transformer_blocks.{b}.{s}" for b in range(20) for s in SINGLE])

DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64}
TAM = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}


class Arquivo:
    """safetensors por mmap somente-leitura: zero de commit charge."""

    def __init__(self, caminho: Path):
        self.caminho = caminho
        self.f = open(caminho, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = int.from_bytes(self.mm[:8], "little")
        self.header = json.loads(self.mm[8:8 + n])
        self.meta = self.header.pop("__metadata__", None)
        self.base = 8 + n

    def cru(self, chave: str) -> memoryview:
        e = self.header[chave]
        a, b = e["data_offsets"]
        return memoryview(self.mm)[self.base + a:self.base + b]

    def tensor(self, chave: str) -> torch.Tensor:
        e = self.header[chave]
        t = torch.frombuffer(bytearray(self.cru(chave)), dtype=DT[e["dtype"]])
        return t.view(*e["shape"])

    def bytes_de(self, chave: str) -> bytes:
        return bytes(self.cru(chave))

    def fecha(self):
        self.mm.close()
        self.f.close()


def auc_por_grupo(orig_abs: torch.Tensor, nao_zero: torch.Tensor) -> tuple[float, int]:
    """AUC de |original| prevendo codigo != 0, calculada DENTRO de cada grupo e agregada.

    Mann-Whitney por ranks: AUC = (soma dos ranks dos nao-zeros - n1(n1+1)/2) / (n0*n1).
    Grupos sem mistura (todos zero ou todos nao-zero) nao informam nada e saem da conta.
    """
    G, g = orig_abs.shape
    ordem = orig_abs.argsort(dim=1)
    ranks = torch.empty_like(ordem)
    alvo = torch.arange(1, g + 1, device=orig_abs.device).expand(G, g)
    ranks.scatter_(1, ordem, alvo)
    n1 = nao_zero.sum(dim=1)
    n0 = g - n1
    mistos = (n0 > 0) & (n1 > 0)
    if not mistos.any():
        return float("nan"), 0
    soma = (ranks * nao_zero).sum(dim=1).double()
    a = (soma - n1 * (n1 + 1) / 2.0) / (n0 * n1).double()
    return float(a[mistos].mean()), int(mistos.sum())


def niveis_por_grupo(w: torch.Tensor, g: int) -> tuple[float, float]:
    """Mediana de magnitudes distintas nao-nulas por grupo, e a fracao de grupos com exatamente 1.

    Ternario g128 verdadeiro: exatamente 1 magnitude nao-nula por grupo (a escala do grupo).
    """
    G = w.numel() // g
    a = w.reshape(G, g).abs()
    # arredondar para 6 digitos significativos absorve o ruido de fp16->fp32
    a = (a * 1e6).round() / 1e6
    ordenado = a.sort(dim=1).values
    dif = ordenado[:, 1:] != ordenado[:, :-1]
    distintos = dif.sum(dim=1) + 1
    tem_zero = (ordenado[:, 0] == 0)
    nao_nulos = distintos - tem_zero.long()
    return float(nao_nulos.median()), float((nao_nulos == 1).double().mean())


def mede(orig: Arquivo, bons: Arquivo, fqn: str, g: int) -> dict:
    chave = f"{fqn}.weight"
    o = orig.tensor(chave).float()
    b = bons.tensor(chave).float()
    assert o.shape == b.shape, f"{fqn}: shape divergente {o.shape} vs {b.shape}"
    n = o.numel()
    frac_zero = float((b == 0).double().mean())

    nv_k, um_k = niveis_por_grupo(b, g)                       # grupos ao longo de K (contiguo)
    nv_n, um_n = niveis_por_grupo(b.t().contiguous(), g)      # grupos ao longo de N

    of, bf = o.reshape(-1), b.reshape(-1)
    nz = bf != 0
    # Um PTQ de magnitude NAO PODE inverter um sinal: ele mapeia w -> s*round(clip(w/d,-1,1)), e o
    # sinal da saida e o sinal de w ou zero. Entao CADA inversao e prova positiva de que aquele peso
    # nao saiu deste original por quantizacao. Guardar a CONTAGEM, nao so a fracao: a fracao some no
    # arredondamento e e a contagem que prova.
    concorda = torch.sign(of[nz]) == torch.sign(bf[nz])
    sinal = float(concorda.double().mean()) if nz.any() else float("nan")
    flips = int((~concorda).sum()) if nz.any() else 0
    cos = float(torch.nn.functional.cosine_similarity(of.unsqueeze(0), bf.unsqueeze(0)).item())

    G = n // g
    auc, grupos = auc_por_grupo(of.reshape(G, g).abs(), nz.reshape(G, g))

    return {
        "fqn": fqn, "shape": list(o.shape), "params": n,
        "frac_zero": frac_zero,
        "niveis_nao_nulos_por_grupo_K": nv_k, "frac_grupos_com_1_nivel_K": um_k,
        "niveis_nao_nulos_por_grupo_N": nv_n, "frac_grupos_com_1_nivel_N": um_n,
        "concordancia_de_sinal": sinal, "flips_de_sinal": flips, "posicoes_nao_nulas": int(nz.sum()),
        "cosseno_com_original": cos,
        "auc_magnitude_preve_codigo": auc, "grupos_informativos": grupos,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--bonsai", required=True)
    p.add_argument("--rotulo", default="bonsai")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--saida", default=None)
    p.add_argument("--limite", type=int, default=0, help="so as N primeiras quantizadas (0 = todas)")
    a = p.parse_args()

    orig = Arquivo(Path(a.original))
    bons = Arquivo(Path(a.bonsai))
    print(f"original {a.original}")
    print(f"  {len(orig.header)} tensores   dtypes {sorted({v['dtype'] for v in orig.header.values()})}"
          f"   __metadata__ {'presente' if orig.meta else 'ausente'}")
    print(f"{a.rotulo} {a.bonsai}")
    print(f"  {len(bons.header)} tensores   dtypes {sorted({v['dtype'] for v in bons.header.values()})}"
          f"   __metadata__ {'presente' if bons.meta else 'ausente'}")

    so_orig = sorted(set(orig.header) - set(bons.header))
    so_bons = sorted(set(bons.header) - set(orig.header))
    print(f"  chaves so no original: {len(so_orig)} {so_orig[:4]}")
    print(f"  chaves so no {a.rotulo}: {len(so_bons)} {so_bons[:4]}")

    # ---- CONTROLE QUE TEM DE PASSAR: as 9 declaradas puladas sao byte a byte identicas?
    print(f"\n=== CONTROLE: as {len(PULADAS_DECLARADAS)} camadas que ELES declaram PULADAS ===")
    print("    (se alguma mudou, 'PTQ com allowlist' cai aqui, antes de qualquer AUC)")
    pulados = []
    for fqn in PULADAS_DECLARADAS:
        chave = f"{fqn}.weight"
        if chave not in orig.header or chave not in bons.header:
            print(f"  {fqn:52s} CHAVE AUSENTE (orig={chave in orig.header} bons={chave in bons.header})")
            pulados.append({"fqn": fqn, "estado": "ausente"})
            continue
        ib = orig.bytes_de(chave) == bons.bytes_de(chave)
        o = orig.tensor(chave).float()
        b = bons.tensor(chave).float()
        rel = float((o - b).norm() / o.norm())
        print(f"  {fqn:52s} {'IDENTICO byte a byte' if ib else 'MUDOU'}   rel-L2 {rel:.3e}")
        pulados.append({"fqn": fqn, "identico": ib, "rel_l2": rel})

    # ---- outro controle: tensores que NENHUMA das duas listas menciona (normas, bias)
    mencionados = {f"{f}.weight" for f in QUANTIZADAS} | {f"{f}.weight" for f in PULADAS_DECLARADAS}
    fora = [k for k in sorted(orig.header) if k not in mencionados and k in bons.header]
    iguais = sum(1 for k in fora if orig.bytes_de(k) == bons.bytes_de(k))
    print(f"\n=== CONTROLE 2: os {len(fora)} tensores fora das duas listas ===")
    print(f"  byte a byte identicos: {iguais}/{len(fora)}")
    difs = [k for k in fora if orig.bytes_de(k) != bons.bytes_de(k)]
    if difs:
        print(f"  MUDARAM ({len(difs)}), primeiros: {difs[:8]}")

    # ---- o teste
    alvos = QUANTIZADAS[:a.limite] if a.limite else QUANTIZADAS
    print(f"\n=== AS {len(alvos)} CAMADAS QUANTIZADAS, uma a uma ===")
    print(f"{'camada':52s} {'zeros':>7s} {'niv/K':>6s} {'1niv/K':>7s} {'sinal':>7s} {'cos':>7s} {'AUC':>7s}")
    linhas = []
    for fqn in alvos:
        if f"{fqn}.weight" not in orig.header:
            print(f"{fqn:52s} CHAVE AUSENTE NO ORIGINAL")
            continue
        r = mede(orig, bons, fqn, a.group_size)
        linhas.append(r)
        print(f"{fqn:52s} {r['frac_zero']:7.4f} {r['niveis_nao_nulos_por_grupo_K']:6.1f} "
              f"{r['frac_grupos_com_1_nivel_K']:7.4f} {r['concordancia_de_sinal']:7.4f} "
              f"{r['cosseno_com_original']:7.4f} {r['auc_magnitude_preve_codigo']:7.4f}", flush=True)

    if linhas:
        def med(k):
            v = sorted(x[k] for x in linhas if not math.isnan(x[k]))
            return v[len(v) // 2] if v else float("nan")

        print(f"\n=== VEREDITO sobre {len(linhas)} camadas ===")
        print(f"  mediana fracao de zeros                  {med('frac_zero'):.4f}")
        print(f"  mediana niveis nao-nulos/grupo, eixo K   {med('niveis_nao_nulos_por_grupo_K'):.2f}"
              f"   (ternario g{a.group_size} verdadeiro = 1,00)")
        print(f"  mediana niveis nao-nulos/grupo, eixo N   {med('niveis_nao_nulos_por_grupo_N'):.2f}")
        print(f"  mediana concordancia de sinal            {med('concordancia_de_sinal'):.4f}"
              f"   (PTQ ~ 1,000  |  treinado ~ 0,500)")
        print(f"  mediana cosseno com o original           {med('cosseno_com_original'):.4f}")
        print(f"  mediana AUC |orig| preve codigo!=0       {med('auc_magnitude_preve_codigo'):.4f}"
              f"   (PTQ = 1,000  |  treinado ~ 0,500)")
        # O limiar CERTO nao e "perto de 1", e "IGUAL a 1". Um PTQ de magnitude nao pode inverter um
        # sinal nem desordenar magnitudes dentro do grupo: ele da sinal 1,0000000 e AUC 1,0000000, por
        # construcao. Qualquer desvio, por pequeno que seja, e prova de que o peso se MOVEU.
        # Esta versao corrige um limiar de 0,98 que eu mesmo escrevi e que classificou 0,9999/0,9826
        # como "PTQ" -- lendo "quase 1" como "1", que e exatamente o erro que a prova exclui.
        s, auc = med("concordancia_de_sinal"), med("auc_magnitude_preve_codigo")
        camadas_limpas = sum(1 for x in linhas if x["flips_de_sinal"] == 0)
        flips_total = sum(x["flips_de_sinal"] for x in linhas)
        nz_total = sum(x["posicoes_nao_nulas"] for x in linhas)
        print(f"  camadas com ZERO inversao de sinal        {camadas_limpas}/{len(linhas)}"
              f"   (PTQ puro = todas)")
        print(f"  inversoes de sinal, total                {flips_total:,} de {nz_total:,} posicoes"
              f" nao-nulas ({100 * flips_total / max(nz_total, 1):.4f}%)")
        if flips_total == 0 and auc >= 0.99999:
            v = ("PTQ do original. Zero inversao de sinal e AUC 1,0 em todas as camadas: os codigos "
                 "sao um arredondamento por magnitude deste original.")
        elif s < 0.75 or auc < 0.80:
            v = ("TREINADO de longe. Sinal e ordem de magnitude nao sobrevivem: os pesos nao guardam "
                 "relacao de arredondamento com este original.")
        else:
            v = (f"TREINADO PARTINDO DESTE ORIGINAL. Existem {flips_total:,} inversoes de sinal, e um "
                 f"PTQ de magnitude nao pode inverter nenhum -- entao os pesos se moveram. Mas "
                 f"moveram POUCO: sinal {s:.4f} e AUC {auc:.4f} contra 0,5 de um treino sem essa "
                 f"inicializacao. E QAT com peso latente (o 'bf16 master' do manifest deles), nao PTQ.")
        print(f"\n  LEITURA: {v}")

    print("\n=== NAO COBERTO ===")
    print("  So peso. Nada de ativacao, nada de imagem, nada de velocidade.")
    print("  O pack gemlite nao foi aberto: nao se sabe se INT2 roda math de 2 bits ou desempacota.")
    print("  O text encoder e o VAE nao foram comparados aqui (tamanhos ja batiam com o original).")
    print(f"  group_size {a.group_size} veio do config DELES, nao foi procurado por varredura.")
    print("  AUC agrega grupos mistos; grupos sem mistura nao informam e ficaram de fora da media.")

    if a.saida:
        Path(a.saida).write_text(json.dumps({
            "original": a.original, "bonsai": a.bonsai, "rotulo": a.rotulo,
            "group_size": a.group_size, "puladas": pulados,
            "fora_das_listas": {"total": len(fora), "identicos": iguais, "mudaram": difs[:64]},
            "quantizadas": linhas,
        }, indent=2), encoding="utf-8")
        print(f"\n  JSON em {a.saida}")

    orig.fecha()
    bons.fecha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
