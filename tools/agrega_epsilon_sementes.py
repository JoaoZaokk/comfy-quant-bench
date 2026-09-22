"""Junta N corridas do `probe_epsilon_ckpt_ab` e reporta o que P3 exige, nao uma media solta.

POR QUE ESTE ARQUIVO EXISTE. O `probe_epsilon_ckpt_ab` roda UMA semente por invocacao e imprime a
media por faixa. Reportar a media de uma semente e o erro que este repo ja cometeu e registrou: um
efeito de 3,4% foi lido como resultado enquanto o espalhamento ENTRE SEMENTES dentro de um unico
braco chegava a 1,39x. O criterio do braco 1 transformou isso em previsao escrita (P3): o relatorio
tem de trazer a diferenca PAREADA, o erro-padrao dela e o placar de vitorias por semente.

PAREADO E A PALAVRA IMPORTANTE. Dentro de uma semente todos os bracos recebem a MESMA trajetoria
imposta do BF16, entao a diferenca entre dois bracos na mesma semente nao carrega a variacao da
semente. Comparar as medias de dois bracos calculadas sobre sementes diferentes jogaria essa
vantagem no lixo.

O ERRO-PADRAO E DA DIFERENCA, nao dos valores. `sd(d) / sqrt(n)` sobre as n diferencas pareadas.
Com n = 8 isso nao e um teste de hipotese e o tool nao finge que e: e a distancia do efeito ao
proprio espalhamento, que e o que este repo usa para decidir se um numero carrega sinal.

LE O TEXTO DA SAIDA DO PROBE, nao um json -- o probe nao escreve json. Entao o parser e frouxo de
proposito num ponto e estrito em outro: aceita qualquer conjunto de rotulos, e RECUSA se as sementes
nao tiverem todas os mesmos rotulos, porque ai a media pareada compararia populacoes diferentes.

NAO COBRE: nenhuma imagem, nenhum tempo. E o BF16 e o alvo, nao a verdade.
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

SEMENTE = re.compile(r"#+\s*SEMENTE\s+(\d+)")
CABECA = re.compile(r"^\s*passo\s+sigma\s+(.+)$")
LINHA = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+((?:[-\d.eE+]+\s*)+)$")


def le_corridas(caminhos: list[Path]) -> dict[int, dict[str, list[tuple[float, float]]]]:
    """{semente: {rotulo: [(sigma, rel_rmse), ...]}}. Uma corrida por bloco de SEMENTE."""
    fora: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for cam in caminhos:
        semente = None
        rotulos: list[str] | None = None
        for ln in cam.read_text(encoding="utf-8", errors="replace").splitlines():
            if (m := SEMENTE.search(ln)):
                semente, rotulos = int(m.group(1)), None
                continue
            if (m := CABECA.match(ln)):
                rotulos = m.group(1).split()
                if semente is None:            # arquivo de uma semente so, sem cabecalho de bloco
                    semente = -len(fora) - 1   # rotulo negativo: identidade vem do --semente
                fora.setdefault(semente, {r: [] for r in rotulos})
                continue
            if rotulos and (m := LINHA.match(ln)):
                vals = m.group(3).split()
                if len(vals) != len(rotulos):
                    continue
                for r, v in zip(rotulos, vals, strict=True):
                    fora[semente][r].append((float(m.group(2)), float(v)))
    return fora


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="saidas do probe_epsilon_ckpt_ab (texto)")
    p.add_argument("--base", help="rotulo da base das diferencas pareadas; default o primeiro")
    p.add_argument("--teto", help="rotulo do teto, para a fracao da distancia base->teto (P2)")
    p.add_argument("--corte-sigma", type=float, default=0.926,
                   help="sigma >= corte e ALTO; o default e o mesmo que o probe usa no klein")
    a = p.parse_args()

    corridas = le_corridas([Path(x) for x in a.logs])
    if not corridas:
        print("RECUSADO: nenhuma corrida reconhecida nos logs.", file=sys.stderr)
        return 2
    conjuntos = {frozenset(v) for v in corridas.values()}
    if len(conjuntos) != 1:
        print(f"RECUSADO: as corridas nao tem os mesmos rotulos: {conjuntos}", file=sys.stderr)
        return 2
    rotulos = list(next(iter(corridas.values())))
    base = a.base or rotulos[0]
    if base not in rotulos:
        print(f"RECUSADO: base {base!r} nao esta em {rotulos}", file=sys.stderr)
        return 2

    faixas = [("todos", lambda s: True),
              (f"sigma ALTO >= {a.corte_sigma:g}", lambda s: s >= a.corte_sigma),
              (f"sigma BAIXO < {a.corte_sigma:g}", lambda s: s < a.corte_sigma)]

    def media(sem, rot, filtro):
        vs = [v for s, v in corridas[sem][rot] if filtro(s)]
        return sum(vs) / len(vs) if vs else float("nan")

    ordem = sorted(corridas)
    print(f"{len(ordem)} sementes: {ordem}")
    print(f"rotulos: {rotulos}   base das diferencas: {base}\n")

    for nome, filtro in faixas:
        print("-" * 92)
        print(f"{nome}")
        print("-" * 92)
        print(f"{'semente':>9}" + "".join(f"{r:>17}" for r in rotulos))
        for sem in ordem:
            print(f"{sem:>9}" + "".join(f"{media(sem, r, filtro):>17.4e}" for r in rotulos))

        print(f"\n{'':>9}" + "".join(f"{r:>17}" for r in rotulos))
        for et, fn in (("media", statistics.fmean), ("min", min), ("max", max)):
            print(f"{et:>9}" + "".join(
                f"{fn([media(s, r, filtro) for s in ordem]):>17.4e}" for r in rotulos))
        print(f"{'espalha':>9}" + "".join(
            f"{max(media(s, r, filtro) for s in ordem) / min(media(s, r, filtro) for s in ordem):>16.3f}x"
            for r in rotulos))

        print("\n  diferenca PAREADA contra", base)
        for r in rotulos:
            if r == base:
                continue
            d = [media(s, base, filtro) - media(s, r, filtro) for s in ordem]
            m = statistics.fmean(d)
            ep = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
            venceu = sum(1 for x in d if x > 0)
            razao = statistics.fmean([media(s, base, filtro) / media(s, r, filtro) for s in ordem])
            print(f"    {r:>17}  d {m:+.4e}  erro-padrao {ep:.4e}  "
                  f"{'-' if math.isnan(ep) or ep == 0 else f'{abs(m) / ep:5.1f}x o ep'}  "
                  f"razao {razao:.4f}x  {r if m > 0 else base} vence {max(venceu, len(d) - venceu)}"
                  f"/{len(d)}")

        if a.teto and a.teto in rotulos:
            print(f"\n  fracao da distancia {base} -> {a.teto}   (P2)")
            for r in rotulos:
                if r in (base, a.teto):
                    continue
                fr = [(media(s, base, filtro) - media(s, r, filtro))
                      / (media(s, base, filtro) - media(s, a.teto, filtro)) for s in ordem]
                print(f"    {r:>17}  {statistics.fmean(fr) * 100:6.2f}%   "
                      f"min {min(fr) * 100:.2f}%  max {max(fr) * 100:.2f}%")
        print()

    print("-" * 92)
    print("placar por semente, contando PASSOS vencidos (menor rel-RMSE) em cada faixa 'todos'")
    print("-" * 92)
    contagem = {r: 0 for r in rotulos}
    for sem in ordem:
        n = len(corridas[sem][base])
        for i in range(n):
            melhor = min(rotulos, key=lambda r: corridas[sem][r][i][1])
            contagem[melhor] += 1
    total = sum(contagem.values())
    print("".join(f"{r:>17}" for r in rotulos))
    print("".join(f"{contagem[r]:>17}" for r in rotulos) + f"   de {total}")

    print("\n=== NAO COBERTO ===")
    print("  Nenhuma imagem, nenhum tempo. O erro-padrao acima e da DIFERENCA pareada e nao e um")
    print("  teste de hipotese: com 8 sementes ele diz a distancia do efeito ao proprio")
    print("  espalhamento, que e o que decide se o numero carrega sinal.")
    print("  Um prompt por semente, uma placa. E o BF16 e o alvo, nao a verdade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
