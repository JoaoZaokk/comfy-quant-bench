"""Testes da ponderacao por sigma em `quant_mixed.py`. Roda direto, sem pytest.

    python_embeded\\python.exe -s tools/test_quant_mixed_sigma.py

O teste que carrega o resto e o primeiro: **peso uniforme tem de dar exatamente o mesmo
numero que nenhum peso.** Se nao der, `--sigma-weight` nao e uma ponderacao da metrica que
esta bancada ja usa -- e uma metrica nova com o mesmo nome de coluna, e toda comparacao
contra medicao antiga fica sem sentido sem ninguem perceber.

O eixo que estes testes seguram, dito porque segurar o eixo errado ja custou trabalho aqui:
eles nao tocam GPU e nao chamam kernel nenhum. Cobrem a ARITMETICA do peso e as recusas.
Se o kernel escolhe outro formato, ou se o sigma gravado na calibracao nao corresponde ao
passo real do sampler, nada aqui percebe -- isso e o probe de ponta a ponta, nao este arquivo.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from quant_mixed import relative, sigma_weights  # noqa: E402

FALHAS: list[str] = []


def check(nome: str, condicao: bool, detalhe: str = "") -> None:
    if condicao:
        print(f"PASS  {nome}")
    else:
        print(f"FAIL  {nome}   {detalhe}")
        FALHAS.append(nome)


def esperar_saida(nome: str, fn, trecho: str) -> None:
    try:
        fn()
    except SystemExit as exc:
        msg = str(exc)
        check(nome, trecho in msg, f"mensagem nao cita {trecho!r}: {msg[:120]}")
        return
    check(nome, False, "nao levantou SystemExit")


def main() -> int:
    g = torch.Generator().manual_seed(20260831)
    ref = torch.randn(64, 32, generator=g, dtype=torch.float32)
    got = ref + 0.05 * torch.randn(64, 32, generator=g, dtype=torch.float32)

    # 1. reducao: peso uniforme == sem peso. O teste que justifica a formula.
    sem = relative(ref, got)
    for valor in (1.0, 0.25, 7.0):
        com = relative(ref, got, torch.full((64,), valor))
        check(f"peso uniforme {valor} reduz ao caso sem peso",
              math.isclose(sem, com, rel_tol=1e-6), f"{sem!r} contra {com!r}")

    # 2. o peso realmente move o numero quando o erro nao e uniforme
    sujo = ref.clone()
    sujo[:32] += 5.0  # metade das linhas com erro grande
    so_sujas = relative(ref, sujo, torch.tensor([1.0] * 32 + [0.0] * 32))
    so_limpas = relative(ref, sujo, torch.tensor([0.0] * 32 + [1.0] * 32))
    check("peso concentrado nas linhas ruins da erro maior",
          so_sujas > relative(ref, sujo) > so_limpas,
          f"{so_sujas!r} / {relative(ref, sujo)!r} / {so_limpas!r}")
    check("peso nas linhas boas zera o erro", math.isclose(so_limpas, 0.0, abs_tol=1e-6),
          f"{so_limpas!r}")

    # 3. modos
    sig = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    check("modo none devolve None", sigma_weights("none", sig, "L") is None)
    check("modo sigma e o proprio sigma",
          torch.allclose(sigma_weights("sigma", sig, "L"), sig))
    check("modo sigma2 e o quadrado",
          torch.allclose(sigma_weights("sigma2", sig, "L"), sig.pow(2)))
    alto = sigma_weights("high", sig, "L")
    check("modo high e um corte na mediana observada",
          torch.equal(alto, (sig >= sig.median()).float()), f"{alto.tolist()}")
    check("modo high nao usa limiar fixo",
          torch.equal(sigma_weights("high", sig * 100, "L"), alto),
          "escalar o sigma mudou o corte, entao o corte e absoluto e nao relativo")

    # 4. recusas -- cada uma existe porque a alternativa e falhar em silencio
    esperar_saida("sem sample_sigma recusa",
                  lambda: sigma_weights("sigma", None, "L0"), "sample_sigma")
    esperar_saida("NaN no sigma recusa em vez de virar zero",
                  lambda: sigma_weights("sigma", torch.tensor([1.0, float("nan")]), "L0"),
                  "sem sigma")
    esperar_saida("peso todo zero recusa",
                  lambda: sigma_weights("sigma", torch.zeros(4), "L0"), "somou")
    esperar_saida("modo desconhecido recusa",
                  lambda: sigma_weights("sigma3", sig, "L0"), "desconhecido")
    esperar_saida("numero de pesos diferente do de linhas recusa",
                  lambda: relative(ref, got, torch.ones(8)), "linhas")

    print()
    print("NAO COBERTO por este arquivo: nenhum kernel, nenhuma GPU, nenhuma calibracao real.")
    print("  Se o sigma gravado pela calibracao nao corresponder ao passo real do sampler, ou se")
    print("  a ponderacao nao mudar decisao nenhuma no modelo de verdade, nada aqui percebe.")
    print(f"\n{'FALHOU' if FALHAS else 'TUDO PASSOU'}  ({len(FALHAS)} falhas)")
    return 1 if FALHAS else 0


if __name__ == "__main__":
    raise SystemExit(main())
