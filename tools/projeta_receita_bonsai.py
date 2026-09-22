"""Se a receita do Bonsai fosse aplicada a ESTE modelo, que arquivo sairia? (ARITMETICA, nao medicao)

Responde "matematicamente, como ficaria noutro modelo" sem treinar nada. Le so o header do
safetensors -- zero commit, nenhum peso carregado -- classifica cada tensor e faz a conta de bytes.

A regra de classificacao **saiu do dado, nao de um chute**, e a primeira versao deste arquivo errou
justamente por chutar. Ela tinha uma lista de nomes (`norm`, `proj_out`, `modulation`, ...) e a
calibracao contra o klein-4B medido reprovou: 105 camadas em vez de 100 e 52.698.624 params densos em
vez de 195.042.816. O que a lista nao pegava: `double_stream_modulation_img` (sufixo depois de
`modulation`), `norm_out.linear` (sufixo depois de `norm`) e `time_guidance_embed` (a lista dizia
`time_text_embed`, nome de outra arquitetura).

Olhando os tensores crus, a regra verdadeira e muito mais simples e nao tem lista de nome nenhuma:

    **quantizado = todo tensor 2-D DENTRO de um bloco do transformer.**
    **denso      = tudo 1-D, e todo 2-D FORA dos blocos.**

No klein-4B isso da exatamente `5 blocos duplos x 12 + 20 blocos simples x 2 = 100` camadas, que e o
numero que o pack deles declara, e 195.042.816 params densos, que e exatamente os 390.085.632 B de
bf16 que o pack carrega. Bate nos dois lados, sem ajuste.

E a regra nao e so da Prism: a Nunchaku (SVDQuant int4, um PTQ, outro metodo, outro time) deixa
**exatamente os mesmos 69 tensores densos** no mesmo modelo, intersecao 69 de 69. Entao "o que fica
denso" e propriedade da ARQUITETURA.

Custo por peso quantizado, MEDIDO nos packs deles (`bonsai_packs_resultado_2026-09-22.md`):

    gemlite int2  2,5000 bit   slot 2 + duas tabelas fp32 por grupo de 128
    MLX 2bit      2,2500 bit   as mesmas tabelas em bf16
    gemlite int1  1,5000 bit
    MLX 1bit      1,2500 bit

**O QUE ISTO NAO E:** nao e previsao de qualidade. A receita deles TREINA, e nada aqui diz que o
treino funcionaria neste modelo. E conta de tamanho de arquivo e de onde fica o piso -- util
justamente porque o piso e a parte que nao depende de treino nenhum.
"""
from __future__ import annotations

import argparse
import json
import mmap
import re
from collections import Counter
from pathlib import Path

# Um "bloco" e um prefixo que termina em `.<numero>.`: e como diffusers nomeia pilha repetida.
# Detectado, nao listado, para nao ter de manter nome por arquitetura.
BLOCO = re.compile(r"^(?P<pilha>[A-Za-z_][\w.]*?)\.(?P<i>\d+)\.")
CUSTO = {
    "gemlite int2 (ternario)": 2.50,
    "MLX 2bit    (ternario)": 2.25,
    "gemlite int1 (binario)": 1.50,
    "MLX 1bit    (binario)": 1.25,
}
BITS_DENSO = 16  # bf16, como nos dois packs


def header(caminho: Path) -> dict:
    """Header por mmap SOMENTE LEITURA: custa 0 de commit, ao contrario de `safe_open`."""
    with open(caminho, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            n = int.from_bytes(mm[:8], "little")
            h = json.loads(mm[8:8 + n])
        finally:
            mm.close()
    h.pop("__metadata__", None)
    return h


def classifica(h: dict, grupo: int) -> dict:
    """quantizado = 2-D dentro de bloco com K divisivel; denso = o resto. Regra medida, nao chutada."""
    # Passo 1: uma pilha de UM nao e pilha. `final_layer.adaLN_modulation.1` casa o padrao
    # `.<numero>.` porque o 1 e indice de nn.Sequential, nao de bloco -- e nos dois Flux isso
    # promovia 18.874.368 params de modulacao para "quantizavel", que e o oposto da regra medida.
    # Uma pilha real tem varios indices distintos.
    indices: dict[str, set[str]] = {}
    for k in h:
        m = BLOCO.match(k)
        if m:
            indices.setdefault(m.group("pilha"), set()).add(m.group("i"))
    pilha_real = {nome for nome, i in indices.items() if len(i) >= 2}

    q = d = nq = 0
    motivo: Counter[str] = Counter()
    pilhas: Counter[str] = Counter()
    fora: list[tuple[str, int]] = []
    for k, v in h.items():
        forma = v["shape"]
        n = 1
        for x in forma:
            n *= x
        m = BLOCO.match(k)
        if len(forma) != 2:
            d += n
            motivo[f"{len(forma)}-D (nao e matriz de Linear)"] += 1
        elif m is None or m.group("pilha") not in pilha_real:
            d += n
            motivo["2-D fora de qualquer pilha de blocos"] += 1
            fora.append((k, n))
        elif forma[1] % grupo:
            d += n
            motivo[f"2-D no bloco mas K={forma[1]} nao divide por {grupo}"] += 1
        else:
            q += n
            nq += 1
            pilhas[m.group("pilha")] += 1
    return {"q": q, "d": d, "nq": nq, "motivo": motivo, "pilhas": pilhas,
            "fora": sorted(fora, key=lambda t: -t[1])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("arquivo", nargs="+")
    p.add_argument("--grupo", type=int, default=128)
    p.add_argument("--esperado-camadas", type=int, default=None,
                   help="calibracao: falha se o numero de camadas quantizaveis diferir")
    p.add_argument("--esperado-densos", type=int, default=None,
                   help="calibracao: falha se os params densos diferirem")
    a = p.parse_args()

    falhas = 0
    for caminho in a.arquivo:
        cam = Path(caminho)
        if not cam.is_file():
            print(f"\n=== {cam.name}: NAO EXISTE, pulado ===")
            continue
        h = header(cam)
        r = classifica(h, a.grupo)
        q, d, nq = r["q"], r["d"], r["nq"]
        total = q + d
        print(f"\n=== {cam.name} ===")
        print(f"  {len(h)} tensores, {total:,} parametros, arquivo atual {cam.stat().st_size/2**30:.2f} GiB")
        print(f"  quantizaveis  {nq:4d} camadas  {q:15,d} params  {q/total*100:5.2f}%")
        for pilha, c in r["pilhas"].most_common():
            print(f"       {c:4d} em {pilha}")
        print(f"  densos                       {d:15,d} params  {d/total*100:5.2f}%")
        for m, c in r["motivo"].most_common():
            print(f"       {c:4d} tensores: {m}")
        if r["fora"]:
            print("     os 2-D de fora dos blocos, que sao o piso:")
            for k, n in r["fora"][:8]:
                print(f"       {n:13,d}  {k}")
        piso = d * BITS_DENSO / 8
        print(f"  {'receita':26s} {'arquivo':>12s} {'bit/peso medio':>15s} {'denso e':>9s}")
        for nome, bits in CUSTO.items():
            b = q * bits / 8 + piso
            print(f"  {nome:26s} {b/2**30:9.2f} GiB {b*8/total:15.4f} {piso/b*100:8.2f}%")
        print(f"  piso (so os densos em bf16): {piso/2**30:.2f} GiB -- nenhuma receita desce disso")

        if a.esperado_camadas is not None and nq != a.esperado_camadas:
            print(f"  CALIBRACAO FALHOU: {nq} camadas, esperado {a.esperado_camadas}")
            falhas += 1
        if a.esperado_densos is not None and d != a.esperado_densos:
            print(f"  CALIBRACAO FALHOU: {d:,} params densos, esperado {a.esperado_densos:,}")
            falhas += 1
        if a.esperado_camadas is not None and not falhas:
            print("  CALIBRACAO OK: reproduz os numeros medidos no pack.")

    print("\n=== NAO COBERTO ===")
    print("  ARITMETICA, nao medicao. Nenhum peso lido alem do header, nada treinado, nada")
    print("  renderizado, GPU nao tocada. NAO diz se o treino a 1,58 bit funcionaria neste modelo:")
    print("  diz o tamanho do arquivo e onde esta o piso, que e a parte que nao depende do treino.")
    print("  A regra foi calibrada NO klein-4B. Noutra arquitetura ela e extrapolacao, e este repo")
    print("  ja registra TRES extrapolacoes entre arquiteturas que morreram na medicao.")
    print("  Um arquivo ja quantizado (fp8, int4) tem os pesos no dtype dele: a coluna de params")
    print("  vale, a comparacao com o 'arquivo atual' nao e contra bf16.")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
