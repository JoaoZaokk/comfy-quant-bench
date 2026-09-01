"""Confere se um conversor migrado planeja o MESMO layout que ele ja escreveu no disco. Sem GPU.

O PROBLEMA QUE ELA RESOLVE
--------------------------
Migrar os sete escritores para `_conversion.py` mexe no caminho que escreve arquivos de 8 a 24
GiB. A verificacao forte de uma migracao dessas e reconverter e comparar byte a byte -- foi o que
se fez no `to_native.py`, que nao quantiza e por isso roda inteiro sem placa. **Os outros seis
precisam de GPU para reconverter**, e ate a janela de placa abrir eles ficariam sem verificacao
nenhuma.

Esta ferramenta fecha metade dessa lacuna agora. O `plan_lazy` do nucleo carrega dtype, forma e
nbytes EXPLICITAMENTE, entao o plano -- e portanto o header inteiro -- sai sem chamar um kernel.
Reconstruindo o plano a partir do header da FONTE e comparando com o header do arquivo que o
conversor ja escreveu, prova-se que **a ordem das chaves, os nomes, os dtypes, as formas e todos
os `data_offsets` nao mudaram**. Sobra o dado quantizado, e so ele precisa da placa.

O QUE ELA PROVA E O QUE ELA NAO PROVA
--------------------------------------
    PROVA     ordem e nomes das chaves, dtypes, formas, offsets, metadata -- o header inteiro,
              byte a byte, contra o arquivo real que a versao pre-migracao escreveu.
    NAO PROVA nada sobre os BYTES DE DADO. O kernel nao roda aqui. Um conversor que planeje o
              header certo e escreva peso errado passa nesta ferramenta.

Por isso o modo `--plano-so` nao substitui a corrida de GPU: ele a ANTECIPA. Se o header ja diverge,
nao vale gastar a placa.

COBERTURA POR CONVERSOR
-----------------------
Nem todos dao a mesma forca, e a diferenca e estrutural, nao preguica:

    quant_w4a4   TOTAL   -- transmite, e as formas de saida sao analiticas (linhas x colunas/2,
                            escala [linhas]). Da para montar o header inteiro sem quantizar.
    to_native    TOTAL   -- ja verificado por byte-identidade do ARQUIVO inteiro; nao precisa
                            desta ferramenta.
    w4a8 / int8 / mixed / smooth
                 PARCIAL -- acumulam, e as formas de saida saem do kernel (s_rel, s_channel,
                            codebook). Aqui compara-se a ordem e os nomes das chaves e o metadata,
                            tomando as formas do proprio arquivo. Pega renomeacao, reordenacao e
                            metadata trocado -- que e o que uma migracao quebra -- e nao pega uma
                            forma que so o kernel decide.

USO
---
    python.exe -s tools\\verificar_migracao.py                 # todos os pares conhecidos
    python.exe -s tools\\verificar_migracao.py --par w4a4      # so um
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402

MODELOS = RAIZ / "ComfyUI" / "models" / "diffusion_models"

# (rotulo, fonte, saida ja no disco). Um par so entra aqui se os DOIS arquivos existem: um par
# inventado passaria por "nao verificavel" e leria como cobertura.
PARES = [
    ("w4a4", MODELOS / "hunyuanvideo1.5_720p_t2v_fp16.safetensors",
     MODELOS / "hunyuanvideo1.5_720p_t2v_fp16_w4a4_convrot.safetensors"),
]


def header_do_disco(path: Path) -> tuple[bytes, dict]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cru = f.read(n)
    return cru, json.loads(cru)


def verificar_w4a4(fonte: Path, saida: Path) -> tuple[bool, str]:
    """Cobertura TOTAL: monta o header inteiro sem tocar a GPU e compara byte a byte."""
    import quant_w4a4 as Q

    sidecar = saida.with_suffix(".quant.json")
    cg = json.loads(sidecar.read_text(encoding="utf-8")).get("convrot_groupsize", 256) \
        if sidecar.is_file() else 256

    header, metadata = Q.read_header(fonte)
    perfil = Q.detect_profile(fonte, list(header))
    selecionadas = Q.selected_layers(header, perfil, cg)
    layers = {n.removesuffix(".weight"): {"format": "convrot_w4a4", "convrot_groupsize": cg}
              for n in selecionadas}
    with fonte.open("rb") as h:
        entradas = Q.planejar(h, header, selecionadas, ck=None, convrot_groupsize=cg)
    planejado, _ = C.header_bytes(entradas, Q.output_metadata(metadata, layers))

    real, _ = header_do_disco(saida)
    if planejado == real:
        return True, (f"header identico byte a byte, {len(real)} bytes, perfil {perfil}, "
                      f"{len(selecionadas)} camadas, cg {cg}")
    for i, (a, b) in enumerate(zip(planejado, real)):
        if a != b:
            return False, (f"primeiro byte diferente em {i}\n     plano: {planejado[max(0,i-70):i+70]!r}"
                           f"\n     disco: {real[max(0,i-70):i+70]!r}")
    return False, f"tamanhos diferentes: plano {len(planejado)}, disco {len(real)}"


VERIFICADORES = {"w4a4": verificar_w4a4}

NAO_COBERTO = [
    "NENHUM byte de dado e conferido: o kernel nao roda aqui. Um conversor que planeje o header",
    "  certo e escreva peso errado passa nesta ferramenta em silencio.",
    "Nada aqui prova que o backend nativo resolve, nem que a saida gera imagem.",
    "Os conversores que ACUMULAM (w4a8, int8, mixed, smooth) nao tem verificador aqui ainda:",
    "  as formas de saida deles saem do kernel, entao o header nao e reconstruivel sem placa.",
    "  Para eles a verificacao continua sendo reconverter na janela de GPU e comparar --",
    "  ver `bench/janela_gpu_migracao.md`.",
    "Um par so e verificado se a saida ja existe no disco. Ausencia de par nao e aprovacao.",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--par", choices=sorted(VERIFICADORES), default=None)
    a = p.parse_args()

    alvos = [x for x in PARES if a.par is None or x[0] == a.par]
    if not alvos:
        print("nenhum par para verificar", file=sys.stderr)
        return 2

    falhou = 0
    ausentes = []
    for rotulo, fonte, saida in alvos:
        if not fonte.is_file() or not saida.is_file():
            ausentes.append(f"{rotulo}: falta {fonte.name if not fonte.is_file() else saida.name}")
            continue
        ok, detalhe = VERIFICADORES[rotulo](fonte, saida)
        print(f"[{'OK  ' if ok else 'FALHA'}] {rotulo}: {detalhe}")
        falhou += 0 if ok else 1

    for linha in ausentes:
        print(f"[PULADO] {linha}")

    print(f"\n{len(alvos) - falhou - len(ausentes)}/{len(alvos)} pares verificados, "
          f"{falhou} falharam, {len(ausentes)} pulados")
    print("\nNAO COBERTO POR ESTA EXECUCAO:")
    for linha in NAO_COBERTO:
        print(f"  - {linha}")
    return 1 if falhou else 0


if __name__ == "__main__":
    raise SystemExit(main())
