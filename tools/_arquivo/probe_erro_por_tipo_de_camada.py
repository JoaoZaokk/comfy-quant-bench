"""Quanto custa quantizar `adaLN_modulation` a 4 bits? Ninguem mediu, so repetiu que nao se faz.

Tres fontes independentes (Comfy-Org oficial em INT8, tritant, nos) evitam 4 bits ali. Consenso nao
e medicao: pode significar que todos copiaram a mesma suposicao. Este arquivo mede.

O erro e de RECONSTRUCAO DO PESO: quantiza, recupera o peso efetivo passando a identidade pelo
kernel real, e compara com o original. Nao e o erro de saida em ativacao real -- e um limite
inferior barato, e serve para comparar tipos de camada entre si.
"""
import json
import statistics as st
import struct
import sys
from pathlib import Path

RAIZ = Path(r"F:\COMFY_PORTABLE")
sys.path.insert(0, str(RAIZ / "ComfyUI"))

import torch  # noqa: E402
import comfy_kitchen as ck  # noqa: E402

MODELO = RAIZ / "ComfyUI" / "models" / "diffusion_models" / "beyond-reality-zimage-v2_native.safetensors"


def carregar():
    with MODELO.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def peso(h, base, k):
    i = h[k]
    a, b = i["data_offsets"]
    with MODELO.open("rb") as f:
        f.seek(base + a)
        cru = bytearray(f.read(b - a))
    return torch.frombuffer(cru, dtype=torch.bfloat16).reshape(i["shape"]).cuda()


def erro(w, cg=256):
    """Erro relativo do peso efetivo que o kernel REAL produz, contra o peso original.

    A identidade e passada pelo `convrot_w4a4_linear` de verdade em vez de desempacotar o INT4 a
    mao: desempacotar seria uma segunda implementacao do decodificador, e ela poderia divergir do
    kernel exatamente onde importa.
    """
    q, s = ck.quantize_convrot_w4a4_weight(w, cg, 64)
    ident = torch.eye(w.shape[1], device="cuda", dtype=torch.bfloat16)
    efetivo = ck.convrot_w4a4_linear(ident, q, s, None, cg).float().T
    return ((efetivo - w.float()).norm() / w.float().norm()).item()


def main() -> int:
    h, base = carregar()
    grupos: dict[str, list[str]] = {}
    for k in h:
        if not k.endswith(".weight") or len(h[k]["shape"]) < 2 or not k.startswith("layers."):
            continue
        grupos.setdefault(k.split(".", 2)[2][: -len(".weight")], []).append(k)

    print("erro relativo de reconstrucao do peso em convrot_w4a4, por tipo de camada")
    print(f"{'tipo':32s} {'n':>3s} {'K':>6s}   mediana     min       max")
    print("-" * 72)
    linhas = []
    for tipo, ks in sorted(grupos.items()):
        amostra = ks[:8]
        es = [erro(peso(h, base, k)) for k in amostra]
        K = h[amostra[0]]["shape"][1]
        linhas.append((tipo, len(ks), K, st.median(es)))
        print(f"{tipo:32s} {len(ks):3d} {K:6d}   {st.median(es):.4f}    {min(es):.4f}   {max(es):.4f}")

    ada = next((x for x in linhas if "adaLN" in x[0]), None)
    outros = [x[3] for x in linhas if "adaLN" not in x[0]]
    if ada and outros:
        print()
        print(f"adaLN vs mediana dos demais: {ada[3]:.4f} contra {st.median(outros):.4f} "
              f"-> {ada[3] / st.median(outros):.2f}x")
    print()
    print("NAO COBERTO: erro de PESO, nao de saida em ativacao real -- e um limite inferior. Nao ha")
    print("  render aqui. Uma camada por tipo em amostra de 8, um modelo, um cg. E erro de peso nao")
    print("  diz o que a modulacao FAZ: ela alimenta escala e deslocamento do bloco inteiro, entao")
    print("  o mesmo erro relativo pode custar mais ali do que numa MLP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
