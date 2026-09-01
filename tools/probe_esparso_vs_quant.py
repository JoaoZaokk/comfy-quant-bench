"""Esparsidade 2:4 contra quantizacao 4 bits, MESMAS camadas, MESMAS ativacoes, MESMA metrica.

Eu disse ao dono que "esparsidade custa ~3x o que a quantizacao custa". Aquilo comparou erro de
PESO da poda contra erro de SAIDA da quantizacao -- duas metricas diferentes, e por isso nao valia.
Aqui tudo e erro de saida na ativacao calibrada real.

Tres bracos por camada, e o quarto e a composicao, que e a pergunta de verdade:

    W4A4        so quantizacao, o que ja fazemos
    2:4 Wanda   so esparsidade, criterio guiado por ativacao
    2:4 cru     so esparsidade, criterio por magnitude -- o controle que mostra quanto o
                criterio importa, sem ele um numero bom de Wanda nao se distingue de sorte
    ambos       poda 2:4 e DEPOIS quantiza o que sobrou
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
CALIB = RAIZ / "calib" / "zimage_v2_sigma.calib.pt"
CG = 256


def peso(k):
    with MODELO.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        i = h[k + ".weight"]
        a, b = i["data_offsets"]
        f.seek(base + a)
        cru = bytearray(f.read(b - a))
    return torch.frombuffer(cru, dtype=torch.bfloat16).reshape(i["shape"]).cuda()


def poda24(w, escore):
    N, K = w.shape
    g = escore.reshape(N, K // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    m = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
    return w.reshape(N, K // 4, 4).mul(m).reshape(N, K)


def q4(x, w):
    q, s = ck.quantize_convrot_w4a4_weight(w.contiguous(), CG, 64)
    return ck.convrot_w4a4_linear(x, q, s, None, CG).float()


def main() -> int:
    d = torch.load(CALIB, map_location="cpu", weights_only=False)
    cam = d["layers"]
    alvos = [k for k in cam if k.startswith("layers.")][:12]

    print(f"{'camada':32s} {'W4A4':>8s} {'2:4 cru':>8s} {'2:4 Wanda':>10s} {'ambos':>8s}")
    print("-" * 72)
    acc = {"q": [], "c": [], "w": [], "a": []}
    for k in alvos:
        e = cam[k]
        X = e["sample"].cuda().to(torch.bfloat16)
        W = peso(k)
        if X.shape[-1] != W.shape[1]:
            continue
        ref = (X.float() @ W.float().T)
        nrm = ref.norm()
        rel = lambda y: ((y - ref).norm() / nrm).item()

        norma = X.float().norm(dim=0)
        Wc = poda24(W.float(), W.float().abs())
        Ww = poda24(W.float(), W.float().abs() * norma.unsqueeze(0))

        r = {
            "q": rel(q4(X, W)),
            "c": rel(X.float() @ Wc.T),
            "w": rel(X.float() @ Ww.T),
            "a": rel(q4(X, Ww.to(torch.bfloat16))),
        }
        for kk, v in r.items():
            acc[kk].append(v)
        print(f"{k:32s} {r['q']:8.4f} {r['c']:8.4f} {r['w']:10.4f} {r['a']:8.4f}")

    if acc["q"]:
        print()
        print(f"{'MEDIANA':32s} {st.median(acc['q']):8.4f} {st.median(acc['c']):8.4f} "
              f"{st.median(acc['w']):10.4f} {st.median(acc['a']):8.4f}")
        print()
        # Razao abaixo de 1 vai INVERTIDA e com a direcao no nome. "0.86x" nao diz quem ganhou,
        # e esta bancada ja registrou o custo de publicar um numero nessa forma.
        def contra_w4a4(chave: str, rotulo: str) -> None:
            r = st.median(acc[chave]) / st.median(acc["q"])
            if r <= 1.0:
                print(f"  {rotulo}: {1/r:.2f}x MAIS FIEL que o W4A4")
            else:
                print(f"  {rotulo}: {r:.2f}x MENOS fiel que o W4A4")
        contra_w4a4("w", "2:4 Wanda")
        contra_w4a4("a", "2:4 Wanda + W4A4")
    print()
    print("NAO COBERTO: erro de saida na ativacao calibrada, NAO imagem -- e esta bancada ja mediu")
    print("  que nenhum corte nesse eixo separa usavel de inutilizavel. Sem SparseGPT, sem treino de")
    print("  recuperacao (a proposta original do dono). Um modelo, uma calibragem, 12 camadas.")
    print("  CORRIGIDO 2026-09-01: esta linha dizia que 2:4 nao executa em tensor core aqui por")
    print("  falta de cuSPARSELt. Executa, e o CUTLASS nao usa cuSPARSELt -- o que barrava era um")
    print("  tile dimensionado para a A100 (139.264 bytes de shared contra os 101.376 desta")
    print("  placa). Medido em tools/sparse24_sm86/: 1,7x a 1,95x mais rapido que o denso bf16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
