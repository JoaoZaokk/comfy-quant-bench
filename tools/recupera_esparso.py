r"""Recuperacao por camada sobre poda 2:4 -- a proposta do dono, e a unica frente aberta na esparsidade.

O QUE ESTA EM JOGO
------------------
Medido em 2026-09-01 nesta bancada: o tensor core esparso da sm_80 executa 2:4 a 1,7x (bf16),
3,2-4,1x (int8) e 5,0-7,6x (int4) sobre o denso, bit-exato. E a foto disse nao. Isolando um eixo
por vez, o culpado e a GRANULARIDADE que o INT4 impoe -- a mascara dele e o PAR, e podar em pares
destroi a imagem mesmo em bf16, sem quantizar nada. O INT8 aceita por ELEMENTO, sobrevive a 4,0x
com 5,0 bits/peso, e o sujeito continua reconhecivel -- so que ruim demais para enviar.

Recuperacao e o que fecha essa distancia, e nunca foi testada aqui.

O METODO, E POR QUE ESTE E NAO OUTRO
-------------------------------------
Nao e fine-tuning. E reconstrucao por camada: com a mascara `M` ja escolhida, achar os pesos
sobreviventes que minimizam o erro de SAIDA nas ativacoes reais,

    min_{Ws}  || X Ws^T - X W^T ||_F^2      sujeito a   Ws = M (*) Ws

Isso e um minimos quadrados por LINHA de saida, e o sistema de cada linha e a Hessiana
`H = X^T X` restrita ao suporte daquela linha. Resolver linha a linha seria [K x K] por linha --
3840 sistemas de 1920x1920 no Z-Image, alguns minutos por camada. Em vez disso roda-se **gradiente
conjugado mascarado**: o operador `V -> M (*) (V H)` e simetrico definido positivo dentro do
suporte, e o CG trata TODAS as linhas em paralelo, porque cada iteracao e um unico produto
[N x K] @ [K x K]. Sai exato no limite e custa dezenas de matmuls, nao milhares de solves.

`H` e `X^T X` acumulada em float64 com amortecimento `lambda = damp * mean(diag(H))`: com menos
linhas amostradas que colunas, `H` e singular por construcao e sem amortecimento o CG anda para o
espaco nulo, onde o erro de treino cai e o de teste explode.

OS CONTROLES, E O QUE CADA UM PODE DERRUBAR
--------------------------------------------
1. **denso** -- a mesma recuperacao com mascara toda de uns. TEM de dar erro ~0. Se nao der, o
   solver esta errado e nenhuma outra linha desta tabela significa nada. Este e o controle que
   tem de passar, no sentido da memoria `controle-que-tem-que-passar`: sem ele, um solver que nao
   faz nada produz "recuperacao nao ajudou" e parece resultado.
2. **aleatorio** -- poda 2:4 com criterio ALEATORIO no lugar do Wanda, recuperada igual. Se a
   recuperacao salvar as duas na mesma medida, ela esta compensando a poda em vez de aproveitar o
   criterio, e o Wanda nao esta fazendo nada -- o que mudaria a conclusao de 2026-09-01 de que
   Wanda vale 2,20x.
3. **par** -- a granularidade que o INT4 forca, recuperada. Diz se recuperacao ressuscita o
   caminho que a foto matou, ou se o buraco e fundo demais.

NAO COBERTO
-----------
Erro de SAIDA por camada nas ativacoes calibradas -- **nao imagem**, e esta bancada ja mediu tres
instrumentos numericos apontando para o lado errado no mesmo dia (erro por camada 1,46x pior com
imagem destruida; divergencia SUBINDO com a imagem melhorando). Este arquivo ordena candidatos
para o render; ele nao decide nada. A reconstrucao aqui e por camada INDEPENDENTE: cada uma ve a
entrada limpa, nao a ja degradada pelas anteriores, que e o que o SparseGPT faz e e estritamente
mais forte. Um modelo, uma calibragem, uma placa. Sem quantizacao junto no braco principal --
`--quantiza` liga o int8 por linha para medir os dois efeitos somados.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import struct
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

import torch


def ler_peso(modelo: Path, chave: str) -> torch.Tensor:
    """Le um tensor pelo intervalo de bytes, sem mmap e sem carregar o arquivo inteiro.

    Mesma razao do resto do repo: mapear os arquivos grandes deste projeto ja derrubou o
    `torch_cpu.dll` com 0xc0000005 neste host.
    """
    with modelo.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        info = head[chave + ".weight"]
        a, b = info["data_offsets"]
        f.seek(base + a)
        cru = bytearray(f.read(b - a))
    return torch.frombuffer(cru, dtype=torch.bfloat16).reshape(info["shape"]).cuda()


def mascara_elemento(w: torch.Tensor, criterio: torch.Tensor) -> torch.Tensor:
    """2 dos 4 de cada grupo de 4 -- a granularidade que o tensor core INT8 aceita."""
    n, k = w.shape
    g = criterio.reshape(n, k // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    m = torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False)
    return m.reshape(n, k)


def mascara_par(w: torch.Tensor, criterio: torch.Tensor) -> torch.Tensor:
    """2 dos 4 PARES de cada grupo de 8 -- a granularidade que o INT4 impoe."""
    n, k = w.shape
    escore = criterio.reshape(n, k // 8, 4, 2).sum(-1)
    idx = escore.argsort(dim=-1)[..., :2]
    m = torch.ones_like(escore, dtype=torch.bool).scatter_(-1, idx, False)
    return m.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(n, k)


def recupera(w: torch.Tensor, m: torch.Tensor, h: torch.Tensor,
             iteracoes: int, tol: float) -> tuple[torch.Tensor, dict]:
    """Gradiente conjugado mascarado: minimiza ||(Ws - W) H^(1/2)||^2 com Ws = M (*) Ws.

    O operador `A(V) = M (*) (V H)` e simetrico definido positivo no subespaco do suporte, entao
    CG vale e converge sem precisar formar nem inverter nada. Cada iteracao e UM matmul
    [N x K] @ [K x K], compartilhado por todas as linhas -- e por isso que isto roda em segundos
    onde um solve por linha levaria minutos.
    """
    w = w.double()
    h = h.double()
    m = m.double()
    # Ponto de partida: o proprio peso podado. O residuo mede quanto do erro ja foi removido.
    ws = w * m
    def A(v):
        return m * (v @ h)
    r = m * ((w - ws) @ h)
    p = r.clone()
    rr = (r * r).sum()
    rr0 = rr.clone()
    historico = []
    for it in range(iteracoes):
        ap = A(p)
        pap = (p * ap).sum()
        if pap <= 0:
            historico.append(("pap<=0", it))
            break
        alfa = rr / pap
        ws = ws + alfa * p
        r = r - alfa * ap
        rr_novo = (r * r).sum()
        historico.append(float((rr_novo / rr0).sqrt()))
        if rr_novo <= tol * tol * rr0:
            break
        p = r + (rr_novo / rr) * p
        rr = rr_novo
    return (ws * m).to(torch.bfloat16), {"iteracoes": it + 1,
                                         "residuo_relativo": historico[-1] if historico else None}


def int8_por_linha(w: torch.Tensor) -> torch.Tensor:
    """Quantiza simetrico por linha para [-127,127] e devolve o RECONSTRUIDO, em float.

    Reconstruido e nao inteiro porque todas as linhas da tabela medem erro de saida contra a
    mesma referencia float32: a quantizacao tem de aparecer como perturbacao do peso, nao como
    mudanca de unidade.
    """
    e = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    return (w / e).round().clamp(-127, 127) * e


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", type=Path,
                   default=RAIZ / "ComfyUI/models/diffusion_models/"
                                  "beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--calib", type=Path, default=RAIZ / "calib/zimage_v2_sigma.calib.pt")
    p.add_argument("--camadas", type=int, default=24)
    p.add_argument("--damp", type=float, default=0.01,
                   help="amortecimento da Hessiana, fracao da media da diagonal")
    p.add_argument("--iteracoes", type=int, default=256)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--quantiza", action="store_true",
                   help="aplica int8 simetrico por linha DEPOIS da recuperacao, para medir os "
                        "dois efeitos somados em vez de so a esparsidade")
    p.add_argument("--saida", type=Path, default=RAIZ / "bench" / "recuperacao_esparsa.json")
    a = p.parse_args()

    d = torch.load(a.calib, map_location="cpu", weights_only=False)
    chaves = [k for k in d["layers"] if k.startswith("layers.")][: a.camadas]

    colunas = ["denso_rec", "elem", "elem_rec", "alea", "alea_rec", "par", "par_rec"]
    acc = {c: [] for c in colunas}
    linhas_json = []
    cab = f"{'camada':28s} " + " ".join(f"{c:>10s}" for c in colunas)
    print(cab)
    print("-" * len(cab))

    for k in chaves:
        x = d["layers"][k]["sample"].cuda().to(torch.bfloat16)
        w = ler_peso(a.modelo, k)
        if x.shape[-1] != w.shape[1] or w.shape[1] % 8:
            continue
        xf = x.float().reshape(-1, x.shape[-1])
        ref = xf @ w.float().T
        nrm = ref.norm()

        def rel(wx):
            return ((xf @ wx.float().T - ref).norm() / nrm).item()

        # A Hessiana em float64: `X^T X` acumula rapido e em float32 a diagonal de camadas com
        # ativacao grande (o Z-Image chega a 344064) perde digitos suficientes para o CG divergir.
        h = (xf.double().T @ xf.double())
        h += torch.eye(h.shape[0], device=h.device, dtype=h.dtype) * (a.damp * h.diag().mean())

        wanda = w.abs().float() * xf.norm(dim=0).unsqueeze(0)
        gerador = torch.Generator(device=w.device).manual_seed(0)
        aleat = torch.rand(w.shape, device=w.device, generator=gerador)

        m_elem = mascara_elemento(w, wanda)
        m_alea = mascara_elemento(w, aleat)
        m_par = mascara_par(w, wanda)
        m_denso = torch.ones_like(m_elem)

        def arma(mask):
            wr, _ = recupera(w, mask, h, a.iteracoes, a.tol)
            if a.quantiza:
                wr = int8_por_linha(wr.float()).to(torch.bfloat16)
            return rel(wr)

        r = {"denso_rec": arma(m_denso),
             "elem": rel(w * m_elem), "elem_rec": arma(m_elem),
             "alea": rel(w * m_alea), "alea_rec": arma(m_alea),
             "par": rel(w * m_par), "par_rec": arma(m_par)}
        for c in colunas:
            acc[c].append(r[c])
        linhas_json.append({"layer": k, **r})
        print(f"{k:28s} " + " ".join(f"{r[c]:10.4f}" for c in colunas))
        del h, xf, ref
        torch.cuda.empty_cache()

    if not acc["elem"]:
        print("nenhuma camada medida", file=sys.stderr)
        return 2
    med = {c: st.median(v) for c, v in acc.items()}
    print()
    print(f"{'MEDIANA':28s} " + " ".join(f"{med[c]:10.4f}" for c in colunas))

    # O CONTROLE QUE TEM DE PASSAR, e ele decide se o resto pode ser lido.
    print()
    ok = med["denso_rec"] < 1e-3
    print(f"CONTROLE denso: recuperacao com mascara de uns mede {med['denso_rec']:.2e}   "
          f"{'PASSOU (< 1e-3)' if ok else 'FALHOU -- o solver esta errado, NAO leia a tabela'}")

    # O segundo controle: se a recuperacao salvar Wanda e aleatorio na mesma medida, o criterio
    # nao esta fazendo nada e o 2,20x de 2026-09-01 nao sobrevive.
    g_elem = med["elem"] / med["elem_rec"] if med["elem_rec"] else float("inf")
    g_alea = med["alea"] / med["alea_rec"] if med["alea_rec"] else float("inf")
    print(f"CONTROLE criterio: Wanda recupera {g_elem:.2f}x, aleatorio recupera {g_alea:.2f}x   "
          f"({'o criterio importa' if med['elem_rec'] < med['alea_rec'] * 0.9 else 'INDISTINGUIVEL: a recuperacao esta compensando a poda'})")

    print()
    print(f"{'formato':34s} {'bits/peso':>9s} {'erro':>9s} {'ganho da recuperacao':>22s}")
    print("-" * 78)
    largura = 8.0 if not a.quantiza else 4.5
    for nome, bruto, rec, bits in (
            ("2:4 por ELEMENTO (kernel int8)", "elem", "elem_rec", largura),
            ("2:4 por PAR (kernel int4)", "par", "par_rec", largura),
            ("2:4 elemento, criterio ALEATORIO", "alea", "alea_rec", largura)):
        razao = med[bruto] / med[rec] if med[rec] else float("inf")
        print(f"{nome:34s} {bits:9.1f} {med[rec]:9.4f} "
              f"{f'{razao:.2f}x mais fiel':>22s}")

    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(
        {"camadas": linhas_json, "mediana": med, "controle_denso_ok": ok,
         "quantiza": bool(a.quantiza), "damp": a.damp, "iteracoes": a.iteracoes},
        indent=2), encoding="utf-8")
    print()
    print("NAO COBERTO: erro de SAIDA por camada nas ativacoes calibradas, NAO imagem -- e esta")
    print("  bancada ja mediu tres instrumentos numericos apontando para o lado errado no mesmo")
    print("  dia. Isto ordena candidatos para o render; nao decide. A reconstrucao e por camada")
    print("  INDEPENDENTE (cada uma ve a entrada limpa), que e mais fraco que o sequencial do")
    print("  SparseGPT. Um modelo, uma calibragem, uma placa.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
