r"""Calcula os pesos 2:4 recuperados e GRAVA num arquivo, para o render nao precisar da calibragem.

POR QUE SEPARAR
---------------
A reconstrucao precisa das ativacoes: 12,9 GiB de amostras residentes. A amostragem precisa que os
pesos remendados sobrevivam, e para isso o ComfyUI tem de rodar com `--disable-dynamic-vram`, que
carrega os 11,5 GiB do modelo de uma vez em vez de streamar do arquivo. As duas coisas juntas
mataram o processo sem stdout nem stderr -- foi a terceira hipotese errada de 2026-09-03 sobre por
que a cirurgia nao sobrevivia, e a unica que era sobre memoria de verdade.

Separando, cada processo paga so a sua metade. E o subproduto e melhor que o efeito: os pesos
recuperados viram um artefato reproduzivel, entao o render pode ser repetido sem recalcular nada.

O QUE GRAVA
-----------
Safetensors com uma entrada por camada tocada, no dtype do peso original, mais um `__metadata__`
dizendo o modo, o groupsize, quantas camadas entraram e quantas foram puladas por falta de amostra.
Uma camada sem amostra e **pulada, nao podada**: podar sem recuperar seria um terceiro formato
misturado no mesmo arquivo, e a folha nao diria qual.

NAO COBERTO
-----------
Nao mede nada e nao renderiza. So calcula e grava. A reconstrucao e por camada INDEPENDENTE -- cada
uma ve a entrada limpa, nao a ja degradada pelas anteriores.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ComfyUI"))

import torch
from safetensors.torch import save_file


def cabecalho(modelo: Path) -> tuple[dict, int]:
    with modelo.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def ler_peso(modelo: Path, head: dict, base: int, chave: str) -> torch.Tensor:
    info = head[chave]
    with modelo.open("rb") as f:
        a, b = info["data_offsets"]
        f.seek(base + a)
        cru = bytearray(f.read(b - a))
    return torch.frombuffer(cru, dtype=torch.bfloat16).reshape(info["shape"]).cuda()


def mascara_elemento(criterio: torch.Tensor) -> torch.Tensor:
    n, k = criterio.shape
    g = criterio.reshape(n, k // 4, 4)
    idx = g.argsort(dim=-1)[..., :2]
    return torch.ones_like(g, dtype=torch.bool).scatter_(-1, idx, False).reshape(n, k)


def recupera_cg(w, m, h, iteracoes=256, tol=1e-6):
    """Gradiente conjugado mascarado -- o mesmo de tools/recupera_esparso.py."""
    w, h, m = w.float(), h.float(), m.float()
    ws = w * m
    r = m * ((w - ws) @ h)
    pd = r.clone()
    rr = (r * r).sum()
    rr0 = rr.clone()
    for _ in range(iteracoes):
        ap = m * (pd @ h)
        pap = (pd * ap).sum()
        if pap <= 0:
            break
        al = rr / pap
        ws = ws + al * pd
        r = r - al * ap
        rn = (r * r).sum()
        if rn <= tol * tol * rr0:
            break
        pd = r + (rn / rr) * pd
        rr = rn
    return ws * m


def int8_por_linha(w: torch.Tensor) -> torch.Tensor:
    e = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    return (w / e).round().clamp(-127, 127) * e


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", type=Path,
                   default=RAIZ / "ComfyUI/models/diffusion_models/"
                                  "beyond-reality-zimage-v2_native.safetensors")
    p.add_argument("--calib", type=Path,
                   default=RAIZ / "calib/zimage_v2_rows8192.calib.pt")
    p.add_argument("--int8", action="store_true", help="aplica int8 por linha depois de recuperar")
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--saida", type=Path, required=True)
    a = p.parse_args()

    if a.saida.exists():
        raise SystemExit(f"{a.saida} ja existe; este script nao sobrescreve")
    torch.backends.cuda.matmul.allow_tf32 = False

    head, base = cabecalho(a.modelo)
    d = torch.load(a.calib, map_location="cpu", weights_only=False)
    camadas = d["layers"]

    saida: dict[str, torch.Tensor] = {}
    puladas = 0
    for i, (nome, entrada) in enumerate(list(camadas.items())):
        chave = nome + ".weight"
        x = entrada.get("sample") if isinstance(entrada, dict) else entrada
        if chave not in head or not torch.is_tensor(x):
            puladas += 1
            continue
        w = ler_peso(a.modelo, head, base, chave)
        if w.dim() != 2 or w.shape[1] % 4 or x.shape[-1] != w.shape[1]:
            puladas += 1
            del w
            continue
        xf = x.cuda().float().reshape(-1, x.shape[-1])
        h = xf.double().T @ xf.double()
        h += torch.eye(h.shape[0], device=h.device, dtype=h.dtype) * (a.damp * h.diag().mean())
        wanda = w.abs().float() * xf.norm(dim=0).unsqueeze(0)
        m = mascara_elemento(wanda)
        rec = recupera_cg(w.float(), m.float(), h.float())
        if a.int8:
            rec = int8_por_linha(rec)
        saida[chave] = rec.to(w.dtype).cpu()
        # A amostra sai da RAM assim que e consumida: com 8192 linhas o dicionario inteiro passa
        # de 12 GiB e nao ha razao para segurar o que ja foi usado.
        if isinstance(entrada, dict):
            entrada["sample"] = None
        del w, xf, h, wanda, m, rec
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            print(f"  [{len(saida)} gravadas, {puladas} puladas]", flush=True)

    if not saida:
        print("nenhuma camada recuperada", file=sys.stderr)
        return 2

    meta = {"modo": "recup_elem_int8" if a.int8 else "recup_elem",
            "criterio": "wanda", "granularidade": "elemento 2:4",
            "camadas": str(len(saida)), "puladas_sem_amostra": str(puladas),
            "calib": str(a.calib.name), "modelo": str(a.modelo.name),
            "nao_coberto": ("reconstrucao por camada INDEPENDENTE; camadas sem amostra sao "
                            "PULADAS e ficam com o peso original, nao podadas")}
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    save_file(saida, str(a.saida), metadata=meta)
    tam = a.saida.stat().st_size / (1 << 30)
    print(f"gravado {a.saida} ({tam:.2f} GiB): {len(saida)} camadas, {puladas} puladas")
    print("NAO COBERTO: nao mede nada e nao renderiza. Camadas sem amostra ficam com o peso")
    print("  ORIGINAL, nao podado -- entao o arquivo e um modelo misto e o render tem de dizer")
    print("  quantas camadas de fato mudaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
