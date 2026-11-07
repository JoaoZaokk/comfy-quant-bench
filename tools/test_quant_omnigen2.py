"""`quant_omnigen2.py`: seleção das 266 camadas, aceitação de K = 2520 e o Q4_1 de grupo 40 em fp16.

EXECUTADO, sem GPU e sem arquivo de modelo: header sintético com os nomes do `omnigen2_fp16` (32 camadas + 3 refiners de 2 blocos)
e pesos aleatórios pequenos. Cobre o que o conversor decide (quais camadas, quais formatos aceitam K = 2520, dtype de escala e zero) e
a ida e volta do quantizador de grupo 40.

NAO COBERTO: não carrega o modelo no ComfyUI nem mede imagem; a dequantização no comfy-kitchen não é exercida aqui.

    python_embeded\\python.exe -s tools/test_quant_omnigen2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _formats as F  # noqa: E402
import quant_omnigen2 as Q  # noqa: E402

FALHAS = 0


def checa(condicao: bool, rotulo: str, detalhe: str = "") -> None:
    global FALHAS
    print(("  PASS  " if condicao else "  FAIL  ") + rotulo + ("" if condicao or not detalhe else f"\n          {detalhe}"))
    FALHAS += not condicao


def nomes() -> list[str]:
    blocos = [f"layers.{i}" for i in range(32)] + [f"{r}.{i}" for r in ("noise_refiner", "context_refiner", "ref_image_refiner") for i in range(2)]
    sub = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "feed_forward.linear_1", "feed_forward.linear_2", "feed_forward.linear_3")
    return [f"{b}.{s}.weight" for b in blocos for s in sub]


def main() -> int:
    print("selecao")
    sel = nomes()
    checa(len(sel) == Q.ESPERADAS == 266, "266 camadas esperadas", str(len(sel)))
    checa(all(Q.CAMADAS.fullmatch(n) for n in sel), "todas casam com o perfil")
    fora = ["layers.0.norm1.linear.weight", "layers.0.attn.norm_q.weight", "x_embedder.weight", "norm_out.linear_1.weight",
            "layers.0.attn.to_q.bias", "layers.0.ffn_norm1.weight", "time_caption_embed.timestep_embedder.linear_1.weight"]
    checa(not any(Q.CAMADAS.fullmatch(n) for n in fora), "modulação, normas, embedders e bias ficam fora", str([n for n in fora if Q.CAMADAS.fullmatch(n)]))

    print("K = 2520")
    k = [64, 2520]
    checa(not F.Int8Tensorwise().accepts(k), "o int8 padrão (ConvRot-256) recusa K = 2520")
    checa(not F.AwqW4A16().accepts(k), "o Q4_1 padrão (grupo 32) recusa K = 2520")
    checa(Q.Int8W8A16().accepts(k), "W8A16 sem rotação aceita K = 2520")
    checa(Q.AwqW4A16Livre().accepts(k) and Q.AwqW4A16Livre().accepts([64, 10240]), "W4A16 grupo 40 aceita K = 2520 e K = 10240")

    print("configuração por camada")
    c8, c4 = Q.Int8W8A16().layer_config(), Q.AwqW4A16Livre().layer_config()
    checa(c8.get("full_precision_matrix_mult") is True and c8["format"] == "int8_tensorwise" and not c8.get("convrot"), "W8A16: int8_tensorwise, sem convrot, peso cheio", str(c8))
    checa(c4.get("full_precision_matrix_mult") is True and c4["format"] == "awq_w4a16" and c4["group_size"] == 40, "W4A16: awq_w4a16, grupo 40, peso cheio", str(c4))
    ts = Q.AwqW4A16Livre().tensors("layers.0.attn.to_q.weight", [2520, 2520])
    dt = {a.rsplit(".", 1)[-1]: d for a, d, _ in ts}
    checa(dt["weight_scale"] == "F16" and dt["weight_zeros"] == "F16", "escala e zero em F16 (bf16 promoveria o linear a fp32)", str(dt))
    forma = {a.rsplit(".", 1)[-1]: sh for a, _, sh in ts}
    checa(forma["weight_scale"] == [63, 2520], "escala [K/40, N]", str(forma))

    print("ida e volta do Q4_1 de grupo 40")
    torch.manual_seed(0)
    n, kk, g = 16, 80, 40
    w = torch.randn(n, kk, dtype=torch.float16)
    w[:, :g] = 0.5  # grupo constante: faixa zero tem de voltar exato
    (packed, esc, zer), extra = Q.AwqW4A16Livre().quantize(w)
    checa(packed.dtype == torch.int8 and list(packed.shape) == [n, kk // 2], "pesos empacotados em int8 [N, K/2]", f"{packed.dtype} {list(packed.shape)}")
    checa(esc.dtype == torch.float16 and zer.dtype == torch.float16 and list(esc.shape) == [kk // g, n], "escala e zero fp16 [K/g, N]", f"{esc.dtype} {list(esc.shape)}")
    u = packed.view(torch.uint8)
    q = torch.stack([u & 0xF, u >> 4], dim=-1).reshape(n, kk).float()
    deq = (q.view(n, kk // g, g) - 8.0) * esc.t().float().unsqueeze(-1) + zer.t().float().unsqueeze(-1)
    deq = deq.view(n, kk)
    rel = float((deq - w.float()).norm() / w.float().norm())
    checa(abs(rel - extra["rel"]) < 1e-5, "o erro relativo gravado é o da dequantização refeita aqui", f"{rel} vs {extra['rel']}")
    checa(rel < 0.12, "erro relativo de 4 bits por grupo abaixo de 0,12 em pesos gaussianos", f"{rel:.4f}")
    checa(torch.equal(deq[:, :g], torch.full((n, g), 0.5)), "grupo constante volta exato")
    print(f"\n{'FALHOU' if FALHAS else 'PASSOU'}: {FALHAS} falha(s)")
    return 1 if FALHAS else 0


if __name__ == "__main__":
    sys.exit(main())
