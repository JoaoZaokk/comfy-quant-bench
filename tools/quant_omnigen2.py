"""OmniGen2 DiT (fp16, ComfyUI) -> W8A16 (int8 por linha, sem rotação) ou W4A16 (Q4_1, grupo 40), só pesos.

Por que não os conversores padrão: o hidden é 2520 (múltiplo de 8, mas não de 16/32/256). ConvRot-256, o GEMM INT8 e o Q4_1 do
gguf (grupo 32) não aceitam K = 2520. Aqui:
  w8a16: int8_tensorwise sem convrot (escala por linha, aceita qualquer K) + full_precision_matrix_mult: o ComfyUI dequantiza o
         peso e multiplica em fp16; as ativações nunca são quantizadas.
  w4a16: awq_w4a16 com grupo 40 (mdc de 2520 e 10240; o kernel de dequantização do comfy-kitchen aceita qualquer grupo). Q4_1: por
         grupo, escala d e zero m + 8d em fp16, código = round((w - zero)/d) + 8 limitado a 0..15.
Camadas: attn.to_q/k/v/to_out.0 e feed_forward.linear_1/2/3 das 32 camadas + 3 refiners (2 blocos cada). Ficam em fp16: normas,
norm1.linear (modulação adaLN), embedders, norm_out, caption/timestep embedder, image_index_embedding.
Escala e zero em fp16 de propósito: com o modelo em fp16, escala bf16 promove o produto a fp32 e o linear falha por dtype.
Streaming (um tensor por vez), escrita atômica .partial, não sobrescreve nada. Uso:
  CUDA_VISIBLE_DEVICES=-1 python quant_omnigen2.py --entrada omnigen2_fp16.safetensors --saida DIR --mode w8a16|w4a16 [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402

G = 40
CAMADAS = re.compile(r"^(?:layers|noise_refiner|context_refiner|ref_image_refiner)\.\d+\."
                     r"(?:attn\.(?:to_q|to_k|to_v|to_out\.0)|feed_forward\.linear_[123])\.weight$")
ESPERADAS = 32 * 7 + 6 * 7


@dataclass(frozen=True)
class Int8W8A16(F.Int8Tensorwise):
    convrot: bool = False

    def layer_config(self) -> dict:
        return {**super().layer_config(), "full_precision_matrix_mult": True}


@dataclass(frozen=True)
class AwqW4A16Livre(F.AwqW4A16):
    """awq_w4a16 com grupo qualquer: Q4_1 por mínimo/máximo do grupo, escala e zero em fp16 (o dtype do modelo)."""
    group_size: int = G

    def layer_config(self) -> dict:
        return {**super().layer_config(), "full_precision_matrix_mult": True}

    def tensors(self, name, shape):
        return [(a, "F16" if d == "BF16" else d, sh) for a, d, sh in super().tensors(name, shape)]

    def quantize(self, weight: torch.Tensor, ck=None):
        n, k = weight.shape
        g = self.group_size
        x = weight.to(torch.float32).view(n, k // g, g)
        lo, hi = x.amin(-1), x.amax(-1)
        d = ((hi - lo) / 15.0).to(torch.float16)
        zeros = (lo + 8.0 * d.float()).to(torch.float16)
        d32, z32 = d.float(), zeros.float()
        seguro = torch.where(d32 > 0, d32, torch.ones_like(d32))
        q = torch.where(d32.unsqueeze(-1) > 0, torch.round((x - z32.unsqueeze(-1)) / seguro.unsqueeze(-1)) + 8.0,
                        torch.full_like(x, 8.0)).clamp_(0, 15).to(torch.uint8).view(n, k)
        deq = ((q.view(n, k // g, g).float() - 8.0) * d32.unsqueeze(-1) + z32.unsqueeze(-1)).view(n, k)
        rel = float((deq - weight.float()).norm() / weight.float().norm())
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous().view(torch.int8)
        return [packed, d.t().contiguous(), zeros.t().contiguous()], {"rel": rel}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, type=Path, help="omnigen2_fp16.safetensors (não é alterado)")
    ap.add_argument("--saida", required=True, type=Path, help="diretório de saída; nunca sobrescreve")
    ap.add_argument("--mode", required=True, choices=["w8a16", "w4a16"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    fmt = Int8W8A16() if args.mode == "w8a16" else AwqW4A16Livre()
    nome = "omnigen2_fp16_w8a16" if args.mode == "w8a16" else f"omnigen2_fp16_w4a16_g{G}"
    saida = args.saida / f"{nome}.safetensors"
    conv = C.Conversion(args.entrada, saida, saida.with_suffix(".quant.json"))
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata
    selec = [k for k, i in header.items() if CAMADAS.fullmatch(k) and len(i["shape"]) == 2 and fmt.accepts(i["shape"])]
    todas = [k for k in header if CAMADAS.fullmatch(k)]
    print(f"{args.mode}: {len(selec)} camadas quantizadas de {len(todas)} no perfil ({len(header)} tensores); saída {saida}")
    if len(selec) != len(todas) or len(selec) != ESPERADAS:
        raise SystemExit(f"seleção inesperada: {len(selec)} / {len(todas)}")
    if args.dry_run:
        return 0
    inicio = time.perf_counter()
    formats = {k: fmt for k in selec}
    descricao = "int8 por linha, só pesos (W8A16)" if args.mode == "w8a16" else f"awq_w4a16 Q4_1 grupo {G}, só pesos (W4A16)"
    novo = F.quant_metadata(metadata, F.layer_configs(formats), descricao)
    erros = {}
    with conv.tensors() as fonte:
        entradas = F.plan_model(header, formats, lambda n: fonte[n], None,
                                on_quantized=lambda n, extra: erros.__setitem__(n, extra.get("rel")))
        conv.guard(conv.planned_size(entradas, novo), accumulated=F.streaming_peak(header, selec), label=nome)

        def manifesto():
            rel = sorted(v for v in erros.values() if v is not None)
            return {"source": args.entrada.name, "source_size": args.entrada.stat().st_size, "output": saida.name,
                    "output_size": conv.output_size, "modo": args.mode, "camadas_quantizadas": len(selec),
                    "tensores_preservados": len(header) - len(selec), "grupo": G if args.mode == "w4a16" else None,
                    "weight_rel_rmse_mean": sum(rel) / len(rel) if rel else None, "weight_rel_rmse_max": rel[-1] if rel else None,
                    "torch": torch.__version__, "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
                    "segundos": round(time.perf_counter() - inicio, 1),
                    "por_que_nao_o_padrao": "K=2520 não é múltiplo de 256 (ConvRot), 32 (Q4_1 gguf) nem 16 (GEMM INT8)"}
        conv.commit(entradas, novo, sidecar=manifesto)
    print("escrito", saida, C.human_size(saida.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
