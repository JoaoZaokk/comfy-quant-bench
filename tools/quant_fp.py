"""bf16 -> FP8 (e4m3, escala por tensor), NVFP4 (blocos de 16, escala global), W4A16 (Q4_1 grupo 32) ou W4A4 (ConvRot), nos formatos nativos do ComfyUI.

Sem calibração de ativação: não grava `input_scale`. ATENÇÃO: nos formatos fp8 e nvfp4 o ComfyUI QUANTIZA as ativações também
(`quantize_input` padrão True), com escala dinâmica quando não há `input_scale`; na sm86 isso roda emulado (quantiza e dequantiza,
matmul em bf16). Então fp8/nvfp4 daqui são W8A8-fp8 / W4A4-fp4 com ativação dinâmica, não só-pesos. Só w4a16
(full_precision_matrix_mult) deixa a ativação em bf16. w4a4 usa o quantizador ConvRot do comfy-kitchen na GPU (K múltiplo de 256).
Camadas: --perfil moss | flux3 (regex abaixo); o resto (normas, embedders, modulações, bias) fica bf16, como nos arquivos oficiais.
fp8/nvfp4 precisam do `comfy.quant_ops` do ComfyUI: aponte COMFYUI_DIR (padrão: ../ComfyUI).
Uso: python quant_fp.py --entrada X.safetensors --perfil moss --formato fp8|nvfp4|w4a16|w4a4 --saida DIR [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402

PERFIS = {
    # MOSS-Image: qkv, o, ffn_up_gate, ffn_down das 42 camadas (mesmo critério do int8 oficial: só os blocos)
    "moss": re.compile(r"^layers\.\d+\.(?:proj\.(?:qkv|o|ffn_up_gate|ffn_down))\.weight$"),
    # FLUX.3 klein (layout do bf16 da BFL): q/k/v/qkv, attn_out, mlp_in, mlp_out de todos os blocos (single_blocks, content_mode_blocks, txt_mode_blocks)
    "flux3": re.compile(r"^(?:single_blocks|content_mode_blocks\.\w+|txt_mode_blocks)\.\d+\.(?:qkv|q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)\.weight$"),
}


def _quant_ops():
    """`comfy.quant_ops` do ComfyUI, importado só quando fp8/nvfp4 quantizam de fato."""
    raiz = os.environ.get("COMFYUI_DIR") or str(Path(__file__).resolve().parent.parent / "ComfyUI")
    sys.path.insert(0, raiz)
    import comfy.quant_ops as Q
    return Q


def _base(name):
    return name[: -len(".weight")]


@dataclass(frozen=True)
class Fp8E4M3:
    name = "float8_e4m3fn"

    def accepts(self, shape): return len(shape) == 2

    def tensors(self, name, shape):
        return [(name, "F8_E4M3", list(shape)), (f"{_base(name)}.weight_scale", "F32", [])]

    def layer_config(self): return {"format": self.name}

    def quantize(self, weight, ck=None):
        Q = _quant_ops()
        qd, p = Q.TensorCoreFP8E4M3Layout.quantize(weight, scale="recalculate")
        sd = Q.TensorCoreFP8E4M3Layout.state_dict_tensors(qd, p)
        deq = Q.TensorCoreFP8E4M3Layout.dequantize(qd, p)
        return [sd[""].contiguous(), sd["_scale"].float().reshape(())], {"rel": float((deq.float() - weight.float()).norm() / weight.float().norm())}


@dataclass(frozen=True)
class Nvfp4:
    name = "nvfp4"

    def accepts(self, shape): return len(shape) == 2 and shape[1] % 16 == 0 and shape[0] % 16 == 0

    def tensors(self, name, shape):
        n, k = shape
        b = _base(name)
        return [(name, "U8", [n, k // 2]), (f"{b}.weight_scale", "F8_E4M3", [n, k // 16]), (f"{b}.weight_scale_2", "F32", [])]

    def layer_config(self): return {"format": self.name}

    def quantize(self, weight, ck=None):
        Q = _quant_ops()
        qd, p = Q.TensorCoreNVFP4Layout.quantize(weight)
        sd = Q.TensorCoreNVFP4Layout.state_dict_tensors(qd, p)
        deq = Q.TensorCoreNVFP4Layout.dequantize(qd, p)
        return [sd[""].contiguous(), sd["_scale"].contiguous(), sd["_scale_2"].float().reshape(())], {"rel": float((deq.float() - weight.float()).norm() / weight.float().norm())}


@dataclass(frozen=True)
class Awq32PesoCheio(F.AwqW4A16):
    """awq_w4a16 grupo 32 com `full_precision_matrix_mult`: o ComfyUI dequantiza o peso e multiplica em bf16. Sem a flag, o caminho
    quantizado do comfy-kitchen 0.2.40 com grupo 32 deu imagem preta (NaN) no FLUX.3."""
    group_size: int = 32

    def layer_config(self):
        return {**super().layer_config(), "full_precision_matrix_mult": True}


@dataclass(frozen=True)
class W4A4Cuda(F.ConvrotW4A4):
    """ConvRot W4A4 (o formato nativo da bancada) com o quantizador do comfy-kitchen na GPU; K tem de ser múltiplo de 256."""

    def quantize(self, weight, ck=None):
        import comfy_kitchen
        return super().quantize(weight.to("cuda", torch.float32), comfy_kitchen)


FORMATOS = {"fp8": Fp8E4M3, "nvfp4": Nvfp4, "w4a16": Awq32PesoCheio, "w4a4": W4A4Cuda}
DESCRICAO = {"fp8": "fp8 e4m3 por tensor, ativação com escala dinâmica (sem input_scale)",
             "nvfp4": "nvfp4 blocos de 16, ativação com escala dinâmica (sem input_scale)",
             "w4a16": "w4a16 só pesos (Q4_1 grupo 32, ativação bf16)", "w4a4": "convrot_w4a4 (ConvRot-256, sem input_scale)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, type=Path)
    ap.add_argument("--perfil", required=True, choices=list(PERFIS))
    ap.add_argument("--formato", required=True, choices=list(FORMATOS))
    ap.add_argument("--saida", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fmt = FORMATOS[a.formato]()
    stem = a.entrada.stem.replace("_bf16", "").replace("-bf16", "")
    saida = a.saida / f"{stem}_{a.formato}.safetensors"
    conv = C.Conversion(a.entrada, saida, saida.with_suffix(".quant.json"))
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata
    selec = [k for k, i in header.items() if PERFIS[a.perfil].fullmatch(k) and len(i["shape"]) == 2 and fmt.accepts(i["shape"])]
    todas = [k for k in header if PERFIS[a.perfil].fullmatch(k)]
    print(f"{a.formato}/{a.perfil}: {len(selec)} de {len(todas)} camadas do perfil ({len(header)} tensores); saída {saida}")
    if not selec or len(selec) != len(todas):
        raise SystemExit("seleção inesperada")
    if a.dry_run:
        return 0
    t0 = time.perf_counter()
    formats = {k: fmt for k in selec}
    novo = F.quant_metadata(metadata, F.layer_configs(formats), DESCRICAO[a.formato])
    erros = {}
    with conv.tensors() as fonte:
        entradas = F.plan_model(header, formats, lambda n: fonte[n].to(torch.bfloat16) if fonte[n].dtype != torch.bfloat16 else fonte[n], None,
                                on_quantized=lambda n, e: erros.__setitem__(n, e.get("rel")))
        conv.guard(conv.planned_size(entradas, novo), accumulated=F.streaming_peak(header, selec), label=stem)

        def manifesto():
            r = sorted(v for v in erros.values() if v is not None)
            return {"source": a.entrada.name, "output": saida.name, "output_size": conv.output_size, "formato": a.formato, "perfil": a.perfil,
                    "camadas": len(selec), "preservadas": len(header) - len(selec),
                    "weight_rel_rmse_mean": sum(r) / len(r) if r else None, "weight_rel_rmse_max": r[-1] if r else None,
                    "sem_input_scale": True, "comfy_kitchen": importlib.metadata.version("comfy-kitchen"), "segundos": round(time.perf_counter() - t0, 1)}
        conv.commit(entradas, novo, sidecar=manifesto)
    r = sorted(v for v in erros.values() if v is not None)
    print("escrito", saida, C.human_size(saida.stat().st_size), (f"erro relativo do peso: média {sum(r)/len(r):.4f}, máx {r[-1]:.4f}" if r else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
