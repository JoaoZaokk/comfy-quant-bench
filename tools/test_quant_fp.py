"""`quant_fp.py`: perfis `moss` e `flux3`, layouts de fp8/nvfp4 e a flag do W4A16.

EXECUTADO, sem GPU, sem ComfyUI e sem arquivo de modelo: o módulo só importa `comfy.quant_ops` ao quantizar, e este teste
confere isso. Cobre quais tensores cada perfil escolhe, o que cada formato grava e que o W4A16 sai com
`full_precision_matrix_mult` (sem ela, o caminho quantizado do comfy-kitchen 0.2.40 deu NaN com grupo 32 no FLUX.3).

NAO COBERTO: a quantização em si (fp8/nvfp4 dependem do ComfyUI), a ConvRot W4A4 na GPU e o erro dos pesos.

    python_embeded\\python.exe -s tools/test_quant_fp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quant_fp as Q  # noqa: E402

FALHAS = 0


def checa(condicao: bool, rotulo: str, detalhe: str = "") -> None:
    global FALHAS
    print(("  PASS  " if condicao else "  FAIL  ") + rotulo + ("" if condicao or not detalhe else f"\n          {detalhe}"))
    FALHAS += not condicao


def main() -> int:
    print("importação")
    checa("comfy" not in sys.modules and "comfy.quant_ops" not in sys.modules, "importar o módulo não puxa o ComfyUI")

    print("perfil moss")
    moss = [f"layers.{i}.proj.{s}.weight" for i in range(42) for s in ("qkv", "o", "ffn_up_gate", "ffn_down")]
    checa(all(Q.PERFIS["moss"].fullmatch(n) for n in moss) and len(moss) == 168, "168 camadas: qkv, o, ffn_up_gate, ffn_down", str(len(moss)))
    fora = ["layers.0.proj.qkv.bias", "layers.0.norm.weight", "text_embedder.layers.0.proj.qkv.weight", "final_layer.linear.weight",
            "layers.0.proj.qkv.weight_scale"]
    checa(not any(Q.PERFIS["moss"].fullmatch(n) for n in fora), "bias, normas, embedder e escalas ficam fora", str([n for n in fora if Q.PERFIS["moss"].fullmatch(n)]))

    print("perfil flux3")
    flux = ["single_blocks.3.qkv.weight", "single_blocks.3.attn_out.weight", "content_mode_blocks.a.2.mlp_in.weight",
            "content_mode_blocks.b.2.mlp_out.weight", "txt_mode_blocks.0.q_proj.weight", "txt_mode_blocks.0.v_proj.weight"]
    checa(all(Q.PERFIS["flux3"].fullmatch(n) for n in flux), "single, content_mode e txt_mode casam")
    fora = ["single_blocks.3.norm.weight", "img_in.weight", "final_layer.linear.weight", "single_blocks.3.qkv.bias", "time_in.in_layer.weight"]
    checa(not any(Q.PERFIS["flux3"].fullmatch(n) for n in fora), "normas, entradas, final_layer e bias ficam fora", str([n for n in fora if Q.PERFIS["flux3"].fullmatch(n)]))

    print("layouts")
    nome = "layers.0.proj.qkv.weight"
    t8 = {a.rsplit(".", 1)[-1]: (d, sh) for a, d, sh in Q.Fp8E4M3().tensors(nome, [64, 128])}
    checa(t8["weight"] == ("F8_E4M3", [64, 128]) and t8["weight_scale"] == ("F32", []), "fp8: peso F8_E4M3 e escala F32 por tensor", str(t8))
    t4 = {a.rsplit(".", 1)[-1]: (d, sh) for a, d, sh in Q.Nvfp4().tensors(nome, [64, 128])}
    checa(t4["weight"] == ("U8", [64, 64]) and t4["weight_scale"] == ("F8_E4M3", [64, 8]) and t4["weight_scale_2"] == ("F32", []),
          "nvfp4: U8 [N, K/2], escala F8 [N, K/16], escala global F32", str(t4))
    checa(not Q.Nvfp4().accepts([64, 120]) and Q.Nvfp4().accepts([64, 128]), "nvfp4 exige K e N múltiplos de 16")
    cfg = Q.Awq32PesoCheio().layer_config()
    checa(cfg == {"format": "awq_w4a16", "group_size": 32, "full_precision_matrix_mult": True}, "w4a16: grupo 32 com full_precision_matrix_mult", str(cfg))
    checa(set(Q.FORMATOS) == set(Q.DESCRICAO) == {"fp8", "nvfp4", "w4a16", "w4a4"}, "todo formato tem descrição")
    checa("só pesos" not in Q.DESCRICAO["fp8"] and "só pesos" not in Q.DESCRICAO["nvfp4"], "fp8 e nvfp4 não se descrevem como só pesos (a ativação é quantizada)")
    print(f"\n{'FALHOU' if FALHAS else 'PASSOU'}: {FALHAS} falha(s)")
    return 1 if FALHAS else 0


if __name__ == "__main__":
    sys.exit(main())
