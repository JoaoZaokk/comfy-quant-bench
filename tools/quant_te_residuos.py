"""Quantiza SO os pesos que ficaram em BF16 num text encoder LTX: a embedding do Gemma e a projecao.

    python_embeded\\python.exe -s tools/quant_te_residuos.py --input gemma_..._w4a8.safetensors --format int8
    python_embeded\\python.exe -s tools/quant_te_residuos.py --input ltx-2.3_text_projection_bf16.safetensors --format fp8

Por que existe. O Gemma W4A8 desta bancada tem 7,53 GiB, dos quais 1,88 GiB sao a tabela de
embedding em BF16; a projecao do LTX (`text_embedding_projection.{video,audio}_aggregate_embed`)
soma mais 2,15 GiB em BF16. `quant_int8.py` recusa a fonte ja quantizada e nao seleciona
embedding; este arquivo toca apenas esses tensores e copia todo o resto byte a byte.

Formatos, ambos lidos nativamente por `comfy/ops.py` (MixedPrecisionOps):
    int8  -> `int8_tensorwise`, escala por LINHA (quantize_int8_rowwise do comfy-kitchen, sem rotacao).
             A embedding le so as linhas pedidas (`dequantize_int8_embedding`).
    fp8   -> `float8_e4m3fn`, UMA escala por tensor (o que `ops.py` aplica na embedding: `x * scale`).

ARMADILHA DO FP8. `_conversion.header_dtype` grava fp8 como U8 cru, que e o certo para escalas
(`pop_scale` faz `.view`), mas o PESO passa por `weight.to(dtype=float8)` no loader: um U8 seria
convertido numericamente e o peso viraria lixo. Aqui o peso fp8 sai com dtype `F8_E4M3` no header.

Cada tensor e conferido antes da escrita pelo dequantizador real do ComfyUI (a mesma classe de
layout que o loader usa), e o erro relativo vai para o sidecar. Isso prova o formato, nao a
qualidade: a aceitacao continua sendo encode real + render.

NAO COBERTO: a projecao quantizada so e lida quantizada se o Gemma carregado junto tambem for
quantizado (a projecao herda `gemma3_12b.operations`); com o Gemma BF16 o loader nao usa
MixedPrecisionOps e este arquivo nao carrega.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import struct
import sys
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "ComfyUI"))

import _conversion as C  # noqa: E402
from quant_w4a8 import HIGH_PRECISION_DTYPES, human_size, read_tensor  # noqa: E402

ALVO = re.compile(r"(.+\.)?embed_tokens\.weight"
                  r"|text_embedding_projection\.((video_|audio_)?aggregate_embed\.)?weight")
FORMATO = {"int8": "int8_tensorwise", "fp8": "float8_e4m3fn"}
FP8_MAX = 448.0
LINHAS_POR_BLOCO = 1 << 24  # elementos por bloco de quantizacao/conferencia (~64 MiB em fp32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--format", choices=sorted(FORMATO), required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def saida_padrao(source: Path, selecionados: list[str], fmt: str) -> Path:
    so_embedding = all(n.endswith("embed_tokens.weight") for n in selecionados)
    stem = source.stem.removesuffix("_bf16")
    return source.with_name(f"{stem}_{'emb' if so_embedding else ''}{fmt}.safetensors")


def blocos(n_linhas: int, n_colunas: int):
    passo = max(1, LINHAS_POR_BLOCO // n_colunas)
    for i in range(0, n_linhas, passo):
        yield slice(i, min(i + passo, n_linhas))


def quantiza(peso: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    if fmt == "int8":
        from comfy_kitchen.backends.eager import quantization as eager
        partes = [eager.quantize_int8_rowwise(peso[b]) for b in blocos(*peso.shape)]
        return torch.cat([q for q, _ in partes]), torch.cat([s for _, s in partes]).float()
    amax = max(peso[b].abs().amax().float().item() for b in blocos(*peso.shape))
    escala = torch.tensor(amax / FP8_MAX, dtype=torch.float32)
    q = torch.empty(peso.shape, dtype=torch.float8_e4m3fn)
    for b in blocos(*peso.shape):
        q[b] = (peso[b].float() / escala).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, escala


def confere(nome: str, peso: torch.Tensor, q: torch.Tensor, escala: torch.Tensor, fmt: str) -> dict:
    """Dequantiza pelo layout real do ComfyUI e mede contra a fonte, bloco a bloco."""
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class

    layout = get_layout_class(QUANT_ALGOS[FORMATO[fmt]]["comfy_tensor_layout"])
    num = den = 0.0
    pior_linha = 0.0
    for b in blocos(*peso.shape):
        s = escala[b] if escala.ndim else escala
        params = layout.Params(scale=s, orig_dtype=torch.bfloat16, orig_shape=tuple(q[b].shape))
        d = layout.dequantize(q[b], params).float()
        ref = peso[b].float()
        err = (d - ref).pow(2).sum(dim=1)
        norma = ref.pow(2).sum(dim=1)
        num += err.sum().item()
        den += norma.sum().item()
        pior_linha = max(pior_linha, (err / norma.clamp_min(1e-30)).sqrt().max().item())
    out = {"rel_rmse": (num / den) ** 0.5, "pior_linha_rel": pior_linha}
    if nome.endswith("embed_tokens.weight"):
        # o caminho da embedding e outro (gather + dequant so das linhas pedidas): conferir tambem
        idx = torch.linspace(0, peso.shape[0] - 1, 257).long()
        params = layout.Params(scale=escala, orig_dtype=torch.bfloat16, orig_shape=tuple(q.shape))
        if fmt == "int8":
            g = layout.dequantize_embedding(q, params, idx[None])[0].float()
        else:
            g = torch.nn.functional.embedding(idx, q).float() * escala  # o que ops.py faz no fp8
        ref = peso[idx].float()
        out["gather_rel_rmse"] = ((g - ref).norm() / ref.norm()).item()
    return out


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("--input precisa ser um .safetensors existente")

    header, metadata = C.read_header(source)
    selecionados = [n for n, i in header.items()
                    if ALVO.fullmatch(n) and i["dtype"] in HIGH_PRECISION_DTYPES and len(i["shape"]) == 2]
    if not selecionados:
        raise SystemExit("nenhuma embedding/projecao em alta precisao nesta fonte")
    quant_meta = json.loads(metadata.get("_quantization_metadata", '{"format_version": "1.0", "layers": {}}'))
    ja = [n for n in selecionados if n.removesuffix(".weight") in quant_meta["layers"]]
    if ja:
        raise SystemExit(f"recusado: ja declarados quantizados no metadata: {ja}")

    output = (args.output or saida_padrao(source, selecionados, args.format)).resolve()
    sidecar = output.with_suffix(".quant.json")
    conv = C.Conversion(source, output, sidecar)
    # a fonte PODE ser quantizada (o Gemma W4A8); os tensores tocados nao, conferido acima
    conv.refuse_unsafe(allow_quantized_source=True)

    print(f"fonte: {source}\nsaida: {output}\nformato: {FORMATO[args.format]}")
    for n in selecionados:
        print(f"  {n}: {header[n]['shape']} {header[n]['dtype']}")
    if args.dry_run:
        return 0

    elem = sum(header[n]["shape"][0] * header[n]["shape"][1] for n in selecionados)
    # fonte bf16 + saida 1 byte, por tensor, mais um bloco fp32 de trabalho
    conv.guard(source.stat().st_size, accumulated=elem * 3 + LINHAS_POR_BLOCO * 16,
               label="quant residuos TE")

    inicio = time.perf_counter()
    quantizados: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    medidas: dict[str, dict] = {}
    with source.open("rb") as handle:
        base = 8 + struct.unpack("<Q", handle.read(8))[0]
        for n in selecionados:
            info = header[n]
            a, b = info["data_offsets"]
            peso = read_tensor(handle, base + a, b - a, info["dtype"], info["shape"])
            q, escala = quantiza(peso, args.format)
            medidas[n] = confere(n, peso, q, escala, args.format)
            print(f"  {n}: {medidas[n]}", flush=True)
            quantizados[n] = (q.contiguous(), escala.contiguous())
            del peso

    entradas = []
    for n, info in header.items():
        if n not in quantizados:
            entradas.append(C.plan_copy(n, info))
            continue
        q, escala = quantizados[n]
        if args.format == "fp8":
            # PESO fp8 com dtype fp8 no header (ver ARMADILHA no topo); plan_write o gravaria como U8
            entradas.append(C.Entry(n, "F8_E4M3", list(q.shape), ("write", q), q.numel()))
        else:
            entradas.append(C.plan_write(n, q))
        entradas.append(C.plan_write(f"{n}_scale", escala))

    for n in selecionados:
        quant_meta["layers"][n.removesuffix(".weight")] = {"format": FORMATO[args.format]}
    out_meta = dict(metadata)
    out_meta["_quantization_metadata"] = json.dumps(quant_meta, separators=(",", ":"))
    out_meta["te_residuos"] = FORMATO[args.format]
    conv.commit(entradas, out_meta)

    segundos = time.perf_counter() - inicio
    sidecar.write_text(json.dumps({
        "source": str(source), "source_size": source.stat().st_size,
        "output": str(output), "output_size": output.stat().st_size,
        "quantization": FORMATO[args.format],
        "escala": "por linha" if args.format == "int8" else "por tensor",
        "quantized_tensors": selecionados,
        "erro_contra_fonte": medidas,
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__,
        "conversion_seconds": round(segundos, 3),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gravado {output} ({human_size(output.stat().st_size)}) em {segundos:.0f} s\ngravado {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
