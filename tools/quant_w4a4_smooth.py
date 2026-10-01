"""Convert a Gemma 3 checkpoint to ConvRot W4A4 with SmoothQuant folded into the norms.

The probe in tools/svdquant_probe.py says channel smoothing is the largest single term in the
SVDQuant recipe -- 1.80x on layer output, against 1.12x for the low-rank branch -- and that it
costs nothing at runtime because `lambda` folds into the RMSNorm that feeds the Linear. This
builds that checkpoint so the claim can be tested on an answer instead of on an error norm.

    Y = (x/lambda) @ (W*lambda).T          exact in full precision
    norm_weight  <- (norm_weight + 1)/lambda - 1      Gemma's RMSNorm is (1 + w), not w
    linear_weight <- linear_weight * lambda

Two details decide whether this is correct rather than merely plausible:

* **A norm has several consumers.** `input_layernorm` feeds q_proj, k_proj and v_proj;
  `pre_feedforward_layernorm` feeds gate_proj and up_proj. One lambda per norm, and every consumer
  must be compensated, so `weight_max` is taken across the whole group.
* **`o_proj` and `down_proj` have no norm directly ahead of them** -- they follow attention output
  and the gated product. They are left unsmoothed here rather than given a runtime multiply, which
  makes this exactly the free configuration: 5 of the 7 projections, no kernel change, no extra op.

Calibration runs the W4A4 checkpoint itself with its weights retyped so ComfyUI dequantizes them,
which gives full-precision activations off 4-bit weights at 7.6 GiB instead of loading the 21.9 GiB
source. Channel structure is a property of the model and survives that approximation; the shipped
`.quant.json` records that it was used.

    python tools/quant_w4a4_smooth.py --source gemma_..._it_heretic.safetensors \\
        --calibrate-with gemma_..._w4a4_convrot.safetensors --alpha 0.5
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _conversion as C  # noqa: E402
import _formats as F  # noqa: E402

import torch  # noqa: E402

from _profiles import HIGH_PRECISION_DTYPES  # noqa: E402

# The two norm groups. Every consumer of a norm shares one lambda.
GROUPS = {
    "input_layernorm": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "pre_feedforward_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
}
# Quantized but never smoothed: no norm feeds them directly.
UNSMOOTHED = ("self_attn.o_proj", "mlp.down_proj")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

CALIBRATION_PROMPTS = [
    "List the first 8 prime numbers, then explain in one sentence what makes a number prime.",
    "Explain in two sentences why the sky appears blue.",
    "Write one short paragraph about a cat who learns to open doors.",
    "What is 47 times 89? Show the steps.",
    "Traduza para o português: the quick brown fox jumps over the lazy dog.",
    "Name three countries in South America and their capitals.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="the BF16 checkpoint")
    parser.add_argument("--calibrate-with", required=True,
                        help="a W4A4 file name inside models/text_encoders, used for calibration")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--calibration-tokens", type=int, default=24)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def calibrate(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Per-input-channel absmax of what each norm hands its consumers."""
    import comfy.sd
    import folder_paths
    from comfy.quant_ops import QuantizedTensor

    path = folder_paths.get_full_path_or_raise("text_encoders", args.calibrate_with)
    clip = comfy.sd.load_clip(ckpt_paths=[path],
                              embedding_directory=folder_paths.get_folder_paths("embeddings"),
                              clip_type=comfy.sd.CLIPType.LTXV)
    model = clip.cond_stage_model
    retyped = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if getattr(module, "quant_format", None) == "convrot_w4a4" and isinstance(weight, QuantizedTensor):
            module.weight = torch.nn.Parameter(weight.to(dtype=torch.float32), requires_grad=False)
            retyped += 1
    print(f"calibrating on {args.calibrate_with}, {retyped} weights forced to dequantize", flush=True)

    # One probe per group: q_proj and gate_proj see exactly their norm's output.
    probes = {"input_layernorm": "self_attn.q_proj", "pre_feedforward_layernorm": "mlp.gate_proj"}
    stats: dict[str, torch.Tensor] = {}
    handles = []
    for module_name, module in model.named_modules():
        match = re.search(r"model\.layers\.(\d+)\.(.+)$", module_name)
        if not match:
            continue
        for norm, probe in probes.items():
            if match.group(2) != probe:
                continue
            key = f"model.layers.{match.group(1)}.{norm}.weight"

            def hook(mod, inputs, key=key):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().abs().amax(dim=0)
                stats[key] = x.cpu() if key not in stats else torch.maximum(stats[key], x.cpu())
            handles.append(module.register_forward_pre_hook(hook))

    for index, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        clip.generate(clip.tokenize(prompt, skip_template=False, min_length=1), do_sample=False,
                      max_length=args.calibration_tokens, temperature=1.0, top_k=0, top_p=1.0,
                      min_p=0.0, repetition_penalty=1.0, seed=0)
        print(f"  [{index}/{len(CALIBRATION_PROMPTS)}] {len(stats)} norms observed", flush=True)
    for handle in handles:
        handle.remove()
    del clip, model
    torch.cuda.empty_cache()
    return stats


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"No such source: {source}")
    output = (args.output or source.with_name(f"{source.stem}_w4a4_smooth.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")

    # A checagem de CUDA vinha AQUI, antes de toda recusa: sem placa visivel nenhuma guarda de
    # entrada rodava, e `test_smooth_guards.py` (seis recusas e um controle, todos de cabecalho)
    # falhava inteiro por ambiente. Ela desceu para logo antes do preflight de backend, que e o
    # primeiro passo que de fato precisa da placa (revisao 2026-09-29, achado 10).
    conv = C.Conversion(source, output, sidecar)
    conv.refuse_unsafe()
    header, metadata = conv.header, conv.metadata

    layer_ids = sorted({int(LAYER_RE.match(k).group(1)) for k in header if LAYER_RE.match(k)})
    selected, norm_keys = [], []
    for layer in layer_ids:
        for norm, members in GROUPS.items():
            norm_keys.append(f"model.layers.{layer}.{norm}.weight")
            selected += [f"model.layers.{layer}.{m}.weight" for m in members]
        selected += [f"model.layers.{layer}.{m}.weight" for m in UNSMOOTHED]
    missing = [k for k in selected + norm_keys if k not in header]
    if missing:
        raise SystemExit(f"Source lacks {len(missing)} expected tensors, e.g. {missing[0]}")
    # `missing` nao pega o caso vazio: numa fonte sem nenhuma chave `model.layers.N.`, `layer_ids`
    # sai vazia, `selected` e `norm_keys` saem vazias e `missing` tambem -- entao a checagem acima
    # aprova. Medido em 2026-09-01 apontando este conversor para um Wan 2.1: imprimiu
    # `Layers: 0   quantized: 0` e saiu com rc=0, ou seja, um `--dry-run` responde SUCESSO para
    # uma conversao que nao tem o que converter. Achado pelo caso de CONTROLE de um teste de
    # recusas, nao por um caso que procurava o defeito.
    if not selected:
        raise SystemExit(
            f"Source has no `model.layers.N.` tensors: this converter is Gemma-3-shaped "
            f"(esperava {'/'.join(GROUPS)} alimentando "
            f"{', '.join(m for members in GROUPS.values() for m in members)}). "
            f"Nada a converter em {source.name}.")
    # A checagem acima confere NOMES; nada aqui conferia o DTYPE. Medido em 2026-09-01 apontando
    # este conversor para o gemma_3_12B_it_heretic_fp8_e4m3fn: passou por toda a `refuse_unsafe`,
    # passou pelo preflight de backend, rodou os seis prompts de calibragem ate o fim (96 normas,
    # ~3 min de GPU) e so entao morreu com `KeyError: 'F8_E4M3'` dentro do leitor. O custo do erro
    # tardio e o trabalho jogado fora; a informacao para recusar ja estava no cabecalho.
    ilegiveis = sorted({header[k]["dtype"] for k in selected + norm_keys
                        if header[k]["dtype"] not in HIGH_PRECISION_DTYPES})
    if ilegiveis:
        raise SystemExit(
            f"Source carries dtypes this converter cannot read: {', '.join(ilegiveis)}. "
            f"Aceita {', '.join(sorted(HIGH_PRECISION_DTYPES))} -- passe o checkpoint de alta "
            f"precisao, nao uma versao ja comprimida.")

    print(f"Source: {source}")
    print(f"Layers: {len(layer_ids)}   quantized: {len(selected)}   "
          f"smoothed: {len(selected) - 2 * len(layer_ids)}   alpha: {args.alpha}")
    print(f"Output: {output}")
    if args.dry_run:
        return 0
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    # This writes the same `convrot_w4a4` format that quant_w4a4.py hard-refuses to produce
    # without the CUDA backend. The eager backend's output is structurally identical -- same
    # packed int4 container, same per-row f32 scales -- so neither verify_w4a4.py nor
    # inspect_quant.py can tell the two apart afterwards. A checkpoint quantized by eager and
    # believed to be CUDA is exactly the failure the whole preflight exists to prevent.
    from _native_probe import native_backend_ready
    from quant_w4a4 import w4a4_probe_ops

    backend = native_backend_ready(PORTABLE_ROOT, w4a4_probe_ops(args.convrot_groupsize))
    if not backend.get("native_ready"):
        raise SystemExit(
            "Refusing: normal ComfyUI resolves the ConvRot ops to "
            + ", ".join(f"{op}={module}"
                        for op, module in sorted(backend["resolved"].items()))
            + ", not a CUDA backend. Producing convrot_w4a4 from the eager path yields a file "
              "nothing downstream can distinguish from a real one.")
    print(f"Backend: {backend['resolved']['quantize_convrot_w4a4_weight']}")

    fmt = F.ConvrotW4A4(args.convrot_groupsize)
    formats = {key: fmt for key in selected}
    out_meta = F.quant_metadata(metadata, F.layer_configs(formats), "ConvRot W4A4",
                                {"smoothquant_alpha": str(args.alpha)})
    norm_set = set(norm_keys)
    # Membro suavizado -> (camada, norma); a norma aponta para o proprio grupo.
    grupo_de: dict[str, tuple[int, str]] = {}
    for layer in layer_ids:
        for norm, members in GROUPS.items():
            grupo_de[f"model.layers.{layer}.{norm}.weight"] = (layer, norm)
            for m in members:
                grupo_de[f"model.layers.{layer}.{m}.weight"] = (layer, norm)

    # Transmite desde 2026-09-29. Antes esta era a ferramenta que MAIS acumulava -- pesos
    # quantizados, escalas e as normas reescritas de todas as camadas -- e a guarda de RAM rodava
    # DEPOIS do acumulo (e da calibragem), onde so podia jogar o trabalho fora. Agora o pico e UM
    # grupo de norma (o maior: q/k/v ou gate/up) mais o que ja saiu dele e ainda espera a vez no
    # header, e as guardas rodam antes da calibragem.
    grupo_bytes = max(sum(header[f"model.layers.{layer}.{m}.weight"]["data_offsets"][1]
                          - header[f"model.layers.{layer}.{m}.weight"]["data_offsets"][0]
                          for m in members)
                      for layer in layer_ids for members in GROUPS.values())
    estado: dict = {"stats": None}
    prontos: dict[str, list[torch.Tensor]] = {}
    lambda_report: list[dict] = []

    import comfy_kitchen as ck

    with conv.tensors() as fonte:
        def calcula_grupo(layer: int, norm: str) -> None:
            stats = estado["stats"]
            norm_key = f"model.layers.{layer}.{norm}.weight"
            member_keys = [f"model.layers.{layer}.{m}.weight" for m in GROUPS[norm]]
            weights = {k: fonte[k].cuda().float() for k in member_keys}
            act_max = stats[norm_key].cuda().clamp(min=1e-5)
            # One lambda per norm: weight_max spans every consumer of that norm.
            weight_max = torch.stack([w.abs().amax(dim=0) for w in weights.values()]).amax(0)
            lam = (act_max.pow(args.alpha) / weight_max.clamp(min=1e-5).pow(1 - args.alpha)
                   ).clamp(min=1e-5)
            for key, weight in weights.items():
                # FP32 na entrada do quantizador desde 2026-09-29 (decisao do dono), como os
                # outros conversores desde 26/09; ate ali este era o unico que ainda passava
                # `.to(torch.bfloat16)` antes da rotacao. Muda os bytes de conversoes FUTURAS.
                prontos[key], _ = fmt.quantize(weight * lam, ck)
            norm_weight = fonte[norm_key].cuda().float()
            # Gemma's RMSNorm applies (1 + w), so the fold has to go through that offset.
            prontos[norm_key] = [((norm_weight + 1.0) / lam - 1.0).to(torch.bfloat16).cpu()]
            lambda_report.append({
                "norm": norm_key,
                "act_channel_ratio_before": (act_max.max() / act_max.median()).item(),
                "act_channel_ratio_after": ((act_max / lam).max() / (act_max / lam).median()).item(),
            })
            del weights, act_max, weight_max, lam, norm_weight
            torch.cuda.empty_cache()

        def produtor(key: str, i: int):
            def produz() -> torch.Tensor:
                if key not in prontos:
                    if key in grupo_de:
                        calcula_grupo(*grupo_de[key])
                    else:  # o_proj / down_proj: quantizados, nunca suavizados
                        prontos[key], _ = fmt.quantize(fonte[key].cuda().float(), ck)
                tensor = prontos[key][i]
                prontos[key][i] = None
                if all(t is None for t in prontos[key]):
                    del prontos[key]
                return tensor
            return produz

        # Tres casos, na mesma ordem de antes: camada quantizada (peso + escala), norma reescrita,
        # ou copia verbatim.
        entradas = []
        for name, info in header.items():
            if name in formats:
                for i, (key, dtype, shape) in enumerate(fmt.tensors(name, info["shape"])):
                    entradas.append(C.plan_lazy(key, dtype, shape, C.nbytes_of(dtype, shape),
                                                produtor(name, i)))
            elif name in norm_set:
                entradas.append(C.plan_lazy(name, "BF16", info["shape"],
                                            C.nbytes_of("BF16", info["shape"]), produtor(name, 0)))
            else:
                entradas.append(C.plan_copy(name, info))
        conv.guard(conv.planned_size(entradas, out_meta), accumulated=3 * grupo_bytes,
                   label="W4A4 SmoothQuant (streaming)")

        estado["stats"] = stats = calibrate(args)
        if len(stats) != len(norm_keys):
            raise SystemExit(f"Calibration saw {len(stats)} norms, expected {len(norm_keys)}")

        started = time.perf_counter()

        def progresso(indice: int, total: int, chave: str) -> None:
            if chave in formats and (indice % 64 == 0 or indice == total):
                print(f"[{indice}/{total}] tensors written", flush=True)

        def manifesto() -> dict:
            before = sum(r["act_channel_ratio_before"] for r in lambda_report) / len(lambda_report)
            after = sum(r["act_channel_ratio_after"] for r in lambda_report) / len(lambda_report)
            estado["ratios"] = (before, after)
            return {
                "source": str(source), "output": str(output), "output_size": conv.output_size,
                "quantization": "ConvRot W4A4 + SmoothQuant", "smoothquant_alpha": args.alpha,
                "calibrated_with": args.calibrate_with,
                "calibration_note": "activations captured from the W4A4 checkpoint with weights "
                                    "retyped so ComfyUI dequantizes them: 4-bit weights, "
                                    "full-precision activations",
                "calibration_prompts": len(CALIBRATION_PROMPTS),
                "smoothed_projections": sorted({m for members in GROUPS.values() for m in members}),
                "quantized_unsmoothed": list(UNSMOOTHED),
                "quantized_tensors": len(selected),
                "act_channel_ratio_before": round(before, 2), "act_channel_ratio_after": round(after, 2),
                "convrot_groupsize": args.convrot_groupsize,
                # Registrado desde 2026-09-29: a entrada do quantizador mudou de BF16 para FP32,
                # entao dois arquivos deste conversor so sao comparaveis se este campo bater.
                "quantizer_input": F.QUANTIZER_INPUT,
                # quant_w4a4.py records this and this file did not, so its outputs were the only
                # convrot_w4a4 checkpoints in the project with no record of which backend produced them.
                "backend": backend["resolved"]["quantize_convrot_w4a4_weight"],
                "backend_linear": backend["resolved"]["convrot_w4a4_linear"],
                "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "conversion_seconds": round(time.perf_counter() - started, 3),
            }

        conv.commit(entradas, out_meta, progress=progresso, sidecar=manifesto)

    before, after = estado["ratios"]
    print(f"activation channel outlier ratio, mean over {len(lambda_report)} norms: "
          f"{before:.0f} -> {after:.1f}")
    elapsed = time.perf_counter() - started
    print(f"Wrote {output} ({C.human_size(output.stat().st_size)}) in {elapsed:.1f} s")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
