"""Choose 4-bit or 8-bit per layer by measuring the real kernels on real activations.

Every converter in this project so far applied one format to every selected layer. That is a bet
that all layers cost the same to quantize, and the measurements say they do not: on Z-Image shapes
ConvRot W4A4's relative error against `F.linear` sat at 0.223 on well-behaved input and 0.246 on
post-activation input, while W4A8 stayed pinned at 0.0737-0.0739 on both. The gap is not uniform,
and where it is small the extra bits buy nothing.

So this measures, per layer, on the tensors the layer was actually handed during generation
(captured by tools/calibrate_activations.py):

    err_bf16   bf16 against a float32 reference    the floor -- error that is not quantization
    err_w4a4   convrot_w4a4_linear                 what the cheap format costs here
    err_w4a8   w4a8_int8_linear                    what the expensive format costs here

and assigns:

    err_w4a4 <= --promote-error            -> convrot_w4a4     (cheap format is good enough)
    otherwise, within --budget             -> asym_w4a8_int8   (promote, ranked by error removed)
    err_w4a8 > --keep-bf16-error           -> no quantization  (neither format is acceptable)

The ranking is by *absolute error removed* (err_w4a4 - err_w4a8), not by ratio: a layer going from
0.30 to 0.07 matters more than one going from 0.004 to 0.001, and a ratio calls them equal.

ComfyUI needs no change to read the result. `comfy/utils.py:1454` turns each entry of
`_quantization_metadata["layers"]` into a per-layer `<layer>.comfy_quant` tensor, and
`comfy/ops.py:1142` dispatches on that layer's own JSON -- a layer with no entry loads as a plain
compute-dtype Parameter. Mixed precision in one file is the format's native behaviour, not a
trick played on it.

Same safety rules as the other converters: refuses to overwrite a source, an existing output, a
stale partial, or to requantize a checkpoint that already carries quantization markers; streams
its output and replaces atomically; and refuses to run at all unless every kernel it measures
resolves to comfy-kitchen's CUDA backend, because the eager backend declares the same
capabilities and would produce numbers that describe dequantized math.

Provenance is checked by the sha256 of the source's safetensors header, not by its filename, and
a calibration or analysis missing any provenance key is refused rather than skipped. Files
written before those keys existed (anything in calib/ from before 2026-08-22) must be
re-measured; there is no flag that waives a missing key, because "nothing to check" was exactly
how the old guard passed.

    python_embeded\\python.exe -s tools/quant_mixed.py \\
        --input ComfyUI/models/diffusion_models/beyond-reality-zimage-v2_bf16.safetensors \\
        --calibration calib/zimage_v2.calib.pt --promote-error 0.15 --dry-run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psutil  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _native_probe import native_backend_ready  # noqa: E402
from calibrate_activations import (  # noqa: E402,F401
    PROFILE_FILE_PATTERNS,
    PROFILE_PATTERNS,
    safetensors_identity_digest,
)

# The error measurement runs at bf16, always, and never at the checkpoint's dtype.
#
# The incident, recorded in CLAUDE.md and in Reservoir.__init__: Z-Image's
# `layers.0.feed_forward.w2` is handed channel magnitudes up to 344064. fp16 caps at 65504. The
# calibration reservoir is bf16 for exactly that reason -- bf16 carries fp32's exponent range --
# but this file then wrote `x = entry["sample"].to(dtype=weight.dtype)`, which on an fp16
# checkpoint pushes the sample straight back through 65504 to inf. Every error for that layer
# comes back nan, `nan > --promote-error` is False, and the layer with the largest activations
# in the model is assigned the *cheapest* format. The reservoir closed that door from one end;
# taking the dtype from the checkpoint at measurement time re-opened it from the other.
#
# So the dtype is a constant here rather than a property of the input, `measure_layer` asserts on
# it rather than casting, and the value is written into the analysis so a reused measurement
# cannot claim to describe a run this one would not reproduce. The float32 reference stays
# float32; bf16 is what the kernels take and what `err_bf16` is defined against.
MEASURE_DTYPE = torch.bfloat16

ERROR_METRICS = ("err_bf16", "err_w4a4", "err_w4a8")

SAFETENSORS_DTYPE = {
    torch.int8: "I8", torch.uint8: "U8", torch.float32: "F32",
    torch.bfloat16: "BF16", torch.float16: "F16",
}
HIGH_PRECISION_DTYPES = {"BF16", "F16", "F32"}
TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# The ops that must resolve to CUDA before any number produced here means anything.
REQUIRED_OPS = ("quantize_convrot_w4a4_weight", "convrot_w4a4_linear",
                "quantize_w4a8_int8_weight", "w4a8_int8_linear")


def human_size(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.2f} TiB"


def mixed_probe_ops(args: argparse.Namespace) -> dict:
    """All four ops this file measures with, each at the configuration it will measure at.

    This used to be a private `normal_comfy_backend()` right here, one of six across `tools/`, and
    of the six it was the closest to right: it already resolved all four ops and already passed
    real quantizer output rather than `torch.empty` placeholders. What it did not do was pass
    `--group-size` and `--convrot-groupsize`; it hardcoded 16 and 256, which happen to be the
    defaults, so `--convrot-groupsize 64` preflighted at 256. Now the args go through.

    Why real kwargs at all: `registry.get_implementation`'s own docstring says "kwargs: Kwargs for
    constraint validation (empty/None skips validation)", so a no-kwargs probe asks which backend
    would be picked *ignoring every constraint*, while the real call drops to eager on a failing
    constraint with only a logger.debug line. READ, not measured -- and see `_native_probe.py`'s
    docstring for the 2026-08-22 run in which dummy and real did NOT differ for the ConvRot pair.
    The kwargs stay real regardless: they are what this file is about to call with.
    """
    ops = {
        "quantize_convrot_w4a4_weight": {"convrot_groupsize": args.convrot_groupsize},
        "convrot_w4a4_linear": {"convrot_groupsize": args.convrot_groupsize},
        "quantize_w4a8_int8_weight": {"convrot_groupsize": args.convrot_groupsize,
                                      "group_size": args.group_size,
                                      # measure_layer() always passes codebook=True
                                      "codebook": True},
        "w4a8_int8_linear": {"convrot_groupsize": args.convrot_groupsize,
                             "group_size": args.group_size, "codebook": True},
    }
    # REQUIRED_OPS is the list this file's own docstring promises to check. Keeping the two in
    # one place would be tidier; keeping them in two with this check is what stops a future op
    # being added to the promise and silently not probed.
    missing = [name for name in REQUIRED_OPS if name not in ops]
    if missing:
        raise SystemExit(f"mixed_probe_ops has no configuration for {missing}, which "
                         "REQUIRED_OPS says must resolve to CUDA before any number here means "
                         "anything")
    return ops


def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    metadata = header.pop("__metadata__", {}) or {}
    return header, metadata


def read_tensor(handle, offset: int, size: int, dtype: str, shape: list[int]) -> torch.Tensor:
    handle.seek(offset)
    raw = handle.read(size)
    if len(raw) != size:
        raise RuntimeError(f"short read: wanted {size}, got {len(raw)}")
    return torch.frombuffer(bytearray(raw), dtype=TORCH_DTYPES[dtype]).reshape(shape)


def copy_range(source_handle, output_handle, offset: int, size: int) -> None:
    source_handle.seek(offset)
    remaining = size
    chunk = 16 * 1024 * 1024
    while remaining:
        block = source_handle.read(min(chunk, remaining))
        if not block:
            raise RuntimeError("unexpected end of source while copying")
        output_handle.write(block)
        remaining -= len(block)


def selected_layers(header: dict, profile: str, convrot_groupsize: int) -> list[str]:
    # File-key pattern, not the module pattern: for HunyuanVideo the checkpoint says
    # img_attn_qkv where the loaded module says img_attn.qkv, and matching the wrong one
    # selects nothing while looking like a profile that simply found no layers.
    pattern = PROFILE_FILE_PATTERNS[profile]
    names = []
    for name, info in header.items():
        if not name.endswith(".weight") or not pattern.match(name.removesuffix(".weight")):
            continue
        if info["dtype"] not in HIGH_PRECISION_DTYPES or len(info["shape"]) != 2:
            continue
        if info["shape"][1] % convrot_groupsize:
            continue
        names.append(name)
    return names


def relative(reference: torch.Tensor, got: torch.Tensor,
             weights: torch.Tensor | None = None) -> float:
    """L2 relativa, opcionalmente com um peso POR LINHA da amostra.

    Sem `weights` isto e exatamente o que sempre foi: uma norma global sobre todas as linhas,
    que trata cada linha igual -- e como o reservoir mistura os passos do sampler, "cada linha
    igual" e na pratica "cada passo com o peso que o numero de linhas dele deu".

    Com `weights` a conta vira uma razao de normas ponderadas:

        sqrt(sum_i w_i * ||ref_i - got_i||^2) / sqrt(sum_i w_i * ||ref_i||^2)

    que reduz exatamente ao caso sem peso quando todos os w_i sao iguais -- verificado em
    `tools/test_quant_mixed_sigma.py`, porque uma formula que NAO reduz seria uma mudanca
    silenciosa de metrica em vez de uma ponderacao.
    """
    ref = reference.float()
    dif = ref - got.float()
    if weights is None:
        return float(dif.norm() / ref.norm().clamp(min=1e-12))
    w = weights.to(device=ref.device, dtype=torch.float32).reshape(-1, 1)
    if w.shape[0] != ref.shape[0]:
        raise SystemExit(f"pesos com {w.shape[0]} linhas para uma referencia de {ref.shape[0]}")
    num = (dif.pow(2) * w).sum().sqrt()
    den = (ref.pow(2) * w).sum().sqrt().clamp(min=1e-12)
    return float(num / den)


SIGMA_WEIGHTS = ("none", "sigma", "sigma2", "high")


def sigma_weights(mode: str, sigma: torch.Tensor | None, layer: str) -> torch.Tensor | None:
    """Peso por linha a partir do sigma em que ela foi observada.

    Motivo, medido e nao suposto: `tools/probe_epsilon_per_step.py` mostrou em 2026-08-30 que o
    erro do epsilon cai monotonico de 6,17e-1 em sigma 1,000 para 5,31e-2 em 0,300. O dano bate
    mais forte onde a estrutura e decidida. Se isso importa para a decisao de formato, um
    criterio que pese as linhas de sigma alto e diferente do que trata todo passo igual.

    `sigma` e `sigma2` sao continuos; `high` e um corte na mediana dos sigmas OBSERVADOS nesta
    camada, e nao um limiar fixo, porque o intervalo de sigma depende do scheduler e um 0,5
    escrito aqui viraria um corte diferente a cada configuracao.

    Um NaN no vetor para a execucao. Nao vira zero: zero descartaria a linha em silencio, que e
    o modo de falha registrado em `finite()` logo abaixo, apontando para outro lado.
    """
    if mode == "none":
        return None
    if sigma is None:
        raise SystemExit(
            f"{layer}: --sigma-weight {mode!r} pedido, mas a calibracao nao tem 'sample_sigma'. "
            "Recalibre com a versao de `calibrate_activations.py` que grava o sigma por linha; "
            "uma calibracao antiga nao pode ser reponderada depois porque o reservoir ja "
            "misturou os passos.")
    s = sigma.float().reshape(-1)
    if not bool(torch.isfinite(s).all()):
        n = int((~torch.isfinite(s)).sum())
        raise SystemExit(
            f"{layer}: {n} de {s.numel()} linhas da amostra estao sem sigma (NaN). Nao sao "
            "tratadas como zero: uma linha sem rotulo descartada em silencio muda a metrica "
            "sem aparecer no resultado.")
    if mode == "sigma":
        w = s
    elif mode == "sigma2":
        w = s.pow(2)
    elif mode == "high":
        w = (s >= s.median()).float()
    else:
        raise SystemExit(f"--sigma-weight desconhecido: {mode!r}")
    total = float(w.sum())
    if not math.isfinite(total) or total <= 0:
        raise SystemExit(
            f"{layer}: o peso por sigma somou {total!r}. Todos os sigmas desta camada sao zero "
            "ou o modo zerou o vetor inteiro; a razao ponderada seria 0/0.")
    return w


def finite(layer: str, metric: str, value) -> float:
    """A non-finite error metric ends the run, naming the layer. It never becomes a number.

    `nan > threshold` is False and `nan < threshold` is False, so a nan that reaches the decision
    does not look like an error -- it looks like a layer whose cheap format was good enough, and
    the layer it happens to is the one with the largest activations in the model. This is the
    failure the whole ticket is about, and the only reliable place to stop it is before the value
    is allowed to exist as a float that something can compare.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SystemExit(
            f"{layer}: {metric} is {value!r}, not a finite number. This is not routed to "
            "--uncalibrated and it is not skipped: a non-finite error means the measurement did "
            "not describe this layer, and a decision made from it would be arbitrary. Check the "
            "calibration sample for this layer (an inf there is the fp16 65504 overflow -- the "
            "reservoir must be bf16), then re-measure.")
    return float(value)


def validate_analysis_rows(rows, origin: str) -> None:
    """Every calibrated row must carry a shape and three finite errors, whoever produced it.

    The measurement branch already cannot emit a non-finite metric (`measure_layer` refuses), but
    an `--analysis` file is not necessarily one this tool wrote: `json.loads` accepts the bare
    tokens `NaN`, `Infinity` and `-Infinity`, so a hand-edited or hand-merged analysis can carry
    a nan into the comparison without ever passing through the measurement path. This runs on
    both branches, once, immediately before the errors are used for anything -- including the
    "worst layers" table, which sorts on `err_w4a4`.
    """
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{origin} has no 'layers' rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "layer" not in row:
            raise SystemExit(f"{origin}: row {index} has no 'layer' name")
        layer = row["layer"]
        if not row.get("shape"):
            raise SystemExit(
                f"{origin}: {layer} has no 'shape'. The shape is what proves the analysis and "
                "the input hold the same weights; a row without one cannot be checked.")
        if not row.get("calibrated"):
            continue
        for metric in ERROR_METRICS:
            if metric not in row:
                raise SystemExit(
                    f"{origin}: {layer} is marked calibrated but has no {metric!r}.")
            finite(f"{origin}: {layer}", metric, row[metric])


def measure_layer(layer: str, weight: torch.Tensor, x: torch.Tensor, ck,
                  group_size: int, convrot_groupsize: int,
                  weights: torch.Tensor | None = None) -> dict:
    """Relative L2 of each format against a float32 reference, on this layer's real input.

    Both operands must already be MEASURE_DTYPE; this function casts neither, because the cast it
    used to do -- the activation to the checkpoint's dtype -- is the bug. See MEASURE_DTYPE.
    """
    if weight.dtype is not MEASURE_DTYPE or x.dtype is not MEASURE_DTYPE:
        raise SystemExit(
            f"{layer}: the error measurement must run at {MEASURE_DTYPE}, got "
            f"weight={weight.dtype} x={x.dtype}. Taking the dtype from the checkpoint is the "
            "344064-vs-65504 overflow described at MEASURE_DTYPE; cast to MEASURE_DTYPE at the "
            "read, not here.")
    reference = F.linear(x.float(), weight.float())
    result = {"err_bf16": finite(layer, "err_bf16",
                                 relative(reference, F.linear(x, weight), weights))}

    qdata4, wscales4 = ck.quantize_convrot_w4a4_weight(weight, convrot_groupsize, 64)
    got4 = ck.convrot_w4a4_linear(x, qdata4, wscales4, None, convrot_groupsize, 64)
    result["err_w4a4"] = finite(layer, "err_w4a4", relative(reference, got4, weights))
    del qdata4, wscales4, got4

    qdata8, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
        weight, group_size=group_size, convrot_groupsize=convrot_groupsize,
        symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True,
        codebook_tensor=None, stochastic_rounding=0)
    if correction is not None:
        raise SystemExit("symmetric=True returned a correction tensor; ComfyUI would drop it")
    got8 = ck.w4a8_int8_linear(x, qdata8, s_rel, s_channel, codebook=codebook,
                               correction=None, bias=None, group_size=group_size,
                               convrot_groupsize=convrot_groupsize, out_dtype=x.dtype)
    result["err_w4a8"] = finite(layer, "err_w4a8", relative(reference, got8, weights))
    del qdata8, s_rel, s_channel, codebook, got8, reference
    return result


def required(mapping: dict, key: str, origin: str):
    """A provenance key that is absent is a refusal, not a skip.

    Every check in the --analysis branch used to read `recorded = analysis.get(field)` and then
    `if recorded is not None and recorded != current`. That made the guard strongest on complete
    files and absent on exactly the hand-edited, hand-merged or older ones it existed to catch: a
    file with fewer keys passed every check by having nothing to check.
    """
    value = mapping.get(key)
    if value is None:
        raise SystemExit(
            f"{origin} carries no {key!r}. Provenance keys are required, not optional -- "
            "'nothing to check' is not 'checked'. Re-run the tool that produced this file "
            "(tools/calibrate_activations.py, or this tool with --save-analysis); files written "
            "before the provenance keys existed cannot be verified against this input.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path,
                        help="a .calib.pt from tools/calibrate_activations.py")
    parser.add_argument("--analysis", type=Path,
                        help="reuse a previously written analysis JSON instead of remeasuring")
    parser.add_argument("--foreign-analysis", action="store_true",
                        help="allow --analysis measured on a DIFFERENT checkpoint of the same "
                             "architecture, skipping calibration entirely. Measured 2026-08-19 on "
                             "three Z-Image checkpoints (turbo, de-turbo, beyond-reality-v2): the "
                             "per-layer err_w4a4 ranking agrees at Spearman +0.981 to +0.997, and "
                             "at --promote-error 0.10 the three select 119-120 of the same 120 "
                             "layers where chance overlap is 84.7. The layer set and every shape "
                             "must still match exactly. This has NOT been shown across "
                             "architectures or vendors -- only across checkpoints of one model.")
    parser.add_argument("--save-analysis", type=Path,
                        help="write the per-layer measurement table here")
    parser.add_argument("--profile", choices=list(PROFILE_PATTERNS),
                        help="defaults to the profile recorded in the calibration")
    parser.add_argument("--promote-error", type=float, default=0.15,
                        help="W4A4 relative error above which a layer is promoted to W4A8")
    parser.add_argument("--keep-bf16-error", type=float, default=None,
                        help="W4A8 relative error above which a layer is left unquantized")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="max fraction of selected layers allowed at W4A8 (0.0-1.0)")
    parser.add_argument("--uncalibrated", choices=["w4a8", "bf16", "fail"], default="w4a8",
                        help="what to do with a selected layer the calibration never saw")
    parser.add_argument("--sigma-weight", choices=list(SIGMA_WEIGHTS), default="none",
                        help="pesa cada linha da amostra pelo sigma em que ela foi observada. "
                             "'none' e o comportamento historico e o default: mudar o default "
                             "reinterpretaria em silencio toda analise ja gravada. Exige uma "
                             "calibracao com 'sample_sigma'.")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise SystemExit("Input must be an existing .safetensors file")
    if not 0.0 <= args.budget <= 1.0:
        raise SystemExit("--budget must be between 0.0 and 1.0")
    if not args.calibration and not args.analysis:
        raise SystemExit("Give --calibration to measure, or --analysis to reuse a measurement")

    output = (args.output or source.with_name(f"{source.stem}_mixed.safetensors")).resolve()
    sidecar = output.with_suffix(".quant.json")
    if output == source:
        raise SystemExit("Refusing to overwrite the source model")
    if not args.dry_run and (output.exists() or sidecar.exists()):
        raise SystemExit(f"Refusing to overwrite existing output or sidecar: {output}")

    header, metadata = read_header(source)
    if metadata.get("_quantization_metadata"):
        raise SystemExit("Refusing to requantize a checkpoint that already has quantization "
                         "metadata")
    inline_quant = sum(1 for name in header if name.endswith(".comfy_quant"))
    if inline_quant:
        raise SystemExit(f"Refusing to requantize: source carries {inline_quant} inline "
                         "'.comfy_quant' markers, so it is already quantized")

    # A diffusers-named checkpoint would match only the feed_forward third of the profile and
    # produce a file whose attention scales land under names no module reads. That failure is
    # silent -- the file loads -- so it is checked here rather than left to be discovered in an
    # image three steps later.
    diffusers_named = [n for n in header if n.endswith("attention.to_q.weight")]
    if diffusers_named:
        raise SystemExit(
            f"{source.name} is in diffusers naming ({len(diffusers_named)} 'attention.to_q' "
            "keys). ComfyUI fuses those into 'attention.qkv' at load and does not carry the "
            "quantization scales across. Run tools/to_native.py first and quantize its output.")

    # Identity of the checkpoint being converted, computed once. A basename does not identify a
    # checkpoint on this bench -- outputs are written beside their source and a second model tree
    # is mounted from D:/ComfyUI-Models -- so every provenance comparison below is against this,
    # and the names only appear in the messages. See safetensors_identity_digest for the incident,
    # and for the follow-up measurement showing that the header ALONE is not an identity either --
    # four groups of checkpoints in this install share a header byte for byte.
    source_digest = safetensors_identity_digest(source)

    analysis = None
    if args.analysis:
        origin = f"analysis {args.analysis}"
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        profile = args.profile or required(analysis, "profile", origin)
        calibration_meta = analysis.get("calibration", {})
        activations = None
        # The measurement branch refuses a calibration captured from a different checkpoint. The
        # --analysis branch checked nothing at all, so a saved analysis could be replayed against
        # another model, or with different group sizes than it was measured under, and the
        # per-layer errors would silently describe a different computation than the one being
        # written.
        recorded_source = Path(required(analysis, "source", origin)).name
        recorded_digest = required(analysis, "source_identity_sha256", origin)
        # ORed, not swapped. The digest samples the body and is strictly better than a basename at
        # telling two *different* files apart -- but it is still a sample, and the check it
        # replaced was already refusing every pairing later measured to collide. Keeping both
        # means the weaker signal can only ever ADD a refusal, never remove one. A legitimate
        # rename or copy still passes, through --foreign-analysis, which is a flag and not a wall.
        foreign = recorded_digest != source_digest or recorded_source != source.name
        if foreign and not args.foreign_analysis:
            raise SystemExit(
                f"The analysis was measured on {recorded_source!r} (identity "
                f"{recorded_digest[:16]}) but the input is {source.name!r} "
                f"({source_digest[:16]}). These are not the same checkpoint even if the names "
                f"match. Re-measure with --calibration, or pass --foreign-analysis if the two "
                f"are the same architecture -- see its help text for what that buys and what it "
                f"costs.")
        if foreign:
            # Loud, every run. The whole risk of this flag is that someone forgets which
            # measurement produced the file they are shipping.
            print(f"--foreign-analysis: promoting layers of {source.name!r} "
                  f"({source_digest[:16]}) using errors measured on {recorded_source!r} "
                  f"({recorded_digest[:16]}).")
        # Required, not `is not None`: the whole point of item 2 of this ticket is that an
        # analysis missing the field it would be checked on used to pass by having nothing to
        # check.
        for field, current in (("group_size", args.group_size),
                               ("convrot_groupsize", args.convrot_groupsize)):
            recorded = required(analysis, field, origin)
            if recorded != current:
                raise SystemExit(
                    f"The analysis was measured with {field}={recorded} but this run uses "
                    f"{current}. The per-layer errors would not describe what gets written; "
                    f"pass --{field.replace('_', '-')} {recorded} or re-measure.")
        # No flag to reconcile this one: a measurement taken at the checkpoint's dtype -- which
        # is what this tool did before MEASURE_DTYPE existed -- is not reusable at any setting,
        # because on an fp16 checkpoint it is the 344064-vs-65504 overflow.
        recorded_dtype = required(analysis, "measure_dtype", origin)
        if recorded_dtype != str(MEASURE_DTYPE):
            raise SystemExit(
                f"The analysis was measured at {recorded_dtype} and this tool measures at "
                f"{MEASURE_DTYPE}. Re-measure with --calibration; see MEASURE_DTYPE for why the "
                "measurement dtype is not negotiable.")
        # Uma analise gravada com outra ponderacao de sigma descreve outra metrica, e as
        # colunas tem o mesmo nome nas duas. Sem esta checagem, reusar uma analise antiga com
        # `--sigma-weight sigma2` produziria uma decisao "ponderada" feita de numeros nao
        # ponderados, sem nenhum sinal na saida. Analise sem o campo e pre-2026-08-31 e portanto
        # 'none' -- o que e verdade, e nao uma suposicao conveniente: a ponderacao nao existia.
        recorded_sw = analysis.get("sigma_weight", "none")
        if recorded_sw != args.sigma_weight:
            raise SystemExit(
                f"A analise foi medida com --sigma-weight {recorded_sw!r} e esta execucao pede "
                f"{args.sigma_weight!r}. Os campos err_* tem o mesmo nome nos dois casos e "
                "significam coisas diferentes. Re-meca com --calibration.")
        validate_analysis_rows(analysis.get("layers"), origin)
        shapes = {row["layer"]: tuple(row["shape"]) for row in analysis["layers"]}
        for name, info in header.items():
            stem = name.removesuffix(".weight")
            if stem in shapes and tuple(info["shape"]) != shapes[stem]:
                raise SystemExit(
                    f"{stem}: the analysis recorded shape {list(shapes[stem])} but the input has "
                    f"{info['shape']}. These are not the same weights.")
        if foreign:
            # Same shapes where both have a layer is not enough when the analysis comes from
            # another file: a layer the analysis never measured would fall to --uncalibrated
            # handling silently, and a layer the analysis has but the input lacks means the two
            # are not the architecture this transfer was measured on. Both must be empty.
            # selected_layers returns tensor names; the analysis keys layers without the suffix.
            present = {n.removesuffix(".weight")
                       for n in selected_layers(header, profile, args.convrot_groupsize)}
            missing = sorted(set(shapes) - present)
            extra = sorted(present - set(shapes))
            if missing or extra:
                raise SystemExit(
                    f"--foreign-analysis requires the same layer set. "
                    f"{len(missing)} in the analysis but not in the input "
                    f"(e.g. {missing[:2]}), {len(extra)} the other way (e.g. {extra[:2]}). "
                    f"These are different architectures; measure this one.")
    else:
        origin = f"calibration {args.calibration}"
        blob = torch.load(args.calibration, map_location="cpu", weights_only=False)
        calibration_meta = blob["meta"]
        activations = blob["layers"]
        profile = args.profile or required(calibration_meta, "profile", origin)
        recorded_source = Path(required(calibration_meta, "source", origin)).name
        recorded_digest = required(calibration_meta, "source_identity_sha256", origin)
        if recorded_digest != source_digest or recorded_source != source.name:
            # Calibrating on one checkpoint and converting another produces a file that looks
            # fine and is tuned for the wrong activations. Refuse rather than warn. This compared
            # basenames until 2026-08-22, and a basename is not an identity here: outputs are
            # written beside their source and D:/ComfyUI-Models is mounted alongside
            # ComfyUI/models, so the same name in two directories is the normal case, not a
            # corner one.
            #
            # But the digest did not REPLACE the basename, it was ORed with it, and that is the
            # correction of 2026-08-22 rather than the change: a header-only digest let three
            # different Z-Image checkpoints -- and the two halves of a Wan i2v pair, and the two
            # passes of one conversion -- compare equal, so for those the basename check being
            # dropped turned a refusal into a silent acceptance. The digest now samples the body
            # too, but it is still a sample, and a weaker signal ORed in can only ever add a
            # refusal. See safetensors_identity_digest for the measurement.
            raise SystemExit(
                f"Calibration was captured from {recorded_source!r} (identity "
                f"{recorded_digest[:16]}) but the input is {source.name!r} "
                f"({source_digest[:16]}). Even with the same filename these are different "
                f"checkpoints. Recalibrate against this file.")

    selected = selected_layers(header, profile, args.convrot_groupsize)
    if not selected:
        raise SystemExit(f"Profile {profile!r} selected no compatible layers")

    print(f"Source:  {source}")
    print(f"Profile: {profile}   group_size={args.group_size} "
          f"convrot_groupsize={args.convrot_groupsize}")
    print(f"Selected Linear weights: {len(selected)}")
    print(f"Output:  {output}\n")

    backend = native_backend_ready(PORTABLE_ROOT, mixed_probe_ops(args))
    resolved = backend["resolved"]
    wrong = {name: module for name, module in resolved.items()
             if "comfy_kitchen.backends.cuda" not in module}
    if wrong:
        raise SystemExit("Refusing: normal ComfyUI would not use the CUDA backend for "
                         + ", ".join(f"{k} -> {v}" for k, v in wrong.items()))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    for name, module in resolved.items():
        print(f"  {name:<32} -> {module}")
    print()

    import comfy_kitchen as ck

    # ---- measurement -------------------------------------------------------------------------
    if analysis is None:
        missing = [n for n in selected if n.removesuffix(".weight") not in activations]
        if missing and args.uncalibrated == "fail":
            raise SystemExit(f"{len(missing)} selected layer(s) absent from the calibration, "
                             f"e.g. {missing[:3]}")
        started = time.perf_counter()
        rows = []
        with source.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            data_start = 8 + header_size
            for index, name in enumerate(selected, 1):
                info = header[name]
                start, end = info["data_offsets"]
                stem = name.removesuffix(".weight")
                entry = activations.get(stem)
                if entry is None or entry["sample"].shape[0] == 0:
                    rows.append({"layer": stem, "shape": info["shape"], "calibrated": False})
                    continue
                # Both operands are cast to MEASURE_DTYPE at the read. The activation used to be
                # cast to `weight.dtype`, i.e. the checkpoint's -- read MEASURE_DTYPE for what
                # that does to a layer whose activations reach 344064 on an fp16 checkpoint.
                weight = read_tensor(handle, data_start + start, end - start,
                                     info["dtype"], info["shape"]).to(device="cuda",
                                                                      dtype=MEASURE_DTYPE)
                x = entry["sample"].to(device="cuda", dtype=MEASURE_DTYPE)
                if not torch.isfinite(x).all():
                    # A calibration that already contains inf/nan is a broken calibration, not a
                    # layer with an interesting error: routing it to --uncalibrated would hide a
                    # capture bug behind a per-layer format choice. Name the layer and stop.
                    raise SystemExit(
                        f"{stem}: the calibration sample contains inf or nan. The reservoir is "
                        "bf16 precisely so real activations (up to 344064 on Z-Image) cannot "
                        "overflow the way fp16's 65504 does, so this is a capture bug, not a "
                        f"property of the layer. Recapture with tools/calibrate_activations.py.")
                w = sigma_weights(args.sigma_weight, entry.get("sample_sigma"), stem)
                measured = measure_layer(stem, weight, x, ck,
                                         args.group_size, args.convrot_groupsize, w)
                measured.update({"layer": stem, "shape": info["shape"], "calibrated": True,
                                 "rows": int(entry["rows"]),
                                 "sigma_weight": args.sigma_weight,
                                 "crest_p99": float(entry["crest_p99"])})
                if w is not None:
                    s = entry["sample_sigma"].float()
                    measured["sigma_span"] = [float(s.min()), float(s.max())]
                    measured["sigma_weight_mass_top_half"] = float(
                        w[s >= s.median()].sum() / w.sum())
                rows.append(measured)
                del weight, x
                torch.cuda.empty_cache()
                if index % 24 == 0 or index == len(selected):
                    print(f"[{index}/{len(selected)}] measured", flush=True)
        analysis = {
            "profile": profile, "source": str(source),
            # Written so a reused analysis can be checked against the checkpoint it describes
            # instead of against a filename, and so the sidecar of any file built from it can
            # carry the same identity forward. Both are what item 1 of the ticket is for.
            "source_identity_sha256": source_digest,
            "measure_dtype": str(MEASURE_DTYPE),
            "sigma_weight": args.sigma_weight,
            "calibration": calibration_meta,
            "group_size": args.group_size, "convrot_groupsize": args.convrot_groupsize,
            "seconds": round(time.perf_counter() - started, 2),
            "layers": rows,
        }
        if args.save_analysis:
            args.save_analysis.parent.mkdir(parents=True, exist_ok=True)
            args.save_analysis.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            print(f"wrote analysis to {args.save_analysis}")

    # Both branches land here, and nothing downstream reads an error metric before this line.
    # The measurement branch cannot produce a non-finite one (`measure_layer` refuses), but a
    # reused --analysis file is not necessarily one this tool wrote.
    validate_analysis_rows(analysis["layers"], "analysis")

    by_layer = {row["layer"]: row for row in analysis["layers"]}

    # ---- decision ----------------------------------------------------------------------------
    decision: dict[str, str] = {}
    for name in selected:
        stem = name.removesuffix(".weight")
        row = by_layer.get(stem)
        if row is None or not row.get("calibrated"):
            # `fail` used to be checked only inside the measurement branch, so reusing a saved
            # analysis with --analysis turned "refuse if any layer is uncalibrated" into a
            # silent bf16 passthrough -- the opposite of what the flag asks for.
            if args.uncalibrated == "fail":
                raise SystemExit(
                    f"{stem} has no usable measurement and --uncalibrated=fail was given. "
                    "Recalibrate, or choose --uncalibrated w4a8 (safe) or bf16 (unquantized).")
            decision[stem] = "asym_w4a8_int8" if args.uncalibrated == "w4a8" else "bf16"
            continue
        if args.keep_bf16_error is not None and row["err_w4a8"] > args.keep_bf16_error:
            decision[stem] = "bf16"
        elif row["err_w4a4"] > args.promote_error:
            decision[stem] = "asym_w4a8_int8"
        else:
            decision[stem] = "convrot_w4a4"

    # Budget caps promotions, keeping the ones where the promotion removes the most error.
    # Uncalibrated layers are held out of the ranking entirely rather than given gain=inf. The
    # inf trick looked safe -- "never demote something that was never measured" -- but the cut is
    # a fixed-size slice, `sorted(...)[:len(promoted) - allowed]`, so once the budget is tight
    # enough the slice runs past the measured layers and reaches the inf keys anyway. Simulated:
    # 170 layers, 12 uncalibrated, --budget 0.05 demotes exactly the layers the comment promised
    # to protect.
    promoted = [s for s, f in decision.items() if f == "asym_w4a8_int8"]
    measured = [s for s in promoted
                if by_layer.get(s) is not None and by_layer[s].get("calibrated")]
    unmeasured_promoted = [s for s in promoted if s not in set(measured)]
    allowed = int(len(selected) * args.budget)
    if len(promoted) > allowed:
        room = max(0, allowed - len(unmeasured_promoted))
        def gain(stem: str) -> float:
            row = by_layer[stem]
            return row["err_w4a4"] - row["err_w4a8"]
        demoted = sorted(measured, key=gain)[:max(0, len(measured) - room)]
        for stem in demoted:
            decision[stem] = "convrot_w4a4"
        print(f"budget {args.budget:.2f} capped promotions at {allowed}; "
              f"demoted {len(demoted)} measured layer(s) with the smallest error removed")
        if unmeasured_promoted:
            print(f"  {len(unmeasured_promoted)} uncalibrated layer(s) kept at W4A8 and excluded "
                  f"from the ranking; they consume budget but are never demoted by it")
        if len(unmeasured_promoted) > allowed:
            print(f"  warning: uncalibrated layers alone ({len(unmeasured_promoted)}) exceed the "
                  f"budget of {allowed}. The budget cannot be honoured without demoting a layer "
                  "that was never measured; it is being exceeded instead.")
        print()

    counts = {"convrot_w4a4": 0, "asym_w4a8_int8": 0, "bf16": 0}
    for fmt in decision.values():
        counts[fmt] += 1
    print(f"{'format':<20}{'layers':>8}")
    for fmt, count in counts.items():
        print(f"{fmt:<20}{count:>8}")

    unmeasured = [row for row in analysis["layers"] if not row.get("calibrated")]
    if unmeasured:
        print(f"\n{len(unmeasured)} layer(s) were not measured and took --uncalibrated="
              f"{args.uncalibrated}:")
        for row in unmeasured[:10]:
            print(f"  {row['layer']}: {row.get('reason', 'absent from the calibration')}")

    measured_rows = [r for r in analysis["layers"] if r.get("calibrated")]
    if measured_rows:
        worst = sorted(measured_rows, key=lambda r: -r["err_w4a4"])[:12]
        print(f"\n{'layer':<40}{'bf16':>9}{'w4a4':>9}{'w4a8':>9}{'crest p99':>11}  format")
        for row in worst:
            print(f"{row['layer']:<40}{row['err_bf16']:>9.4f}{row['err_w4a4']:>9.4f}"
                  f"{row['err_w4a8']:>9.4f}{row['crest_p99']:>11.2f}  {decision[row['layer']]}")
        w4a4_rows = [r for r in measured_rows if decision[r["layer"]] == "convrot_w4a4"]
        if w4a4_rows:
            print(f"\nworst W4A4 error left in the model: "
                  f"{max(r['err_w4a4'] for r in w4a4_rows):.4f}")

    if args.dry_run:
        return 0

    # ---- quantize ----------------------------------------------------------------------------
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise SystemExit(f"Refusing to overwrite stale partial output: {partial}")
    quant_names = [n for n in selected if decision[n.removesuffix(".weight")] != "bf16"]
    if not quant_names:
        raise SystemExit("Every layer was left at bf16; there is nothing to write")
    # This converter is two-pass: every selected layer is quantized into `quantized` and only
    # then written. The old guard was `largest * 3 + 2 GiB`, copied from quant_w4a4.py, which is
    # correct for a streaming design and understates a two-pass one by the layer count -- it
    # asked for 2.25 GiB against an accumulation of 2.92 GiB on the smallest model in this
    # project, and would understate a large model by several times.
    from _ram_guard import check, convrot_w4a4_bytes, w4a8_bytes

    accumulated = 0
    for name in quant_names:
        rows, cols = header[name]["shape"]
        if decision[name.removesuffix(".weight")] == "convrot_w4a4":
            accumulated += convrot_w4a4_bytes(rows, cols)
        else:
            accumulated += w4a8_bytes(rows, cols, args.group_size, codebook=True)
    refusal = check(psutil.virtual_memory().available, accumulated, label="mixed conversion")
    if refusal:
        raise SystemExit(refusal)
    if shutil.disk_usage(output.parent).free < source.stat().st_size:
        raise SystemExit("Insufficient disk space")

    started = time.perf_counter()
    quantized: dict[str, dict] = {}
    with source.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        data_start = 8 + header_size
        for index, name in enumerate(quant_names, 1):
            info = header[name]
            start, end = info["data_offsets"]
            stem = name.removesuffix(".weight")
            weight = read_tensor(handle, data_start + start, end - start,
                                 info["dtype"], info["shape"]).to("cuda")
            if decision[stem] == "convrot_w4a4":
                qdata, scale = ck.quantize_convrot_w4a4_weight(
                    weight, args.convrot_groupsize, 64)
                quantized[name] = {"format": "convrot_w4a4",
                                   "qdata": qdata.cpu().contiguous(),
                                   "scale": scale.cpu().contiguous()}
                del qdata, scale
            else:
                qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
                    weight, group_size=args.group_size,
                    convrot_groupsize=args.convrot_groupsize, symmetric=True,
                    scale_dtype=torch.float8_e4m3fn, codebook=True,
                    codebook_tensor=None, stochastic_rounding=0)
                if correction is not None:
                    raise SystemExit("symmetric=True returned a correction tensor; "
                                     "ComfyUI would drop it")
                quantized[name] = {"format": "asym_w4a8_int8",
                                   "qdata": qdata.cpu().contiguous(),
                                   "s_rel": s_rel.cpu().contiguous(),
                                   "s_channel": s_channel.cpu().contiguous(),
                                   "codebook": None if codebook is None
                                   else codebook.cpu().contiguous()}
                del qdata, s_rel, s_channel, codebook
            del weight
            torch.cuda.empty_cache()
            if index % 24 == 0 or index == len(quant_names):
                print(f"[{index}/{len(quant_names)}] quantized", flush=True)

    # ---- write -------------------------------------------------------------------------------
    layers_meta = {}
    for name in quant_names:
        stem = name.removesuffix(".weight")
        if decision[stem] == "convrot_w4a4":
            layers_meta[stem] = {"format": "convrot_w4a4",
                                 "convrot_groupsize": args.convrot_groupsize}
        else:
            layers_meta[stem] = {"format": "asym_w4a8_int8",
                                 "group_size": args.group_size,
                                 "convrot_groupsize": args.convrot_groupsize}
    output_metadata = dict(metadata)
    output_metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": layers_meta}, separators=(",", ":"))
    output_metadata["quantization"] = "mixed convrot_w4a4 / asym_w4a8_int8"

    target = {"__metadata__": output_metadata}
    offset = 0
    plan = []
    quant_set = set(quant_names)
    for name, info in header.items():
        if name in quant_set:
            stem = name.removesuffix(".weight")
            entry = quantized[name]
            if entry["format"] == "convrot_w4a4":
                pieces = [(name, entry["qdata"]), (f"{stem}.weight_scale", entry["scale"])]
            else:
                pieces = [(name, entry["qdata"]),
                          (f"{stem}.weight_s_rel", entry["s_rel"]),
                          (f"{stem}.weight_s_channel", entry["s_channel"])]
                if entry["codebook"] is not None:
                    pieces.append((f"{stem}.weight_codebook", entry["codebook"]))
            for key, tensor in pieces:
                store = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
                nbytes = store.numel() * store.element_size()
                target[key] = {"dtype": SAFETENSORS_DTYPE[store.dtype],
                               "shape": list(tensor.shape),
                               "data_offsets": [offset, offset + nbytes]}
                plan.append(("write", store))
                offset += nbytes
        else:
            start, end = info["data_offsets"]
            size = end - start
            target[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + size]}
            plan.append(("copy", (start, size)))
            offset += size

    payload = json.dumps(target, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * (-len(payload) % 8)

    try:
        with source.open("rb") as source_handle, partial.open("xb") as out_handle:
            header_size = struct.unpack("<Q", source_handle.read(8))[0]
            data_start = 8 + header_size
            out_handle.write(struct.pack("<Q", len(payload)))
            out_handle.write(payload)
            body_start = out_handle.tell()
            for kind, item in plan:
                if kind == "write":
                    out_handle.write(memoryview(item.numpy()).cast("B"))
                else:
                    start, size = item
                    copy_range(source_handle, out_handle, data_start + start, size)
            written = out_handle.tell() - body_start
            if written != offset:
                raise RuntimeError(f"length mismatch: wrote {written}, planned {offset}")
            out_handle.flush()
            os.fsync(out_handle.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()

    elapsed = time.perf_counter() - started
    manifest = {
        "source": str(source), "source_size": source.stat().st_size,
        # The ticket's actual complaint: given only a produced checkpoint there was no way to
        # tell whether the analysis it was built from described that checkpoint. These three
        # answer it -- `analysis_source_identity_sha256` differs from `source_identity_sha256` only
        # under --foreign-analysis, and then it names which measurement was borrowed.
        "source_identity_sha256": source_digest,
        "analysis_source_identity_sha256": analysis.get("source_identity_sha256"),
        "measure_dtype": str(MEASURE_DTYPE),
        "output": str(output), "output_size": output.stat().st_size,
        "architecture": profile,
        "quantization": "mixed convrot_w4a4 / asym_w4a8_int8",
        "selection": {
            "promote_error": args.promote_error,
            "keep_bf16_error": args.keep_bf16_error,
            "budget": args.budget,
            "uncalibrated": args.uncalibrated,
            "sigma_weight": args.sigma_weight,
        },
        "layer_counts": counts,
        "calibration": analysis.get("calibration", {}),
        "group_size": args.group_size, "convrot_groupsize": args.convrot_groupsize,
        # {op: module}, the same shape this sidecar has always carried -- not the whole probe
        # payload, which would put a `native_ready` flag next to it that reads as a second claim.
        "backend": resolved,
        "preserved_tensors": len(header) - len(quant_names),
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "conversion_seconds": round(elapsed, 3),
        "layers": {stem: fmt for stem, fmt in sorted(decision.items())},
    }
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    source_size = source.stat().st_size
    out_size = output.stat().st_size
    print(f"\nWrote {output} ({human_size(out_size)}) in {elapsed:.1f} s")
    print(f"{human_size(source_size)} -> {human_size(out_size)}   "
          f"{source_size / out_size:.2f}x lighter")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
