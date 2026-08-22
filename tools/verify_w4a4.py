"""Verify a ComfyUI quantized checkpoint against its high-precision source.

Three output formats produced by this bench are understood:

    convrot_w4a4     quant_w4a4.py, quant_w4a4_smooth.py, quant_mixed.py
    asym_w4a8_int8   quant_w4a8.py, quant_mixed.py
    int8_tensorwise  quant_int8.py  (both --convrot and --no-convrot)

The file is still named verify_w4a4.py because scripts, notes and the two handoff documents refer
to it by that name; only `validate_structure`'s per-layer expectations were ever W4A4-specific.

Why it grew past one format: `zimage-v2-mixed.safetensors` on this bench is 115 `convrot_w4a4`
layers plus 55 `asym_w4a8_int8` layers, and the old code rejected the 55 by format and returned 1
before `validate_preserved_bytes` ran. The byte-identical preserved-tensor comparison -- the one
check here that can prove the converter did not corrupt anything it was supposed to copy -- was
therefore unavailable on a real, on-disk, produced-by-our-own-tool checkpoint. Measured
2026-08-22, and written up in `.scratch/varredura-2026-08-22/issues/06-verify-cobre-1-de-5.md`.

Provenance of the format table below: READ, not executed. Every dtype/shape rule was derived from
the writing code in quant_w4a4.py, quant_w4a8.py, quant_int8.py and quant_mixed.py and then
checked against the headers of real checkpoints on this bench (see `Format` docstrings).

**All three smoke halves were EXECUTED on 2026-08-22**, on the RTX 3090 with the GPU lock held,
against real checkpoints rather than fixtures. Every op resolved to `comfy_kitchen.backends.cuda`
and every output was finite:

    int8_tensorwise   0.0127   ltx-2.5-22b-distilled-transformer-bf16_int8_convrot  (1440 layers)
    asym_w4a8_int8    0.0701   zimage-v2-mixed                     (55 of its 170 layers)
    convrot_w4a4      0.2333   zimage-v2-mixed                     (115 of its 170 layers)

Read that ladder as a liveness bound per format on ONE layer of random input at M=2, not as a
quality ranking -- but it is monotone in the direction the formats predict, which is a cheap
independent cross-check on `quant_mixed.py`'s per-layer assignment.

Before that run this file understood exactly one of the three formats, and `zimage-v2-mixed` --
produced by this bench's own converter -- had no verifier at all: it was rejected by format
before `validate_preserved_bytes` could run.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kernel-smoke", action="store_true")
    parser.add_argument("--structural-only", action="store_true",
                        help="skip backend resolution (which needs CUDA) and the smoke. The "
                             "header/shape/dtype checks and the byte-identical preserved-tensor "
                             "comparison run unchanged, so this is the half that can be run on a "
                             "machine whose GPU is busy or held by another session. The caveat "
                             "block says so on the way out.")
    parser.add_argument("--smoke-rmse-ceiling", type=float, default=0.9,
                        help="relative RMSE above which --kernel-smoke fails. A liveness bound, "
                             "not a quality one: ~0.25 is normal for Gemma W4A4 on random input. "
                             "Below 1.0 on purpose -- an all-zero output scores exactly 1.0 "
                             "(error equals the reference), so a ceiling of 1.0 would let the "
                             "most obvious failure through. Measured 2026-08-22 on the 3090: a "
                             "convrot groupsize mismatch (made at 64, run at 256) scores 1.023 "
                             # %% because argparse runs help strings through %-formatting.
                             "against this 0.9 -- it fails, but with only 14%% of headroom, so "
                             "raising this above ~1.0 would convert that loud failure into a "
                             "silent PASS for a comparison that never happened.")
    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# Format adapters. Everything else in this file is format-agnostic.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorSpec:
    """What one tensor of a quantized layer must look like in the output header.

    `shapes` is computed from the *output's own* packed weight shape, never from the source, so
    every auxiliary-tensor check keeps working when --source is absent or the source drive is not
    mounted -- the Gemma source lives on D:/ComfyUI-Models. An earlier version of this file had
    the scale-shape check inside the `if source_header is not None` branch, where verifying
    without a source made it silently vanish rather than degrade.
    """

    dtypes: frozenset[str]
    # None where the format legitimately allows more than one rank -- int8_tensorwise's
    # weight_scale is [N, 1] on both LTX checkpoints on this bench but eager.quantize_int8_rowwise
    # is free to return [N]; `shapes` below carries the real constraint in that case.
    ndim: int | None = None
    required: bool = True
    shapes: Callable[[list[int], dict], list[list[int]]] | None = None


@dataclass(frozen=True)
class Format:
    name: str
    # [rows, cols] of the source weight -> the packed shape the output must carry.
    packed_shape: Callable[[list[int]], list[int]]
    # layer name + its own metadata config -> {tensor name: TensorSpec}
    tensors: Callable[[str, dict], dict[str, TensorSpec]]
    # comfy-kitchen op that executes this format, used both by the backend probe and the smoke.
    linear_op: str
    # source for a subprocess that resolves `linear_op` against ck.registry.
    probe_snippet: str
    # (layer, config, load, x, options) -> kwargs for `linear_op`
    smoke_kwargs: Callable[..., dict]

    def expected_tensors(self, layer: str, config: dict) -> dict[str, TensorSpec]:
        return self.tensors(layer, config)


def _require(config: dict, layer: str, key: str) -> int:
    """Read a kernel parameter out of the layer's own metadata, or refuse to guess.

    This is the fix for the bug that motivated the ticket: `kernel_smoke` used to hardcode
    `convrot_groupsize=256, quant_group_size=64` and never look at the metadata, so a file written
    by `quant_w4a4.py --convrot-groupsize 64` -- a supported, documented flag -- was smoke-tested
    at 256. Measured 2026-08-22 on the 3090: made-at-64/run-at-256 scores 1.0230 and made-at-256/
    run-at-64 scores 1.0236, against matched runs at 0.2136-0.2137. So the old behaviour produced
    a false ALARM on a correct file rather than a false PASS -- but only by 14% of the 0.9 ceiling.
    """
    value = config.get(key)
    if value is None:
        raise SystemExit(
            f"{layer}: the layer's quantization metadata has no {key!r}, and this tool will not "
            f"guess it -- guessing is what it used to do, and a mismatched groupsize scores 1.023 "
            f"against a 0.9 ceiling, i.e. 14% away from passing silently. Metadata present: "
            f"{sorted(config)}."
        )
    return int(value)


def _rows_only(packed: list[int], config: dict) -> list[list[int]]:
    """A per-output-channel scale: one entry per row of the packed weight."""
    return [[packed[0]]]


# ---- convrot_w4a4 ---------------------------------------------------------------------------
# Checked against zimage-v2-w4a4.safetensors (170/170 layers) and the convrot half of
# zimage-v2-mixed.safetensors (115 layers), both headers read 2026-08-22:
#   <l>.weight        I8  [3840, 1920]     (source [3840, 3840])
#   <l>.weight_scale  F32 [3840]
def _w4a4_tensors(layer: str, config: dict) -> dict[str, TensorSpec]:
    return {
        f"{layer}.weight": TensorSpec(frozenset({"I8"}), 2),
        f"{layer}.weight_scale": TensorSpec(frozenset({"F32", "F16", "BF16"}), 1,
                                            shapes=_rows_only),
    }


# The loader's own value, and the reason it is a literal here rather than a metadata read.
#
# `ComfyUI/comfy/ops.py:1195-1203` builds the convrot_w4a4 kwargs like this:
#
#     "convrot_groupsize": int(layer_conf.get("convrot_groupsize",
#                              params_conf.get("convrot_groupsize", 256))),
#     "quant_group_size": 64,
#     "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
#
# Two of the three are read from the file. **`quant_group_size` is a literal with no metadata
# lookup at all**, so it is not a property of a checkpoint, cannot vary at execution time, and is
# written by no converter -- confirmed 2026-08-22 by reading the headers of the produced files,
# whose per-layer configs are exactly `{format, convrot_groupsize}` (plus `group_size` on w4a8
# layers).
#
# So requiring it from the metadata makes this verifier STRICTER THAN THE RUNTIME IT VERIFIES,
# and the cost is not theoretical: it refuses --kernel-smoke on 100% of existing convrot_w4a4
# checkpoints, including the invocation printed in CLAUDE.md. The point of this smoke is to
# reproduce what the loader does; where the loader hardcodes, reproducing it means hardcoding the
# same value and citing where it came from.
LOADER_QUANT_GROUP_SIZE = 64


def _w4a4_smoke(layer, config, load, x, options) -> dict:
    convrot_groupsize = _require(config, layer, "convrot_groupsize")
    return {
        "x": x,
        "qweight": load(".weight"),
        "wscales": load(".weight_scale"),
        "bias": None,
        # From the file: this is the one the ticket measured, and getting it wrong scores 1.023
        # against a 0.9 ceiling.
        "convrot_groupsize": convrot_groupsize,
        # From the loader, per the comment above. If ops.py ever starts reading this from the
        # metadata, this line is what has to change, and the grep that finds it is this constant.
        "quant_group_size": LOADER_QUANT_GROUP_SIZE,
        "linear_dtype": config.get("linear_dtype", "int4"),
    }


# ---- asym_w4a8_int8 -------------------------------------------------------------------------
# Checked against capybara_v0.1_w4a8.safetensors (432 layers) and the w4a8 half of
# zimage-v2-mixed.safetensors (55 layers), both headers read 2026-08-22:
#   <l>.weight            I8  [2048, 1024]  (source [2048, 2048])
#   <l>.weight_s_rel      U8  [2048, 128]   fp8_e4m3fn stored as raw U8, K/group_size wide
#   <l>.weight_s_channel  F32 [2048]
#   <l>.weight_codebook   F32 [16]          absent under quant_w4a8.py --no-codebook
def _w4a8_tensors(layer: str, config: dict) -> dict[str, TensorSpec]:
    group_size = config.get("group_size")
    if not group_size:
        # FAIL CLOSED. This returned `[]` from s_rel_shapes when `group_size` was absent, under a
        # comment saying the missing key was "reported separately". It was not reported anywhere:
        # `group_size` was required only by `_w4a8_smoke`, i.e. only on the GPU path -- which is
        # the path this tool was just changed to make optional. Measured 2026-08-22: a config of
        # `{format: asym_w4a8_int8, convrot_groupsize: 256}` with a `weight_s_rel` of shape [8, 3]
        # against a packed weight of [8, 128] -- structurally nonsense -- returned zero errors.
        #
        # A verifier that returns clean because it could not work out what to check is worse than
        # one that refuses, and this repo's whole standard is that a column of PASS lines must not
        # be able to mean "I did not look".
        raise SystemExit(
            f"{layer}: asym_w4a8_int8 metadata has no 'group_size', so the expected shape of "
            f"'{layer}.weight_s_rel' cannot be computed and this tool will not skip the check "
            f"instead. Metadata present: {sorted(config)}. Every file quant_w4a8.py and "
            f"quant_mixed.py have written records it; a file without it was not written by them.")

    def s_rel_shapes(packed: list[int], cfg: dict) -> list[list[int]]:
        return [[packed[0], (packed[1] * 2) // int(group_size)]]

    return {
        f"{layer}.weight": TensorSpec(frozenset({"I8"}), 2),
        # U8 because both converters `view(torch.uint8)` the fp8 scale before writing
        # (quant_w4a8.py header_dtype(), quant_mixed.py's write loop). F8_E4M3 is accepted so a
        # future writer that stores it natively is not reported as corrupt.
        f"{layer}.weight_s_rel": TensorSpec(frozenset({"U8", "F8_E4M3"}), 2, shapes=s_rel_shapes),
        f"{layer}.weight_s_channel": TensorSpec(frozenset({"F32", "F16", "BF16"}), 1,
                                                shapes=_rows_only),
        f"{layer}.weight_codebook": TensorSpec(frozenset({"F32", "F16", "BF16"}), 1,
                                               required=False,
                                               shapes=lambda packed, cfg: [[16]]),
    }


def _w4a8_smoke(layer, config, load, x, options) -> dict:
    group_size = _require(config, layer, "group_size")
    convrot_groupsize = _require(config, layer, "convrot_groupsize")
    return {
        "x": x,
        "qdata": load(".weight"),
        "s_rel": load(".weight_s_rel", view="float8_e4m3fn"),
        "s_channel": load(".weight_s_channel"),
        "codebook": load(".weight_codebook", optional=True),
        # quant_w4a8.py and quant_mixed.py both refuse to write asymmetric weights, because
        # comfy/ops.py drops the correction tensor. There is nothing on disk to load here.
        "correction": None,
        "bias": None,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "out_dtype": x.dtype,
    }


# ---- int8_tensorwise ------------------------------------------------------------------------
# Checked against ltx-2.5-22b-distilled-transformer-bf16_int8{,_convrot}.safetensors (1440 layers
# each), headers read 2026-08-22:
#   <l>.weight        I8  [2048, 2048]   NOT halved -- int8, not packed int4
#   <l>.weight_scale  F32 [2048, 1]      2D, unlike convrot_w4a4's 1D scale
# `convrot` is absent from the metadata on the --no-convrot file and true on the other; comfy's
# own loader reads it as `layer_conf.get("convrot", False)` (ops.py:1279), so absence *means*
# false here and is not a guess.
def _int8_tensors(layer: str, config: dict) -> dict[str, TensorSpec]:
    return {
        f"{layer}.weight": TensorSpec(frozenset({"I8"}), 2),
        f"{layer}.weight_scale": TensorSpec(frozenset({"F32", "F16", "BF16"}),
                                            shapes=lambda packed, cfg: [[packed[0]],
                                                                        [packed[0], 1]]),
    }


def _int8_smoke(layer, config, load, x, options) -> dict:
    convrot = bool(config.get("convrot", False))
    # `input_act` is not optional in the kwargs dict even though it is optional in the signature:
    # ck.int8_linear() at comfy_kitchen/__init__.py:838 always puts it in the dict it hands to
    # registry.get_implementation, and the registry matches on the kwargs it is given. Omitting it
    # here would resolve against a different kwarg set than the one the call actually runs under,
    # which is the "A/B only counts if both arms took the same dispatch" failure this bench has
    # hit four times in a day.
    return {
        "x": x,
        "weight": load(".weight"),
        "weight_scale": load(".weight_scale"),
        "bias": None,
        "out_dtype": x.dtype,
        "convrot": convrot,
        # 256 on the non-convrot path is inert, not a guess: int8_linear only reads
        # convrot_groupsize inside `if convrot` (backends/cuda/__init__.py:1873). When the rotation
        # IS on, the value changes the result and must come from the file.
        "convrot_groupsize": _require(config, layer, "convrot_groupsize") if convrot else 256,
        "input_act": None,
    }


# The convrot_w4a4 snippet is byte-for-byte the probe that has been in this file since it was
# written and whose output the ticket records ("comfy_kitchen.backends.cuda"). The other two are
# modelled on quant_w4a8.py:97 and quant_int8.py:102 and HAVE NOT BEEN EXECUTED -- both were
# written on a session forbidden the GPU. `tools/probe_backend_resolution.py` would settle them.
_PROBE_W4A4 = """
q = torch.empty((64, 32), device='cuda', dtype=torch.int8)
s = torch.empty((64,), device='cuda', dtype=torch.float32)
x = torch.empty((2, 64), device='cuda', dtype=torch.float16)
resolved['convrot_w4a4_linear'] = ck.registry.get_implementation('convrot_w4a4_linear', kwargs={'x': x, 'qweight': q, 'wscales': s, 'bias': None, 'convrot_groupsize': 64, 'quant_group_size': 64, 'linear_dtype': 'int4'}).__module__
"""

_PROBE_W4A8 = """
w = torch.empty((64, 256), device='cuda', dtype=torch.bfloat16)
p8 = ck.quantize_w4a8_int8_weight(w, group_size=16, convrot_groupsize=256, symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True, codebook_tensor=None, stochastic_rounding=0)
x8 = torch.empty((2, 256), device='cuda', dtype=torch.bfloat16)
resolved['w4a8_int8_linear'] = ck.registry.get_implementation('w4a8_int8_linear', kwargs={'x': x8, 'qdata': p8[0], 's_rel': p8[1], 's_channel': p8[2], 'codebook': p8[4], 'correction': p8[3], 'bias': None, 'group_size': 16, 'convrot_groupsize': 256, 'out_dtype': torch.bfloat16}).__module__
"""

_PROBE_INT8 = """
wi = torch.empty((64, 256), device='cuda', dtype=torch.int8)
si = torch.empty((64, 1), device='cuda', dtype=torch.float32)
xi = torch.empty((2, 256), device='cuda', dtype=torch.bfloat16)
resolved['int8_linear'] = ck.registry.get_implementation('int8_linear', kwargs={'x': xi, 'weight': wi, 'weight_scale': si, 'bias': None, 'out_dtype': torch.bfloat16, 'convrot': False, 'convrot_groupsize': 256, 'input_act': None}).__module__
"""


FORMATS: dict[str, Format] = {
    "convrot_w4a4": Format(
        name="convrot_w4a4",
        packed_shape=lambda shape: [shape[0], shape[1] // 2],
        tensors=_w4a4_tensors,
        linear_op="convrot_w4a4_linear",
        probe_snippet=_PROBE_W4A4,
        smoke_kwargs=_w4a4_smoke,
    ),
    "asym_w4a8_int8": Format(
        name="asym_w4a8_int8",
        packed_shape=lambda shape: [shape[0], shape[1] // 2],
        tensors=_w4a8_tensors,
        linear_op="w4a8_int8_linear",
        probe_snippet=_PROBE_W4A8,
        smoke_kwargs=_w4a8_smoke,
    ),
    "int8_tensorwise": Format(
        name="int8_tensorwise",
        packed_shape=lambda shape: list(shape),
        tensors=_int8_tensors,
        linear_op="int8_linear",
        probe_snippet=_PROBE_INT8,
        smoke_kwargs=_int8_smoke,
    ),
}


# --------------------------------------------------------------------------------------------
# Format-agnostic machinery
# --------------------------------------------------------------------------------------------

def read_header(path: Path) -> tuple[dict, dict[str, str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(header_size))
    metadata = dict(raw.pop("__metadata__", {}) or {})
    return raw, metadata


def load_tensor_cuda(path: Path, info: dict, view: str | None = None):
    import torch

    dtypes = {"I8": torch.int8, "U8": torch.uint8, "I16": torch.int16,
              "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
              "F8_E4M3": torch.uint8, "F8_E5M2": torch.uint8}
    start, end = info["data_offsets"]
    size = end - start
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_size + start)
        raw = bytearray(size)
        buffer = memoryview(raw)
        position = 0
        while position < size:
            count = handle.readinto(buffer[position:])
            if not count:
                raise EOFError(f"Unexpected end of {path} with {size - position} bytes remaining")
            position += count
    tensor = torch.frombuffer(raw, dtype=dtypes[info["dtype"]]).reshape(info["shape"]).cuda()
    if view is not None:
        # weight_s_rel is fp8_e4m3fn written as raw U8 (quant_w4a8.py header_dtype()); the kernel
        # wants it back as fp8 or it validates the wrong element count.
        tensor = tensor.view(getattr(torch, view))
    return tensor


def normal_comfy_backend(portable_root: Path, formats: list[Format]) -> dict:
    """Ask a fresh interpreter which implementation ck.registry picks for each format present.

    Format-aware on purpose. This used to resolve `convrot_w4a4_linear` unconditionally, which was
    fine while the tool only understood one format and became actively misleading the moment it
    understood three: printing a CUDA implementation of the W4A4 op says nothing about whether
    `w4a8_int8_linear` or `int8_linear` resolve natively, and quant_int8.py:108 already carries
    the same warning in its own words.
    """
    body = "".join(fmt.probe_snippet for fmt in formats)
    code = (
        "import json, sys, torch, comfy_kitchen as ck\n"
        f"sys.path.insert(0, {str(portable_root / 'ComfyUI')!r})\n"
        "import comfy.quant_ops\n"
        "resolved = {}\n"
        f"{body}"
        "print(json.dumps({'resolved': resolved, 'backends': ck.list_backends()}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=portable_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["warning"] = result.stderr.strip()
    payload["native_ready"] = all(".backends.cuda" in module
                                  for module in payload["resolved"].values())
    return payload


def resolve_source(model: Path, explicit_source: Path | None) -> Path | None:
    if explicit_source:
        return explicit_source.resolve()
    sidecar = model.with_suffix(".quant.json")
    if sidecar.is_file():
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        source = Path(manifest.get("source", ""))
        return source if source.is_file() else None
    return None


def validate_structure(model_header: dict, metadata: dict, source_header: dict | None) -> list[str]:
    errors = []
    raw_quant = metadata.get("_quantization_metadata")
    if not raw_quant:
        return ["missing _quantization_metadata"]
    try:
        quant = json.loads(raw_quant)
    except json.JSONDecodeError as error:
        return [f"invalid _quantization_metadata JSON: {error}"]
    layers = quant.get("layers", {})
    if not layers:
        return ["quantization metadata contains no layers"]
    expected_extra = set()
    for layer_name, config in layers.items():
        fmt = FORMATS.get(config.get("format"))
        if fmt is None:
            errors.append(f"{layer_name}: unexpected format {config.get('format')!r} "
                          f"(known: {', '.join(sorted(FORMATS))})")
            continue
        specs = fmt.expected_tensors(layer_name, config)
        expected_extra.update(specs)
        expected_extra.add(f"{layer_name}.comfy_quant")
        weight_name = f"{layer_name}.weight"
        weight = model_header.get(weight_name)
        if weight is None:
            errors.append(f"{layer_name}: missing packed weight")
            continue
        for tensor_name, spec in specs.items():
            info = model_header.get(tensor_name)
            if info is None:
                if spec.required:
                    errors.append(f"{tensor_name}: missing, required by format {fmt.name}")
                continue
            if info.get("dtype") not in spec.dtypes:
                errors.append(f"{tensor_name}: dtype {info.get('dtype')!r} not one of "
                              f"{sorted(spec.dtypes)} for format {fmt.name}")
            if spec.ndim and len(info.get("shape", [])) != spec.ndim:
                errors.append(f"{tensor_name}: expected a {spec.ndim}D tensor, got shape "
                              f"{info.get('shape')}")
            if spec.shapes is not None:
                allowed = spec.shapes(weight["shape"], config)
                if allowed and list(info.get("shape", [])) not in allowed:
                    errors.append(f"{tensor_name}: shape {info.get('shape')} is not one of "
                                  f"{allowed} for a packed weight of {weight['shape']}")
        if source_header is not None:
            source_weight = source_header.get(weight_name)
            if not source_weight:
                errors.append(f"{layer_name}: weight missing from source")
            elif list(weight["shape"]) != fmt.packed_shape(source_weight["shape"]):
                errors.append(f"{layer_name}: packed shape {weight['shape']} does not match "
                              f"source {source_weight['shape']} under format {fmt.name}")
    if source_header is not None:
        quantized_weights = {f"{name}.weight" for name in layers}
        for name, source_info in source_header.items():
            if name in quantized_weights:
                continue
            output_info = model_header.get(name)
            if output_info is None or any(
                output_info.get(field) != source_info.get(field)
                for field in ("dtype", "shape")
            ):
                errors.append(f"{name}: preserved tensor dtype/shape changed")

        # The loop above only walks the *source*, so it can never notice a tensor the output has
        # and the source does not. A stray key smuggled into the output passed PASS/PASS. Walk
        # the other direction too; the only keys legitimately introduced by conversion are the
        # auxiliaries each layer's own format declares, which is why `expected_extra` is built
        # from the adapters above rather than from a fixed union of every suffix anyone writes.
        for name in model_header:
            if name not in source_header and name not in expected_extra:
                errors.append(f"{name}: present in the output but not in the source, and not a "
                              "quantization auxiliary of a selected layer")
    return errors


def data_start(path: Path) -> int:
    with path.open("rb") as handle:
        return 8 + struct.unpack("<Q", handle.read(8))[0]


def compare_ranges(left_handle, left_start: int, right_handle, right_start: int, size: int) -> bool:
    left_handle.seek(left_start)
    right_handle.seek(right_start)
    remaining = size
    while remaining:
        chunk_size = min(16 * 1024**2, remaining)
        if left_handle.read(chunk_size) != right_handle.read(chunk_size):
            return False
        remaining -= chunk_size
    return True


def validate_preserved_bytes(
    model: Path,
    source: Path,
    model_header: dict,
    source_header: dict,
    layers: dict,
) -> list[str]:
    errors = []
    quantized_weights = {f"{name}.weight" for name in layers}
    model_base = data_start(model)
    source_base = data_start(source)
    with model.open("rb") as model_handle, source.open("rb") as source_handle:
        for name, source_info in source_header.items():
            if name in quantized_weights:
                continue
            output_info = model_header[name]
            source_start, source_end = source_info["data_offsets"]
            output_start, output_end = output_info["data_offsets"]
            size = source_end - source_start
            if output_end - output_start != size or not compare_ranges(
                model_handle,
                model_base + output_start,
                source_handle,
                source_base + source_start,
                size,
            ):
                errors.append(f"{name}: preserved tensor bytes changed")
    return errors


def kernel_smoke(
    model: Path,
    source: Path,
    layer_name: str,
    config: dict,
    model_header: dict,
    source_header: dict,
    options: dict,
) -> dict:
    import torch
    import comfy_kitchen as ck

    fmt = FORMATS[config["format"]]

    def load(suffix: str, view: str | None = None, optional: bool = False):
        info = model_header.get(f"{layer_name}{suffix}")
        if info is None:
            if optional:
                return None
            raise SystemExit(f"{layer_name}{suffix}: missing, needed by the {fmt.name} smoke")
        return load_tensor_cuda(model, info, view=view)

    weight = load_tensor_cuda(source, source_header[f"{layer_name}.weight"])
    x = torch.randn((2, weight.shape[1]), device="cuda", dtype=weight.dtype)
    kwargs = fmt.smoke_kwargs(layer_name, config, load, x, options)
    implementation = ck.registry.get_implementation(fmt.linear_op, kwargs=kwargs)
    output = getattr(ck, fmt.linear_op)(**kwargs)
    reference = torch.nn.functional.linear(x, weight)
    error = output.float() - reference.float()
    return {
        "layer": layer_name,
        "format": fmt.name,
        "op": fmt.linear_op,
        "backend": f"{implementation.__module__}.{implementation.__name__}",
        "output_dtype": str(output.dtype),
        "relative_rmse": error.square().mean().sqrt().div(reference.float().square().mean().sqrt()).item(),
        "max_abs_error": error.abs().max().item(),
    }


@dataclass
class Coverage:
    """What this run did NOT check, accumulated as the run goes and printed on every exit path.

    It prints on failure too. A column of PASS lines followed by a JSON blob reads as "verified"
    to whoever pastes it into the next session, and the numbers out of this bench do get pasted --
    see CLAUDE.md's table of five claims that travelled as measurements when they were traces.
    """

    lines: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        self.lines.append(line)

    def render(self) -> str:
        lines = self.lines or ["everything: the run stopped before it checked anything."]
        body = "\n".join(f"  - {line}" for line in lines)
        return f"\nNOT COVERED BY THIS RUN:\n{body}"


def run(args: argparse.Namespace, coverage: Coverage) -> int:
    model = args.model.resolve()
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    portable_root = Path(__file__).resolve().parent.parent
    source = resolve_source(model, args.source)
    model_header, metadata = read_header(model)
    source_header = read_header(source)[0] if source else None

    coverage.note("numerical quality: nothing here compares generated images or latents against "
                  "the source model. A checkpoint can pass every check below and still be worse.")

    errors = validate_structure(model_header, metadata, source_header)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        coverage.note("everything after the structural stage was skipped because it failed.")
        return 1

    quant = json.loads(metadata["_quantization_metadata"])
    layers = quant["layers"]
    counts: dict[str, int] = {}
    for config in layers.values():
        counts[config["format"]] = counts.get(config["format"], 0) + 1
    present = [FORMATS[name] for name in sorted(counts)]

    if source:
        errors = validate_preserved_bytes(model, source, model_header, source_header, layers)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
    else:
        coverage.note("the byte-identical preserved-tensor comparison DID NOT RUN -- no source "
                      "was given and no sidecar pointed at a readable one. That is the check "
                      "that proves the converter copied everything it did not quantize.")

    summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    print(f"Structural verification: PASS ({summary})")
    print(f"Source comparison: {'PASS' if source else 'SKIPPED (no source supplied/found)'}")

    if args.structural_only:
        if args.kernel_smoke:
            # Silently winning would be the worse outcome: --kernel-smoke is the flag people pass
            # when they want proof a kernel ran, and it would have produced a clean exit 0 with
            # no kernel anywhere near it.
            print("ERROR: --structural-only and --kernel-smoke contradict each other")
            return 1
        coverage.note("--structural-only: normal-ComfyUI backend resolution did not run, so "
                      "nothing here shows this checkpoint would execute natively rather than "
                      "dequantized.")
        coverage.note("no kernel was executed and no tensor data was read; only the header, the "
                      "metadata and the preserved byte ranges were touched.")
        return 0

    backend = normal_comfy_backend(portable_root, present)
    for op, module in sorted(backend["resolved"].items()):
        print(f"Normal ComfyUI backend: {op} -> {module}")
    if not backend["native_ready"]:
        print("ERROR: normal ComfyUI is not selecting the CUDA backend for every format present")
        return 1
    coverage.note("backend resolution is a registry lookup on dummy tensors in a subprocess. It "
                  "shows which implementation would be chosen, NOT that this file's own shapes "
                  "resolve the same way, and not that any kernel ran.")
    # There used to be a note here saying the w4a8 and int8 probe snippets had never been
    # executed. All three have now been run against real checkpoints (see the module docstring),
    # so the note is gone rather than left standing -- a caveat that overstates teaches the reader
    # to skim the block, which costs exactly as much as one that understates.

    if not args.kernel_smoke:
        coverage.note("--kernel-smoke was not passed, so no kernel ran and no output was compared "
                      "against F.linear on the source weights.")
        return 0

    if not source:
        print("ERROR: --kernel-smoke requires --source or a valid sidecar source")
        return 1

    options: dict = {}
    # One layer per format present, not one layer overall: on the mixed checkpoint a single
    # sample would have exercised whichever format happens to sort first and said nothing at all
    # about the other 55 layers.
    smoked = []
    smoke_errors = []
    for fmt_name in sorted(counts):
        layer_name = next(name for name, config in layers.items()
                          if config["format"] == fmt_name)
        result = kernel_smoke(model, source, layer_name, layers[layer_name],
                              model_header, source_header, options)
        smoked.append(result)
        print(json.dumps(result, indent=2))
        # The smoke used to print its numbers and unconditionally `return 0`: nothing compared
        # relative_rmse or max_abs_error against anything, so a NaN result, or a run that
        # resolved to the eager backend, still exited 0 under a "PASS" heading. These three
        # checks are the minimum that makes the flag able to fail.
        impl = str(result.get("backend", ""))
        if "comfy_kitchen.backends.cuda" not in impl:
            smoke_errors.append(f"{layer_name} executed through {impl or '?'}, not the CUDA backend")
        rmse = result.get("relative_rmse")
        if rmse is None or rmse != rmse:
            smoke_errors.append(f"{layer_name}: relative_rmse came back {rmse!r}")
        elif rmse > args.smoke_rmse_ceiling:
            # Deliberately loose. This is a liveness signal, not a quality metric -- ~0.25 is
            # normal for Gemma W4A4 on random input, and 0.2402 was measured on the real
            # zimage-v2-w4a4 file. What it catches is a kernel returning zeros, garbage, or a
            # dequantized path, all of which land far outside.
            smoke_errors.append(
                f"{layer_name}: relative_rmse {rmse:.4f} exceeds the liveness ceiling "
                f"{args.smoke_rmse_ceiling:.2f}; this is not a quality threshold, so exceeding "
                "it means the kernel returned something structurally wrong")

    coverage.note(f"the smoke ran {len(smoked)} of {len(layers)} layers (one per format) on "
                  "random input, at M=2. Its RMSE is a liveness bound, not a quality metric, and "
                  "it says nothing about the other layers or about real activations.")
    if smoke_errors:
        for error in smoke_errors:
            print(f"ERROR: {error}")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    coverage = Coverage()
    try:
        return run(args, coverage)
    finally:
        print(coverage.render())


if __name__ == "__main__":
    raise SystemExit(main())
