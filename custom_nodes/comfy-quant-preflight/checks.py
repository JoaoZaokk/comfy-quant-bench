"""What can be known about a quantized checkpoint before a single tensor is loaded.

Every check here reads only the safetensors header (a JSON blob at the front of the file, tens of
KB) plus process state. Nothing allocates, nothing touches the GPU, nothing loads a model. The
whole point is to fail in the UI in milliseconds instead of failing in twenty minutes, or worse,
not failing at all.

The checks are not invented. Each one comes from a failure that actually happened on this machine
or from an audit finding with a named mechanism, and each carries where it came from -- because a
check whose provenance is forgotten is a check nobody dares delete when it turns out to be wrong.

Severity has a rule, and it is narrow on purpose:

  ERROR  two statements are both being obeyed, in different places, and they contradict.
         The file wins for the weights and the widget wins for the compute dtype, so the run
         produces a mongrel nobody asked for.
  WARN   one statement is simply ignored. Nothing is corrupted; the user is just wrong about
         what is happening.
  SILENT the mismatch is the feature. A BF16 file with weight_dtype=fp8 is the widget doing its
         job.

If this blocked every mismatch it would block the widget's intended use, and it would be switched
off within a week.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

ERROR = "error"
WARN = "warn"

# Formats whose weights carry their own per-layer precision. For these the file has already
# decided, so a dtype widget is at best redundant.
WEIGHT_ONLY_FORMATS = {"convrot_w4a4", "asym_w4a8_int8", "int8_tensorwise"}


def read_header(path: Path) -> tuple[dict, dict]:
    """(tensors, metadata) from the safetensors header, bounds-checked.

    The bounds check is not decoration: without it a truncated or non-safetensors file makes this
    allocate whatever the first eight bytes happened to say.
    """
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("shorter than a safetensors header prefix")
        size = struct.unpack("<Q", prefix)[0]
        limit = min(path.stat().st_size - 8, 1024 ** 3)
        if size <= 2 or size > limit:
            raise ValueError(f"header size {size} outside 2..{limit}")
        blob = handle.read(size)
        if len(blob) != size:
            raise ValueError("header truncated")
        header = json.loads(blob)
    metadata = header.pop("__metadata__", {}) or {}
    return header, metadata


def quant_layers(tensors: dict, metadata: dict) -> dict[str, dict]:
    """Per-layer quantization config, from either of the two places ComfyUI accepts it.

    `_quantization_metadata` in `__metadata__`, and inline `<layer>.comfy_quant` tensors. Both are
    real: Comfy-Org and Lightricks ship the inline form with no `__metadata__` at all, which is
    why a checker that reads only the first one sees an unquantized file.
    """
    layers: dict[str, dict] = {}
    raw = metadata.get("_quantization_metadata")
    if raw:
        try:
            layers.update(json.loads(raw).get("layers", {}) or {})
        except (TypeError, ValueError):
            pass
    for name in tensors:
        if name.endswith(".comfy_quant"):
            layers.setdefault(name[: -len(".comfy_quant")], {})
    return layers


def formats_present(layers: dict[str, dict]) -> set[str]:
    return {str(conf.get("format")) for conf in layers.values() if conf.get("format")}


# --------------------------------------------------------------------------------------------
# File-level checks. Each returns (severity, message) or None.
# --------------------------------------------------------------------------------------------

def check_diffusers_named_quantized(tensors: dict, metadata: dict, layers: dict) -> tuple | None:
    """A quantized checkpoint in diffusers naming loads, and loads WRONG, with no error.

    Provenance: cost this project several hours on 2026-08-18. ComfyUI fuses
    `attention.to_{q,k,v}` into `attention.qkv` at load (`model_detection.py` →
    `convert_diffusers_mmdit`), and the map it uses only names `.weight`/`.bias`. Everything
    beside the weight -- `weight_scale`, `weight_s_rel`, `weight_s_channel`, `comfy_quant` --
    falls through the z-image branch's identity passthrough and keeps its old name, while the
    module that needs it is now called `qkv`. The layer loads with no scale.

    Detection is exact, not heuristic: diffusers naming and quantization markers cannot both be
    correct in the same file.
    """
    if not layers:
        return None
    diffusers = [n for n in tensors if n.endswith("attention.to_q.weight")]
    if not diffusers:
        return None
    return (ERROR,
            f"quantized checkpoint in diffusers naming ({len(diffusers)} 'attention.to_q' keys). "
            "ComfyUI fuses to_q/to_k/to_v into 'attention.qkv' at load and does not rename the "
            "quantization scales with them, so those layers load without scales and produce "
            "wrong output silently. Convert the names first (tools/to_native.py).")


def check_weight_correction_dropped(tensors: dict, metadata: dict, layers: dict) -> tuple | None:
    """`asym_w4a8_int8` can carry an asymmetric correction term that ComfyUI never reads.

    Provenance: audit finding on `ops.py:1203`, confirmed here by grep -- `weight_correction`
    appears nowhere in `comfy/ops.py`. This project's own converters refuse to emit it
    (`quant_w4a8.py` and `quant_mixed.py` both raise on a non-None correction), so a file that
    has one came from somewhere else, and its weights will decode with a systematic bias.
    """
    corrections = [n for n in tensors if n.endswith(".weight_correction")]
    if not corrections:
        return None
    return (ERROR,
            f"{len(corrections)} layer(s) carry a '.weight_correction' tensor. ComfyUI's loader "
            "never reads it, so the asymmetric correction is dropped and those weights decode "
            "with a systematic offset. There is no error at load; the output is just wrong.")


def check_inert_full_precision_flag(tensors: dict, metadata: dict, layers: dict) -> tuple | None:
    """`full_precision_matrix_mult: true` does nothing for the weight-only formats.

    Provenance: audit finding on `ops.py:1373`, marked unverified. The flag is meant to keep a
    sensitive layer in BF16; for convrot_w4a4 / asym_w4a8_int8 / int8_tensorwise the weight stays
    a QuantizedTensor and the kernel runs anyway. A converter that marked adaLN or proj_out as
    "runs in BF16" is then wrong about its own file.

    WARN, not ERROR: nothing is corrupted, the intent is simply not honoured -- and the finding
    has not been confirmed by execution.
    """
    flagged = [name for name, conf in layers.items()
               if conf.get("full_precision_matrix_mult")
               and str(conf.get("format")) in WEIGHT_ONLY_FORMATS]
    if not flagged:
        return None
    return (WARN,
            f"{len(flagged)} layer(s) request full_precision_matrix_mult, but for "
            f"{'/'.join(sorted(WEIGHT_ONLY_FORMATS))} that flag does not keep the layer in BF16 "
            "-- the quantized kernel runs regardless. The file's own intent is not being "
            "honoured. (Audit finding, not yet confirmed by execution.)")


FILE_CHECKS = (
    check_diffusers_named_quantized,
    check_weight_correction_dropped,
    check_inert_full_precision_flag,
)


def check_file(path: Path) -> list[tuple]:
    """Every file-level check. Returns a list of (severity, message)."""
    try:
        tensors, metadata = read_header(path)
    except Exception as exc:
        # A file we cannot parse is not a file we should block on: GGUF, .pt and friends come
        # through here too. Say nothing rather than invent a verdict.
        del exc
        return []
    layers = quant_layers(tensors, metadata)
    found = []
    for check in FILE_CHECKS:
        result = check(tensors, metadata, layers)
        if result:
            found.append(result)
    return found


# --------------------------------------------------------------------------------------------
# Widget-vs-file
# --------------------------------------------------------------------------------------------

def check_dtype_widget(path: Path, weight_dtype: str) -> tuple | None:
    """A dtype widget pointed at a checkpoint that already carries its own per-layer format.

    Provenance: traced through `comfy/sd.py` on 2026-08-19. The loader knows the file is
    quantized -- it tests `model_config.quant_config is not None` twice -- but protects only half
    of the decision:

        if model_config.quant_config is not None:   # 2305
            manual_cast_dtype = unet_manual_cast(None, ...)      # ignores the widget
        else:
            manual_cast_dtype = unet_manual_cast(unet_dtype, ...)

    while a few lines earlier `unet_dtype = dtype` takes the widget unconditionally (2303). So
    the widget IS obeyed, as the compute dtype, on a model whose weights are 4-bit. That is the
    definition of a contradiction used in this file: both statements are honoured, in different
    places, and they disagree.

    A BF16 file with weight_dtype=fp8 is NOT this. There the widget is the whole point.

    Caveat kept deliberately visible: the trace is a reading of the code, not an execution. What
    the mongrel actually does -- crash, garbage, or work by accident -- has not been measured.
    """
    if weight_dtype in (None, "", "default"):
        return None
    try:
        tensors, metadata = read_header(path)
    except Exception:
        return None
    layers = quant_layers(tensors, metadata)
    if not layers:
        return None            # BF16/fp16 source: the widget is doing its job.
    formats = formats_present(layers) or {"unknown"}
    return (ERROR,
            f"weight_dtype={weight_dtype} on a checkpoint that is already quantized as "
            f"{'+'.join(sorted(formats))} ({len(layers)} layers). The file decides the weight "
            "format and the widget still becomes the compute dtype, so the two disagree inside "
            "the same model. Set weight_dtype to 'default'.")


# --------------------------------------------------------------------------------------------
# Graph and process state
# --------------------------------------------------------------------------------------------

NUNCHAKU_LOADER_MARKERS = ("Nunchaku",)


def check_nunchaku_needs_disable_dynamic_vram(class_types: list[str]) -> tuple | None:
    """A Nunchaku SVDQuant loader with ComfyUI's lazy Linear enabled crashes, unhelpfully.

    Provenance: this one is not from the audit, it is from losing an afternoon to it. Documented
    in the project's CLAUDE.md. ComfyUI 0.33 added a Windows-only lazy `Linear` (`comfy/ops.py`,
    `self.weight = None` until `_load_from_state_dict`), enabled by `main.py:289` setting
    `comfy.memory_management.aimdo_enabled = True`. ComfyUI-nunchaku reads
    `orig_attn.qkv.weight.dtype` in `patch_model` before that happens and dies with

        AttributeError: 'NoneType' object has no attribute 'dtype'

    Nothing in that traceback points at the loader, the checkpoint, or the flag, and the same
    node called directly in-process works fine -- which is what makes it expensive to diagnose
    rather than merely annoying. The fix is `--disable-dynamic-vram` on the server command line.

    Both halves are readable here: the graph says whether a Nunchaku loader is present, and
    `aimdo_enabled` is a module-level flag set only by main.py.
    """
    nunchaku = [c for c in class_types if any(m in c for m in NUNCHAKU_LOADER_MARKERS)]
    if not nunchaku:
        return None
    try:
        import comfy.memory_management
        if not getattr(comfy.memory_management, "aimdo_enabled", False):
            return None
    except Exception:
        return None
    return (ERROR,
            f"this workflow uses {', '.join(sorted(set(nunchaku)))} and the server is running "
            "without --disable-dynamic-vram. ComfyUI's lazy Linear leaves weight=None until the "
            "state dict loads, and the Nunchaku loader reads qkv.weight.dtype before that, so "
            "this will fail with \"'NoneType' object has no attribute 'dtype'\" -- a traceback "
            "that points nowhere near the cause. Restart with --disable-dynamic-vram.")


def check_lora_over_quantized(class_types: list[str], quantized_files: int) -> tuple | None:
    """A LoRA over a quantized model was suspected of dequantizing every patched layer. Measured.

    Provenance: audit finding on `ops.py:1377`, which requires `len(self.weight_function) == 0`
    to take the quantized path. The reasoning was that `ModelPatcher` populates `weight_function`,
    so a LoRA would send every patched layer down the dequantize branch and run it in BF16 with
    no log line.

    **Measured 2026-08-19 and it did not reproduce.** `tools/dispatch_census.py` counted every
    dispatch during a real 4-step generation of the mixed Z-Image checkpoint with
    `char_Liria_zimage.safetensors` at strength 1.0, 150 of whose patched keys land on quantized
    layers: 680 calls, all of them through the native kernel, none dequantized -- identical to the
    run without the LoRA. And the LoRA was genuinely in effect: the latent moved (norm 747.06 ->
    728.99). A layer that had fallen back would have vanished from the layout dispatch entirely
    rather than showing up as a dequantized call, so the count is the right instrument for this.

    Kept as a WARN rather than deleted, narrowed to what is still unknown: one model, one LoRA,
    one strength, one resolution. And nothing here measures *quality* -- whether applying a LoRA
    delta to an already-quantized weight costs accuracy is a separate question that was not asked.
    """
    if not quantized_files:
        return None
    loras = [c for c in class_types if "Lora" in c or "LoRA" in c]
    if not loras:
        return None
    return (WARN,
            f"{', '.join(sorted(set(loras)))} is applied to a quantized model. The audit "
            "hypothesis that this silently dequantizes the patched layers was MEASURED on "
            "2026-08-19 and did not reproduce: 680 of 680 dispatches stayed on the native kernel "
            "with the LoRA applied and in effect. What remains unverified is whether the LoRA "
            "delta costs accuracy once the weight is already 4-bit -- that was not measured. "
            "Check your output, not your throughput.")
