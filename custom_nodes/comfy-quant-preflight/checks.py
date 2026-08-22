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
import logging
import struct
from pathlib import Path

logger = logging.getLogger("quant-preflight")

ERROR = "error"
WARN = "warn"

# Distinguishes "the attribute is absent" from "the attribute is falsy". `getattr(x, n, False)`
# collapses those two, and for a flag whose False value means "everything is fine" that collapse
# is a fail-open with a rename as its trigger.
_MISSING = object()


def unresolved(symbol: str, exc: BaseException, consequence: str) -> tuple:
    """The WARN a check emits when it could not run at all.

    Every fail-open in this package had the same shape: an `except` returned None, None meant
    "nothing found", and "nothing found" reached the user as a green light. A check that could not
    execute is not a check that passed, so the difference has to survive into the message the user
    reads -- naming the symbol, because the usual cause is an upstream rename and the person
    reading the log is the person who has to update the table.

    WARN and not ERROR on purpose: an unresolvable symbol is this package being broken, and this
    package breaking must not stop the user's workflow. It must only stop the user believing they
    were checked.
    """
    return (WARN,
            f"{symbol} could not be resolved ({type(exc).__name__}: {exc}), so {consequence} DID "
            "NOT RUN. This is a missing check, not a clean result -- most likely an upstream "
            "rename that this package has not caught up with.")

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
        except (TypeError, ValueError) as exc:
            # Swallowing this used to mean a file with malformed quantization metadata came back
            # as "no layers", which every caller below reads as "not quantized" -- the dtype-widget
            # ERROR and the LoRA WARN both go silent on the one file most likely to need them.
            # The parse still degrades to the inline scan; it just no longer does so quietly.
            logger.warning(
                "quant-preflight: _quantization_metadata did not parse (%s: %s); falling back to "
                "inline .comfy_quant markers only, so any layer declared ONLY in __metadata__ is "
                "invisible to every check below.", type(exc).__name__, exc)
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

    The message below says "not confirmed by execution" in the message itself, and states the
    consequence as *would*, not *does*. It said it as fact once, and a fact in a WARN is how an
    audit hypothesis becomes repo lore: the person who reads it in the UI does not have this
    docstring in front of them, and the caveat that lives only next to the source is a caveat that
    does not travel.
    """
    flagged = [name for name, conf in layers.items()
               if conf.get("full_precision_matrix_mult")
               and str(conf.get("format")) in WEIGHT_ONLY_FORMATS]
    if not flagged:
        return None
    return (WARN,
            f"{len(flagged)} layer(s) request full_precision_matrix_mult. Reading ops.py:1373, "
            f"for {'/'.join(sorted(WEIGHT_ONLY_FORMATS))} that flag would not keep the layer in "
            "BF16 -- the weight stays a QuantizedTensor and the quantized kernel would run "
            "regardless, so the file's own intent would not be honoured. Audit finding, not "
            "confirmed by execution: nothing has been run with the flag set. "
            "tools/dispatch_census.py on such a file would settle it.")


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
        # A file we cannot parse is still not a file to block on: GGUF, .pt and friends come
        # through here too, and inventing a verdict for them would get this package uninstalled.
        # But returning [] said "checked, clean" for a file on which nothing was checked, and the
        # caller cannot tell those apart. WARN keeps it non-blocking and stops it reading green.
        # In practice VALIDATE_INPUTS filters to .safetensors before calling this, so a GGUF only
        # reaches here through a direct call.
        return [(WARN,
                 f"could not read the safetensors header of {path.name} "
                 f"({type(exc).__name__}: {exc}), so NONE of the file-level checks ran on it. "
                 "Not a clean result -- an unchecked one.")]
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

    **Measured 2026-08-19, and the trace was right about the mechanism and wrong about the
    symptom.** It does not crash and it does not produce garbage. `tools/dispatch_census.py` on
    the mixed Z-Image checkpoint, 4 steps at 512px:

        widget        unet_dtype           unquantized tensors        latent norm
        default       bfloat16             bf16 x283                  747.06
        fp8_e4m3fn    float8_e4m3fn        bf16 x76, fp8_e4m3fn x207  713.98
        fp8_e5m2      float8_e5m2          bf16 x76, fp8_e5m2  x207   828.08

    All three ran 680 of 680 dispatches on the native kernel: the widget cannot touch the 170
    quantized layers, which are already 4-bit. What it casts is the 207 tensors the profile
    deliberately kept in high precision -- norms, embeddings, modulation -- and the output moves.
    For scale, a LoRA at strength 1.0 moved the same latent from 747.06 to 728.99; fp8_e5m2 moves
    it to 828.08, roughly four times further, with no error, no warning, and no slowdown to notice.

    Silence is what makes this an ERROR rather than a WARN. A user who sets this widget sees a
    model that loads, runs at full speed, and returns a different picture.
    """
    if weight_dtype in (None, "", "default"):
        return None
    try:
        tensors, metadata = read_header(path)
    except Exception as exc:
        # This one mattered more than the check_file twin: the user has explicitly set a widget,
        # and a silent None told them the setting was fine on a file whose header was never read.
        return (WARN,
                f"weight_dtype={weight_dtype} was NOT checked against {path.name}: its "
                f"safetensors header did not read ({type(exc).__name__}: {exc}). If that file is "
                "quantized, the contradiction this check exists to catch is still there.")
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

    The second half used to be `getattr(mod, "aimdo_enabled", False)`, which made a renamed flag
    indistinguishable from a flag that is off -- and the "off" reading is the one that returns a
    clean pass on exactly the workflow this check exists for. Absence is now its own answer,
    because in this checkout the name is unconditional: `memory_management.py:173` is a bare
    `aimdo_enabled = False` at module scope (traced by grep 2026-08-22, not executed), so it is
    present whether dynamic VRAM is on or off. If it is missing, upstream renamed it.
    """
    nunchaku = [c for c in class_types if any(m in c for m in NUNCHAKU_LOADER_MARKERS)]
    if not nunchaku:
        return None
    try:
        import comfy.memory_management
    except Exception as exc:
        return unresolved("comfy.memory_management", exc,
                          "the dynamic-VRAM half of the Nunchaku check")
    flag = getattr(comfy.memory_management, "aimdo_enabled", _MISSING)
    if flag is _MISSING:
        return (WARN,
                "this workflow uses "
                f"{', '.join(sorted(set(nunchaku)))}, but comfy.memory_management.aimdo_enabled "
                "no longer exists, so the dynamic-VRAM check DID NOT RUN. Upstream renamed the "
                "flag; until this package is updated, confirm --disable-dynamic-vram yourself.")
    if not flag:
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


# --------------------------------------------------------------------------------------------
# Scope. What a pass from this package does and does not mean.
# --------------------------------------------------------------------------------------------
#
# The table in __init__ covers five widgets across four loader classes. ComfyUI's own nodes.py
# defines fifteen classes whose name ends in "Loader" (grep, 2026-08-22, not executed), and the
# custom_nodes tree adds more. A workflow built on any of the others got a completely clean
# preflight, which is indistinguishable from a workflow that was checked and found sound. Naming
# the gap on every run is the cheapest way to keep those two apart, and it costs one log line.
#
# **The class is the wrong unit, and that is not theoretical.** `DualCLIPLoader` has two file
# widgets. While only `clip_name1` was in the table, the class was reported as covered and a
# quantized encoder in slot 2 was never opened -- a coverage line that CONCEALED the hole rather
# than merely missing it. Adding the second table entry did not close it either. Measured
# 2026-08-22 against the suite's fake registry (EXECUTED, `scratchpad/probe_dual.py`): with both
# entries in the table, `folder_paths.get_full_path` was called exactly once, for
# `clip_name1`, and the boot log blamed "another package got there first" for a validator this
# package had installed one loop iteration earlier. So scope is stated per WIDGET below, and
# `_inject` groups the table by class so one validator covers all of a class's widgets.

LOADER_CLASS_HINT = "Loader"

# A widget whose value ends in one of these is a widget this package could in principle check.
# Read off the VALUE in the submitted graph rather than off INPUT_TYPES, for two reasons: the
# graph is available without importing ComfyUI (and importing `nodes` here would pull in torch
# and a CUDA context), and it is the widget the user actually filled in, so the list cannot
# inflate with widgets nobody used.
MODEL_FILE_SUFFIXES = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx")


def _live_widgets(covered) -> dict[str, set[str]]:
    """{class: {widget, ...}} -- the widgets an installed validator actually reads.

    Tolerates the older shape, a bare sequence of class names, by mapping each to an EMPTY widget
    set: "covered for no widget I can name". That reads as a gap rather than as coverage, which is
    the direction an unknown has to fail in here -- the whole point of the widget unit is that
    "the class is in the table" must stop being an answer to "was this file checked".
    """
    if isinstance(covered, dict):
        return {str(name): set(widgets or ()) for name, widgets in covered.items()}
    return {str(name): set() for name in covered}


def _pairs(nodes) -> list[tuple[str, dict]]:
    """Normalise the graph argument to (class_type, inputs) pairs.

    A bare class name is accepted and becomes a node with no inputs, so the widget clause reports
    nothing rather than raising. This function exists only because the alternative is a
    `ValueError` raised inside the wrapped `execution.validate_prompt`, where nothing catches it:
    the package would go from "states its scope" to "no prompt validates at all". A preflight that
    can break the server is worse than one that under-reports, every time.
    """
    # A raw prompt dict is the one shape that iterates into silent nonsense rather than into
    # nothing: `for node in {"3": {...}, "7": {...}}` yields the KEYS, so every node id gets
    # reported as a class type and the scope line names "3" and "7" as node types nobody covers.
    # Wrong and confident, which is worse than the ValueError this function exists to avoid --
    # so it is unwrapped here rather than raised. ComfyUI's prompt is exactly {id: {class_type,
    # inputs}}, and that is the shape a caller most plausibly passes by mistake.
    if isinstance(nodes, dict):
        nodes = [(n.get("class_type"), n.get("inputs")) for n in nodes.values()
                 if isinstance(n, dict)]

    normalised = []
    for node in nodes:
        if isinstance(node, str):
            normalised.append((node, {}))
        elif isinstance(node, dict):
            # A single node object, or one plucked out of a prompt. Same reasoning as above.
            class_type = node.get("class_type")
            if isinstance(class_type, str):
                inputs = node.get("inputs")
                normalised.append((class_type, inputs if isinstance(inputs, dict) else {}))
        elif isinstance(node, (tuple, list)) and len(node) == 2:
            class_type, inputs = node
            if isinstance(class_type, str):
                normalised.append((class_type, inputs if isinstance(inputs, dict) else {}))
    return normalised


# The suffixes `_make_validator` will actually open. It reads safetensors headers, and a header
# is the only thing every check here inspects -- so a .gguf or a .pth in a covered widget is seen,
# recognised as a model, and then skipped.
#
# It lives HERE, next to the coverage line, and `__init__.py` imports it, because the two drifted
# apart the moment there were two of them: the validator gated on `.safetensors` while
# `split_file_widgets` gated on the widget name alone, so the log said `opened` about six of the
# seven suffixes it never opens. On this bench that is not hypothetical -- 35 of the 159 files in
# `quantization_inventory.json` are non-safetensors (10 .gguf, 14 .pth, 7 .onnx, 4 .pt, 3 .ckpt,
# 1 .bin). Widen this tuple and the log widens with it, in one edit.
CHECKED_FILE_SUFFIXES = (".safetensors",)


def file_widgets(nodes) -> list[tuple[str, str, str]]:
    """(class_type, widget, suffix) for every widget in this graph whose value names a model file.

    `nodes` is an iterable of (class_type, inputs) pairs taken straight from the submitted prompt.
    Deduplicated on all three: two `CheckpointLoaderSimple` nodes pointing at two `.safetensors`
    are one widget to report, because the question is "which widget does this package open", not
    "how many nodes are there" -- but the same widget carrying a `.gguf` in one node and a
    `.safetensors` in another is genuinely two coverage situations and is reported as two.
    """
    found: set[tuple[str, str, str]] = set()
    for class_type, inputs in _pairs(nodes):
        if not isinstance(class_type, str):
            continue
        for widget, value in (inputs or {}).items():
            if isinstance(value, str) and value.lower().endswith(MODEL_FILE_SUFFIXES):
                suffix = "." + value.rsplit(".", 1)[-1].lower()
                found.add((class_type, str(widget), suffix))
    return sorted(found)


def split_file_widgets(nodes, covered) -> tuple[list[str], list[str]]:
    """(checked, unchecked) file widgets in this graph, as 'ClassName.widget' strings.

    A widget lands in `checked` only if it carries a live validator **and** the file is one this
    package opens. Gating on the widget name alone was a false claim about a whole extension
    class, and the test that was meant to catch it varied the widget NAME while holding the
    extension constant -- the same blind spot as an earlier fixture that varied tensor shape while
    the bug lived in identical shapes. So the unchecked label names the suffix: an operator
    reading `UNETLoader.unet_name (.gguf)` knows why, without reading this file.
    """
    live = _live_widgets(covered)
    checked, skipped = [], []
    for class_type, widget, suffix in file_widgets(nodes):
        if widget in live.get(class_type, ()) and suffix in CHECKED_FILE_SUFFIXES:
            checked.append(f"{class_type}.{widget}")
        else:
            label = f"{class_type}.{widget}"
            skipped.append(label if suffix in CHECKED_FILE_SUFFIXES else f"{label} ({suffix})")
    return checked, skipped


def uncovered_loader_classes(class_types, covered) -> list[str]:
    """Loader-shaped class names that this package does not check.

    The name is the only signal available without importing ComfyUI, and it is a heuristic: it
    misses a loader called something else, which is why the caller also prints the full uncovered
    list and not just this subset. A heuristic that under-reports is acceptable here only because
    it never decides anything -- it just makes a line louder.

    Note what this CANNOT see, and why `check_uncovered_loaders` below no longer relies on it
    alone: a class that is in the table is absent from this list even when only some of its
    widgets carry a live check.
    """
    covered = set(_live_widgets(covered))
    return sorted({c for c in class_types
                   if isinstance(c, str) and LOADER_CLASS_HINT in c and c not in covered})


def check_uncovered_loaders(class_types, covered, nodes=()) -> tuple | None:
    """WARN naming what this graph contains that no check looked at.

    Two populations, because one of them used to hide inside the other. Loader-shaped CLASSES
    nothing covers, and file WIDGETS nothing opened -- including widgets on a class that is
    otherwise covered, which is the `DualCLIPLoader.clip_name2` shape and the only one that reads
    as a pass while being a miss.
    """
    skipped = uncovered_loader_classes(class_types, covered)
    _, widgets = split_file_widgets(nodes, covered)
    if not skipped and not widgets:
        return None
    parts = []
    if skipped:
        parts.append(
            f"{len(skipped)} loader-shaped node(s) in this workflow are OUTSIDE this package's "
            f"scope and were not checked at all: {', '.join(skipped)}. A pass here says nothing "
            "about the files they load. Extend LOADER_TABLE in this package's __init__ to cover "
            "one.")
    if widgets:
        parts.append(
            f"{len(widgets)} file widget(s) in this workflow were NOT opened: "
            f"{', '.join(widgets)}. The unit is the widget, not the class: a loader this package "
            "covers can still carry a second file widget it does not, and that one reads as a "
            "pass. Add the missing widget to LOADER_TABLE.")
    return (WARN, " ".join(parts))


def scope_line(nodes, covered, unaudited=()) -> str:
    """One line stating what a pass covered, for the log, on every run.

    `nodes` is an iterable of (class_type, inputs) pairs from the submitted prompt.

    Two units, and both are needed. **Node types** say which boxes this package looked at at all.
    **File widgets** say which files it actually opened -- and the widget is the unit a
    partially-covered class hides in: with only the class clause, `DualCLIPLoader` printed as
    checked while `clip_name2` never reached `check_file`. A coverage line that can say "covered"
    about a partially-checked class does not merely miss the hole, it argues against looking for
    it.

    Deliberately prints the whole uncovered list rather than a count. A count is the same
    reassurance a silent pass gives, in a smaller font.
    """
    nodes = _pairs(nodes)
    present = sorted({c for c, _ in nodes if isinstance(c, str)})
    covered_here = sorted(set(present) & set(_live_widgets(covered)))
    uncovered = [c for c in present if c not in set(covered_here)]
    checked, skipped = split_file_widgets(nodes, covered)
    line = ("opened {nw} file widget(s): {cw} | SAW AND DID NOT CHECK {mw} file widget(s): {uw} "
            "| node types checked {n}: {c} | node types seen and not covered {m}: {u}".format(
                nw=len(checked), cw=", ".join(checked) or "none",
                mw=len(skipped), uw=", ".join(skipped) or "none",
                n=len(covered_here), c=", ".join(covered_here) or "none",
                m=len(uncovered), u=", ".join(uncovered) or "none"))

    # A class reaches `covered` even when `INPUT_TYPES` could not be read -- `_inject` installs on
    # the table's widget names rather than declining, because declining would turn an unreadable
    # class into an unchecked one. That is the right trade and it has a cost that has to travel:
    # if the audit failed *because* upstream renamed the widget, the validator reads
    # `kwargs.get(<old name>)`, gets None, and returns True forever -- and every word above would
    # count it as opened. Naming it here is the difference between a coverage line that reports
    # and one that vouches.
    unverified = sorted(set(unaudited) & set(present))
    if unverified:
        line += (" | WIDGET NAMES NOT CONFIRMED against INPUT_TYPES for {k}: {v}"
                 " -- counted as opened above, but nothing proved those widgets exist"
                 .format(k=len(unverified), v=", ".join(unverified)))
    return line
