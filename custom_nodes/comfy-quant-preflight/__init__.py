"""Refuse to start a workflow whose quantization configuration cannot be true.

Not a fork. ComfyUI imports every `custom_nodes/*/__init__.py` at boot
(`nodes.py:init_external_custom_nodes`), which is enough to reach both validation hooks without
editing a single upstream file:

    VALIDATE_INPUTS injected onto existing loader classes
        `execution.py` calls `getattr(obj_class, "VALIDATE_INPUTS")` per node during
        validate_prompt. A returned string becomes an error attributed to that node in the UI.
        Sees one node's widgets. Used for file-versus-widget.

    validate_prompt wrapped
        Sees the whole graph and returns `(ok, error, good_outputs, node_errors)` where
        node_errors is keyed by node id. Used for anything that needs more than one node, or
        needs process state.

Why not a fork, in numbers taken from this checkout: upstream lands ~307 commits per 60 days, and
~35 of those touch the four files a verification fork would have to modify. That is a merge tax
forever, in exactly the lines you changed.

The decisive argument is not cost, though. **This fails loudly and a fork fails quietly.** If an
upgrade renames `UNETLoader`, the injection below finds nothing and says so at boot. The same
upgrade against a fork merges cleanly and leaves the check sitting in a code path nobody calls
any more -- silently inert, which is the exact failure class this whole effort exists to catch.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("quant-preflight")

# ComfyUI requires these to exist. This package adds no nodes of its own -- it only teaches the
# existing ones to check themselves.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# (node class name, widget carrying a filename, folder to resolve it in, widget carrying a dtype
# claim or None). Kept as an explicit table rather than guessed from INPUT_TYPES: each loader
# treats dtype differently, and a wrong guess here produces a false refusal, which is worse than
# no check at all.
#
# `DiffusionModelLoader` was in this table on the first run and does not exist in ComfyUI 0.33 --
# the "NOT FOUND and therefore UNCHECKED" warning below caught it within seconds of the first
# boot. That is the property this design was chosen for, demonstrated on its own author: a fork
# carrying the same mistake would have merged cleanly and left a check pointed at nothing.
LOADER_TABLE = (
    ("UNETLoader", "unet_name", "diffusion_models", "weight_dtype"),
    ("CLIPLoader", "clip_name", "text_encoders", None),
    ("DualCLIPLoader", "clip_name1", "text_encoders", None),
    ("CheckpointLoaderSimple", "ckpt_name", "checkpoints", None),
)


def _resolve(folder: str, name: str) -> Path | None:
    try:
        import folder_paths
        found = folder_paths.get_full_path(folder, name)
        return Path(found) if found else None
    except Exception:
        return None


def _format(findings: list[tuple]) -> str | None:
    """Render findings, or None when nothing blocks.

    Warnings go to the log and never block. Only an error becomes a returned string, because a
    returned string stops the run.
    """
    from .checks import ERROR

    errors = [message for severity, message in findings if severity == ERROR]
    for severity, message in findings:
        if severity != ERROR:
            logger.warning("quant-preflight: %s", message)
    if not errors:
        return None
    return "quant-preflight: " + " | ".join(errors)


def _make_validator(file_widget: str, folder: str, dtype_widget: str | None):
    def VALIDATE_INPUTS(cls=None, **kwargs):
        # Injected as a plain function and read through getattr, so it must tolerate being called
        # with or without the class. ComfyUI filters kwargs down to the names in the signature,
        # which is why this takes **kwargs rather than naming them.
        from . import checks

        name = kwargs.get(file_widget)
        if not isinstance(name, str) or not name:
            return True
        path = _resolve(folder, name)
        if path is None or not path.is_file() or path.suffix.lower() != ".safetensors":
            return True
        findings = checks.check_file(path)
        if dtype_widget:
            result = checks.check_dtype_widget(path, kwargs.get(dtype_widget))
            if result:
                findings.append(result)
        message = _format(findings)
        return message if message else True

    return VALIDATE_INPUTS


def _inject() -> None:
    import nodes

    injected, missing = [], []
    for class_name, file_widget, folder, dtype_widget in LOADER_TABLE:
        node_class = nodes.NODE_CLASS_MAPPINGS.get(class_name)
        if node_class is None:
            missing.append(class_name)
            continue
        if getattr(node_class, "VALIDATE_INPUTS", None) is not None:
            # Refusing to replace an existing validator is not politeness. Overwriting one would
            # silently delete whatever check it was performing.
            logger.info("quant-preflight: %s already has VALIDATE_INPUTS, leaving it alone",
                        class_name)
            continue
        validator = _make_validator(file_widget, folder, dtype_widget)
        node_class.VALIDATE_INPUTS = classmethod(validator)
        injected.append(class_name)

    logger.info("quant-preflight: injected into %d loader(s): %s",
                len(injected), ", ".join(injected) or "none")
    if missing:
        # Loud on purpose. A silently-absent check is the failure mode this package exists to
        # prevent, so it must not be the failure mode of the package itself.
        logger.warning(
            "quant-preflight: %d loader class(es) NOT FOUND and therefore UNCHECKED: %s. "
            "ComfyUI probably renamed or removed them; the table in this package's __init__ "
            "needs updating.", len(missing), ", ".join(missing))


def _wrap_validate_prompt() -> None:
    """Graph-level checks: things no single node can see."""
    import execution

    original = execution.validate_prompt
    if getattr(original, "_quant_preflight", False):
        return

    async def validate_prompt(prompt_id, prompt, partial_execution_list=None):
        result = await original(prompt_id, prompt, partial_execution_list)
        ok = result[0]
        if not ok:
            return result          # already failing for another reason; do not pile on

        from . import checks

        class_types, quantized_files, first_node = [], 0, None
        for node_id, node in (prompt or {}).items():
            class_type = node.get("class_type")
            if not class_type:
                continue
            class_types.append(class_type)
            if first_node is None:
                first_node = node_id
            for _, file_widget, folder, _ in LOADER_TABLE:
                value = (node.get("inputs") or {}).get(file_widget)
                if isinstance(value, str) and value.endswith(".safetensors"):
                    path = _resolve(folder, value)
                    if path and path.is_file():
                        try:
                            tensors, metadata = checks.read_header(path)
                            if checks.quant_layers(tensors, metadata):
                                quantized_files += 1
                        except Exception:
                            pass

        findings = []
        for check in (
            lambda: checks.check_nunchaku_needs_disable_dynamic_vram(class_types),
            lambda: checks.check_lora_over_quantized(class_types, quantized_files),
        ):
            found = check()
            if found:
                findings.append(found)

        message = _format(findings)
        if message is None:
            return result
        error = {
            "type": "quant_preflight",
            "message": "Quantization preflight refused this workflow",
            "details": message,
            "extra_info": {},
        }
        # node_errors keyed by id is how the UI attributes a failure to a box. Graph-level
        # findings have no single owner, so they are pinned to the first node and repeated in
        # the top-level message, which is what the user actually reads.
        node_errors = {first_node: {"errors": [error]}} if first_node else {}
        return (False, error, [], node_errors)

    validate_prompt._quant_preflight = True
    execution.validate_prompt = validate_prompt
    logger.info("quant-preflight: wrapped execution.validate_prompt")


try:
    _inject()
    _wrap_validate_prompt()
except Exception:
    # A broken preflight must never stop ComfyUI from starting. It logs the traceback and gets
    # out of the way; the cost of being wrong here is a missing check, not a dead server.
    logger.exception("quant-preflight: failed to install, continuing without it")
