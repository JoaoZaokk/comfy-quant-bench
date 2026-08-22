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
#
# **Several entries may name the same class.** `_inject` groups by class before installing, and
# that grouping is load-bearing rather than tidy: `VALIDATE_INPUTS` is one attribute per class, so
# one entry per iteration meant the SECOND entry for a class found the attribute its own package
# had just set and took the "someone else got here first" branch. Measured 2026-08-22, EXECUTED
# against the suite's fake registry -- with both DualCLIPLoader rows present,
# `folder_paths.get_full_path` was called once, for clip_name1, and the boot log named another
# package as the reason clip_name2 was skipped.
LOADER_TABLE = (
    ("UNETLoader", "unet_name", "diffusion_models", "weight_dtype"),
    ("CLIPLoader", "clip_name", "text_encoders", None),
    # DualCLIPLoader has TWO file widgets. Only clip_name1 was listed, so a quantized encoder in
    # slot 2 went unchecked while the class itself was reported as covered -- a fail-open the
    # coverage line actively concealed, which is worse than one it merely misses.
    ("DualCLIPLoader", "clip_name1", "text_encoders", None),
    ("DualCLIPLoader", "clip_name2", "text_encoders", None),
    ("CheckpointLoaderSimple", "ckpt_name", "checkpoints", None),
)


# What actually got installed, and what did not. Read by the graph-level wrapper so that a run
# whose file-level checks never installed says so on that run, instead of at a boot line that
# scrolled past three hours ago.
#
# `covered` is keyed by class and holds the WIDGET NAMES the installed validator actually reads,
# not a bare list of class names. The class was the unit until 2026-08-22 and it is the unit a
# half-covered loader hides in -- see the scope section of checks.py for the measurement.
_STATUS: dict = {
    "covered": {},          # {loader class: (widget, ...)} carrying a live injected validator
    "file_checks": "not installed yet",
    "graph_checks": "not installed yet",
}


def _resolve(folder: str, name: str) -> tuple[Path | None, tuple | None]:
    """(path, failure-finding). A path of None means "not found"; a failure means "did not look".

    Those were the same value before, and the difference is the whole ticket: `folder_paths`
    changing shape under us returned None, None meant "no file", "no file" meant "nothing to
    check", and the node passed clean. The caller now has to handle the two separately because the
    signature no longer lets it confuse them.
    """
    try:
        import folder_paths
        found = folder_paths.get_full_path(folder, name)
    except Exception as exc:
        from .checks import unresolved
        return None, unresolved(f"folder_paths.get_full_path({folder!r}, ...)", exc,
                                f"every file-level check on {name!r}")
    return (Path(found) if found else None), None


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


def _make_validator(entries: list):
    """One validator for ALL of a class's file widgets. `entries` is [(file, folder, dtype), ...].

    Per-widget rather than per-entry because `VALIDATE_INPUTS` is a single class attribute: a
    second installation for the same class cannot exist, so the loop has to live inside the one
    validator. A widget that resolves to nothing no longer ends the check for the rest of them
    either -- `clip_name1` pointing at a missing file used to `return True` before `clip_name2`
    was looked at.
    """
    def VALIDATE_INPUTS(cls=None, **kwargs):
        # Injected as a plain function and read through getattr, so it must tolerate being called
        # with or without the class. ComfyUI inspects the signature and passes every input when
        # `varkw` is not None, which is why this takes **kwargs rather than naming them --
        # `grep -n "validate_has_kwargs" ComfyUI/execution.py` is the line, wherever it has
        # drifted to. (Traced 2026-08-22, not executed: no prompt was submitted this session.)
        from . import checks

        findings: list[tuple] = []
        for file_widget, folder, dtype_widget in entries:
            name = kwargs.get(file_widget)
            if not isinstance(name, str) or not name:
                continue
            path, failure = _resolve(folder, name)
            if failure:
                findings.append(failure)    # a WARN; logged by _format below, never blocks
                continue
            # Same tuple the coverage line reads, imported rather than restated. When these were
            # two literals the log claimed `opened` about six suffixes this branch skips.
            if path is None or not path.is_file():
                continue
            if path.suffix.lower() not in checks.CHECKED_FILE_SUFFIXES:
                continue
            findings.extend(checks.check_file(path))
            if dtype_widget:
                result = checks.check_dtype_widget(path, kwargs.get(dtype_widget))
                if result:
                    findings.append(result)
        message = _format(findings)
        return message if message else True

    return VALIDATE_INPUTS


def _widget_names(node_class) -> tuple[set[str], tuple | None]:
    """(widget names this class accepts, failure-finding).

    An empty set with no failure means the class declares no widgets; an empty set *with* a
    failure means the audit below could not run, which must not be read as "the widget is fine".
    """
    from .checks import unresolved

    try:
        spec = node_class.INPUT_TYPES()
        names = set()
        for group in ("required", "optional"):
            names.update((spec.get(group) or {}).keys())
    except Exception as exc:
        return set(), unresolved(f"{getattr(node_class, '__name__', node_class)}.INPUT_TYPES()",
                                 exc, "the widget-name audit for this loader")
    return names, None


def _inject() -> None:
    import nodes

    from . import checks

    # Grouped by class BEFORE anything is installed. `VALIDATE_INPUTS` is one attribute per class,
    # so a per-entry loop physically cannot install a second validator for a class -- it finds the
    # one it set on the previous iteration and reports the class as somebody else's. See the
    # LOADER_TABLE comment for the measurement.
    grouped: dict[str, list[tuple[str, str, str | None]]] = {}
    for class_name, file_widget, folder, dtype_widget in LOADER_TABLE:
        grouped.setdefault(class_name, []).append((file_widget, folder, dtype_widget))

    injected: dict[str, tuple[str, ...]] = {}
    live_entries, missing, stale, deferred = 0, [], [], []
    for class_name, entries in grouped.items():
        node_class = nodes.NODE_CLASS_MAPPINGS.get(class_name)
        if node_class is None:
            missing.append(class_name)
            continue
        # The class surviving a rename is not the same as its widgets surviving one. A validator
        # keyed on `unet_name` after upstream renames that widget reads `kwargs.get("unet_name")`,
        # gets None, and returns True forever -- installed, reported as injected, and structurally
        # incapable of ever firing. That is the fork failure mode this package was built to avoid,
        # reproduced inside the package. So the widget names are audited against INPUT_TYPES, and
        # a validator that could never fire is not installed and is reported as UNCHECKED.
        #
        # Audited per ENTRY, not per class: one renamed widget must cost that widget's check and
        # not its siblings'. A class whose entries all went stale installs nothing, exactly as
        # before.
        widgets, audit_failure = _widget_names(node_class)
        if audit_failure:
            logger.warning("quant-preflight: %s", audit_failure[1])
            live = list(entries)
        elif widgets:
            live = []
            for file_widget, folder, dtype_widget in entries:
                gone = [w for w in (file_widget, dtype_widget) if w and w not in widgets]
                if gone:
                    stale.append(f"{class_name}(missing widget: {', '.join(gone)})")
                    continue
                live.append((file_widget, folder, dtype_widget))
        else:
            live = list(entries)
        if not live:
            continue
        if getattr(node_class, "VALIDATE_INPUTS", None) is not None:
            # Refusing to replace an existing validator is not politeness. Overwriting one would
            # silently delete whatever check it was performing. It does mean this loader is not
            # covered by us, though, so it is reported as such rather than logged at INFO and
            # forgotten.
            deferred.append(class_name)
            continue
        node_class.VALIDATE_INPUTS = classmethod(_make_validator(live))
        # Every widget the installed validator reads, dtype widget included -- this is what the
        # per-run scope line subtracts the graph's file widgets from, so an omission here reads
        # as "not checked", which is the safe direction.
        injected[class_name] = tuple(
            w for file_widget, _, dtype_widget in live
            for w in (file_widget, dtype_widget) if w)
        live_entries += len(live)

    _STATUS["covered"] = injected
    _STATUS["file_checks"] = (
        f"{live_entries} of {len(LOADER_TABLE)} table entries live" if injected
        else "NO loader is checked")

    logger.info("quant-preflight: injected into %d loader(s), %d widget(s): %s",
                len(injected), live_entries,
                ", ".join(f"{c}.{w}" for c, ws in injected.items() for w in ws) or "none")
    for label, names, why in (
        ("NOT FOUND", missing,
         "ComfyUI probably renamed or removed them; the table in this package's __init__ needs "
         "updating"),
        ("WIDGET RENAMED", stale,
         "the class is still there but the widget the check reads is not, so an injected "
         "validator would have returned True forever; it was not installed"),
        ("ALREADY VALIDATED BY SOMEONE ELSE", deferred,
         "another package got there first and its validator was left alone, so these files are "
         "not checked by us"),
    ):
        if names:
            # Loud on purpose. A silently-absent check is the failure mode this package exists to
            # prevent, so it must not be the failure mode of the package itself.
            logger.warning("quant-preflight: %d loader class(es) %s and therefore UNCHECKED: %s. "
                           "%s.", len(names), label, ", ".join(names), why)

    # The classes in LOADER_TABLE are not the loaders that exist. Everything else in the registry
    # that looks like a loader is out of scope, and until this line existed a workflow built on
    # one of them got a clean preflight that meant nothing. This line is CLASS-level and cannot be
    # anything else -- at boot there is no graph, so there are no widget values to read. The
    # widget-level statement is the per-run one below.
    #
    # This list UNDER-REPORTS, deliberately and unavoidably: `init_external_custom_nodes` loads
    # packages in name order, so anything sorting after "comfy-quant-preflight" has not registered
    # its nodes yet. The per-run scope line in the wrapper below is the one that is complete,
    # because it reads the graph the user actually submitted. Do not read this boot line as the
    # full set of things not covered -- read it as a floor.
    outside = checks.uncovered_loader_classes(nodes.NODE_CLASS_MAPPINGS, injected)
    if outside:
        logger.warning(
            "quant-preflight: %d further loader-shaped node class(es) are OUT OF SCOPE and get no "
            "check at all: %s", len(outside), ", ".join(outside))


def _wrap_validate_prompt() -> None:
    """Graph-level checks: things no single node can see."""
    import execution

    original = execution.validate_prompt
    if getattr(original, "_quant_preflight", False):
        _STATUS["graph_checks"] = "installed (already wrapped)"
        return

    async def validate_prompt(prompt_id, prompt, partial_execution_list=None):
        from . import checks

        result = await original(prompt_id, prompt, partial_execution_list)
        ok = result[0]
        if not ok:
            # Already failing for another reason; do not pile on. Still state scope, because "on
            # every run" includes the runs where somebody else spoke first -- otherwise the only
            # runs that say what was skipped are the ones that got far enough to pass.
            logger.info("quant-preflight: %s (run already failed elsewhere)", checks.scope_line(
                [(n.get("class_type"), n.get("inputs"))
                 for n in (prompt or {}).values() if isinstance(n, dict)],
                _STATUS["covered"]))
            return result

        findings: list[tuple] = []
        seen_messages: set[str] = set()

        def note(finding: tuple | None) -> None:
            # Deduplicated because a graph with eight loaders behind one broken folder_paths would
            # otherwise repeat the same WARN eight times, and a wall of identical lines is read
            # the same way as no line at all.
            if finding and finding[1] not in seen_messages:
                seen_messages.add(finding[1])
                findings.append(finding)

        class_types, graph_nodes, quantized_files, first_node = [], [], 0, None
        for node_id, node in (prompt or {}).items():
            class_type = node.get("class_type")
            if not class_type:
                continue
            class_types.append(class_type)
            graph_nodes.append((class_type, node.get("inputs")))
            if first_node is None:
                first_node = node_id
            for _, file_widget, folder, _ in LOADER_TABLE:
                value = (node.get("inputs") or {}).get(file_widget)
                if isinstance(value, str) and value.endswith(".safetensors"):
                    path, failure = _resolve(folder, value)
                    note(failure)
                    if path and path.is_file():
                        try:
                            tensors, metadata = checks.read_header(path)
                            if checks.quant_layers(tensors, metadata):
                                quantized_files += 1
                        except Exception as exc:
                            # This count feeds check_lora_over_quantized. Swallowing the failure
                            # undercounted it, and an undercount of zero turns that check off
                            # without saying so.
                            note((checks.WARN,
                                  f"could not read the header of {path.name} "
                                  f"({type(exc).__name__}: {exc}), so it was NOT counted as "
                                  "quantized. Any check below that keys off 'is a quantized "
                                  "model in this graph' ran on an undercount."))

        # Every run states its own scope, pass or fail. A green light from a package that looked at
        # two of a workflow's twenty nodes is worth exactly as much as the list of the eighteen.
        logger.info("quant-preflight: %s", checks.scope_line(graph_nodes, _STATUS["covered"]))
        note(checks.check_uncovered_loaders(class_types, _STATUS["covered"], graph_nodes))
        if not _STATUS["covered"]:
            note((checks.WARN,
                  f"no per-file check is installed ({_STATUS['file_checks']}), so nothing in this "
                  "workflow was checked against its checkpoint headers. Only the graph-level "
                  "checks below ran."))

        for check in (
            lambda: checks.check_nunchaku_needs_disable_dynamic_vram(class_types),
            lambda: checks.check_lora_over_quantized(class_types, quantized_files),
        ):
            note(check())

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
    _STATUS["graph_checks"] = "installed"
    logger.info("quant-preflight: wrapped execution.validate_prompt")


# Installed in two independent steps. One `try` around both meant a failure in _inject() skipped
# the wrapper entirely, so the half that still worked was lost along with the half that broke --
# and the resulting server ran with no preflight at all while the only evidence was one boot line.
for _step in (_inject, _wrap_validate_prompt):
    try:
        _step()
    except Exception as exc:
        # A broken preflight must never stop ComfyUI from starting. It logs the traceback and gets
        # out of the way; the cost of being wrong here is a missing check, not a dead server.
        # The cost of being wrong QUIETLY is a user who thinks they were checked.
        _STATUS["file_checks" if _step is _inject else "graph_checks"] = (
            f"FAILED to install ({type(exc).__name__}: {exc})")
        logger.exception("quant-preflight: %s failed to install (%s: %s); THAT HALF OF THE "
                         "PREFLIGHT IS NOT RUNNING", _step.__name__, type(exc).__name__, exc)

if not _STATUS["graph_checks"].startswith("installed"):
    # Nothing else will ever say this again: with the wrapper gone there is no per-run line, so
    # the last chance to tell anyone is here.
    logger.warning("quant-preflight: graph-level checks are NOT active (%s). Nunchaku/dynamic-VRAM "
                   "and LoRA-over-quantized will not be checked on any run.",
                   _STATUS["graph_checks"])
