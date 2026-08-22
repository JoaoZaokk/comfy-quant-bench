"""CPU tests for the preflight checks, run against the real files on this machine.

    python_embeded\\python.exe -s custom_nodes\\comfy-quant-preflight\\test_checks.py

(That is the tracked package at the portable root, not the three-line loader stub under
ComfyUI\\custom_nodes\\. Editing the tracked copy is what changes what runs.)

Synthetic headers are built for the cases no file here exhibits (a weight_correction tensor, an
inert full-precision flag), because "we have no example" is not evidence that a check works.

Nothing here imports torch, ComfyUI, or `nodes`. The tests that need ComfyUI to exist fake it --
see `fake_comfyui` -- for two reasons: importing the real `nodes` pulls in torch and initialises a
CUDA context on a card this bench shares, and a fake registry is the only way to exercise the
upstream-rename paths, which by definition do not occur in the checkout as it stands today.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
import struct
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checks  # noqa: E402

# parents[2], not [3]. This package lives at F:/COMFY_PORTABLE/custom_nodes/comfy-quant-preflight,
# so [0]=the package, [1]=custom_nodes, [2]=the portable root. The [3] this used to carry was the
# arithmetic for the *stub* inside ComfyUI/custom_nodes/, one level deeper, and it resolved to
# F:/ -- so MODELS pointed at F:/ComfyUI/models/diffusion_models, which does not exist, and the
# real-checkpoint test skipped every file and printed "(0 real checkpoint(s) exercised)" while
# reporting PASS.
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "ComfyUI" / "models" / "diffusion_models"
PACKAGE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------
# Harness. Every fail-open below is triggered by something upstream changing, so the tests have to
# be able to change it.
# --------------------------------------------------------------------------------------------

class Captured(logging.Handler):
    """Collects the package's log records so a test can assert a WARN was actually emitted.

    Asserting on a return value is not enough for these: half the fail-opens are paths whose whole
    correct behaviour is "return the same harmless thing, but say so". The saying-so is the fix, so
    the saying-so is what gets asserted.
    """

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def messages(self, level=logging.WARNING) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


@contextlib.contextmanager
def capture_logs():
    logger = logging.getLogger("quant-preflight")
    handler = Captured()
    previous, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)      # the scope line is INFO; the default root level eats it
    logger.propagate = False            # keep the test output readable
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
        logger.propagate = previous_propagate


class FakeUNETLoader:
    """Stands in for ComfyUI's UNETLoader, with the widget names it has in 0.33."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (["m.safetensors"],), "weight_dtype": (["default"],)}}


class RenamedUNETLoader:
    """The same class after a hypothetical upstream rename of the file widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (["m.safetensors"],), "weight_dtype": (["default"],)}}


class FakeDualCLIPLoader:
    """Two file widgets on one class -- the shape LOADER_TABLE has five entries for four classes.

    ComfyUI's DualCLIPLoader in 0.33 (traced, not executed). The point of the fake is that both
    widgets are present, so a validator that reads only one of them is visible as a miss rather
    than as an absent widget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name1": (["a.safetensors"],),
                             "clip_name2": (["b.safetensors"],),
                             "type": (["flux"],)}}


@contextlib.contextmanager
def fake_comfyui(node_classes=None, get_full_path=None, break_nodes=False, prompt_ok=True):
    """Import the package against a fake ComfyUI. Yields (module, nodes, execution).

    The package installs itself at import time, so the only way to observe what it installs is to
    control what it installs into. `break_nodes` makes `import nodes` fail, which is how the
    "one half failed to install" path is reached without breaking anything real.
    """
    keys = ("nodes", "execution", "folder_paths",
            "comfy_quant_preflight", "comfy_quant_preflight.checks")
    saved = {key: sys.modules.get(key) for key in keys}

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.NODE_CLASS_MAPPINGS = dict(node_classes or {})

    execution_mod = types.ModuleType("execution")

    async def validate_prompt(prompt_id, prompt, partial_execution_list=None):
        return (True, None, [], {}) if prompt_ok else (False, {"type": "other"}, [], {})

    execution_mod.validate_prompt = validate_prompt

    folder_paths_mod = types.ModuleType("folder_paths")
    folder_paths_mod.get_full_path = get_full_path or (lambda folder, name: None)

    sys.modules["nodes"] = None if break_nodes else nodes_mod
    sys.modules["execution"] = execution_mod
    sys.modules["folder_paths"] = folder_paths_mod
    for key in ("comfy_quant_preflight", "comfy_quant_preflight.checks"):
        sys.modules.pop(key, None)

    spec = importlib.util.spec_from_file_location(
        "comfy_quant_preflight", PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["comfy_quant_preflight"] = module
    # Injection mutates the class object, and these fakes are module-level and shared. Without
    # this the second test to use one would find a VALIDATE_INPUTS already there, take the
    # "someone else got here first" branch, and fail for a reason that has nothing to do with what
    # it was testing.
    already = {name: getattr(cls, "VALIDATE_INPUTS", None)
               for name, cls in (node_classes or {}).items()}
    try:
        spec.loader.exec_module(module)
        yield module, nodes_mod, execution_mod
    finally:
        for name, cls in (node_classes or {}).items():
            if already[name] is None and "VALIDATE_INPUTS" in cls.__dict__:
                delattr(cls, "VALIDATE_INPUTS")
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@contextlib.contextmanager
def fake_memory_management(**attributes):
    """A stand-in `comfy.memory_management`. Pass no attributes to simulate the flag being gone.

    `break_import=True` puts None in sys.modules, which is what makes `import` itself fail.
    """
    break_import = attributes.pop("break_import", False)
    saved = {key: sys.modules.get(key) for key in ("comfy", "comfy.memory_management")}
    comfy_mod = types.ModuleType("comfy")
    mm_mod = types.ModuleType("comfy.memory_management")
    for name, value in attributes.items():
        setattr(mm_mod, name, value)
    comfy_mod.memory_management = mm_mod
    sys.modules["comfy"] = comfy_mod
    sys.modules["comfy.memory_management"] = None if break_import else mm_mod
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def write_safetensors(entries: dict, metadata: dict | None = None) -> Path:
    """A header-only safetensors file: offsets are declared, payload is zeros."""
    header, offset = {}, 0
    for name, (dtype, shape) in entries.items():
        size = 1
        for dim in shape:
            size *= dim
        size *= {"BF16": 2, "F32": 4, "I8": 1, "U8": 1}[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path = Path(tempfile.mkdtemp()) / "synthetic.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    return path


def test_diffusers_named_quantized_is_caught():
    path = write_safetensors(
        {"layers.0.attention.to_q.weight": ("I8", [8, 4]),
         "layers.0.attention.to_q.weight_scale": ("F32", [8])},
        {"_quantization_metadata": json.dumps(
            {"layers": {"layers.0.attention.to_q": {"format": "convrot_w4a4"}}})})
    found = checks.check_file(path)
    assert any(s == checks.ERROR and "diffusers naming" in m for s, m in found), found


def test_native_named_quantized_is_clean():
    """The same file after tools/to_native.py must not trip anything."""
    path = write_safetensors(
        {"layers.0.attention.qkv.weight": ("I8", [24, 4]),
         "layers.0.attention.qkv.weight_scale": ("F32", [24])},
        {"_quantization_metadata": json.dumps(
            {"layers": {"layers.0.attention.qkv": {"format": "convrot_w4a4"}}})})
    assert checks.check_file(path) == [], checks.check_file(path)


def test_weight_correction_is_caught():
    path = write_safetensors(
        {"layers.0.mlp.weight": ("I8", [8, 4]),
         "layers.0.mlp.weight_correction": ("F32", [8, 4])},
        {"_quantization_metadata": json.dumps(
            {"layers": {"layers.0.mlp": {"format": "asym_w4a8_int8"}}})})
    found = checks.check_file(path)
    assert any(s == checks.ERROR and "weight_correction" in m for s, m in found), found


def test_inert_full_precision_flag_warns_but_does_not_block():
    path = write_safetensors(
        {"layers.0.mlp.weight": ("I8", [8, 4])},
        {"_quantization_metadata": json.dumps({"layers": {"layers.0.mlp": {
            "format": "convrot_w4a4", "full_precision_matrix_mult": True}}})})
    found = checks.check_file(path)
    assert any(s == checks.WARN and "full_precision" in m for s, m in found), found
    assert not any(s == checks.ERROR for s, m in found), "a warning must not block"


def test_inline_comfy_quant_markers_count_as_quantized():
    """Comfy-Org and Lightricks ship these with no __metadata__ at all."""
    path = write_safetensors({"layers.0.mlp.weight": ("I8", [8, 4]),
                              "layers.0.mlp.comfy_quant": ("U8", [40])})
    tensors, metadata = checks.read_header(path)
    assert checks.quant_layers(tensors, metadata), "inline markers were not detected"


def test_dtype_widget_default_is_always_silent():
    path = write_safetensors(
        {"layers.0.mlp.weight": ("I8", [8, 4])},
        {"_quantization_metadata": json.dumps(
            {"layers": {"layers.0.mlp": {"format": "convrot_w4a4"}}})})
    assert checks.check_dtype_widget(path, "default") is None
    assert checks.check_dtype_widget(path, "") is None


def test_dtype_widget_on_unquantized_file_is_silent():
    """BF16 + fp8 is the widget doing its job. Blocking it would get this package uninstalled."""
    path = write_safetensors({"layers.0.mlp.weight": ("BF16", [8, 4])})
    assert checks.check_dtype_widget(path, "fp8_e4m3fn") is None


def test_dtype_widget_on_quantized_file_is_an_error():
    path = write_safetensors(
        {"layers.0.mlp.weight": ("I8", [8, 4])},
        {"_quantization_metadata": json.dumps(
            {"layers": {"layers.0.mlp": {"format": "convrot_w4a4"}}})})
    result = checks.check_dtype_widget(path, "fp8_e4m3fn")
    assert result and result[0] == checks.ERROR, result
    assert "convrot_w4a4" in result[1]


def test_full_precision_message_says_it_was_not_confirmed_by_execution():
    """The caveat has to be in the message, not only in the docstring beside it.

    It drifted out once. The user reading this WARN in the ComfyUI log has the message and nothing
    else, and an audit hypothesis stated as fact in a UI string is how it becomes repo lore.
    """
    path = write_safetensors(
        {"layers.0.mlp.weight": ("I8", [8, 4])},
        {"_quantization_metadata": json.dumps({"layers": {"layers.0.mlp": {
            "format": "convrot_w4a4", "full_precision_matrix_mult": True}}})})
    found = checks.check_file(path)
    message = next(m for s, m in found if "full_precision" in m)
    assert "not confirmed by execution" in message.lower(), message
    assert "dispatch_census" in message, "name the thing that would settle it"


def test_unparseable_file_warns_that_nothing_was_checked():
    """GGUF and .pt come through the same call. Inventing a verdict for them is still worse than
    staying quiet -- but returning [] said 'checked, clean' about a file nothing was read from,
    and the caller could not tell that from a real pass."""
    path = Path(tempfile.mkdtemp()) / "not.safetensors"
    path.write_bytes(b"this is not a safetensors file at all")
    found = checks.check_file(path)
    assert found, "a file that could not be read must not come back as a clean pass"
    assert all(s == checks.WARN for s, _ in found), f"must not block: {found}"
    assert "NONE of the file-level checks ran" in found[0][1], found


def test_dtype_widget_warns_when_the_header_cannot_be_read():
    """The user set the widget explicitly here, so silence is a direct answer to a direct
    question -- and it was the wrong one."""
    path = Path(tempfile.mkdtemp()) / "broken.safetensors"
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00not json")
    result = checks.check_dtype_widget(path, "fp8_e4m3fn")
    assert result and result[0] == checks.WARN, result
    assert "NOT checked" in result[1], result


def test_malformed_quantization_metadata_is_logged_not_swallowed():
    """A file whose metadata does not parse looks exactly like an unquantized file to every check
    downstream, which is the one file most likely to need them."""
    path = write_safetensors({"layers.0.mlp.weight": ("I8", [8, 4])},
                             {"_quantization_metadata": "{not json at all"})
    with capture_logs() as log:
        tensors, metadata = checks.read_header(path)
        assert checks.quant_layers(tensors, metadata) == {}
    assert any("did not parse" in m for m in log.messages()), log.messages()
    assert any("JSONDecodeError" in m or "ValueError" in m for m in log.messages()), \
        "the WARN must name the exception"


def test_nunchaku_check_warns_when_the_upstream_flag_is_gone():
    """A renamed flag used to be indistinguishable from a flag that is off, and 'off' is the
    reading that returns a clean pass on exactly the workflow this check exists for."""
    with fake_memory_management():          # module present, aimdo_enabled absent
        found = checks.check_nunchaku_needs_disable_dynamic_vram(["NunchakuFluxDiTLoader"])
    assert found and found[0] == checks.WARN, found
    assert "aimdo_enabled" in found[1] and "DID NOT RUN" in found[1], found


def test_nunchaku_check_warns_when_the_upstream_module_is_gone():
    with fake_memory_management(break_import=True):
        found = checks.check_nunchaku_needs_disable_dynamic_vram(["NunchakuFluxDiTLoader"])
    assert found and found[0] == checks.WARN, found
    assert "comfy.memory_management" in found[1] and "DID NOT RUN" in found[1], found


def test_nunchaku_check_still_blocks_when_the_flag_is_actually_on():
    """The WARN paths above must not have cost the ERROR this check exists for."""
    with fake_memory_management(aimdo_enabled=True):
        found = checks.check_nunchaku_needs_disable_dynamic_vram(["NunchakuFluxDiTLoader"])
    assert found and found[0] == checks.ERROR, found
    assert "--disable-dynamic-vram" in found[1]


def test_nunchaku_check_is_silent_when_the_flag_is_genuinely_off():
    with fake_memory_management(aimdo_enabled=False):
        assert checks.check_nunchaku_needs_disable_dynamic_vram(["NunchakuFluxDiTLoader"]) is None


def test_uncovered_loaders_are_named_not_counted():
    found = checks.check_uncovered_loaders(
        ["UNETLoader", "KSampler", "GGUFLoader", "NunchakuTextEncoderLoader"],
        covered=["UNETLoader"])
    assert found and found[0] == checks.WARN, found
    assert "GGUFLoader" in found[1] and "NunchakuTextEncoderLoader" in found[1], found
    assert "UNETLoader," not in found[1], "a covered loader must not be listed as skipped"


def test_scope_line_lists_what_was_not_checked():
    line = checks.scope_line(
        [("UNETLoader", {"unet_name": "m.safetensors"}), ("KSampler", {"seed": 1}),
         ("VAELoader", {"vae_name": "v.safetensors"})],
        covered={"UNETLoader": ("unet_name", "weight_dtype")})
    assert "node types seen and not covered 2" in line, line
    assert "KSampler" in line and "VAELoader" in line, line


def test_scope_line_names_an_unchecked_widget_on_a_covered_class():
    """The general form of the DualCLIPLoader hole, and the thing the class unit cannot say.

    `DualCLIPLoader` is covered -- for clip_name1. A line that stops at the class prints that as a
    pass. The widget clause is what makes the second slot visible without anyone already knowing
    to look for it.
    """
    line = checks.scope_line(
        [("DualCLIPLoader", {"clip_name1": "a.safetensors", "clip_name2": "b.safetensors"})],
        covered={"DualCLIPLoader": ("clip_name1",)})
    assert "SAW AND DID NOT CHECK 1 file widget(s): DualCLIPLoader.clip_name2" in line, line
    assert "opened 1 file widget(s): DualCLIPLoader.clip_name1" in line, line


def test_scope_line_does_not_claim_to_have_opened_a_gguf():
    """The blind spot in the test above, and it is the third instance of one shape.

    The test above varies the widget NAME and holds the extension constant, so it cannot see a
    widget that is fully covered and still never opened. `_make_validator` opens only
    `.safetensors`; `MODEL_FILE_SUFFIXES` recognises seven. So six of the seven reached the
    `checked` bucket -- the line said `opened` about files nothing ever read.

    Not hypothetical on this bench: 35 of the 159 files in `quantization_inventory.json` are
    non-safetensors (10 .gguf, 14 .pth, 7 .onnx, 4 .pt, 3 .ckpt, 1 .bin), and a .gguf UNET is a
    normal workflow here.

    The shape to notice, because it has now cost three fixtures: **a test that varies the axis the
    bug is not on.** An earlier one varied tensor shape while the collision lived in identical
    shapes; this one varied the widget name while the hole lived in the suffix. Vary the other
    axis, and say which axis you varied.
    """
    line = checks.scope_line(
        [("UNETLoader", {"unet_name": "flux-q4.gguf"})],
        covered={"UNETLoader": ("unet_name",)})
    assert "opened 0 file widget(s)" in line, line
    assert "SAW AND DID NOT CHECK 1 file widget(s): UNETLoader.unet_name (.gguf)" in line, line

    # ...and a covered widget carrying a file it DOES open is still reported as opened, so the
    # fix narrows the claim rather than emptying it.
    line = checks.scope_line(
        [("UNETLoader", {"unet_name": "model.safetensors"})],
        covered={"UNETLoader": ("unet_name",)})
    assert "opened 1 file widget(s): UNETLoader.unet_name" in line, line

    # One widget, two nodes, two extensions: genuinely two coverage situations, reported as two.
    line = checks.scope_line(
        [("UNETLoader", {"unet_name": "a.safetensors"}),
         ("UNETLoader", {"unet_name": "b.pth"})],
        covered={"UNETLoader": ("unet_name",)})
    assert "opened 1 file widget(s): UNETLoader.unet_name" in line, line
    assert "SAW AND DID NOT CHECK 1 file widget(s): UNETLoader.unet_name (.pth)" in line, line


def test_a_raw_prompt_dict_does_not_become_node_ids():
    """`_pairs` iterating a dict yields its KEYS, so every node id printed as a class type.

    Wrong and confident, which is worse than the ValueError `_pairs` exists to avoid -- so the
    prompt shape is unwrapped rather than raised on. ComfyUI's prompt is exactly
    `{id: {class_type, inputs}}`, which is the shape a caller most plausibly passes by mistake.
    """
    prompt = {"3": {"class_type": "UNETLoader", "inputs": {"unet_name": "a.safetensors"}},
              "7": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    line = checks.scope_line(prompt, covered={"UNETLoader": ("unet_name",)})
    assert "opened 1 file widget(s): UNETLoader.unet_name" in line, line
    assert "node types seen and not covered 1: KSampler" in line, line
    assert '"3"' not in line and " 3," not in line and " 7," not in line, (
        "node ids leaked into the scope line as class types")


def test_a_class_installed_without_an_audit_is_marked_not_counted():
    """`covered` includes classes whose widget names INPUT_TYPES could not confirm.

    That fail-open is deliberate -- declining to install would turn an unreadable class into an
    unchecked one. But if the audit failed BECAUSE upstream renamed the widget, the validator
    reads the old name, gets None, and returns True forever, while the scope line counts it as
    opened. The line has to say so, or it vouches instead of reporting.
    """
    graph = [("UNETLoader", {"unet_name": "a.safetensors"})]
    plain = checks.scope_line(graph, covered={"UNETLoader": ("unet_name",)})
    assert "NOT CONFIRMED" not in plain, plain

    marked = checks.scope_line(graph, covered={"UNETLoader": ("unet_name",)},
                               unaudited=("UNETLoader",))
    assert "opened 1 file widget(s): UNETLoader.unet_name" in marked, marked
    assert "WIDGET NAMES NOT CONFIRMED against INPUT_TYPES for 1: UNETLoader" in marked, marked

    # A class that is unaudited but absent from THIS graph must not be named: the line describes
    # the run, not the registry.
    elsewhere = checks.scope_line(graph, covered={"UNETLoader": ("unet_name",)},
                                  unaudited=("CLIPLoader",))
    assert "NOT CONFIRMED" not in elsewhere, elsewhere


def test_the_validator_and_the_coverage_line_read_one_tuple():
    """They were two literals and they drifted. This is the assertion that they cannot again."""
    import __init__ as pkg  # noqa: F401 -- imported for the side effect of being importable
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "checks.CHECKED_FILE_SUFFIXES" in source, (
        "the validator must read the same tuple the coverage line does, not its own literal")
    assert '.suffix.lower() != ".safetensors"' not in source, (
        "the old literal is back; the log will start claiming `opened` about files it skips")


def test_scope_line_survives_a_bare_class_name_list():
    """It must degrade, not raise. `scope_line` runs inside the wrapped `execution.validate_prompt`
    where nothing catches an exception, so a shape it does not expect must cost the widget clause
    and not every prompt validation on the server."""
    line = checks.scope_line(["UNETLoader", "KSampler"], covered={"UNETLoader": ("unet_name",)})
    assert "node types checked 1: UNETLoader" in line, line
    assert "SAW AND DID NOT CHECK 0 file widget(s)" in line, line


def test_scope_line_treats_a_bare_class_list_as_covering_no_widget():
    """An older `covered` shape must read as a gap, not as coverage.

    This is the fail direction that matters: if the widget map is ever lost or half-populated,
    the line has to say the files were not opened, not stay quiet about them.
    """
    line = checks.scope_line([("UNETLoader", {"unet_name": "m.safetensors"})],
                             covered=["UNETLoader"])
    assert "SAW AND DID NOT CHECK 1 file widget(s): UNETLoader.unet_name" in line, line


def test_uncovered_loaders_warns_about_a_widget_on_a_covered_class():
    found = checks.check_uncovered_loaders(
        ["DualCLIPLoader"], covered={"DualCLIPLoader": ("clip_name1",)},
        nodes=[("DualCLIPLoader", {"clip_name1": "a.safetensors",
                                   "clip_name2": "b.safetensors"})])
    assert found and found[0] == checks.WARN, found
    assert "DualCLIPLoader.clip_name2" in found[1], found
    assert "DualCLIPLoader.clip_name1" not in found[1], "a checked widget must not be listed"


def test_unresolved_names_the_symbol_and_the_exception():
    finding = checks.unresolved("folder_paths.get_full_path", KeyError("diffusion_models"),
                                "every file-level check")
    assert finding[0] == checks.WARN
    assert "folder_paths.get_full_path" in finding[1]
    assert "KeyError" in finding[1]
    assert "DID NOT RUN" in finding[1]


def test_nunchaku_check_is_silent_without_a_nunchaku_loader():
    assert checks.check_nunchaku_needs_disable_dynamic_vram(["UNETLoader", "KSampler"]) is None


def test_lora_check_needs_both_halves():
    assert checks.check_lora_over_quantized(["LoraLoader"], quantized_files=0) is None
    assert checks.check_lora_over_quantized(["KSampler"], quantized_files=3) is None
    found = checks.check_lora_over_quantized(["LoraLoader"], quantized_files=1)
    # Still WARN, but now for the opposite reason: the dequantization it warned about was measured
    # on 2026-08-19 and did not reproduce, so blocking on it would be blocking on a refuted claim.
    # What survives is the unmeasured half -- accuracy of a LoRA delta over a 4-bit weight.
    assert found and found[0] == checks.WARN, "a refuted finding must not block either"
    assert "did not reproduce" in found[1], "the message must carry the measurement, not the fear"


# --------------------------------------------------------------------------------------------
# Installation-time fail-opens. These need a registry to inject into, so they get a fake one.
# --------------------------------------------------------------------------------------------

def test_injection_refuses_a_loader_whose_widget_was_renamed():
    """The class surviving an upgrade is not the same as its widget surviving one.

    A validator keyed on a widget that no longer exists reads None out of kwargs and returns True
    on every workflow forever -- installed, counted as injected, and structurally unable to fire.
    """
    with capture_logs() as log:
        with fake_comfyui({"UNETLoader": RenamedUNETLoader}) as (module, nodes_mod, _):
            assert getattr(RenamedUNETLoader, "VALIDATE_INPUTS", None) is None, \
                "a check that could never fire must not be installed"
            assert "UNETLoader" not in module._STATUS["covered"]
    assert any("WIDGET RENAMED" in m and "unet_name" in m for m in log.messages()), log.messages()


def test_a_loader_class_that_vanished_is_named():
    with capture_logs() as log:
        with fake_comfyui({}) as (module, _, _e):
            assert module._STATUS["covered"] == {}
    warnings = log.messages()
    assert any("NOT FOUND" in m and "UNETLoader" in m for m in warnings), warnings


def test_out_of_scope_loaders_are_named_at_boot():
    """The table covers four classes. ComfyUI's own nodes.py defines fifteen classes ending in
    'Loader' and custom_nodes adds more; a workflow on any of them got a clean pass that meant
    nothing."""
    with capture_logs() as log:
        with fake_comfyui({"UNETLoader": FakeUNETLoader,
                           "GGUFLoader": FakeUNETLoader}) as (module, _, _e):
            assert module._STATUS["covered"] == {
                "UNETLoader": ("unet_name", "weight_dtype")}, module._STATUS["covered"]
    assert any("OUT OF SCOPE" in m and "GGUFLoader" in m for m in log.messages()), log.messages()


def test_both_widgets_of_a_two_widget_loader_are_actually_checked():
    """The regression test for the fix that was not one.

    LOADER_TABLE gained a `DualCLIPLoader.clip_name2` row and the hole stayed open: injection ran
    per table entry, so the second row found the VALIDATE_INPUTS the first row had just installed,
    took the "another package got there first" branch, and clip_name2 was never resolved -- while
    the class read as covered. Measured before the fix: get_full_path called once.

    Asserts on the RESOLUTIONS, not on the return value. A validator that silently skipped a
    widget would return True here exactly like one that checked it and found nothing.
    """
    resolved = []

    def spy(folder, name):
        resolved.append((folder, name))
        return None

    with capture_logs() as log:
        with fake_comfyui({"DualCLIPLoader": FakeDualCLIPLoader},
                          get_full_path=spy) as (module, _, _e):
            assert module._STATUS["covered"] == {
                "DualCLIPLoader": ("clip_name1", "clip_name2")}, module._STATUS["covered"]
            FakeDualCLIPLoader.VALIDATE_INPUTS(clip_name1="a.safetensors",
                                               clip_name2="b.safetensors")
    assert resolved == [("text_encoders", "a.safetensors"),
                        ("text_encoders", "b.safetensors")], resolved
    assert not any("ALREADY VALIDATED" in m for m in log.messages()), \
        "the package must not report its own validator as another package's"


def test_a_dead_widget_costs_its_own_check_and_not_its_siblings():
    """One renamed widget on a two-widget class used to drop that entry; grouping must not turn
    that into dropping the class, and must not turn it into pretending the dead one is covered."""
    class HalfRenamedDualCLIPLoader:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"clip_name1": (["a.safetensors"],), "type": (["flux"],)}}

    with capture_logs() as log:
        with fake_comfyui({"DualCLIPLoader": HalfRenamedDualCLIPLoader}) as (module, _, _e):
            assert module._STATUS["covered"] == {
                "DualCLIPLoader": ("clip_name1",)}, module._STATUS["covered"]
            line = checks.scope_line(
                [("DualCLIPLoader", {"clip_name1": "a.safetensors",
                                     "clip_name2": "b.safetensors"})],
                module._STATUS["covered"])
    assert "DualCLIPLoader.clip_name2" in line and "SAW AND DID NOT CHECK 1" in line, line
    assert any("WIDGET RENAMED" in m and "clip_name2" in m for m in log.messages()), \
        log.messages()


def test_resolve_failure_warns_instead_of_passing_the_node_clean():
    """`folder_paths` changing shape used to return None, and None meant 'no file to check'."""
    def broken(folder, name):
        raise RuntimeError("folder_paths moved")

    # capture_logs wraps the import too, so the boot lines land in the handler instead of on
    # stderr, where a reader scanning for FAIL sees a wall of warnings and assumes the worst.
    with capture_logs() as log:
        with fake_comfyui({"UNETLoader": FakeUNETLoader}, get_full_path=broken):
            result = FakeUNETLoader.VALIDATE_INPUTS(unet_name="whatever.safetensors",
                                                    weight_dtype="fp8_e4m3fn")
    assert result is True, "an unresolvable path must not block the user's run"
    assert any("DID NOT RUN" in m and "RuntimeError" in m for m in log.messages()), log.messages()


def test_every_run_states_which_node_types_it_did_not_check():
    prompt = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m.safetensors"}},
              "2": {"class_type": "KSampler", "inputs": {}},
              "3": {"class_type": "GGUFLoader", "inputs": {}}}
    with capture_logs() as log:
        with fake_comfyui({"UNETLoader": FakeUNETLoader}) as (module, _, execution_mod):
            result = asyncio.run(execution_mod.validate_prompt("p", prompt))
    assert result[0] is True, "stating scope must not block anything"
    scope = [m for m in log.messages(logging.INFO) if "node types seen and not covered" in m]
    assert scope, log.messages(logging.INFO)
    assert "KSampler" in scope[0] and "GGUFLoader" in scope[0], scope
    assert "opened 1 file widget(s): UNETLoader.unet_name" in scope[0], scope
    assert any("OUTSIDE this package's scope" in m and "GGUFLoader" in m
               for m in log.messages()), log.messages()


def test_scope_is_stated_even_when_the_run_already_failed_elsewhere():
    """Otherwise the only runs that admit what was skipped are the ones that got far enough to
    pass, and the criterion is 'every run'."""
    with capture_logs() as log:
        with fake_comfyui({"UNETLoader": FakeUNETLoader},
                          prompt_ok=False) as (module, _, execution_mod):
            result = asyncio.run(execution_mod.validate_prompt(
                "p", {"1": {"class_type": "KSampler", "inputs": {}}}))
    assert result[0] is False, "our wrapper must not rescue a run that failed upstream"
    assert any("node types seen and not covered" in m and "KSampler" in m
               for m in log.messages(logging.INFO)), log.messages(logging.INFO)


def test_one_half_failing_to_install_does_not_cancel_the_other():
    """A single try around both meant a broken _inject() also skipped the graph wrapper, leaving a
    server with no preflight at all and one boot line as the only evidence."""
    with capture_logs() as log:
        with fake_comfyui({"UNETLoader": FakeUNETLoader}, break_nodes=True) as (module, _, _e):
            assert module._STATUS["file_checks"].startswith("FAILED"), module._STATUS
            assert module._STATUS["graph_checks"].startswith("installed"), module._STATUS
    assert any("NOT RUNNING" in m for m in log.messages()), log.messages()


def test_a_run_with_no_file_checks_installed_says_so_on_that_run():
    with capture_logs() as log:
        with fake_comfyui({}) as (module, _, execution_mod):
            asyncio.run(execution_mod.validate_prompt(
                "p", {"1": {"class_type": "KSampler", "inputs": {}}}))
    assert any("no per-file check is installed" in m for m in log.messages()), log.messages()


def test_against_the_real_checkpoints_on_this_machine():
    """The mixed checkpoint this project produced must pass; a bare BF16 must pass."""
    seen = 0
    for name in ("zimage-v2-mixed.safetensors", "zimage-v2-w4a4.safetensors",
                 "beyond-reality-zimage-v2_native.safetensors"):
        path = MODELS / name
        if not path.is_file():
            continue
        seen += 1
        found = checks.check_file(path)
        blocking = [m for s, m in found if s == checks.ERROR]
        assert not blocking, f"{name} was blocked by {blocking}"

    # And the one that genuinely is diffusers-named must be caught.
    source = MODELS / "beyond-reality-zimage-v2_bf16.safetensors"
    if source.is_file():
        seen += 1
        # It is not quantized, so it must NOT be flagged -- the check is about quantized files in
        # diffusers naming, not about diffusers naming as such.
        assert checks.check_file(source) == [], "an unquantized diffusers file was flagged"
    # A test named "against the real checkpoints" that silently exercises none is worse than no
    # test: it prints PASS. If the models directory is genuinely absent this should be visible as
    # a failure and fixed, not skipped.
    assert MODELS.is_dir(), f"{MODELS} does not exist; this test checked nothing"
    assert seen, f"no checkpoint found under {MODELS}; this test checked nothing"
    print(f"      ({seen} real checkpoint(s) exercised)")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    # This used to say check_dtype_widget was a reading and had never been executed. It has been,
    # on 2026-08-19 -- see the docstring for the table. What is still untested is listed instead,
    # because the point of the line is to name the gap, not to be reassuring once one gap closes.
    print("\nNOT COVERED BY THESE TESTS:")
    print("  * Every check here runs against file headers and node lists. None of them loads a "
          "model, so none proves what the loader does -- that lives in tools/dispatch_census.py.")
    print("  * check_lora_over_quantized: dispatch was measured, accuracy was not. A LoRA delta "
          "over an already-4-bit weight may cost quality that no count would show.")
    print("  * Only ConvRot W4A4 and AsymW4A8Int8 have been exercised on a real model. The other "
          "five formats in QUANT_ALGOS have not.")
    print("  * The injection and graph-wrapper tests run against a FAKE `nodes` and `execution`. "
          "They prove the package's own logic, not that ComfyUI 0.33 still calls VALIDATE_INPUTS "
          "the way this package assumes -- that needs a real boot and a real prompt submission.")
    print("  * The rename paths are simulated. No upstream rename has actually occurred here; "
          "what is tested is what this package does when told one has.")
    print("  * The widget scope line finds file widgets by the SUFFIX of the value in the graph "
          "(MODEL_FILE_SUFFIXES). A loader whose widget holds a bare name, a directory, or an "
          "extension not in that tuple is invisible to it -- so the line under-reports the gap "
          "rather than over-reporting it, and no test here exercises such a loader because none "
          "of the four in LOADER_TABLE behaves that way in ComfyUI 0.33.")
    raise SystemExit(1 if failures else 0)
