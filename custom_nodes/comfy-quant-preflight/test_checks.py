"""CPU tests for the preflight checks, run against the real files on this machine.

    python_embeded\\python.exe -s ComfyUI\\custom_nodes\\comfy-quant-preflight\\test_checks.py

Synthetic headers are built for the cases no file here exhibits (a weight_correction tensor, an
inert full-precision flag), because "we have no example" is not evidence that a check works.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
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


def test_unparseable_file_says_nothing():
    """GGUF and .pt come through the same call. Inventing a verdict for them is worse than
    staying quiet."""
    path = Path(tempfile.mkdtemp()) / "not.safetensors"
    path.write_bytes(b"this is not a safetensors file at all")
    assert checks.check_file(path) == []


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
    raise SystemExit(1 if failures else 0)
