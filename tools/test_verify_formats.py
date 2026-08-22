"""Tests for verify_w4a4.py's three format adapters, run without a GPU and without torch.

Why synthetic files instead of the real checkpoints: the check that matters most here is item 3 of
the ticket -- flip one byte inside a *preserved* tensor and confirm the byte-identical comparison
catches it. Doing that on a real pair means copying `zimage-v2-mixed.safetensors` (3.41 GB) and its
source `beyond-reality-zimage-v2_native.safetensors` (12.31 GB), and this bench's hard rule is
never to write near a model file. The synthetic pairs below carry the same header structure, the
same metadata, and the same preserved/quantized split at 4 KiB a layer, so the flipped byte lands
in exactly the code path a 3 GB copy would have exercised.

What these do NOT cover: any kernel. `--kernel-smoke`, backend resolution and the numeric ceiling
all need CUDA and none of them is exercised here. Run with:

    .\\python_embeded\\python.exe -s .\\tools\\test_verify_formats.py

There is no pytest in the embedded interpreter (CLAUDE.md), hence the hand-rolled runner at the
bottom rather than a fixture.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_w4a4 import (  # noqa: E402
    Coverage,
    FORMATS,
    data_start,
    read_header,
    validate_preserved_bytes,
    validate_structure,
)

ELEMENT_SIZE = {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4}

ROWS, COLS, GROUP_SIZE, CONVROT = 8, 256, 16, 256

# Two layers get quantized, three tensors are preserved verbatim. `blocks.0.mlp.bias` is
# deliberately a sibling of a quantized weight: the extra-key walk has to tell "auxiliary of a
# selected layer" from "tensor that merely shares a prefix".
QUANT_LAYERS = ["blocks.0.mlp", "blocks.1.attn.out"]
PRESERVED = {
    "final_norm.weight": ("BF16", [ROWS]),
    "blocks.0.mlp.bias": ("BF16", [ROWS]),
    "pos_embed": ("F32", [4, 4]),
}


def write_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    """`tensors` maps name -> (dtype string, shape, payload bytes)."""
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(payload)]}
        blobs.append(payload)
        offset += len(payload)
    if metadata:
        header = {"__metadata__": metadata, **header}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def filler(dtype: str, shape: list[int], seed: int) -> bytes:
    count = 1
    for dim in shape:
        count *= dim
    size = count * ELEMENT_SIZE[dtype]
    return bytes((seed * 37 + index * 11) % 251 + 1 for index in range(size))


def build_source(path: Path) -> None:
    tensors = {}
    for index, layer in enumerate(QUANT_LAYERS):
        tensors[f"{layer}.weight"] = ("BF16", [ROWS, COLS],
                                      filler("BF16", [ROWS, COLS], index + 1))
    for index, (name, (dtype, shape)) in enumerate(PRESERVED.items()):
        tensors[name] = (dtype, shape, filler(dtype, shape, index + 40))
    write_safetensors(path, tensors)


def quantized_tensors(fmt_name: str, layer: str, seed: int) -> dict:
    """The auxiliary tensors each converter really writes, at the shapes read off real files."""
    if fmt_name == "convrot_w4a4":
        return {
            f"{layer}.weight": ("I8", [ROWS, COLS // 2], filler("I8", [ROWS, COLS // 2], seed)),
            f"{layer}.weight_scale": ("F32", [ROWS], filler("F32", [ROWS], seed + 1)),
        }
    if fmt_name == "asym_w4a8_int8":
        groups = COLS // GROUP_SIZE
        return {
            f"{layer}.weight": ("I8", [ROWS, COLS // 2], filler("I8", [ROWS, COLS // 2], seed)),
            f"{layer}.weight_s_rel": ("U8", [ROWS, groups], filler("U8", [ROWS, groups], seed + 1)),
            f"{layer}.weight_s_channel": ("F32", [ROWS], filler("F32", [ROWS], seed + 2)),
            f"{layer}.weight_codebook": ("F32", [16], filler("F32", [16], seed + 3)),
        }
    if fmt_name == "int8_tensorwise":
        return {
            f"{layer}.weight": ("I8", [ROWS, COLS], filler("I8", [ROWS, COLS], seed)),
            f"{layer}.weight_scale": ("F32", [ROWS, 1], filler("F32", [ROWS, 1], seed + 1)),
        }
    raise AssertionError(fmt_name)


LAYER_CONFIG = {
    "convrot_w4a4": {"format": "convrot_w4a4", "convrot_groupsize": CONVROT},
    "asym_w4a8_int8": {"format": "asym_w4a8_int8", "group_size": GROUP_SIZE,
                       "convrot_groupsize": CONVROT},
    "int8_tensorwise": {"format": "int8_tensorwise", "convrot": True,
                        "convrot_groupsize": CONVROT},
}


def build_output(path: Path, source: Path, formats: list[str]) -> None:
    """One output per format, plus a mixed one when `formats` names more than one.

    Preserved tensors are copied byte-for-byte out of the source file, which is what makes the
    flipped-byte test meaningful: without a real copy the comparison would be trivially satisfied.
    """
    source_header, _ = read_header(source)
    base = data_start(source)
    tensors, layers = {}, {}
    with source.open("rb") as handle:
        for index, layer in enumerate(QUANT_LAYERS):
            fmt_name = formats[index % len(formats)]
            tensors.update(quantized_tensors(fmt_name, layer, index + 1))
            layers[layer] = dict(LAYER_CONFIG[fmt_name])
        for name in PRESERVED:
            info = source_header[name]
            start, end = info["data_offsets"]
            handle.seek(base + start)
            tensors[name] = (info["dtype"], info["shape"], handle.read(end - start))
    write_safetensors(path, tensors, metadata={
        "_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers},
                                             separators=(",", ":")),
        "quantization": "+".join(formats),
    })


def flip_one_preserved_byte(path: Path, tensor: str = "pos_embed") -> None:
    header, _ = read_header(path)
    offset = data_start(path) + header[tensor]["data_offsets"][0]
    with path.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([original[0] ^ 0x01]))


def check(model: Path, source: Path) -> tuple[list[str], list[str]]:
    model_header, metadata = read_header(model)
    source_header, _ = read_header(source)
    structural = validate_structure(model_header, metadata, source_header)
    if structural:
        return structural, []
    layers = json.loads(metadata["_quantization_metadata"])["layers"]
    return [], validate_preserved_bytes(model, source, model_header, source_header, layers)


# ---- the tests -------------------------------------------------------------------------------

def test_each_format_passes(tmp: Path) -> None:
    source = tmp / "src.safetensors"
    build_source(source)
    for fmt_name in FORMATS:
        model = tmp / f"out_{fmt_name}.safetensors"
        build_output(model, source, [fmt_name])
        structural, preserved = check(model, source)
        assert not structural, (fmt_name, structural)
        assert not preserved, (fmt_name, preserved)


def test_mixed_file_passes(tmp: Path) -> None:
    """The case the old verifier could not reach at all: two formats in one checkpoint."""
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "out_mixed.safetensors"
    build_output(model, source, ["convrot_w4a4", "asym_w4a8_int8"])
    structural, preserved = check(model, source)
    assert not structural, structural
    assert not preserved, preserved


def test_flipped_preserved_byte_is_caught(tmp: Path) -> None:
    """Criterion item 3. The corruption is one bit in a tensor the converter only copies.

    Structural verification must still pass -- dtype, shape and offsets are all untouched -- so
    this is precisely the failure only the byte comparison can see.
    """
    source = tmp / "src.safetensors"
    build_source(source)
    for formats in (["convrot_w4a4"], ["asym_w4a8_int8"], ["int8_tensorwise"],
                    ["convrot_w4a4", "asym_w4a8_int8"]):
        model = tmp / ("corrupt_" + "_".join(formats) + ".safetensors")
        build_output(model, source, formats)
        flip_one_preserved_byte(model)
        structural, preserved = check(model, source)
        assert not structural, (formats, "corruption leaked into the structural stage", structural)
        assert preserved == ["pos_embed: preserved tensor bytes changed"], (formats, preserved)


def test_missing_required_auxiliary_is_caught(tmp: Path) -> None:
    source = tmp / "src.safetensors"
    build_source(source)
    for fmt_name, dropped in (("convrot_w4a4", "weight_scale"),
                              ("asym_w4a8_int8", "weight_s_channel"),
                              ("int8_tensorwise", "weight_scale")):
        model = tmp / f"missing_{fmt_name}.safetensors"
        build_output(model, source, [fmt_name])
        model_header, metadata = read_header(model)
        target = f"{QUANT_LAYERS[0]}.{dropped}"
        del model_header[target]
        errors = validate_structure(model_header, metadata, read_header(source)[0])
        assert any(target in error and "missing" in error for error in errors), (fmt_name, errors)


def test_optional_codebook_may_be_absent(tmp: Path) -> None:
    """quant_w4a8.py --no-codebook writes no weight_codebook; that is not corruption."""
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "nocodebook.safetensors"
    build_output(model, source, ["asym_w4a8_int8"])
    model_header, metadata = read_header(model)
    for layer in QUANT_LAYERS:
        del model_header[f"{layer}.weight_codebook"]
    assert not validate_structure(model_header, metadata, read_header(source)[0])


def test_wrong_scale_shape_is_caught(tmp: Path) -> None:
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "badscale.safetensors"
    build_output(model, source, ["asym_w4a8_int8"])
    model_header, metadata = read_header(model)
    # s_rel must be K/group_size wide; claim a different group count.
    model_header[f"{QUANT_LAYERS[0]}.weight_s_rel"]["shape"] = [ROWS, 4]
    errors = validate_structure(model_header, metadata, read_header(source)[0])
    assert any("weight_s_rel" in error and "shape" in error for error in errors), errors


def test_int8_scale_is_not_forced_to_one_dimension(tmp: Path) -> None:
    """[N, 1] is what both LTX int8 checkpoints on this bench carry; [N] must also pass.

    The old per-layer rule demanded a 1D scale, which would have rejected every int8_tensorwise
    file on disk on a shape that is correct.
    """
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "int8_flat.safetensors"
    build_output(model, source, ["int8_tensorwise"])
    model_header, metadata = read_header(model)
    for layer in QUANT_LAYERS:
        model_header[f"{layer}.weight_scale"]["shape"] = [ROWS]
    assert not validate_structure(model_header, metadata, read_header(source)[0])


def test_stray_output_key_is_caught(tmp: Path) -> None:
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "stray.safetensors"
    build_output(model, source, ["convrot_w4a4"])
    model_header, metadata = read_header(model)
    model_header[f"{QUANT_LAYERS[0]}.weight_smuggled"] = {"dtype": "F32", "shape": [ROWS],
                                                          "data_offsets": [0, 32]}
    errors = validate_structure(model_header, metadata, read_header(source)[0])
    assert any("weight_smuggled" in error for error in errors), errors


def test_wrong_packed_width_is_caught(tmp: Path) -> None:
    """int8_tensorwise does not halve the columns and the other two do; the adapter must not
    accept a W4A4 file that forgot to pack."""
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "unpacked.safetensors"
    build_output(model, source, ["convrot_w4a4"])
    model_header, metadata = read_header(model)
    model_header[f"{QUANT_LAYERS[0]}.weight"]["shape"] = [ROWS, COLS]
    errors = validate_structure(model_header, metadata, read_header(source)[0])
    assert any("packed shape" in error for error in errors), errors


def test_unknown_format_still_rejected(tmp: Path) -> None:
    source = tmp / "src.safetensors"
    build_source(source)
    model = tmp / "unknown.safetensors"
    build_output(model, source, ["convrot_w4a4"])
    model_header, metadata = read_header(model)
    quant = json.loads(metadata["_quantization_metadata"])
    quant["layers"][QUANT_LAYERS[0]]["format"] = "nvfp4"
    metadata["_quantization_metadata"] = json.dumps(quant)
    errors = validate_structure(model_header, metadata, read_header(source)[0])
    assert any("nvfp4" in error for error in errors), errors


def test_smoke_refuses_to_guess_a_missing_groupsize(tmp: Path) -> None:
    """Criterion item 1, and the line between the parameter that IS in the file and the one that is not.

    Rewritten 2026-08-22. The first version required BOTH `convrot_groupsize` and
    `quant_group_size` from the metadata. That is stricter than the runtime and it broke
    `--kernel-smoke` on every convrot_w4a4 checkpoint on this bench, including the invocation
    printed in CLAUDE.md.

    `ComfyUI/comfy/ops.py:1195-1203` reads `convrot_groupsize` and `linear_dtype` from the file
    and sets `"quant_group_size": 64` as a **literal**. Confirmed by reading the produced headers:
    every per-layer config is `{format, convrot_groupsize}`. So `quant_group_size` is a property of
    the loader, not of a checkpoint, and this smoke reproduces the loader.

    Calls the adapter directly rather than kernel_smoke(), which would need CUDA.
    """
    fmt = FORMATS["convrot_w4a4"]

    def explode(*args, **kwargs):  # noqa: ARG001 - must never be called
        raise AssertionError("the smoke read a tensor before checking the groupsize")

    # convrot_groupsize IS in the file, and getting it wrong is what the ticket measured
    # (1.023 against a 0.9 ceiling). Absent means refuse.
    try:
        fmt.smoke_kwargs("layers.0.attn.out", {"format": "convrot_w4a4"}, explode, None, {})
    except SystemExit as error:
        assert "will not guess" in str(error), error
    else:
        raise AssertionError("no refusal for a config with no convrot_groupsize")

    # quant_group_size is NOT in the file and must not be demanded from it.
    seen = {}
    kwargs = fmt.smoke_kwargs("layers.0.attn.out",
                              {"format": "convrot_w4a4", "convrot_groupsize": 64},
                              lambda suffix, **kw: seen.setdefault(suffix, object()), None, {})
    assert seen.keys() == {".weight", ".weight_scale"}, seen
    assert kwargs["convrot_groupsize"] == 64, "must come from the file"
    assert kwargs["quant_group_size"] == 64, "must come from the loader's literal"

    # And it must track ops.py rather than a coincidence: both are 64 above, so pin the source.
    import verify_w4a4
    assert verify_w4a4.LOADER_QUANT_GROUP_SIZE == 64, (
        "if ComfyUI/comfy/ops.py stops hardcoding 64, this constant and its comment are what "
        "have to change")


def test_smoke_kwargs_match_the_comfy_kitchen_signatures(tmp: Path) -> None:
    """A wrong kwarg name would only surface on a machine with a free GPU, so pin the names here.

    The expected sets were READ off comfy_kitchen 0.2.31 on 2026-08-22, not executed:
    tensor/convrot_w4a4.py:80, tensor/w4a8_int8.py:93, __init__.py:807. `input_act` is in the
    int8 set because ck.int8_linear always passes it to registry.get_implementation, so a kwargs
    dict without it resolves against a different signature than the call runs under. This test
    does not import comfy_kitchen -- doing so loads the CUDA extension.
    """
    expected = {
        "convrot_w4a4": {"x", "qweight", "wscales", "bias", "convrot_groupsize",
                         "quant_group_size", "linear_dtype"},
        "asym_w4a8_int8": {"x", "qdata", "s_rel", "s_channel", "codebook", "correction", "bias",
                           "group_size", "convrot_groupsize", "out_dtype"},
        "int8_tensorwise": {"x", "weight", "weight_scale", "bias", "out_dtype", "convrot",
                            "convrot_groupsize", "input_act"},
    }

    class FakeTensor:
        dtype = "bfloat16"

    for fmt_name, names in expected.items():
        config = dict(LAYER_CONFIG[fmt_name])
        if fmt_name == "convrot_w4a4":
            config["quant_group_size"] = 64
        kwargs = FORMATS[fmt_name].smoke_kwargs(
            QUANT_LAYERS[0], config, lambda suffix, **kw: FakeTensor(), FakeTensor(),
            {"assume_quant_group_size": None})
        assert set(kwargs) == names, (fmt_name, set(kwargs) ^ names)


def test_coverage_block_renders_on_every_path(tmp: Path) -> None:
    coverage = Coverage()
    coverage.note("something")
    rendered = coverage.render()
    assert "NOT COVERED BY THIS RUN" in rendered and "- something" in rendered


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="verify_formats_") as raw:
        tmp = Path(raw)
        for test in tests:
            try:
                test(tmp)
            except Exception as error:  # noqa: BLE001 - a runner, not a library
                failures += 1
                print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
            else:
                print(f"PASS {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    print("\nNOT COVERED BY THIS RUN:\n"
          "  - no kernel executed and no CUDA touched: --kernel-smoke, backend resolution and the\n"
          "    RMSE ceiling are all untested here. The w4a8 and int8 smoke paths in verify_w4a4.py\n"
          "    have never been run at all.\n"
          "  - synthetic 4 KiB checkpoints, not the real 3.41 GB zimage-v2-mixed.safetensors. The\n"
          "    structural half against the real files is a separate, manual invocation:\n"
          "    python_embeded\\python.exe -s tools\\verify_w4a4.py <model> --structural-only")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
