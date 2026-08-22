"""Provenance and non-finite-error guards in quant_mixed.py, on synthetic inputs only.

No GPU, no checkpoint. Everything here builds tiny safetensors files in a temp directory and
calls the guards directly, because the failures being covered are decisions made *before* any
kernel runs -- which is the only reason they can be tested on this bench at all while the 3090
is shared. What is NOT covered: `measure_layer`'s actual kernel calls, and therefore whether a
real Z-Image layer still measures the same errors after the dtype change. That needs the card:

    python_embeded\\python.exe -s tools/quant_mixed.py --input <native.safetensors> \\
        --calibration calib/<fresh>.calib.pt --save-analysis calib/<fresh>.analysis.json --dry-run

pytest is not installed in the embedded interpreter and installing it is the owner's decision,
so this carries its own runner:

    python_embeded\\python.exe -s tools/test_quant_mixed_provenance.py
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import quant_mixed as qm  # noqa: E402
from calibrate_activations import (  # noqa: E402
    IDENTITY_SAMPLE_BYTES,
    safetensors_identity_digest,
)


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata=None) -> Path:
    """Minimal writer -- the real one is 60 lines of offset planning and is not what is tested."""
    header = {}
    offset = 0
    blobs = []
    for name, tensor in tensors.items():
        blob = memoryview(tensor.contiguous().numpy()).cast("B").tobytes()
        header[name] = {"dtype": qm.SAFETENSORS_DTYPE[tensor.dtype],
                        "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)
    return path


def test_same_basename_different_content_gets_a_different_digest(tmp: Path) -> None:
    """The whole point of item 1: this bench has the same filename in two directories."""
    a_dir, b_dir = tmp / "models", tmp / "mounted"
    a_dir.mkdir()
    b_dir.mkdir()
    a = write_safetensors(a_dir / "same_name.safetensors",
                          {"layers.0.attention.qkv.weight": torch.zeros(8, 4)})
    b = write_safetensors(b_dir / "same_name.safetensors",
                          {"layers.0.attention.qkv.weight": torch.zeros(16, 4)})
    assert a.name == b.name, "the fixture must reproduce the basename collision"
    assert safetensors_identity_digest(a) != safetensors_identity_digest(b)

    # Byte-identical files must agree, or every legitimate re-run would be refused.
    c = write_safetensors(tmp / "copy.safetensors",
                          {"layers.0.attention.qkv.weight": torch.zeros(8, 4)})
    assert safetensors_identity_digest(a) == safetensors_identity_digest(c)


def test_identical_layout_different_weights_gets_a_different_digest(tmp: Path) -> None:
    """The case that broke, and that the test above cannot see.

    The fixture above varies the tensor SHAPE, so the two files get different headers and a
    header-only digest separates them. Two checkpoints of the same architecture have byte-identical
    headers and different weights, and the first version of this guard hashed the header alone --
    so it called them the same file.

    MEASURED 2026-08-22 over the 45 `.safetensors` in ComfyUI/models/diffusion_models,
    ComfyUI/models/unet and D:/ComfyUI-Models/diffusion_models: header-only produced four colliding
    groups covering nine files, each group also identical in byte size --

        beyond-reality-zimage-v2_native / z_image_de_turbo_v1_bf16 / z_image_turbo_bf16
        beyond-reality-zimage-v2_bf16 / beyond-reality-recovered-bf16
        void_pass2 / void_pass1
        wan2.2_i2v_high_noise_14B_fp8_scaled / wan2.2_i2v_low_noise_14B_fp8_scaled

    -- the first group being the three checkpoints `--foreign-analysis`'s own help text names as
    different models. With the body sampled: 45 distinct, zero collisions.

    This synthetic pair reproduces the shape of that failure at 4 KiB, so the next time somebody
    decides the header is enough, a run of this file says otherwise.
    """
    torch.manual_seed(0)
    shape = (64, IDENTITY_SAMPLE_BYTES // 2 // 64 * 8)  # big enough that the sample windows land in it
    a = write_safetensors(tmp / "arch_a.safetensors",
                          {"layers.0.attention.qkv.weight": torch.randn(*shape)})
    b = write_safetensors(tmp / "arch_b.safetensors",
                          {"layers.0.attention.qkv.weight": torch.randn(*shape)})

    def header_of(path: Path) -> bytes:
        with path.open("rb") as fh:
            return fh.read(struct.unpack("<Q", fh.read(8))[0])

    assert header_of(a) == header_of(b), (
        "the fixture must produce byte-identical headers, or it is not testing the collision")
    assert a.stat().st_size == b.stat().st_size, "and identical sizes, as the real cases are"
    assert safetensors_identity_digest(a) != safetensors_identity_digest(b), (
        "identical layout, different weights, same digest -- the guard is header-only again")


def test_digest_refuses_a_file_that_is_not_safetensors(tmp: Path) -> None:
    junk = tmp / "not_a_checkpoint.ckpt"
    junk.write_bytes(b"\x80\x04\x95" + b"\xff" * 64)
    expect_refusal(lambda: safetensors_identity_digest(junk), "not a safetensors header")


def test_missing_provenance_key_is_a_refusal(tmp: Path) -> None:
    """Item 2. The old code read `if recorded is not None`, so absence passed every check."""
    assert qm.required({"group_size": 16}, "group_size", "x") == 16
    expect_refusal(lambda: qm.required({}, "source_identity_sha256", "analysis foo.json"),
                   "carries no 'source_identity_sha256'")
    # Present-but-null must refuse too: json.dumps of a None writes `null`, and a file that
    # records the key with no value has told us nothing.
    expect_refusal(lambda: qm.required({"measure_dtype": None}, "measure_dtype", "analysis foo"),
                   "carries no 'measure_dtype'")


def test_nan_and_inf_error_metrics_are_refused_by_name(tmp: Path) -> None:
    """Item 5, at the function that produces the numbers."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        message = expect_refusal(
            lambda: qm.finite("layers.0.feed_forward.w2", "err_w4a4", bad), "err_w4a4")
        assert "layers.0.feed_forward.w2" in message, "the failure must name the layer"
    assert qm.finite("layers.0.attention.qkv", "err_w4a4", 0.0459) == 0.0459


def test_json_nan_reaches_the_comparison_unless_the_analysis_is_validated(tmp: Path) -> None:
    """Item 5, at the reuse path -- the entry vector the measurement guard cannot see.

    `json.loads` accepts the bare token `NaN`. Asserted here rather than assumed, because the
    whole reason validate_analysis_rows exists is that this is true.
    """
    loaded = json.loads('{"err_w4a4": NaN}')
    assert loaded["err_w4a4"] != loaded["err_w4a4"], "json.loads no longer accepts NaN"
    # And `nan > threshold` is False, which is what routes the worst layer to the cheapest format.
    assert not (loaded["err_w4a4"] > 0.15)

    rows = [{"layer": "layers.0.feed_forward.w2", "shape": [3840, 10240], "calibrated": True,
             "err_bf16": 0.0015, "err_w4a4": loaded["err_w4a4"], "err_w4a8": 0.013}]
    message = expect_refusal(lambda: qm.validate_analysis_rows(rows, "analysis test.json"),
                             "err_w4a4")
    assert "layers.0.feed_forward.w2" in message


def test_validate_analysis_rows_accepts_a_real_row_and_an_uncalibrated_one(tmp: Path) -> None:
    qm.validate_analysis_rows(
        [{"layer": "context_refiner.0.attention.out", "shape": [3840, 3840],
          "calibrated": True, "err_bf16": 0.0015, "err_w4a4": 0.0459, "err_w4a8": 0.0135},
         # An uncalibrated row legitimately carries no error metrics.
         {"layer": "layers.29.feed_forward.w3", "shape": [3840, 10240], "calibrated": False}],
        "analysis test.json")

    # A calibrated row missing a metric is a refusal, not a row with two thirds of a decision.
    expect_refusal(
        lambda: qm.validate_analysis_rows(
            [{"layer": "a", "shape": [4, 4], "calibrated": True,
              "err_bf16": 0.1, "err_w4a4": 0.2}], "analysis test.json"),
        "has no 'err_w4a8'")
    # No shape means the analysis cannot be checked against the input's weights at all.
    expect_refusal(
        lambda: qm.validate_analysis_rows(
            [{"layer": "a", "calibrated": True,
              "err_bf16": 0.1, "err_w4a4": 0.2, "err_w4a8": 0.3}], "analysis test.json"),
        "has no 'shape'")


def test_measure_layer_refuses_the_checkpoint_dtype(tmp: Path) -> None:
    """Item 3. Runs on CPU: the dtype assertion fires before `ck` is touched, so a `ck` that
    would explode if called is the strongest available proof that it is not called."""
    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"measure_layer reached ck.{name} despite a bad dtype")

    weight = torch.zeros(8, 4, dtype=torch.float16)
    x = torch.zeros(2, 4, dtype=torch.float16)
    message = expect_refusal(
        lambda: qm.measure_layer("layers.0.feed_forward.w2", weight, x, Exploding(), 16, 256),
        "344064")
    assert "torch.float16" in message

    # And the half-cast case: weight promoted, activation left at the checkpoint's dtype, which
    # is precisely the shape of the original bug.
    expect_refusal(
        lambda: qm.measure_layer("layers.0.feed_forward.w2",
                                 weight.to(torch.bfloat16), x, Exploding(), 16, 256),
        "must run at torch.bfloat16")


def test_measure_dtype_is_not_fp16(tmp: Path) -> None:
    """A constant, but the one constant whose value the 344064 incident is about."""
    assert qm.MEASURE_DTYPE is torch.bfloat16
    assert torch.finfo(qm.MEASURE_DTYPE).max > 344064, (
        "the measurement dtype must hold real Z-Image activations; fp16's 65504 does not")


def expect_refusal(call, needle: str) -> str:
    try:
        call()
    except SystemExit as exit_:
        message = str(exit_)
        assert needle in message, f"expected {needle!r} in the refusal, got: {message}"
        return message
    raise AssertionError(f"expected a SystemExit mentioning {needle!r}, nothing was raised")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    with tempfile.TemporaryDirectory() as raw:
        for test in tests:
            root = Path(raw) / test.__name__
            root.mkdir()
            try:
                test(root)
                print(f"PASS  {test.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL  {test.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    print("NOT covered here: any kernel call in measure_layer, the real analysis/calibration "
          "round trip, and whether a real checkpoint still measures the same per-layer errors "
          "at bf16. Those need the GPU and a multi-GiB source.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
