"""Tests for the one native-backend probe, run without a GPU and without comfy_kitchen.

What this can and cannot check, stated up front because the answer is unusual for this file's
subject: `native_backend_ready()` resolves comfy-kitchen implementations on a CUDA device, so its
*answer* is untestable here. What IS testable, and is the whole content of ticket 05, is the
**question it asks** -- which ops, at which groupsizes, built from whose arguments. Every test
below reads the generated subprocess program rather than running it.

So a green run here means "every converter now asks about the configuration it is about to run".
It does NOT mean the backend resolves, that any kernel ran, or that the probe would pass on this
machine. `tools/probe_backend_resolution.py` is what settles those, and needs the card.

    .\\python_embeded\\python.exe -s tools\\test_native_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _native_probe  # noqa: E402
from _native_probe import (  # noqa: E402
    _OP_RECIPES,
    _need,
    _shape,
    build_probe_source,
)

ROOT = Path(r"F:/COMFY_PORTABLE")

# Every op some tool here dispatches, with a config that exercises a NON-default groupsize --
# 512/32 rather than 256/16 -- so a recipe that quietly ignores its argument and emits the old
# hardcoded literal fails instead of accidentally agreeing.
ALL_OPS = {
    "quantize_convrot_w4a4_weight": {"convrot_groupsize": 512},
    "convrot_w4a4_linear": {"convrot_groupsize": 512},
    "quantize_w4a8_int8_weight": {"convrot_groupsize": 512, "group_size": 32, "codebook": False},
    # `codebook` is required, not defaulted: it decides which implementation the registry picks,
    # and a probe that guesses it resolves a call the caller is not about to make. See
    # `_recipe_w4a8_linear`.
    "w4a8_int8_linear": {"convrot_groupsize": 512, "group_size": 32, "codebook": True},
    "quantize_int8_convrot_weight": {"convrot_groupsize": 512},
    "int8_linear": {"convrot": True, "convrot_groupsize": 512},
}


def test_every_recipe_is_exercised_here():
    """A new op added to _OP_RECIPES with no entry above would otherwise be tested by nothing."""
    assert set(ALL_OPS) == set(_OP_RECIPES), (
        f"ALL_OPS and _OP_RECIPES disagree: {set(ALL_OPS) ^ set(_OP_RECIPES)}")


def test_generated_source_compiles():
    source = build_probe_source(ROOT, ALL_OPS)
    compile(source, "<probe>", "exec")


def test_no_control_bytes_in_the_generated_source():
    """The Windows-path hazard, measured on this bench four times.

    A backslash that goes through a shell heredoc or sed becomes TAB, formfeed or 0x0F without
    anything raising. The path here is interpolated with `!r`, which is safe -- this test is what
    keeps it that way if someone switches it to concatenation.
    """
    source = build_probe_source(ROOT, ALL_OPS)
    bad = sorted({c for c in source if ord(c) < 0x20 and c != "\n"})
    assert not bad, f"control bytes in generated probe source: {[hex(ord(c)) for c in bad]}"


def test_the_comfyui_path_survives_as_a_real_path():
    source = build_probe_source(ROOT, ALL_OPS)
    line = next(l for l in source.splitlines() if l.startswith("sys.path.insert"))
    literal = line[line.index("(") + 1:line.rindex(")")].split(", ", 1)[1]
    assert Path(eval(literal)) == ROOT / "ComfyUI", literal  # noqa: S307 -- our own repr


def test_callers_groupsize_reaches_every_recipe():
    """The regression ticket 05 is about: a probe answering for a configuration nobody runs.

    `quant_w4a4.py` preflighted `convrot_groupsize: 64` while converting at `--convrot-groupsize`
    (default 256); `quant_w4a8.py` preflighted `group_size: 16` and `codebook: True` regardless of
    `--group-size` and `--no-codebook`. Measured 2026-08-22 on the 3090, none of that changed
    which implementation resolved -- so this is a test of the question, not of any known wrong
    answer, and it is worth exactly that much.
    """
    for op, config in ALL_OPS.items():
        setup, kwargs = _OP_RECIPES[op]("0", dict(config))
        rendered = setup + kwargs
        assert "512" in rendered, f"{op}: convrot_groupsize 512 does not appear in {rendered!r}"
        assert "256," not in kwargs and "256}" not in kwargs, (
            f"{op}: a hardcoded 256 survives in {kwargs!r}")
    setup, kwargs = _OP_RECIPES["quantize_w4a8_int8_weight"]("0", dict(
        ALL_OPS["quantize_w4a8_int8_weight"]))
    assert "'group_size': 32" in kwargs, kwargs
    assert "'codebook': False" in kwargs, kwargs


def test_a_missing_config_key_refuses_rather_than_defaulting():
    """`_need` is the whole guard. A `.get(key, 256)` here would rebuild the bug silently."""
    try:
        _need({}, "convrot_w4a4_linear", "convrot_groupsize")
    except SystemExit as error:
        assert "convrot_groupsize" in str(error) and "default" in str(error), error
    else:
        raise AssertionError("_need accepted a missing key")


def test_probe_weight_is_divisible_by_every_groupsize_it_uses():
    """K must divide by the rotation size and by the scale group size or the op raises before the
    registry is ever consulted -- a probe that dies on its own shape says nothing about backends."""
    for groupsizes in ((64,), (256,), (512,), (256, 16), (512, 32), (64, 16), (128, 48)):
        rows, cols = _shape(*groupsizes)
        assert rows == 256 and cols >= 256, (groupsizes, rows, cols)
        for size in groupsizes:
            assert cols % size == 0, (groupsizes, cols, size)


def test_default_shape_is_the_one_that_was_measured():
    """256x256 at cg=256 is literally the configuration probe_backend_resolution.py ran on the
    3090 on 2026-08-22. Keeping the default identical is what lets that run stand as evidence."""
    assert _shape(256) == (256, 256)
    assert _shape(64) == (256, 256)


def test_unknown_op_and_empty_op_set_both_refuse():
    for ops, needle in (({"not_an_op": {}}, "no probe recipe"), ({}, "nothing to preflight")):
        try:
            build_probe_source(ROOT, ops)
        except SystemExit as error:
            assert needle in str(error), error
        else:
            raise AssertionError(f"build_probe_source accepted {ops!r}")


def test_default_ops_still_serve_the_one_argument_callers():
    """`tools/probe_backend_resolution.py:63` calls this with the root alone. That file is not
    ours to edit, and it is the script that produced the only measurement anyone has."""
    source = build_probe_source(ROOT)
    assert "quantize_convrot_w4a4_weight" in source and "convrot_w4a4_linear" in source
    compile(source, "<probe>", "exec")


# ---- the callers ---------------------------------------------------------------------------
# Each converter builds its own op/config map. These check the map is renderable and carries the
# caller's flags -- not that the backend resolves, which needs the card.

def _renders(ops: dict) -> str:
    source = build_probe_source(ROOT, ops)
    compile(source, "<probe>", "exec")
    return source


def test_quant_w4a4_passes_its_convrot_groupsize():
    import quant_w4a4

    ops = quant_w4a4.w4a4_probe_ops(64)
    assert set(ops) == {"quantize_convrot_w4a4_weight", "convrot_w4a4_linear"}, ops
    assert all(c["convrot_groupsize"] == 64 for c in ops.values()), ops
    _renders(ops)


def test_quant_w4a8_passes_group_size_and_no_codebook():
    import quant_w4a8

    args = argparse.Namespace(group_size=32, convrot_groupsize=64, no_codebook=True)
    ops = quant_w4a8.w4a8_probe_ops(args)
    # One op, unchanged from before the consolidation -- see w4a8_probe_ops' docstring for why
    # widening it to w4a8_int8_linear was left as a flagged gap rather than done here.
    assert set(ops) == {"quantize_w4a8_int8_weight"}, ops
    config = ops["quantize_w4a8_int8_weight"]
    assert config == {"group_size": 32, "convrot_groupsize": 64, "codebook": False}, config
    assert "'codebook': False" in _renders(ops)


def test_quant_int8_probes_the_convrot_quantizer_only():
    import quant_int8

    ops = quant_int8.int8_probe_ops(128)
    assert set(ops) == {"quantize_int8_convrot_weight"}, ops
    # This op's `group_size` IS the rotation size; quant_int8 passes --convrot-groupsize into it.
    assert "'group_size': 128" in _renders(ops)


def test_quant_int8_no_convrot_is_still_exempt():
    """The one deliberate exemption in the tree, and the consolidation must not have eaten it.

    `--no-convrot` calls `eager.quantize_int8_rowwise` outside `ck.registry` entirely, so there is
    no native backend for it to resolve to. The preflight is gated on `args.convrot` at the call
    site and prints why instead of pretending. Read from the source rather than executed: running
    main() needs a real checkpoint and a card.
    """
    body = (Path(__file__).resolve().parent / "quant_int8.py").read_text(encoding="utf-8")
    call = body.index("native_backend_ready(portable_root,")
    gate = body.rindex("if args.convrot:", 0, call)
    assert call - gate < 200, "the int8 preflight is no longer inside `if args.convrot:`"
    assert "no native-backend\n" in body or "no native-backend " in body, (
        "the --no-convrot branch no longer explains why it is exempt")


def test_quant_mixed_probes_all_four_ops_at_its_own_sizes():
    import quant_mixed

    args = argparse.Namespace(group_size=32, convrot_groupsize=64)
    ops = quant_mixed.mixed_probe_ops(args)
    assert set(ops) == set(quant_mixed.REQUIRED_OPS), ops
    assert all(c["convrot_groupsize"] == 64 for c in ops.values()), ops
    _renders(ops)


def test_verify_probes_the_formats_configuration_not_a_literal():
    import verify_w4a4

    layers = {
        "a": {"format": "convrot_w4a4", "convrot_groupsize": 64},
        "b": {"format": "asym_w4a8_int8", "convrot_groupsize": 64, "group_size": 32},
        "c": {"format": "int8_tensorwise", "convrot_groupsize": 64, "convrot": True},
    }
    first = {config["format"]: name for name, config in reversed(list(layers.items()))}
    header = {"b.weight": {}, "b.weight_s_rel": {}, "b.weight_codebook": {}}
    ops = verify_w4a4.probe_ops_for(layers, first, header)
    assert set(ops) == {"convrot_w4a4_linear", "w4a8_int8_linear", "int8_linear"}, ops
    assert ops["convrot_w4a4_linear"]["convrot_groupsize"] == 64
    assert ops["w4a8_int8_linear"]["group_size"] == 32
    # Was hardcoded False, while the smoke below it used the file's real value.
    assert ops["int8_linear"]["convrot"] is True
    _renders(ops)


def test_the_w4a8_probe_reads_the_codebook_off_the_file():
    """The probe must resolve the call the smoke is about to make, not a likelier-looking one.

    `_w4a8_smoke` loads `.weight_codebook` with `optional=True`, so on a
    `quant_w4a8.py --no-codebook` file it passes `None`. The probe hardcoded `codebook: True`, so
    the two arms asked the registry about two different calls -- the exact defect
    `_int8_probe_ops` was rewritten to fix, one format above it in the same file.

    Latent rather than live on this bench: measured 2026-08-22, all nine w4a8 checkpoints across
    both drives carry `weight_codebook` on every w4a8 layer. That is a fact about what has been
    converted so far, not about what the flag permits.
    """
    import verify_w4a4

    layers = {"x": {"format": "asym_w4a8_int8", "convrot_groupsize": 64, "group_size": 32}}
    with_cb = verify_w4a4.probe_ops_for(layers, {"asym_w4a8_int8": "x"},
                                        {"x.weight": {}, "x.weight_codebook": {}})
    without = verify_w4a4.probe_ops_for(layers, {"asym_w4a8_int8": "x"},
                                        {"x.weight": {}, "x.weight_s_rel": {}})
    assert with_cb["w4a8_int8_linear"]["codebook"] is True, with_cb
    assert without["w4a8_int8_linear"]["codebook"] is False, without
    # A prefix must not count as the tensor: `x.weight_codebook_extra` is a different tensor, and
    # `x2.weight_codebook` belongs to a different layer.
    other = verify_w4a4.probe_ops_for(layers, {"asym_w4a8_int8": "x"},
                                      {"x.weight": {}, "x2.weight_codebook": {}})
    assert other["w4a8_int8_linear"]["codebook"] is False, other
    _renders(with_cb)
    _renders(without)


def test_verify_does_not_become_stricter_than_the_loader():
    """The round-1 regression, in the same file, from the same instinct.

    `comfy/ops.py:1195` falls back to `convrot_groupsize` 256 when the key is absent, and
    `ops.py:1279` reads `convrot` as `.get("convrot", False)`. A probe that refused a config the
    loader would happily run is how a verifier came to reject 100% of existing checkpoints.
    """
    ops = verify_probe({"format": "convrot_w4a4"}, "convrot_w4a4")
    assert ops["convrot_w4a4_linear"]["convrot_groupsize"] == 256
    ops = verify_probe({"format": "int8_tensorwise"}, "int8_tensorwise")
    assert ops["int8_linear"] == {"convrot": False, "convrot_groupsize": 256}


def verify_probe(config: dict, fmt: str) -> dict:
    import verify_w4a4

    # A header with the layer's own tensors present, so w4a8 resolves with a codebook. The
    # codebook-absent case has its own test above rather than riding on this helper's default.
    return verify_w4a4.probe_ops_for({"x": config}, {fmt: "x"},
                                     {"x.weight": {}, "x.weight_codebook": {}})


def test_verify_smoke_and_probe_agree_on_which_layer():
    """An A/B only counts if both arms took the same dispatch -- four incidents in one day here.

    The probe and the smoke both index `first_layer`, so they cannot disagree. Checked by reading
    `run()`, which cannot be executed without a checkpoint.
    """
    body = (Path(__file__).resolve().parent / "verify_w4a4.py").read_text(encoding="utf-8")
    # `model_header` joined the call on 2026-08-22: w4a8's `codebook` kwarg is a property of the
    # FILE (is `<layer>.weight_codebook` there?) and not of the per-layer config, which carries
    # only `{format, convrot_groupsize, group_size}`. The probe hardcoded `codebook: True` while
    # the smoke passes `optional=True`, so on a `--no-codebook` file they resolved and called two
    # different things -- the same defect this test is named after, in the same file.
    assert "probe_ops_for(layers, first_layer, model_header)" in body
    assert "layer_name = first_layer[fmt_name]" in body


def test_one_definition_of_the_probe_remains():
    """Closing criterion item 3, mechanised. The count is 0, not 1, because the surviving function
    is `native_backend_ready` -- criterion item 1 names it -- so there is no old-name definition
    left to count. Six existed before.

    The needle is assembled from two pieces so that `grep -c` for it over `tools/*.py`, which is
    how the criterion is checked by hand, does not count this file's own source as a hit.
    """
    needle = "def " + "normal_comfy_backend"
    here = Path(__file__).resolve()
    others = [p for p in here.parent.glob("*.py") if p.resolve() != here]
    hits = [p.name for p in others if needle in p.read_text(encoding="utf-8")]
    assert hits == [], hits
    assert sum("def native_backend_ready" in p.read_text(encoding="utf-8")
               for p in others) == 1


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001 -- a test runner reports, it does not raise
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    print("\nNOT COVERED BY THIS RUN:")
    print("  - no CUDA device was touched and comfy_kitchen was never imported. Nothing here "
          "shows the backend resolves, that any op is native, or that a kernel ran.")
    print("  - the generated probe program is compiled, never executed. Whether comfy-kitchen "
          "accepts these kwargs at an exotic groupsize is unmeasured; cg=64 and cg=256 are the "
          "only two anyone has run (2026-08-22, RTX 3090).")
    print("  - `_native_probe.instrument()` is not tested here at all, and has no recorded "
          "execution anywhere.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
