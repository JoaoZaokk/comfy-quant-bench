"""Shared native-backend probes for ConvRot W4A4 / W4A8 / INT8 tooling.

Two questions this project's probe scripts kept reimplementing, and drifting on
(`AUDITORIA_2026-08-18.md` items 17/18; `.scratch/estado-entregavel/issues/20-sonda-de-backend-nativo-duplicada.md`):

1. **"Is the native CUDA backend actually going to be picked?"** -- `native_backend_ready()`, a
   subprocess-based check that resolves whichever ops the caller names, with the real kwargs the
   caller is about to use. Every converter here and `verify_w4a4.py` call this one. Until
   2026-08-22 there were six private copies of it, and they disagreed about what they asked:
   which ops (one, two, or four), whether the probe tensors were `torch.empty` or real quantizer
   output, and at which groupsize -- `quant_w4a4.py` preflighted at 64/64 while converting at
   whatever `--convrot-groupsize` said, default 256, and `quant_w4a8.py` preflighted
   `codebook: True` while `--no-codebook` converted without one.

   **MEASURED 2026-08-22, RTX 3090 (cc 8.6), GPU lock held**, by `tools/probe_backend_resolution.py`:
   none of those disagreements changed the answer. All four combinations of {cg=64, cg=256} x
   {`torch.empty` dummies, real quantized tensors} resolved BOTH `quantize_convrot_w4a4_weight`
   and `convrot_w4a4_linear` to `comfy_kitchen.backends.cuda` -- one distinct implementation
   across all four -- and the real `ck.convrot_w4a4_linear` call returned finite bf16 of the
   right shape at both groupsizes. So the argument for consolidating is NOT "the copies give
   different answers": measured, they do not, and claiming they do would overstate this. It is
   that six copies existed, asked six different questions, and nothing justified six.

   The kwargs are real quantizer output rather than placeholders because
   `registry.get_implementation` docstrings its own `kwargs` argument as "for constraint
   validation (empty/None skips validation)" (`comfy_kitchen/registry.py:246`), so dummy kwargs
   could make this check pass where a real call drops to eager. **That sentence is a READING of
   `registry.py`, not a measurement** -- an earlier version of this docstring stated it as though
   it were one. The run above did not reproduce it: for these two ops, on this build, at these
   two groupsizes, dummy and real resolved identically. Two ops / one build / one card is not
   "the mechanism is false", it is "it did not reproduce under the only conditions anyone has
   tried" -- so the real kwargs stay, because they cost nothing and are what the conversion is
   about to pass anyway.

2. **`instrument()`** -- monkeypatches `convrot_w4a4_linear`, `w4a8_int8_linear`, the INT8
   dispatch-table handler, and all three layouts' `dequantize`, to count native-dispatcher calls
   against weight dequantizations, AND records which implementation module actually resolved
   (the `impls` set) for each distinct call shape seen. Counting the DISPATCHER call is not proof
   of CUDA: the dispatcher also routes to eager, gated by per-call constraint checks and by
   `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`. Reading `impl.__module__` back is what tells the two
   apart -- `.backends.cuda.*` vs `.backends.eager.*` -- and it also tells W4A4 apart from W4A8,
   which a bare call count cannot.

Both functions were duplicated with copy-paste drift across 6 files. One drift was real, not just
style: `convrot_w4a4_linear`'s second positional argument is named `qweight`
(`comfy_kitchen/tensor/convrot_w4a4.py:80-89`, matched by the installed cuda and eager backend
implementations), while `w4a8_int8_linear`'s equivalent argument is named `qdata`
(`comfy_kitchen/tensor/w4a8_int8.py:93-98`, matched by the installed cuda, eager, hip, and triton
backends). `gemma_chat.py`'s old copy of `instrument()` had these backwards for the W4A8 wrap --
it labelled the packed-weight argument `qweight` where `w4a8_int8_linear` wants `qdata`.
`AUDITORIA_2026-08-18.md:174` cites the right file and lines for this (`w4a8_int8.py:93-96`) but
the ticket built from it attached the citation to the wrong call
(`convrot_w4a4_linear`/gemma_chat.py:105, whose own argument really is `qweight` -- confirmed
correct by the same reading). The real bug was one line down, on the `w4a8_int8_linear` wrap.
LIDO, not medido: confirmed by reading `comfy_kitchen/tensor/{convrot_w4a4,w4a8_int8}.py` and all
four installed `comfy_kitchen/backends/{cuda,eager,hip,triton}/*.py` signatures, and cross-checked
against `quant_mixed.py`'s own real-kwargs probe (which already had both names right). Whether
`dict(zip(arg_names, args))` mislabelling actually changed which implementation resolved for any
real checkpoint is NOT covered here; that needs a live run with `--kernel-smoke` or an equivalent
probe against the GPU.

EXECUTION STATUS of the two functions, kept here because it changes what a caller may claim:
`native_backend_ready()` was EXECUTED for the first time on 2026-08-22 on the 3090 and works --
`native_ready: true`, both ConvRot ops on `comfy_kitchen.backends.cuda`. (This paragraph used to
say neither function had ever been run; that was true when written and stopped being true.)
`instrument()` below still has no recorded execution: it was checked by `py_compile` and by
reading the four installed backends' signatures, nothing more.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


# The loader's own value, and the reason it is a literal rather than something read from a file.
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
# written by no converter here -- confirmed 2026-08-22 by reading the headers of the produced
# files, whose per-layer configs are exactly `{format, convrot_groupsize}` (plus `group_size` on
# w4a8 layers). Round 1 of this sweep shipped a verifier that demanded it from the metadata; that
# made the tool stricter than the runtime it verifies and refused 100% of existing checkpoints.
# Where the loader hardcodes, reproducing the loader means hardcoding the same value and citing
# where it came from. `verify_w4a4.py` imports this constant rather than keeping its own copy.
LOADER_QUANT_GROUP_SIZE = 64

# Same reasoning, same lines: `linear_dtype` defaults to "int4" in the loader, and no converter
# here writes it, so this default reproduces ops.py rather than guessing on its behalf.
LOADER_LINEAR_DTYPE = "int4"


# --------------------------------------------------------------------------------------------
# Op recipes. One per comfy-kitchen op this bench dispatches, each building the REAL kwargs the
# caller is about to pass -- because a preflight that resolves a configuration the tool is not
# going to run answers a question nobody asked. See the module docstring for what was and was not
# measured about that on 2026-08-22.
# --------------------------------------------------------------------------------------------

def _need(config: dict, op: str, key: str):
    """Read one kernel parameter out of the caller's own config, or refuse to substitute one.

    Deliberately not `config.get(key, <default>)`. The six copies this function replaced each
    hardcoded their groupsizes, which is how `quant_w4a4.py` came to preflight at 64/64 while
    converting at 256 and `quant_w4a8.py` to preflight `codebook: True` under `--no-codebook`.
    A missing key here is a caller bug, and it is louder as an exception than as a silent 256.
    """
    if key not in config:
        raise SystemExit(
            f"native_backend_ready: probing {op!r} needs {key!r} in its config and will not "
            f"substitute a default -- preflighting a configuration the caller is not about to "
            f"run is the bug this function exists to remove. Config given: {sorted(config)}.")
    return config[key]


def _shape(*groupsizes: int) -> tuple[int, int]:
    """[N, K] for a probe weight, with K a multiple of every groupsize the op will use.

    K must be divisible by the rotation groupsize (and, for W4A8, by the scale group size) or the
    op raises before the registry is ever consulted. Smallest such K at or above 256, so the
    default case is literally the 256x256 shape the 2026-08-22 run measured.
    """
    step = 1
    for size in groupsizes:
        step = step * int(size) // math.gcd(step, int(size))
    return 256, step * max(1, -(-256 // step))


def _recipe_quantize_w4a4(tag: str, config: dict) -> tuple[str, str]:
    op = "quantize_convrot_w4a4_weight"
    cg = int(_need(config, op, "convrot_groupsize"))
    rows, cols = _shape(cg)
    setup = f"w{tag} = torch.randn({rows}, {cols}, device='cuda', dtype=torch.bfloat16)\n"
    return setup, (f"{{'weight': w{tag}, 'convrot_groupsize': {cg}, "
                   f"'quant_group_size': {LOADER_QUANT_GROUP_SIZE}, 'stochastic_rounding': 0}}")


def _recipe_w4a4_linear(tag: str, config: dict) -> tuple[str, str]:
    op = "convrot_w4a4_linear"
    cg = int(_need(config, op, "convrot_groupsize"))
    dtype = config.get("linear_dtype", LOADER_LINEAR_DTYPE)
    rows, cols = _shape(cg)
    # The packed weight comes from a real quantize call at the caller's own groupsize, not from
    # `torch.empty`. Two reasons, only one of them measured: (a) it is what the conversion is
    # about to hand the kernel, and (b) it means the probe also fails when the quantizer itself
    # raises at that groupsize. Measured 2026-08-22: the call succeeds at cg=64 and cg=256.
    setup = (f"w{tag} = torch.randn({rows}, {cols}, device='cuda', dtype=torch.bfloat16)\n"
             f"x{tag} = torch.randn(64, {cols}, device='cuda', dtype=torch.bfloat16)\n"
             f"q{tag}, s{tag} = ck.quantize_convrot_w4a4_weight("
             f"w{tag}, {cg}, {LOADER_QUANT_GROUP_SIZE})\n")
    return setup, (f"{{'x': x{tag}, 'qweight': q{tag}, 'wscales': s{tag}, 'bias': None, "
                   f"'convrot_groupsize': {cg}, 'quant_group_size': {LOADER_QUANT_GROUP_SIZE}, "
                   f"'linear_dtype': {dtype!r}}}")


def _recipe_quantize_w4a8(tag: str, config: dict) -> tuple[str, str]:
    op = "quantize_w4a8_int8_weight"
    cg = int(_need(config, op, "convrot_groupsize"))
    gs = int(_need(config, op, "group_size"))
    # `codebook` is a real flag (`quant_w4a8.py --no-codebook`), so it is the caller's, not a
    # constant. Every copy of this probe hardcoded True.
    codebook = bool(_need(config, op, "codebook"))
    rows, cols = _shape(cg, gs)
    setup = f"w{tag} = torch.randn({rows}, {cols}, device='cuda', dtype=torch.bfloat16)\n"
    return setup, (f"{{'weight': w{tag}, 'group_size': {gs}, 'convrot_groupsize': {cg}, "
                   f"'symmetric': True, 'scale_dtype': torch.float8_e4m3fn, "
                   f"'codebook': {codebook}, 'codebook_tensor': None, 'stochastic_rounding': 0}}")


def _recipe_w4a8_linear(tag: str, config: dict) -> tuple[str, str]:
    op = "w4a8_int8_linear"
    cg = int(_need(config, op, "convrot_groupsize"))
    gs = int(_need(config, op, "group_size"))
    # `_need`, not `config.get(..., True)`. A silent default here is a guess about which
    # implementation the registry would pick, in the one module that exists to stop guessing --
    # and its sibling `_recipe_quantize_w4a8` already reads the same key with `_need`.
    codebook = bool(_need(config, op, "codebook"))
    rows, cols = _shape(cg, gs)
    # `symmetric=True` is not a caller knob: both writers refuse asymmetric weights outright,
    # because comfy/ops.py drops the `correction` tensor, so nothing downstream could decode one.
    setup = (f"w{tag} = torch.randn({rows}, {cols}, device='cuda', dtype=torch.bfloat16)\n"
             f"x{tag} = torch.randn(64, {cols}, device='cuda', dtype=torch.bfloat16)\n"
             f"p{tag} = ck.quantize_w4a8_int8_weight(w{tag}, group_size={gs}, "
             f"convrot_groupsize={cg}, symmetric=True, scale_dtype=torch.float8_e4m3fn, "
             f"codebook={codebook}, codebook_tensor=None, stochastic_rounding=0)\n")
    return setup, (f"{{'x': x{tag}, 'qdata': p{tag}[0], 's_rel': p{tag}[1], "
                   f"'s_channel': p{tag}[2], 'codebook': p{tag}[4], 'correction': p{tag}[3], "
                   f"'bias': None, 'group_size': {gs}, 'convrot_groupsize': {cg}, "
                   f"'out_dtype': torch.bfloat16}}")


def _recipe_quantize_int8_convrot(tag: str, config: dict) -> tuple[str, str]:
    op = "quantize_int8_convrot_weight"
    # This op's `group_size` IS the rotation groupsize -- `quant_int8.py` passes its
    # `--convrot-groupsize` straight into it -- so it is read under that name and passed under
    # this one, rather than being a second independent knob.
    cg = int(_need(config, op, "convrot_groupsize"))
    rows, cols = _shape(cg)
    setup = f"w{tag} = torch.randn({rows}, {cols}, device='cuda', dtype=torch.bfloat16)\n"
    return setup, f"{{'weight': w{tag}, 'group_size': {cg}}}"


def _recipe_int8_linear(tag: str, config: dict) -> tuple[str, str]:
    op = "int8_linear"
    convrot = bool(_need(config, op, "convrot"))
    # Only read when the rotation is on: `int8_linear` touches convrot_groupsize solely inside
    # `if convrot` (backends/cuda/__init__.py:1873), and the --no-convrot files carry no such key
    # -- comfy's own loader reads it as `layer_conf.get("convrot", False)` (ops.py:1279).
    cg = int(_need(config, op, "convrot_groupsize")) if convrot else 256
    rows, cols = _shape(cg)
    # The only recipe here whose tensors are constructed rather than quantizer output. Nothing
    # measured says real tensors change int8 resolution, and the quantizer for the --no-convrot
    # half is `eager.quantize_int8_rowwise`, reached outside the registry -- adding a never-run
    # call to a probe that HAS run cleanly (2026-08-22, 1440-layer LTX file) buys nothing and can
    # only add ways to fail. `zeros`, not `empty`: uninitialized memory can hold NaN, and a probe
    # that fails on the contents of stale VRAM is a flake, not a refusal.
    setup = (f"w{tag} = torch.zeros(({rows}, {cols}), device='cuda', dtype=torch.int8)\n"
             f"s{tag} = torch.zeros(({rows}, 1), device='cuda', dtype=torch.float32)\n"
             f"x{tag} = torch.zeros((64, {cols}), device='cuda', dtype=torch.bfloat16)\n")
    return setup, (f"{{'x': x{tag}, 'weight': w{tag}, 'weight_scale': s{tag}, 'bias': None, "
                   f"'out_dtype': torch.bfloat16, 'convrot': {convrot}, "
                   f"'convrot_groupsize': {cg}, 'input_act': None}}")


_OP_RECIPES = {
    "quantize_convrot_w4a4_weight": _recipe_quantize_w4a4,
    "convrot_w4a4_linear": _recipe_w4a4_linear,
    "quantize_w4a8_int8_weight": _recipe_quantize_w4a8,
    "w4a8_int8_linear": _recipe_w4a8_linear,
    "quantize_int8_convrot_weight": _recipe_quantize_int8_convrot,
    "int8_linear": _recipe_int8_linear,
}

# What `native_backend_ready(root)` asks when the caller names no ops: the ConvRot W4A4 pair at
# the loader's own defaults. Kept so the standalone GPU probes that call it with one argument
# (`tools/probe_backend_resolution.py:63`) keep working. A converter must NOT rely on this -- it
# is a configuration nobody is about to run, which is the whole complaint.
DEFAULT_OPS = {
    "quantize_convrot_w4a4_weight": {"convrot_groupsize": 256},
    "convrot_w4a4_linear": {"convrot_groupsize": 256},
}


def build_probe_source(portable_root: Path, ops: dict | None = None) -> str:
    """The subprocess program, as text. Separate from running it so it can be tested off-GPU.

    `tools/test_native_probe.py` compiles this and reads the groupsizes back out of it, which is
    the only part of this module a machine without a free card can check at all.
    """
    ops = dict(DEFAULT_OPS if ops is None else ops)
    if not ops:
        raise SystemExit("native_backend_ready: no ops named; there is nothing to preflight")
    unknown = sorted(name for name in ops if name not in _OP_RECIPES)
    if unknown:
        raise SystemExit(f"native_backend_ready: no probe recipe for {unknown} "
                         f"(known: {sorted(_OP_RECIPES)})")

    setup, kwargs_lines = [], []
    for index, (name, config) in enumerate(ops.items()):
        block, expression = _OP_RECIPES[name](str(index), dict(config))
        setup.append(block)
        kwargs_lines.append(f"kw[{name!r}] = {expression}\n")

    # The one interpolated string that is not a number: a Windows path. It goes in through `!r`,
    # never through concatenation -- a backslash that reaches a shell or a heredoc on this bench
    # has turned into TAB, formfeed and 0x0F, four separate times. test_native_probe.py scans the
    # result for bytes below 0x20 for that reason.
    return (
        "import json, sys\n"
        f"sys.path.insert(0, {str(portable_root / 'ComfyUI')!r})\n"
        "import torch\n"
        "import comfy.quant_ops\n"  # registers the ops; without it the registry is empty
        "import comfy_kitchen as ck\n"
        "from comfy_kitchen import registry as R\n"
        "kw = {}\n"
        + "".join(setup)
        + "".join(kwargs_lines)
        + "resolved = {n: getattr(R.get_implementation(n, kwargs=k), '__module__', '?')\n"
          "            for n, k in kw.items()}\n"
        "print(json.dumps({'resolved': resolved, 'backends': ck.list_backends()}))\n"
    )


def native_backend_ready(portable_root: Path, ops: dict | None = None) -> dict:
    """Ask a fresh interpreter which implementation normal ComfyUI would pick, per op.

    `ops` maps each comfy-kitchen op name to the real configuration the caller is about to run it
    under -- `{"convrot_w4a4_linear": {"convrot_groupsize": args.convrot_groupsize}}` and so on.
    Each op gets its own probe tensors, built at its own groupsizes, so a caller mixing formats
    (quant_mixed, or verify_w4a4 on a mixed checkpoint) cannot have one format's configuration
    silently answer for another's.

    Runs in a subprocess, not in-process, because this module (or its caller) may already have
    imported and poked at `comfy_kitchen` by the time it asks -- the subprocess sees what a fresh
    `main.py` would see.

    Requires a CUDA device (the probe tensors are allocated on `cuda`). Raises SystemExit with the
    subprocess's stderr on failure, including "no CUDA device" -- this function does not run
    without the GPU, and does not pretend to.

    Returns `{"resolved": {op: module}, "backends": [...], "warning": str, "native_ready": bool}`.
    """
    code = build_probe_source(portable_root, ops)
    result = subprocess.run(
        [sys.executable, "-s", "-c", code],
        cwd=str(portable_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(f"Backend probe failed:\n{result.stderr.strip()}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["warning"] = result.stderr.strip()
    payload["native_ready"] = all(".backends.cuda" in module
                                  for module in payload["resolved"].values())
    return payload


def instrument() -> dict:
    """Wrap every shipped 4-bit-family linear/dequant so callers can count native vs. dequantized.

    Returns a counters dict: `native_calls` and `dequant_calls` are running totals the caller
    resets between probes (`counters["native_calls"] = 0`, etc.), and `impls` is a set of
    `"<module>.<qualname>"` strings -- the real implementations `registry.get_implementation`
    resolved for the distinct call shapes seen (capped at 4 distinct entries per reset, matching
    the cap already used by every prior copy of this function, to avoid re-resolving on a hot loop
    once the answer has stopped changing).

    Covers `convrot_w4a4_linear` (ConvRot W4A4), `w4a8_int8_linear` (asymmetric W4A8), and the
    INT8 tensor-wise layout's linear dispatch (`comfy_kitchen.tensor.int8`, routed through
    `comfy_kitchen.tensor.base._LAYOUT_DISPATCH_TABLE` rather than `registry`, so it is wrapped at
    the dispatch-table handler instead of a module attribute). Without the INT8 branch, a
    checkpoint quantized only in that format reports zero native calls and zero dequantizations,
    which reads as "nothing ran" rather than "this probe does not watch that layout" -- the same
    failure `diffusion_smoke.py` documented and fixed for itself before this module existed.
    """
    import comfy_kitchen.tensor.base as tensor_base
    import comfy_kitchen.tensor.convrot_w4a4 as convrot
    import comfy_kitchen.tensor.int8 as int8
    import comfy_kitchen.tensor.w4a8_int8 as w4a8
    from comfy_kitchen.registry import registry

    counters = {"native_calls": 0, "dequant_calls": 0, "impls": set()}

    def wrap_linear(module, name: str, arg_names: tuple[str, ...]):
        original = getattr(module, name)

        def counting(*args, **kwargs):
            counters["native_calls"] += 1
            if len(counters["impls"]) < 4:
                probe = dict(zip(arg_names, args))
                probe.update(kwargs)
                try:
                    impl = registry.get_implementation(name, kwargs=probe)
                    counters["impls"].add(f"{impl.__module__}.{impl.__name__}")
                except Exception as error:  # probing must never break the forward/generation
                    counters["impls"].add(f"<probe failed: {type(error).__name__}>")
            return original(*args, **kwargs)

        setattr(module, name, counting)

    def wrap_dequant(layout):
        original = layout.dequantize.__func__

        def counting(cls, qdata, params):
            counters["dequant_calls"] += 1
            return original(cls, qdata, params)

        layout.dequantize = classmethod(counting)

    # Argument names below are each op's real positional-parameter names, confirmed by reading
    # comfy_kitchen/tensor/{convrot_w4a4,w4a8_int8}.py and the cuda/eager/hip/triton backend
    # signatures. ConvRot's packed-weight kwarg is `qweight`; W4A8's is `qdata` -- see the module
    # docstring. They are not interchangeable: `dict(zip(arg_names, args))` uses these names to
    # build the kwargs dict `registry.get_implementation` validates constraints against, so the
    # wrong name here means a constraint check silently looks at a missing/misnamed kwarg instead
    # of the real one.
    wrap_linear(convrot, "convrot_w4a4_linear", ("x", "qweight", "wscales", "bias"))
    wrap_linear(w4a8, "w4a8_int8_linear", ("x", "qdata", "s_rel", "s_channel"))
    wrap_dequant(convrot.TensorCoreConvRotW4A4Layout)
    wrap_dequant(w4a8.AsymW4A8Int8Layout)

    # INT8 cannot be wrapped the same way: TensorWiseINT8Layout's linear calls
    # `torch.ops.comfy_kitchen.int8_linear` directly rather than going through a registry
    # function, so the registered dispatch-table handler is wrapped instead of a module attribute.
    wrap_dequant(int8.TensorWiseINT8Layout)
    table = getattr(tensor_base, "_LAYOUT_DISPATCH_TABLE", {})
    for op, by_layout in list(table.items()):
        if int8.TensorWiseINT8Layout not in by_layout:
            continue
        original_handler = by_layout[int8.TensorWiseINT8Layout]

        def counting_handler(qt, args, kwargs, _original=original_handler, _op=op):
            counters["native_calls"] += 1
            counters["impls"].add(f"comfy_kitchen.tensor.int8.{str(_op).split('.')[-2]}")
            return _original(qt, args, kwargs)

        by_layout[int8.TensorWiseINT8Layout] = counting_handler

    return counters
