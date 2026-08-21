"""Shared native-backend probes for ConvRot W4A4 / W4A8 / INT8 tooling.

Two questions this project's probe scripts kept reimplementing, and drifting on
(`AUDITORIA_2026-08-18.md` items 17/18; `.scratch/estado-entregavel/issues/20-sonda-de-backend-nativo-duplicada.md`):

1. **"Is the native CUDA backend actually going to be picked?"** -- `native_backend_ready()`, a
   subprocess-based check that resolves BOTH `quantize_convrot_w4a4_weight` and
   `convrot_w4a4_linear` through `comfy_kitchen.registry` -- the stricter pair `quant_w4a4.py`
   already checks, not the linear op alone. The kwargs are real tensors from a real
   `quantize_convrot_w4a4_weight` call, not empty/zero placeholders: `registry.get_implementation`
   docstrings its own `kwargs` argument as "for constraint validation (empty/None skips
   validation)" (`comfy_kitchen/registry.py:246`), so dummy kwargs make this check pass while a
   real call, with real constraints, could still drop to eager. That is the exact bug
   `quant_mixed.py` already fixed in its own copy of this probe (`tools/quant_mixed.py:85-133`);
   this function follows that fixed pattern, not `quant_w4a4.py`'s dummy-tensor one.

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
against `quant_mixed.py`'s own real-kwargs probe (which already had both names right). No GPU was
available in the session that wrote this module, so neither function below has been executed by
it -- only `py_compile` and import-without-CUDA-op were run. Whether `dict(zip(arg_names, args))`
mislabelling actually changed which implementation resolved for any real checkpoint is NOT
covered here; that needs a live run with `--kernel-smoke` or an equivalent probe against the GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def native_backend_ready(portable_root: Path) -> dict:
    """Ask a fresh interpreter which implementation normal ComfyUI would pick for ConvRot W4A4.

    Resolves both `quantize_convrot_w4a4_weight` and `convrot_w4a4_linear`. Runs in a subprocess,
    not in-process, because this module (or its caller) may already have imported and poked at
    `comfy_kitchen` by the time it asks -- the subprocess sees what a fresh `main.py` would see.

    Requires a CUDA device (the probe tensors are allocated on `cuda`). Raises SystemExit with the
    subprocess's stderr on failure, including "no CUDA device" -- this function does not run
    without the GPU, and does not pretend to.
    """
    probe = (
        "import json,sys;"
        f"sys.path.insert(0, {str(portable_root / 'ComfyUI')!r});"
        "import torch;"
        "import comfy.quant_ops;"
        "import comfy_kitchen as ck;"
        "from comfy_kitchen import registry as R;"
        "w = torch.zeros(256, 256, device='cuda', dtype=torch.bfloat16);"
        "x = torch.zeros(64, 256, device='cuda', dtype=torch.bfloat16);"
        "q4, s4 = ck.quantize_convrot_w4a4_weight(w, 256, 64);"
        "kw = {"
        "  'quantize_convrot_w4a4_weight': {'weight': w, 'convrot_groupsize': 256,"
        "      'quant_group_size': 64, 'stochastic_rounding': 0},"
        "  'convrot_w4a4_linear': {'x': x, 'qweight': q4, 'wscales': s4, 'bias': None,"
        "      'convrot_groupsize': 256, 'quant_group_size': 64, 'linear_dtype': 'int4'},"
        "};"
        "names = ['quantize_convrot_w4a4_weight', 'convrot_w4a4_linear'];"
        "modules = {n: getattr(R.get_implementation(n, kwargs=kw[n]), '__module__', '?') for n in names};"
        "print(json.dumps({**modules, 'backends': ck.list_backends()}))"
    )
    result = subprocess.run(
        [sys.executable, "-s", "-c", probe],
        cwd=str(portable_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(f"Backend probe failed:\n{result.stderr.strip()}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["warning"] = result.stderr.strip()
    payload["native_ready"] = all(
        ".backends.cuda" in payload[key]
        for key in ("quantize_convrot_w4a4_weight", "convrot_w4a4_linear")
    )
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
