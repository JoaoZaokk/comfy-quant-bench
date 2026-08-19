"""Regression tests for `svdq_to_bf16`, on CPU, with no checkpoint and no GPU.

Two things are locked down here.

**The recovery's central claim.** Feeding the identity recovers the kernel's exact weight,
because every row of `I` holds a single nonzero and a symmetric per-group scale represents a
lone nonzero (and the zeros beside it) with no error. Confirmed on the real kernel too --
`layer(8I)/8` matched `layer(I)` to exactly 0 on every Z-Image layer sampled -- but that needs a
GPU and this does not.

**Why the diagnostic scores must never become a gate again.** They were a gate twice:

    v1  require relL2 <= 0.02 on random input      rejected correct reconstructions at 0.0988
    v2  require cosine >= 0.99 on random input     calibrated on a stand-in with no smoothing

v2 was measured against the real kernel afterwards and does not survive it. Sampled across
Z-Image INT4 r32:

    correct reconstruction    cos 0.704 .. 0.998
    correct x 1.5             cos 0.995 max
    correct + 20% noise       cos 0.976 max

A correct `feed_forward.net.2` scores 0.704 and a matrix 20% wrong scores 0.976, so the wrong
matrix outranks the right one and no absolute threshold separates them.

`test_no_absolute_threshold_separates` reproduces that on CPU. The cause is the per-channel
`smooth_factor`: activations are divided by it before quantisation, so a layer whose factors span
a wide range has one channel setting each group's scale and the rest crushed. Sweeping that
spread walks a *correct* reconstruction from cos 0.994 down to 0.586, straight through where a
noised matrix on an easier layer sits. It is the same category error as an FBCache threshold that
does not port between architectures: the number describes the layer, not the correctness.

    python_embeded\\python.exe -s tools/test_svdq_verify.py

Run it directly. There is no pytest in this embedded interpreter -- the commands in CLAUDE.md
that invoke one do not work as written -- so the file carries its own runner. It is still plain
`test_*` functions with bare asserts, so pytest collects it unchanged wherever one exists.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

TOOLS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("svdq_to_bf16", TOOLS / "svdq_to_bf16.py")
svdq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svdq)


class FakeW4A4(nn.Module):
    """A linear layer with the two properties that matter: INT4 activations, and a per-channel
    smooth factor applied before quantisation.

    Not the Nunchaku kernel -- no low-rank branch, no weight quantisation, no packing. It
    reproduces the behaviour the tests turn on: the layer is not a linear function of its input,
    and how badly `x @ W.T` fails to match depends on `smooth_spread`, not on whether `W` is
    right. Both were confirmed against the real kernel; neither needs a GPU to demonstrate.
    """

    def __init__(self, in_features: int, out_features: int, smooth_spread: float = 0.0,
                 seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.randn(out_features, in_features) / in_features ** 0.5
        self.bias = nn.Parameter(torch.randn(out_features) * 0.1, requires_grad=False)
        self.smooth = torch.exp(torch.randn(in_features) * smooth_spread)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smoothed = x / self.smooth
        grouped = smoothed.reshape(*smoothed.shape[:-1], -1, 64)
        scale = grouped.abs().amax(-1, keepdim=True).clamp(min=1e-9) / 7
        quantised = ((grouped / scale).round().clamp(-8, 7) * scale).reshape(x.shape)
        return (quantised * self.smooth) @ self.W.T + self.bias


def _recover(layer: FakeW4A4) -> tuple[torch.Tensor, float]:
    """The identity probe from `recover_weight`, with the bias suppressed the same way."""
    saved = layer.bias.detach().clone()
    layer.bias = nn.Parameter(torch.zeros_like(saved), requires_grad=False)
    eye = torch.eye(layer.in_features).unsqueeze(0)
    recovered = layer(eye)[0].T.contiguous()
    zero_leak = layer(torch.zeros(1, 1, layer.in_features)).abs().max().item()
    layer.bias = nn.Parameter(saved, requires_grad=False)
    return recovered, zero_leak


def _score(layer, weight, probe):
    reference = layer(probe)[0] - layer.bias
    replayed = (probe[0] @ weight.T).float()
    return torch.nn.functional.cosine_similarity(
        reference.float().flatten().unsqueeze(0), replayed.flatten().unsqueeze(0)).item()


def test_identity_probe_is_exact():
    """The claim the whole tool rests on. If this stops holding, the recovered matrix is an
    estimate rather than the kernel's weight, and everything downstream is approximate."""
    layer = FakeW4A4(1024, 512)
    recovered, zero_leak = _recover(layer)
    assert (recovered - layer.W).abs().max().item() == 0.0
    assert zero_leak == 0.0

    smoothed = FakeW4A4(1024, 512, smooth_spread=1.5)
    recovered, _ = _recover(smoothed)
    # Not bit-exact here only because the stand-in divides by the smooth factor in float32; the
    # real kernel folds it into the scale and matched to exactly 0.
    assert (recovered - smoothed.W).abs().max().item() < 1e-6


def test_no_absolute_threshold_separates():
    """The reason `verify` is reported and not gated.

    An easy layer and a hard one, differing only in how wide their smooth factors spread. The
    hard layer's *correct* reconstruction must score below the easy layer's *wrong* one -- which
    means any absolute cosine floor either rejects the correct matrix or accepts the wrong one.
    """
    easy = FakeW4A4(1024, 512, smooth_spread=0.0, seed=0)
    hard = FakeW4A4(1024, 512, smooth_spread=1.5, seed=0)

    scores = {}
    for name, layer in (("easy", easy), ("hard", hard)):
        weight, _ = _recover(layer)
        torch.manual_seed(1)
        probe = torch.randn(1, 64, layer.in_features)
        noised = weight + torch.randn_like(weight) * weight.std() * 0.2
        scores[name] = (_score(layer, weight, probe), _score(layer, noised, probe))

    hard_correct = scores["hard"][0]
    easy_wrong = scores["easy"][1]
    assert hard_correct < easy_wrong, (
        f"expected a correct reconstruction on a hard layer ({hard_correct:.4f}) to score below "
        f"a 20%-wrong one on an easy layer ({easy_wrong:.4f}); if this no longer holds, re-check "
        f"whether a threshold has become defensible")
    # And within one layer it does separate -- which is exactly the trap: it looks like a
    # working gate until the layers differ.
    assert scores["hard"][0] > scores["hard"][1]
    assert scores["easy"][0] > scores["easy"][1]


def test_check_recovery_accepts_a_good_layer():
    layer = FakeW4A4(1024, 512)
    weight, zero_leak = _recover(layer)
    assert svdq.check_recovery(layer, weight, "cpu", torch.float32, zero_leak) is None


def test_check_recovery_rejects_the_failures_it_claims_to():
    layer = FakeW4A4(1024, 512)
    weight, zero_leak = _recover(layer)

    leaked = svdq.check_recovery(layer, weight, "cpu", torch.float32, 0.5)
    assert leaked is not None and "zero input" in leaked

    zeroed = svdq.check_recovery(layer, torch.zeros_like(weight), "cpu", torch.float32, zero_leak)
    assert zeroed is not None and "all zeros" in zeroed

    broken = weight.clone()
    broken[0, 0] = float("nan")
    nonfinite = svdq.check_recovery(layer, broken, "cpu", torch.float32, zero_leak)
    assert nonfinite is not None and "nan" in nonfinite


def test_check_recovery_catches_a_nondeterministic_kernel():
    """A kernel whose output moves between two identical probes is one whose output cannot be
    called 'the weight'. Nothing later in the pipeline would attribute that to this step."""

    class Flaky(FakeW4A4):
        # Input-scaled, so a zero input still returns zero. A constant offset would be caught
        # by the bias check first and this test would pass for the wrong reason -- it did, on
        # the first attempt.
        def forward(self, x):
            out = super().forward(x)
            return out + torch.randn_like(out) * x.abs().mean() * 1e-3

    layer = Flaky(256, 128)
    weight, zero_leak = _recover(layer)
    why = svdq.check_recovery(layer, weight, "cpu", torch.float32, zero_leak)
    assert why is not None and "deterministic" in why


def _fused_weight(rows: int, cols: int) -> torch.Tensor:
    """A weight whose every row is identifiable, so a wrong split is visible, not just unequal."""
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)


def test_split_fused_qkv_calls_the_real_function():
    """The suite used to have no test for `split_fused` at all.

    That is the function that decides which half of a fused `net.0.proj` becomes w3 and which
    becomes w1 -- and svdq_to_bf16's own docstring records that getting it backwards moves the
    relative error from 0.0995 to 1.4199. It was the least tested and most consequential piece
    in the file. These tests call `svdq.split_fused` directly rather than reimplementing it,
    which is what made the rest of this suite stay green under mutation.
    """
    weight = _fused_weight(3 * 8, 4)
    out = svdq.split_fused("noise_refiner.0.attention.to_qkv", weight)
    assert set(out) == {
        "noise_refiner.0.attention.to_q.weight",
        "noise_refiner.0.attention.to_k.weight",
        "noise_refiner.0.attention.to_v.weight",
    }, f"unexpected keys {sorted(out)}"
    # Order matters and is checked against the rows themselves, not just against shapes.
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        chunk = out[f"noise_refiner.0.attention.{name}.weight"]
        expected = weight[index * 8:(index + 1) * 8]
        assert torch.equal(chunk, expected), f"{name} got rows {chunk[0, 0].item()}"
        assert chunk.is_contiguous(), f"{name} is not contiguous"


def test_split_fused_ff_keeps_w3_before_w1():
    """w3 is the TOP half. Measured, not assumed: the docstring records w1+w3 scoring 1.4199
    against w3+w1 at 0.0995 on a real layer. A swap here is exactly the mutation that left the
    old suite green."""
    weight = _fused_weight(2 * 8, 4)
    out = svdq.split_fused("layers.0.feed_forward.net.0.proj", weight)
    assert set(out) == {"layers.0.feed_forward.w3.weight", "layers.0.feed_forward.w1.weight"}
    assert torch.equal(out["layers.0.feed_forward.w3.weight"], weight[:8]), "w3 is not the top half"
    assert torch.equal(out["layers.0.feed_forward.w1.weight"], weight[8:]), "w1 is not the bottom half"


def test_split_fused_net2_is_a_rename_only():
    weight = _fused_weight(8, 4)
    out = svdq.split_fused("layers.0.feed_forward.net.2", weight)
    assert set(out) == {"layers.0.feed_forward.w2.weight"}
    assert torch.equal(out["layers.0.feed_forward.w2.weight"], weight)


def test_split_fused_passes_through_unfused_names():
    weight = _fused_weight(8, 4)
    out = svdq.split_fused("layers.0.attention.to_out.0", weight)
    assert set(out) == {"layers.0.attention.to_out.0.weight"}
    assert torch.equal(out["layers.0.attention.to_out.0.weight"], weight)


def test_split_fused_refuses_indivisible_qkv():
    """A GQA layer can have out_features not divisible by 3. Splitting anyway would silently
    produce three wrong matrices, so the function must fall through to passthrough instead."""
    weight = _fused_weight(10, 4)   # 10 % 3 != 0
    out = svdq.split_fused("layers.0.attention.to_qkv", weight)
    assert set(out) == {"layers.0.attention.to_qkv.weight"}, (
        f"an indivisible qkv was split anyway into {sorted(out)}")


# NOT a test, deliberately not named test_*, and printed by the runner instead.
#
# `recover_weight(tensors, stem, device, dtype)` builds the layer itself from a state dict via
# nunchaku's own module, so it cannot be handed a FakeW4A4 and cannot run on CPU. Everything in
# this file that exercises "the probe" therefore exercises `_recover`, the reimplementation
# above -- not the shipped function. That is why the suite stayed green when the audit mutated
# svdq_to_bf16 three ways at once (w3/w1 swapped, `.T` dropped, q/k/v inverted): none of the
# three is reachable from here.
#
# What the new split_fused tests above DO cover: the w3/w1 order and the q/k/v order, because
# split_fused is a pure tensor operation and the real function is called directly. The `.T`
# mutation lives inside recover_weight and remains untested.
UNTESTED_ON_CPU = (
    "recover_weight() is NOT covered by the CPU tests: it builds its layer from a state dict "
    "through nunchaku and needs a GPU. The probe tests above exercise the in-file "
    "reimplementation `_recover`, so a bug in the shipped recover_weight passes them. The "
    "gpu_ tests below close the `.T` and the scale; everything else about it is still open."
)

# ---------------------------------------------------------------------------------------------
# GPU tests. Named gpu_* rather than test_* so the CPU runner does not collect them, and driven
# by the runner below only when a GPU, nunchaku, and both checkpoints are present. A missing
# prerequisite prints SKIP and is counted -- it must never look like a pass, which is the failure
# mode that let `test_against_the_real_checkpoints_on_this_machine` in the preflight package
# exercise zero files and report PASS for weeks.
# ---------------------------------------------------------------------------------------------

SVDQ_CANDIDATES = (
    Path("D:/ComfyUI-Models/diffusion_models/svdq-int4_r32-beyond-reality-zimage-v2.safetensors"),
    Path(__file__).resolve().parent.parent
    / "ComfyUI/models/diffusion_models/svdq-int4_r32-beyond-reality-zimage-v2.safetensors",
)
NATIVE_REFERENCE = (Path(__file__).resolve().parent.parent
                    / "ComfyUI/models/diffusion_models"
                    / "beyond-reality-zimage-v2_native.safetensors")

# SVDQ keeps diffusers names; the native reference is what `to_native.py` produces.
GPU_PAIRS = (
    ("context_refiner.0.attention.to_out.0", "context_refiner.0.attention.out.weight"),
    ("noise_refiner.0.attention.to_out.0", "noise_refiner.0.attention.out.weight"),
    ("layers.0.attention.to_out.0", "layers.0.attention.out.weight"),
    ("layers.17.attention.to_out.0", "layers.17.attention.out.weight"),
)


def _gpu_prerequisites() -> str | None:
    try:
        import torch as _t
    except Exception as exc:
        return f"torch unavailable ({exc!r})"
    if not _t.cuda.is_available():
        return "no CUDA device"
    try:
        import nunchaku  # noqa: F401
    except Exception as exc:
        return f"nunchaku unavailable ({exc!r})"
    if not any(p.is_file() for p in SVDQ_CANDIDATES):
        return f"no SVDQ checkpoint at {[str(p) for p in SVDQ_CANDIDATES]}"
    if not NATIVE_REFERENCE.is_file():
        return f"no native reference at {NATIVE_REFERENCE}"
    return None


def _load_stem(stem: str):
    from safetensors.torch import safe_open
    path = next(p for p in SVDQ_CANDIDATES if p.is_file())
    out = {}
    with safe_open(str(path), framework="pt") as f:
        available = set(f.keys())
        if f"{stem}.qweight" not in available:
            return None
        for key in available:
            if key.startswith(stem + "."):
                out[key] = f.get_tensor(key)
    return out


def gpu_recover_weight_returns_the_layers_own_linear_map():
    """The `.T` in `recover_weight`, tested against the layer it came from.

    The assertion is a *contrast*, not a threshold, because the absolute number cannot be small:
    `SVDQW4A4Linear` quantizes the activation too, so `layer(x)` on random x carries W4A4
    activation error that `F.linear(x, recovered)` does not. Measured on
    context_refiner.0.attention.to_out.0: 0.101 in the right orientation, 1.413 in the wrong one.
    A threshold tight enough to reject the transpose would also reject the correct answer; the
    ratio between the two separates them by more than a factor of ten.
    """
    import torch
    recover_weight = svdq.recover_weight

    stem, _ = GPU_PAIRS[0]
    tensors = _load_stem(stem)
    assert tensors, f"{stem} not in the SVDQ checkpoint"
    layer, recovered, zero_leak = recover_weight(tensors, stem, "cuda", torch.bfloat16)
    assert recovered.shape[0] == recovered.shape[1], "this stem must be square for the transpose "\
                                                     "contrast to be meaningful"
    assert zero_leak == 0.0, f"bias was not suppressed: zero response {zero_leak}"

    torch.manual_seed(0)
    x = torch.randn(1, 64, recovered.shape[1], dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        got = layer(x).float()
        right = torch.nn.functional.linear(x, recovered, layer.bias).float()
        wrong = torch.nn.functional.linear(x, recovered.T.contiguous(), layer.bias).float()
    rel_right = float((got - right).norm() / right.norm())
    rel_wrong = float((got - wrong).norm() / wrong.norm())
    assert rel_right < 0.25, f"recovered does not reproduce the layer: rel {rel_right:.4f}"
    assert rel_wrong > 10 * rel_right, (
        f"the transposed orientation is not clearly worse (right {rel_right:.4f}, "
        f"wrong {rel_wrong:.4f}); this test would not catch a dropped .T")


def gpu_recovered_weight_has_no_systematic_scale_error():
    """The scale, tested against the published BF16 rather than against another recovery.

    Relative error alone cannot separate a quantizer's noise from a scale bug -- both raise it.
    The least-squares alpha can: with a correct recovery alpha sits at 1 and rescaling by it
    barely moves the residual; with a scale bug alpha absorbs the factor and the residual
    collapses. Measured across four layers: alpha 0.9922 to 0.9981, and rescaling improves the
    relative error by under 0.4% of itself. That is noise, not a factor.
    """
    import torch
    from safetensors.torch import safe_open
    recover_weight = svdq.recover_weight

    checked = 0
    for stem, ref_key in GPU_PAIRS:
        tensors = _load_stem(stem)
        if not tensors:
            continue
        _, recovered, _ = recover_weight(tensors, stem, "cuda", torch.bfloat16)
        with safe_open(str(NATIVE_REFERENCE), framework="pt") as f:
            if ref_key not in set(f.keys()):
                continue
            reference = f.get_tensor(ref_key).cuda().float()
        r = recovered.float()
        assert r.shape == reference.shape, f"{stem}: {tuple(r.shape)} vs {tuple(reference.shape)}"
        alpha = float((r * reference).sum() / (r * r).sum())
        rel_one = float((r - reference).norm() / reference.norm())
        rel_alpha = float((alpha * r - reference).norm() / reference.norm())
        assert 0.97 < alpha < 1.03, f"{stem}: systematic scale factor alpha={alpha:.6f}"
        assert rel_alpha > 0.9 * rel_one, (
            f"{stem}: rescaling by alpha={alpha:.4f} cut the error from {rel_one:.4f} to "
            f"{rel_alpha:.4f}, which is what a scale bug looks like")
        assert rel_one < 0.2, f"{stem}: recovery is {rel_one:.4f} off the published BF16"
        checked += 1
        del recovered, r, reference, tensors
        torch.cuda.empty_cache()
    assert checked >= 2, f"only {checked} layer(s) compared; this test needs the reference file"


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
    skipped = _gpu_prerequisites()
    for name, fn in sorted(globals().items()):
        if not name.startswith("gpu_") or not callable(fn):
            continue
        if skipped:
            # Counted as a failure of coverage, printed as SKIP, never as PASS.
            print(f"SKIP  {name}: {skipped}")
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")

    # Printed on every run, pass or fail. A row of PASS lines reads as "the tool is verified",
    # and for one specific function it is not.
    print(f"\nGAP   {UNTESTED_ON_CPU}")
    if skipped:
        print(f"GAP   the GPU tests did not run ({skipped}), so the `.T` and the scale in the "
              "shipped recover_weight are unverified in this run.")
    raise SystemExit(1 if failures else 0)
