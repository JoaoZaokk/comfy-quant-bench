"""Unit-test CachedTransformerBlocks against a synthetic dual-stream model, with no GPU or weights.

Two properties of the wrapper cannot be checked on HunyuanVideo 1.5, because that checkpoint has
54 double blocks and zero single blocks:

  1. the concatenation order fed to the single-stream blocks. With an empty single-block list the
     cat/split round trip is the identity, so `cat_hidden_states_first` has no observable effect
     and a wrong setting passes unnoticed.
  2. that the first residual survives blocks which update their input in place.

This builds a model small enough to run on CPU that reproduces both: DoubleBlock does `img += ...`
exactly as comfy/ldm/flux/layers.py:246 does, and SingleBlock adds a position ramp so that feeding
it [img, txt] instead of [txt, img] changes the result. ComfyUI's HunyuanVideo concatenates txt
first (comfy/ldm/hunyuan_video/model.py:416), so the wrapper must do the same.

The test drives the wrapper with the cache disabled by threshold, where it is required to be
numerically identical to the plain block loop -- any difference is plumbing, not approximation.

    python tools/fbcache_catorder_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def load_first_block_cache():
    import importlib.util

    node_dir = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "Comfy-WaveSpeed-Fixed"
    spec = importlib.util.spec_from_file_location(
        "wavespeed_fbc", node_dir / "first_block_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DoubleBlock(torch.nn.Module):
    """Dual-stream block that updates both streams in place, like flux's DoubleStreamBlock."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.wi = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)
        self.wt = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, img, txt, vec=None, pe=None):
        img += torch.tanh(img @ self.wi + vec)
        txt += torch.tanh(txt @ self.wt + vec)
        return img, txt


class SingleBlock(torch.nn.Module):
    """Single-stream block whose output depends on token position, like an attention block."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, x, vec=None, pe=None):
        # `pe` is indexed by absolute position, so the order the caller concatenated the two
        # streams in is observable here. That is what makes the order testable at all.
        return x + torch.tanh(x @ self.w + pe[:, :x.shape[1]])


class SyntheticModel(torch.nn.Module):
    """Mimics comfy/ldm/hunyuan_video/model.py: txt-first concatenation, then slice img back out."""

    def __init__(self, dim: int, doubles: int, singles: int):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList(
            [DoubleBlock(dim, 100 + i) for i in range(doubles)])
        self.single_blocks = torch.nn.ModuleList(
            [SingleBlock(dim, 200 + i) for i in range(singles)])

    def forward(self, img, txt, vec, pe):
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        return img[:, txt.shape[1]:]


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def run(model, fbc, cat_first: bool, threshold: float, inputs):
    """Run the model with its double blocks swapped for the cached wrapper."""
    import unittest.mock

    img, txt, vec, pe = inputs
    wrapper = torch.nn.ModuleList([
        fbc.CachedTransformerBlocks(
            model.double_blocks, model.single_blocks,
            residual_diff_threshold=threshold,
            cat_hidden_states_first=cat_first,
            return_hidden_states_only=False,
            return_hidden_states_first=True,
            accept_hidden_states_first=True,
        )
    ])
    fbc.set_current_cache_context(fbc.create_cache_context())
    with unittest.mock.patch.object(model, "double_blocks", wrapper), \
            unittest.mock.patch.object(model, "single_blocks", torch.nn.ModuleList()):
        out = model(img.clone(), txt.clone(), vec, pe)
    fbc.set_current_cache_context(None)
    return out


def main() -> int:
    fbc = load_first_block_cache()
    torch.manual_seed(0)
    dim, img_len, txt_len = 32, 24, 8
    img = torch.randn(1, img_len, dim)
    txt = torch.randn(1, txt_len, dim)
    vec = torch.randn(1, 1, dim) * 0.1
    pe = torch.arange(img_len + txt_len).float().reshape(1, -1, 1).repeat(1, 1, dim) * 0.01
    inputs = (img, txt, vec, pe)

    failures = 0
    for singles in (0, 4):
        model = SyntheticModel(dim, doubles=6, singles=singles)
        reference = model(img.clone(), txt.clone(), vec, pe)

        print(f"\n{'=' * 70}")
        print(f"{len(model.double_blocks)} double blocks, {singles} single blocks")
        print(f"{'=' * 70}")

        # Two separate code paths, and only one of them is reachable from the node:
        #   threshold <= 0  -> the plain loop at the top of forward(). ApplyFBCacheOnModel returns
        #                      the model unpatched at this setting, so nothing reaches it in
        #                      normal use, but it is public API on the class.
        #   threshold > 0   -> the cached path. A threshold this small never satisfies the
        #                      similarity gate, so every step is a miss and the wrapper is
        #                      required to reproduce the plain loop exactly.
        for branch, threshold in (("no-cache branch", 0.0), ("cached branch", 1e-9)):
            print(f"  {branch} (threshold={threshold:g})")
            for cat_first in (False, True):
                out = run(model, fbc, cat_first, threshold=threshold, inputs=inputs)
                error = relative_l2(out, reference)
                verdict = "exact" if error < 1e-6 else f"DIVERGES {error:.4f}"
                print(f"    cat_hidden_states_first={str(cat_first):<5}  "
                      f"relL2 {error:.6f}   {verdict}")
                # txt-first is what the model does, so False must always reproduce it.
                if not cat_first and error >= 1e-6:
                    print("      FAIL: must match the unpatched model")
                    failures += 1
            print()

    # The residual must be non-zero even though DoubleBlock updates its input in place.
    model = SyntheticModel(dim, doubles=6, singles=4)
    residuals = []
    original = fbc.get_can_use_cache

    def record(residual, threshold, *args, **kwargs):
        residuals.append(residual.abs().max().item())
        return original(residual, threshold, *args, **kwargs)

    fbc.get_can_use_cache = record
    run(model, fbc, cat_first=False, threshold=0.2, inputs=inputs)
    fbc.get_can_use_cache = original

    print(f"\n{'=' * 70}")
    print("FIRST RESIDUAL through an in-place block")
    print(f"{'=' * 70}")
    peak = max(residuals) if residuals else 0.0
    print(f"  max|block0_out - block0_in| = {peak:.6f}")
    if peak == 0.0:
        print("  FAIL: aliased to the block's in-place output, so the cache can never fire")
        failures += 1
    else:
        print("  ok: the reference was copied before the block ran")

    print(f"\n{'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
