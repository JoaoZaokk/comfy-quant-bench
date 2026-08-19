"""Unit-test CachedTransformerBlocks against a tuple hidden state, LTX-2.5's shape, on CPU.

LTX-2.5's BasicAVTransformerBlock takes `x: Tuple[Tensor, Tensor]` -- video and audio travelling
together -- reaches its text through v_context/a_context keywords that never move, and updates
both parts in place before returning the pair. Three things the wrapper originally could not do:
subtract a tuple, run with no encoder stream to thread, and survive in-place updates.

Testing this on the real 20 GB checkpoint would take minutes per run and would not isolate the
mechanism; a synthetic block with the same shape does it in a second, and the properties checked
here are exactly the ones that would be silently wrong on the real model:

  * with the cache unable to fire, the wrapper must reproduce the plain loop exactly
  * the first residual must be non-zero despite the in-place update
  * with the cache firing, both parts must change -- an implementation that carried only the
    video part would still look plausible on a video-only comparison

    python tools/fbcache_tuple_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

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


class AVBlock(torch.nn.Module):
    """Mirrors BasicAVTransformerBlock: tuple in, tuple out, both parts updated in place."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.wv = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)
        self.wa = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None,
                transformer_options=None):
        vx, ax = x
        vx.add_(torch.tanh(vx @ self.wv + v_context))
        if ax.numel():
            ax.add_(torch.tanh(ax @ self.wa + a_context))
        return vx, ax


class AVModel(torch.nn.Module):
    """Calls its blocks the way LTXAVModel does: block((vx, ax), v_context=..., a_context=...)."""

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [AVBlock(dim, 300 + i) for i in range(depth)])

    def forward(self, vx, ax, v_context, a_context):
        for block in self.transformer_blocks:
            vx, ax = block((vx, ax), v_context=v_context, a_context=a_context,
                           transformer_options={})
        return vx, ax


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def run(model, fbc, threshold: float, inputs, steps: int = 1):
    """Drive the model with its block list replaced by the cached wrapper."""
    import unittest.mock

    vx, ax, v_context, a_context = inputs
    wrapper = torch.nn.ModuleList([
        fbc.CachedTransformerBlocks(
            model.transformer_blocks, None,
            residual_diff_threshold=threshold,
            return_hidden_states_only=True,
        )
    ])
    fbc.set_current_cache_context(fbc.create_cache_context())
    outputs = []
    with unittest.mock.patch.object(model, "transformer_blocks", wrapper):
        for _ in range(steps):
            outputs.append(model(vx.clone(), ax.clone(), v_context, a_context))
    fbc.set_current_cache_context(None)
    return outputs


def main() -> int:
    fbc = load_first_block_cache()
    torch.manual_seed(0)
    dim, v_len, a_len = 32, 40, 12
    vx = torch.randn(1, v_len, dim)
    ax = torch.randn(1, a_len, dim)
    v_context = torch.randn(1, 1, dim) * 0.1
    a_context = torch.randn(1, 1, dim) * 0.1
    inputs = (vx, ax, v_context, a_context)

    failures = 0
    model = AVModel(dim, depth=6)
    ref_v, ref_a = model(vx.clone(), ax.clone(), v_context, a_context)

    print(f"{'=' * 70}\ntuple hidden state (video {v_len} tokens, audio {a_len} tokens)\n"
          f"{'=' * 70}")

    # A threshold this small never satisfies the gate, so every step misses and the wrapper is
    # required to reproduce the plain loop exactly.
    (out_v, out_a), = run(model, fbc, 1e-9, inputs)
    err_v, err_a = relative_l2(out_v, ref_v), relative_l2(out_a, ref_a)
    ok = err_v < 1e-6 and err_a < 1e-6
    failures += not ok
    print(f"  cache cannot fire: video relL2 {err_v:.6f}, audio relL2 {err_a:.6f}   "
          f"{'exact' if ok else 'FAIL -- must match the unpatched loop'}")

    # The first residual must survive the in-place update.
    residuals = []
    original = fbc.get_can_use_cache

    def record(residual, threshold, *args, **kwargs):
        residuals.append(tuple(p.abs().max().item() for p in fbc.parts_of(residual)))
        return original(residual, threshold, *args, **kwargs)

    fbc.get_can_use_cache = record
    run(model, fbc, 0.2, inputs, steps=3)
    fbc.get_can_use_cache = original
    peak = max((max(r) for r in residuals), default=0.0)
    ok = peak > 0.0
    failures += not ok
    print(f"  first residual through in-place blocks: max {peak:.6f}   "
          f"{'ok' if ok else 'FAIL -- aliased, cache can never fire'}")
    print(f"    per part: {residuals[0] if residuals else '-'}")

    # With the cache firing, the cached residual must be applied to BOTH parts. Checked with a
    # drifting input, the way a sampler actually moves: feeding the same tensors every step makes
    # the stored residual exactly the true one, so a cached step reproduces the real answer and
    # the comparison proves nothing. An implementation that carried only the video part would
    # sail through any video-only check while audio silently kept the uncached value.
    import unittest.mock

    drift = [torch.randn(1, v_len, dim) * 0.05 for _ in range(4)]
    audio_drift = [torch.randn(1, a_len, dim) * 0.05 for _ in range(4)]

    def sweep(threshold):
        wrapper = torch.nn.ModuleList([
            fbc.CachedTransformerBlocks(model.transformer_blocks, None,
                                        residual_diff_threshold=threshold,
                                        return_hidden_states_only=True)
        ])
        fbc.set_current_cache_context(fbc.create_cache_context())
        results = []
        with unittest.mock.patch.object(model, "transformer_blocks", wrapper):
            for dv, da in zip(drift, audio_drift):
                results.append(model(vx + dv, ax + da, v_context, a_context))
        context = fbc.get_current_cache_context()
        hits = context.cache_hits if context else 0
        fbc.set_current_cache_context(None)
        return results, hits

    exact, exact_hits = sweep(1e-9)
    cached, cached_hits = sweep(5.0)
    moved_v = max(relative_l2(c[0], e[0]) for c, e in zip(cached, exact))
    moved_a = max(relative_l2(c[1], e[1]) for c, e in zip(cached, exact))
    ok = cached_hits > 0 and moved_v > 0 and moved_a > 0
    failures += not ok
    print(f"  cache firing ({cached_hits} hits vs {exact_hits}): video differs by {moved_v:.6f}, "
          f"audio by {moved_a:.6f}")
    if not ok:
        print(f"    FAIL: {'no hits at all' if not cached_hits else 'a part never received the '
                          'cached residual'}")

    # Empty audio is a real LTX-2.5 state (the block guards on ax.numel()).
    empty_inputs = (vx, torch.zeros(1, 0, dim), v_context, a_context)
    ref_empty = model(vx.clone(), torch.zeros(1, 0, dim), v_context, a_context)
    try:
        (ev, ea), = run(model, fbc, 1e-9, empty_inputs)
        err = relative_l2(ev, ref_empty[0])
        ok = err < 1e-6 and ea.numel() == 0
        failures += not ok
        print(f"  empty audio stream: video relL2 {err:.6f}, audio numel {ea.numel()}   "
              f"{'ok' if ok else 'FAIL'}")
    except Exception as error:
        print(f"  empty audio stream: RAISED {type(error).__name__}: {error} <-- FAIL")
        failures += 1

    print(f"\n{'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
