"""Demonstrate the bug that made this node decorative for every model except LTX 2.3, on CPU.

FBCache decides everything from

    first_residual = (output of block 0) - (input to block 0)

so it needs the input as it was *before* block 0 ran. ComfyUI's transformer blocks update their
input in place -- `img += ...` at comfy/ldm/flux/layers.py:246 -- and return the very object they
were handed, so without a copy the subtraction is `x - x`.

The fork gated that copy on the model's class name:

    clone_original_hidden_states=model_class_name == "LTXVModel"    # fbcache_nodes.py:301 at HEAD

Zero is the worst value this can take, because it fails silently in both directions at once:
relative MAE is 0/clamp(0) = 0, which passes any threshold, while the cosine similarity of a null
vector is 0, which fails the direction gate. Every step reports a miss, the output stays
bit-identical to the unpatched model, and nothing raises. The user sees a node that is switched
on and a run that is not one second faster.

This reproduces the old behaviour by aliasing instead of copying -- the same thing the flag did --
on a NextDiT-shaped stand-in, which is the route the fork's own headline commit shipped:

    8e75e34  "ApplyFirstBlockCache is now compatible with Z-Image Turbo | Thereshold tested: 0.200"

Z-Image is a NextDiT, so it took the generic path with the flag off.

    python tools/fbcache_clone_test.py
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
        "wavespeed_fbc_clone", node_dir / "first_block_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InPlaceBlock(torch.nn.Module):
    """A single-stream block that updates its input in place, as ComfyUI's blocks do."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, x, transformer_options=None):
        x += torch.tanh(x @ self.w)
        return x


def build(fbc, threshold: float, dim: int = 24, depth: int = 8):
    return fbc.CachedTransformerBlocks(
        torch.nn.ModuleList([InPlaceBlock(dim, 700 + i) for i in range(depth)]),
        None,
        residual_diff_threshold=threshold,
        return_hidden_states_only=True,      # NextDiT / Z-Image: one stream, no encoder
    )


def drive(fbc, wrapper, steps: int = 6, dim: int = 24):
    """A descending schedule whose states drift slowly, so a working cache should fire."""
    fbc.set_current_cache_context(fbc.create_cache_context())
    generator = torch.Generator().manual_seed(99)
    base = torch.randn(1, 12, dim, generator=generator)
    outputs, residuals = [], []
    for step in range(steps):
        state = base * (1.0 + 0.02 * step)
        # Positionally, the way NextDiT calls its own layers -- `layer(x, ...)`.
        outputs.append(wrapper(state, transformer_options={}))
        buffered = fbc.get_buffer("first_hidden_states_residual")
        if buffered is not None:
            residuals.append(buffered.abs().max().item())
    context = fbc.get_current_cache_context()
    tally = (context.cache_hits, context.cache_misses)
    fbc.set_current_cache_context(None)
    return outputs, tally, residuals


def main() -> int:
    fbc = load_first_block_cache()
    torch.manual_seed(0)
    failures = 0
    dim = 24

    torch.manual_seed(0)
    fixed = build(fbc, 0.5, dim)
    torch.manual_seed(0)
    aliased = build(fbc, 0.5, dim)
    aliased.load_state_dict(fixed.state_dict())
    torch.manual_seed(0)
    plain = build(fbc, 0.0, dim)          # threshold 0 is the plain loop by definition
    plain.load_state_dict(fixed.state_dict())

    plain_out, _, _ = drive(fbc, plain, dim=dim)

    print(f"{'=' * 74}\nthe copy is made (current behaviour)\n{'=' * 74}")
    fixed_out, fixed_tally, fixed_residuals = drive(fbc, fixed, dim=dim)
    fixed_drift = max((a - b).abs().max().item() for a, b in zip(fixed_out, plain_out))
    biggest = max(fixed_residuals) if fixed_residuals else 0.0
    print(f"  first residual, largest component : {biggest:.6f}")
    print(f"  (hits, misses)                    : {fixed_tally}")
    print(f"  deviation from the unpatched loop : {fixed_drift:.6f}")

    print(f"\n{'=' * 74}\nthe copy is skipped (clone_original_hidden_states=False at HEAD)"
          f"\n{'=' * 74}")
    original_clone = fbc.clone_state
    fbc.clone_state = lambda state: state          # alias instead of copy -- the old flag, exactly
    try:
        aliased_out, aliased_tally, aliased_residuals = drive(fbc, aliased, dim=dim)
    finally:
        fbc.clone_state = original_clone
    aliased_drift = max((a - b).abs().max().item() for a, b in zip(aliased_out, plain_out))
    aliased_biggest = max(aliased_residuals) if aliased_residuals else 0.0
    print(f"  first residual, largest component : {aliased_biggest:.6f}")
    print(f"  (hits, misses)                    : {aliased_tally}")
    print(f"  deviation from the unpatched loop : {aliased_drift:.6f}")

    print(f"\n{'=' * 74}\nverdict\n{'=' * 74}")
    checks = [
        ("with the copy, the first residual is non-zero", biggest > 0.0),
        ("with the copy, the cache fires", fixed_tally[0] > 0),
        ("with the copy, the output actually changes", fixed_drift > 0.0),
        ("without it, the first residual is exactly zero", aliased_biggest == 0.0),
        ("without it, the cache never fires", aliased_tally[0] == 0),
        ("without it, the output is bit-identical to unpatched", aliased_drift == 0.0),
    ]
    for label, ok in checks:
        failures += not ok
        print(f"  {label:<58} {'ok' if ok else '<-- FAIL'}")

    print("\n  Bit-identical output plus zero hits plus no error is the whole failure: the node\n"
          "  reports success, costs the full compute, and changes nothing. That is what shipped\n"
          "  for Z-Image, SD3.5, HunyuanVideo and every other non-LTXVModel on this path.")

    fbc.clear_all_cache_contexts(report=False)
    print(f"\n{'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
