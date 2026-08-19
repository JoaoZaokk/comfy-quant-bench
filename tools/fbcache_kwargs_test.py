"""Cover three behaviours of CachedTransformerBlocks that no other test touches, on CPU.

1. Double-stream-only kwargs must not reach the single-stream blocks. HunyuanVideo splits its
   modulation dims per stream (`modulation_dims_img` / `modulation_dims_txt`) while its
   SingleStreamBlock takes one `modulation_dims`, so forwarding the double blocks' kwargs
   verbatim raises TypeError. That was fixed upstream in #120 and lost when this fork rewrote
   the file; both the cached and the uncached path have to drop them.

2. Spatiotemporal Guidance must switch caching off. LTX passes `stg_self_attn_blocks` through
   transformer_options and rewrites the options for those block indices only. This wrapper
   stands in for the whole list, so it cannot reproduce a per-index change -- it has to run
   every block plainly instead of silently applying the perturbation to none of them.

3. `direction_threshold` must actually gate, and -1 must switch the gate off. The cosine test
   was hardcoded at 0.95 and invisible: a model whose residuals rotate faster than that refuses
   every hit, and raising residual_diff_threshold cannot help because it is not what refused.

    python tools/fbcache_kwargs_test.py
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
        "wavespeed_fbc_kwargs", node_dir / "first_block_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DoubleBlock(torch.nn.Module):
    """Takes the per-stream modulation dims, and updates its input in place as FLUX's does."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, img, txt, modulation_dims_img=None, modulation_dims_txt=None,
                transformer_options=None):
        img += torch.tanh(img @ self.w)
        return img, txt


class SingleBlock(torch.nn.Module):
    """Takes no per-stream modulation dims at all -- passing them is a TypeError, as upstream."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, x, modulation_dims=None, transformer_options=None):
        return x + torch.tanh(x @ self.w)


def build(fbc, threshold: float):
    dim = 16
    return fbc.CachedTransformerBlocks(
        torch.nn.ModuleList([DoubleBlock(dim, 10 + i) for i in range(3)]),
        torch.nn.ModuleList([SingleBlock(dim, 30 + i) for i in range(2)]),
        residual_diff_threshold=threshold,
        return_hidden_states_only=False,
        cat_hidden_states_first=False,
    ), dim


def call(wrapper, img, txt, options=None):
    return wrapper(img=img, txt=txt,
                   modulation_dims_img=[(0, 4, None)],
                   modulation_dims_txt=[(0, 2, None)],
                   transformer_options=options or {})


def main() -> int:
    fbc = load_first_block_cache()
    torch.manual_seed(0)
    failures = 0

    print(f"{'=' * 72}\n1. modulation_dims_* must be dropped before the single blocks\n{'=' * 72}")
    for label, threshold, steps in (("uncached path (threshold 0)", 0.0, 1),
                                    ("cached path (threshold 0.5)", 0.5, 3)):
        wrapper, dim = build(fbc, threshold)
        fbc.set_current_cache_context(fbc.create_cache_context())
        try:
            for step in range(steps):
                img = torch.randn(1, 6, dim) * (1.0 + 0.1 * step)
                call(wrapper, img, torch.randn(1, 2, dim))
            print(f"  {label:<32} ran, single blocks never saw them")
        except TypeError as error:
            print(f"  {label:<32} TypeError: {error}")
            print("    FAIL: this is upstream #120 -- the kwargs were forwarded verbatim")
            failures += 1
        fbc.set_current_cache_context(None)

    print(f"\n{'=' * 72}\n2. STG must disable caching, not silently drop the "
          f"perturbation\n{'=' * 72}")
    wrapper, dim = build(fbc, 0.5)
    img_seed = torch.randn(1, 6, dim)
    txt = torch.randn(1, 2, dim)

    # Same schedule twice: once with STG on, once through a wrapper whose threshold is 0 (which
    # is the plain loop by definition). With STG on they must agree exactly.
    fbc.set_current_cache_context(fbc.create_cache_context())
    stg_out = [call(wrapper, img_seed.clone() * (1 + 0.05 * s), txt,
                    {"stg_self_attn_blocks": [1]})[0] for s in range(4)]
    stg_context = fbc.get_current_cache_context()
    stg_hits = stg_context.cache_hits
    fbc.set_current_cache_context(None)

    plain, _ = build(fbc, 0.0)
    plain.load_state_dict(wrapper.state_dict())
    fbc.set_current_cache_context(fbc.create_cache_context())
    plain_out = [call(plain, img_seed.clone() * (1 + 0.05 * s), txt)[0] for s in range(4)]
    fbc.set_current_cache_context(None)

    drift = max((a - b).abs().max().item() for a, b in zip(stg_out, plain_out))
    ok = stg_hits == 0 and drift == 0.0
    failures += not ok
    print(f"  cache hits with STG active : {stg_hits}   {'ok' if not stg_hits else '<-- FAIL'}")
    print(f"  max deviation from the plain loop: {drift:.3e}   "
          f"{'ok' if drift == 0.0 else '<-- FAIL, STG output was altered'}")

    print(f"\n{'=' * 72}\n3. direction_threshold gates, and -1 turns it off\n{'=' * 72}")
    # A difference concentrated in one huge component: small in relative L1, large in angle.
    # This is the case that made the hardcoded gate invisible -- raising residual_diff_threshold
    # cannot rescue it, because magnitude was never what refused.
    previous = torch.ones(1, 1000)
    current = previous.clone()
    current[0, 0] += 200.0
    mae = (current - previous).abs().mean() / previous.abs().mean()
    cos = torch.nn.functional.cosine_similarity(previous.flatten(), current.flatten(), dim=0)
    print(f"  crafted residual pair: relative L1 {mae:.4f}, cosine {cos:.4f}")

    for label, direction, expected in (("direction 0.95 (default)", 0.95, False),
                                       ("direction -1 (disabled)", -1.0, True)):
        fbc.set_current_cache_context(fbc.create_cache_context())
        verdict = fbc.compute_tensor_similarity(previous, current, 0.25, direction)
        context = fbc.get_current_cache_context()
        ok = verdict == expected
        failures += not ok
        print(f"  {label:<26} similar={verdict!s:<5} expected={expected!s:<5} "
              f"blocked_by_direction={context.blocked_by_direction}  "
              f"{'ok' if ok else '<-- FAIL'}")
        fbc.set_current_cache_context(None)

    fbc.clear_all_cache_contexts(report=False)
    print(f"\n{'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
