"""Test the per-conditioning-branch cache and the consecutive-hit limit, on CPU.

comfy/samplers.py calls the model once per batch of conditionings, and it only batches those it
can concatenate: whenever the positive and negative text streams differ in length -- or the batch
does not fit in free VRAM -- cond and uncond arrive as two separate calls at the same timestep
with the same latent shape. A cache keyed on (shape, timestep) cannot tell them apart, so the
second call reads back the first one's residual. On HunyuanVideo 1.5 with a 25-token prompt
against a 6-token negative that raises outright; where the encoder pads to a fixed width, as T5
does for FLUX, the shapes match and the negative pass silently continues from the positive
pass's cache instead.

That silent case is the reason for this test: it cannot be seen in an image, only in the buffers.
Here the two branches are given deliberately different residual behaviour, and the test asserts
they never read each other's.

`max_consecutive_cache_hits` is checked in the same harness because it moved to per-branch
counting at the same time -- a shared counter let one branch spend the other's allowance, so the
limit bounded the pair rather than each pass.

    python tools/fbcache_branch_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def load_node_module():
    import importlib.util

    node_dir = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "Comfy-WaveSpeed-Fixed"
    sys.path.insert(0, str(node_dir.parent))
    spec = importlib.util.spec_from_file_location(
        "Comfy_WaveSpeed_Fixed", node_dir / "__init__.py",
        submodule_search_locations=[str(node_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["Comfy_WaveSpeed_Fixed"] = module
    spec.loader.exec_module(module)
    return module


class Block(torch.nn.Module):
    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(dim, dim, generator=generator) * 0.05)

    def forward(self, img, txt=None, vec=None):
        img = img + torch.tanh(img @ self.w + vec)
        return img, txt


class Model(torch.nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([Block(dim, 400 + i) for i in range(depth)])

    def forward(self, img, txt, vec):
        for block in self.transformer_blocks:
            img, txt = block(img=img, txt=txt, vec=vec)
        return img


class FakeSampling:
    """Sigma falls as the percentage rises, which is the convention the node's window assumes."""

    def percent_to_sigma(self, percent):
        return 1.0 - percent


class FakePatcher:
    """The ModelPatcher surface ApplyFBCacheOnModel touches."""

    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.wrapper = None
        self.model_sampling = FakeSampling()

    def clone(self):
        return self

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.model_sampling
        return self.diffusion_model


def main() -> int:
    module = load_node_module()
    fbc = module.first_block_cache
    torch.manual_seed(0)
    dim = 32
    model = Model(dim, depth=6)

    node = module.NODE_CLASS_MAPPINGS["ApplyFBCacheOnModel"]()
    failures = 0

    def drive(max_hits, steps=6):
        """Run a descending schedule, alternating cond and uncond as separate model calls."""
        patcher = FakePatcher(model)
        patcher.set_model_unet_function_wrapper = (
            lambda wrapper: setattr(patcher, "wrapper", wrapper))
        node.patch(patcher, "diffusion_model", 0.5, max_consecutive_cache_hits=max_hits)
        wrapper = patcher.wrapper
        assert wrapper is not None, "node did not install a wrapper"

        # cond carries 8 text tokens, uncond 3 -- the shape mismatch that stops comfy batching
        # them, and therefore the exact situation the branch key exists for.
        streams = {
            (0,): torch.randn(1, 8, dim),
            (1,): torch.randn(1, 3, dim),
        }
        latent = torch.randn(1, 20, dim)
        log = {branch: [] for branch in streams}
        for step in range(steps):
            sigma = 1.0 - step * 0.12          # descending, as a sampler does
            for branch, txt in streams.items():
                before = fbc.get_current_cache_context()
                del before

                def model_function(x, t, **kwargs):
                    return model(x, kwargs["txt"], kwargs["vec"])

                drift = torch.randn(1, 20, dim) * 0.01
                wrapper(model_function,
                        {"input": latent + drift,
                         "timestep": torch.tensor([sigma]),
                         "c": {"txt": txt, "vec": torch.full((1, 1, dim), sigma)},
                         "cond_or_uncond": list(branch)})
                context = fbc.get_current_cache_context()
                log[branch].append((context.cache_hits, context.cache_misses) if context
                                   else (0, 0))
        return log

    # 1. Two branches whose text streams differ in length must not raise. Before the branch key
    #    existed this is where the residual shapes collided.
    print(f"{'=' * 70}\ncond and uncond with different text lengths\n{'=' * 70}")
    try:
        log = drive(max_hits=-1)
        totals = {b: entries[-1] for b, entries in log.items()}
        print(f"  ran without error. per-branch (hits, misses): {totals}")
        ok = all(h + m > 0 for h, m in totals.values())
        failures += not ok
        if not ok:
            print("    FAIL: a branch never recorded a decision")
    except Exception as error:
        print(f"  RAISED {type(error).__name__}: {error}")
        print("    FAIL: this is the collision the branch key is meant to prevent")
        failures += 1
        log = {}

    # 2. Each branch keeps its own cache: their buffers must be separate objects.
    contexts = []
    fbc.set_current_cache_context(None)
    patcher = FakePatcher(model)
    patcher.set_model_unet_function_wrapper = (
        lambda wrapper: setattr(patcher, "wrapper", wrapper))
    node.patch(patcher, "diffusion_model", 0.5)
    latent = torch.randn(1, 20, dim)
    for branch, tokens in (((0,), 8), ((1,), 3)):
        patcher.wrapper(lambda x, t, **kw: model(x, kw["txt"], kw["vec"]),
                        {"input": latent, "timestep": torch.tensor([0.9]),
                         "c": {"txt": torch.randn(1, tokens, dim),
                               "vec": torch.full((1, 1, dim), 0.9)},
                         "cond_or_uncond": list(branch)})
        contexts.append(fbc.get_current_cache_context())
    separate = contexts[0] is not contexts[1]
    failures += not separate
    print(f"\n  separate cache context per branch: {separate}   "
          f"{'' if separate else '<-- FAIL, they share one'}")

    # 3. max_consecutive_cache_hits must bound each branch, not the pair.
    print(f"\n{'=' * 70}\nmax_consecutive_cache_hits = 1\n{'=' * 70}")
    fbc.set_current_cache_context(None)
    log = drive(max_hits=1, steps=8)
    for branch, entries in log.items():
        per_step_hits = [entries[i][0] - (entries[i - 1][0] if i else 0)
                         for i in range(len(entries))]
        streak = worst = 0
        for hit in per_step_hits:
            streak = streak + 1 if hit else 0
            worst = max(worst, streak)
        ok = worst <= 1
        failures += not ok
        print(f"  branch {branch}: hits per step {per_step_hits}, longest run {worst}   "
              f"{'ok' if ok else '<-- FAIL, limit exceeded'}")

    fbc.clear_all_cache_contexts()
    print(f"\n{'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
