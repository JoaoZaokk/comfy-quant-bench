"""Trace every decision point in WaveSpeed's FBCache and report what each one actually did.

A cache that never fires looks exactly like a cache that is switched off: same output, same time,
no error. `get_buffer` returns None when there is no context, `get_can_use_cache` turns that into
False, and the sampler runs to completion reporting nothing. So "0 hits" on its own does not say
whether the mechanism was bypassed, mis-wired, or correctly deciding not to fire.

This wraps each function in the chain and records, per call, whether it ran and what it returned,
then prints the two similarity metrics `compute_tensor_similarity` compares against the threshold.
Those two numbers are the answer: the gate is `mae_relative < threshold AND cos_sim >
direction_threshold`. When the direction bound is the one refusing, no value of the tolerance
slider can ever produce a hit -- which is exactly the shape a dead cache takes.

    python tools/fbcache_audit.py --model hunyuanvideo1.5_720p_t2v_fp16.safetensors \
        --steps 20 --threshold 0.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


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


class Audit:
    """Records every call through the cache decision chain."""

    def __init__(self, fbc):
        self.fbc = fbc
        self.calls = {}
        self.similarity = []      # (mae, cos, threshold, verdict, direction_threshold)
        self.buffer_reads = []    # (name, hit_or_miss)
        self.buffer_writes = []   # name
        self.residual_norms = []  # max |first_residual| per decision
        self.context_none = 0

    def bump(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def install(self) -> None:
        fbc = self.fbc

        original_get_buffer = fbc.get_buffer
        original_set_buffer = fbc.set_buffer
        original_similarity = fbc.compute_tensor_similarity
        original_can_use = fbc.get_can_use_cache
        original_apply = fbc.apply_cached_residual
        original_context = fbc.get_current_cache_context
        original_forward = fbc.CachedTransformerBlocks.forward
        original_remaining = fbc.CachedTransformerBlocks.call_remaining_transformer_blocks

        def get_buffer(name):
            self.bump("get_buffer")
            out = original_get_buffer(name)
            self.buffer_reads.append((name, out is not None))
            return out

        def set_buffer(name, buffer):
            self.bump("set_buffer")
            self.buffer_writes.append(name)
            return original_set_buffer(name, buffer)

        def compute_tensor_similarity(t1, t2, threshold, *args, **kwargs):
            self.bump("compute_tensor_similarity")
            # Recomputed here rather than read out of the node, so the audit stays valid even if
            # the node's internals change. Same formulas as the original.
            t1_f, t2_f = t1.float(), t2.float()
            mae = ((t1_f - t2_f).abs().mean() / t1_f.abs().mean().clamp(min=1e-8)).item()
            cos = F.cosine_similarity(t1_f.flatten().unsqueeze(0),
                                      t2_f.flatten().unsqueeze(0)).item()
            verdict = original_similarity(t1, t2, threshold, *args, **kwargs)
            direction = args[0] if args else kwargs.get("direction_threshold", 0.95)
            self.similarity.append((mae, cos, threshold, verdict, direction))
            return verdict

        def get_can_use_cache(residual, threshold, *args, **kwargs):
            self.bump("get_can_use_cache")
            # An all-zero first residual is the signature of the input being aliased to the
            # block's in-place output, and it cannot be told apart from a legitimate miss by
            # looking at hit counts alone. A tuple hidden state (LTX-2.5) reports every part.
            parts = self.fbc.parts_of(residual)
            self.residual_norms.append(max(p.float().abs().max().item() for p in parts))
            return original_can_use(residual, threshold, *args, **kwargs)

        def apply_cached_residual(hidden_states, first_residual, encoder_hidden_states=None):
            self.bump("apply_cached_residual")
            if original_get_buffer("hidden_states_residual") is None:
                self.bump("apply_cached_residual:NO_BUFFER_returned_unchanged")
            return original_apply(hidden_states, first_residual, encoder_hidden_states)

        def get_current_cache_context():
            out = original_context()
            if out is None:
                self.context_none += 1
            return out

        def forward(inner_self, *a, **kw):
            self.bump("CachedTransformerBlocks.forward")
            return original_forward(inner_self, *a, **kw)

        def call_remaining(inner_self, *a, **kw):
            self.bump("call_remaining_transformer_blocks")
            return original_remaining(inner_self, *a, **kw)

        fbc.get_buffer = get_buffer
        fbc.set_buffer = set_buffer
        fbc.compute_tensor_similarity = compute_tensor_similarity
        fbc.get_can_use_cache = get_can_use_cache
        fbc.apply_cached_residual = apply_cached_residual
        fbc.get_current_cache_context = get_current_cache_context
        fbc.CachedTransformerBlocks.forward = forward
        fbc.CachedTransformerBlocks.call_remaining_transformer_blocks = call_remaining

    def report(self, steps: int) -> None:
        print(f"\n{'=' * 78}")
        print("CALL CHAIN -- was each stage reached, and how often\n")
        expected = {
            "CachedTransformerBlocks.forward": "once per model call; 0 means the blocks were "
                                               "never swapped in",
            "get_can_use_cache": "once per forward; 0 means the cache decision never ran",
            "get_buffer": "at least once per decision",
            "compute_tensor_similarity": "runs only when a previous residual exists, so it is "
                                         "one fewer than the forwards",
            "set_buffer": "two per cache miss",
            "call_remaining_transformer_blocks": "one per cache MISS",
            "apply_cached_residual": "one per cache HIT",
        }
        for name, note in expected.items():
            count = self.calls.get(name, 0)
            mark = "  " if count else "!!"
            print(f" {mark} {name:<38} {count:>5}   {note}")
        for name, count in sorted(self.calls.items()):
            if name not in expected:
                print(f" ?? {name:<38} {count:>5}")

        misses = self.calls.get("call_remaining_transformer_blocks", 0)
        hits = self.calls.get("apply_cached_residual", 0)
        print(f"\n forwards {self.calls.get('CachedTransformerBlocks.forward', 0)}  "
              f"= misses {misses} + hits {hits}   (over {steps} sampler steps)")
        print(f" get_current_cache_context returned None: {self.context_none} times "
              f"(any non-zero during sampling silently disables the cache)")

        reads = {}
        for name, found in self.buffer_reads:
            slot = reads.setdefault(name, [0, 0])
            slot[found] += 1
        print("\nBUFFER TRAFFIC -- a read that always misses means nothing was ever stored\n")
        for name, (empty, found) in sorted(reads.items()):
            print(f"    read  {name:<34} found {found:>4}   empty {empty:>4}")
        writes = {}
        for name in self.buffer_writes:
            writes[name] = writes.get(name, 0) + 1
        for name, count in sorted(writes.items()):
            print(f"    write {name:<34} {count:>4}")

        if self.residual_norms:
            low, high = min(self.residual_norms), max(self.residual_norms)
            print(f"\nFIRST RESIDUAL -- max|block0_out - block0_in| per step: "
                  f"{low:.6f} .. {high:.6f}")
            if high == 0.0:
                print("    ALL ZERO. The reference was aliased to a tensor the block updates in "
                      "place,\n    so the cache compares nothing against nothing and can never fire.")

        print(f"\n{'=' * 78}")
        print("THE GATE -- compute_tensor_similarity returns "
              "(mae_relative < threshold) AND (cos_sim > direction_threshold)")
        print("            both are tunable now; the direction bound used is shown per row\n")
        if not self.similarity:
            print("  never evaluated -- no two consecutive steps were ever compared")
            return
        print(f"  {'#':>3}  {'mae_rel':>9}  {'thresh':>7}  {'mae ok':>7}  "
              f"{'cos_sim':>9}  {'cos ok':>7}  {'verdict':>8}")
        for index, (mae, cos, threshold, verdict, direction) in enumerate(self.similarity, 1):
            print(f"  {index:>3}  {mae:9.4f}  {threshold:7.3f}  "
                  f"{'yes' if mae < threshold else 'NO':>7}  {cos:9.4f}  "
                  f"{'yes' if cos > direction else 'NO':>7}  "
                  f"{'HIT' if verdict else 'miss':>8}")
        mae_blocked = sum(1 for m, _, t, _, _ in self.similarity if m >= t)
        cos_blocked = sum(1 for _, c, _, _, d in self.similarity if c <= d)
        print(f"\n  blocked by mae     : {mae_blocked}/{len(self.similarity)}")
        print(f"  blocked by cos_sim : {cos_blocked}/{len(self.similarity)}   "
              f"<-- not reachable by any threshold setting")
        maes = [m for m, _, _, _, _ in self.similarity]
        coss = [c for _, c, _, _, _ in self.similarity]
        print(f"\n  mae_relative range : {min(maes):.4f} .. {max(maes):.4f}")
        print(f"  cos_sim range      : {min(coss):.4f} .. {max(coss):.4f}")
        needed = max(maes)
        print(f"\n  a threshold above {needed:.4f} would clear the mae gate on every step"
              if cos_blocked == 0 else
              f"\n  raising the threshold cannot help: cos_sim blocks {cos_blocked} steps")


def main() -> int:
    args = parse_args()

    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import folder_paths

    extra_paths = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra_paths.is_file():
        from utils.extra_config import load_extra_path_config

        load_extra_path_config(str(extra_paths))

    path = folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    model = comfy.sd.load_diffusion_model(path)
    diffusion_model = model.get_model_object("diffusion_model")
    print(f"model  : {type(diffusion_model).__name__}")

    module = load_node_module()
    audit = Audit(module.first_block_cache)
    audit.install()

    ApplyFBCacheOnModel = module.NODE_CLASS_MAPPINGS["ApplyFBCacheOnModel"]
    patched = ApplyFBCacheOnModel().patch(model, "diffusion_model", args.threshold,
                                          start=0.0, end=1.0,
                                          max_consecutive_cache_hits=-1)[0]

    params = getattr(diffusion_model, "params", None)
    context_dim = getattr(params, "context_in_dim", None) or 4096
    vec_in_dim = getattr(params, "vec_in_dim", None) or 768
    side = max(args.size // 8, 8)
    channels = model.model.latent_format.latent_channels
    # 4D for image models, 5D for video. Building a temporal axis unconditionally makes an image
    # model fail inside its own forward -- comfy/ldm/flux/model.py does `bs, c, h, w = x.shape`.
    shape = ([1, channels, side, side] if args.frames == 1
             else [1, channels, (args.frames - 1) // 4 + 1, side, side])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.zeros(shape)
    context = torch.randn([1, args.context_tokens, context_dim],
                          generator=generator, dtype=torch.float32) * 0.02
    pooled = torch.zeros([1, vec_in_dim])
    positive = [[context, {"pooled_output": pooled}]]
    negative = [[torch.zeros_like(context), {"pooled_output": pooled.clone()}]]

    print(f"latent : {shape}, {args.steps} steps, threshold {args.threshold}\n", flush=True)
    noise = comfy.sample.prepare_noise(latent, args.seed, None)
    started = time.perf_counter()
    comfy.sample.sample(patched, noise, args.steps, 1.0, "euler", "simple",
                        positive, negative, latent, denoise=1.0, disable_pbar=True,
                        seed=args.seed)
    torch.cuda.synchronize()
    print(f"sampled in {time.perf_counter() - started:.1f}s")

    audit.report(args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
