"""A/B a diffusion model with and without WaveSpeed FBCache, headless.

Detection is not correctness. `ApplyFBCacheOnModel` routes a model to one of two very different
code paths, and the routing is structural: a model that merely *looks* like FLUX (has both
`double_blocks` and `single_blocks`) gets FLUX's whole `forward_orig` replaced by WaveSpeed's
rewrite, arguments and all. If that rewrite disagrees with the real model's forward in any
argument, the model is patched into something that either raises or silently computes the wrong
thing. So this runs the sampler for real rather than trusting the node's own log line.

Conditioning is synthesised at the checkpoint's own `context_in_dim` instead of loading a 16 GB
text encoder: the question here is whether the patched forward survives a real sampling loop with
correctly shaped tensors, and a random context answers that at a fraction of the cost. Output
quality is meaningless with random conditioning -- for that, rerun with a real encoder.

    python tools/fbcache_probe.py --model hunyuanvideo1.5_720p_t2v_fp16.safetensors \
        --steps 4 --frames 9 --size 256
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="file name inside models/diffusion_models (UNet only)")
    source.add_argument("--checkpoint",
                        help="file name inside models/checkpoints. Loaded through "
                             "CheckpointLoaderSimple and prompted with its own CLIP, which is "
                             "the only way to reach the UNet route: SD1.5 and SDXL ship as "
                             "all-in-one checkpoints, not bare diffusion models.")
    parser.add_argument("--prompt", default="a red apple on a wooden table",
                        help="used only with --checkpoint")
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.12],
                        help="one or more residual_diff_threshold values, swept against a single "
                             "baseline so the model is loaded once")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--frames", type=int, default=9, help="video frames (1 for an image model)")
    parser.add_argument("--size", type=int, default=256, help="square pixel side")
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="above 1.0 the sampler also runs the negative pass, which is what "
                             "exercises the per-conditioning-branch cache")
    parser.add_argument("--highvram", action="store_true",
                        help="keep weights resident, as ComfyUI's --highvram does. Required for "
                             "any timing on a model that does not comfortably fit: when the "
                             "weights are streamed back in per step, that transfer dominates and "
                             "the two baselines disagree by more than the cache is worth.")
    return parser.parse_args()


def describe(diffusion_model, fbc) -> dict:
    """Report where the node will send this model -- by asking it, not by guessing.

    This used to reimplement the routing rules, and promptly disagreed with the node about two
    models: it labelled Wan "UNSUPPORTED" in the same run where the node routed it successfully.
    That is the exact duplication the node itself was suffering from, so ask the one function
    that decides.
    """
    names = ("double_blocks", "single_blocks", "input_blocks", "middle_block",
             "output_blocks", "transformer_blocks", "layers", "joint_blocks", "blocks")
    present = {n: hasattr(diffusion_model, n) for n in names}
    try:
        route = fbc.detect_model_route(diffusion_model).description
    except ValueError as error:
        route = f"UNSUPPORTED -- {str(error).splitlines()[0]}"
    return {"class": type(diffusion_model).__name__,
            "present": [n for n, v in present.items() if v],
            "route": route}


def resolve_context_dim(diffusion_model, params) -> int:
    """Width of the text stream this model expects, asked of the model rather than assumed.

    A flat 4096 works for the FLUX family and fails on LTX-2.5, whose text embedding carries the
    video and audio captions side by side. Its `preprocess_text_embeds` splits the context at
    `cross_attention_dim` and pushes each half through its own connector -- so a context of the
    wrong width does not merely mis-condition, it raises in `extra_conds` before sampling starts:

        RuntimeError: Sizes of tensors must match except in dimension 1.
                      Expected size 4096 but got size 2048

    At exactly `cross_attention_dim + audio_cross_attention_dim` the same function short-circuits
    (av_model.py:587) and returns the context untouched, which is what a synthetic probe wants:
    the connectors are not what is under test here, the block stack is.
    """
    context_in_dim = getattr(params, "context_in_dim", None)
    if context_in_dim:
        return int(context_in_dim)
    video = getattr(diffusion_model, "cross_attention_dim", None)
    if video:
        audio = getattr(diffusion_model, "audio_cross_attention_dim", None) or 0
        return int(video) + int(audio)
    return 4096


def run_once(model, latent, positive, negative, steps: int, seed: int,
             cfg: float = 1.0) -> tuple[torch.Tensor, float]:
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview  # noqa: F401  (import keeps node-side monkeypatches consistent)

    noise = comfy.sample.prepare_noise(latent, seed, None)
    started = time.perf_counter()
    out = comfy.sample.sample(
        model, noise, steps, cfg, "euler", "simple",
        positive, negative, latent,
        denoise=1.0, disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=None, callback=None,
        disable_pbar=True, seed=seed)
    torch.cuda.synchronize()
    return out, time.perf_counter() - started


def main() -> int:
    args = parse_args()

    import comfy.model_management
    import comfy.sd
    import folder_paths

    extra_paths = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra_paths.is_file():
        from utils.extra_config import load_extra_path_config

        load_extra_path_config(str(extra_paths))

    if args.highvram:
        # Set before the first load, because load_diffusion_model consults vram_state to decide
        # whether the weights may live on the GPU at all.
        comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
        comfy.model_management.set_vram_to = comfy.model_management.VRAMState.HIGH_VRAM
        device = comfy.model_management.get_torch_device()
        total = comfy.model_management.get_total_memory(device)
        free = comfy.model_management.get_free_memory(device)
        print(f"highvram: weights pinned to the GPU "
              f"({free / 2**30:.1f} of {total / 2**30:.1f} GiB free)", flush=True)

    clip = None
    load_started = time.perf_counter()
    if args.checkpoint:
        path = folder_paths.get_full_path_or_raise("checkpoints", args.checkpoint)
        print(f"loading checkpoint {path}", flush=True)
        model, clip, _vae = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=False, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))[:3]
    else:
        candidate = Path(args.model)
        path = str(candidate) if candidate.is_file() else \
            folder_paths.get_full_path_or_raise("diffusion_models", args.model)
        print(f"loading {path}", flush=True)
        model = comfy.sd.load_diffusion_model(path)
    print(f"loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

    # The package directory has dashes, so it cannot be imported by name. ComfyUI itself loads it
    # by file path, and that is what nodes.py does at boot -- mirror it here. Loaded before the
    # route is reported, because reporting the route means asking this module.
    import importlib.util

    node_dir = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "Comfy-WaveSpeed-Fixed"
    sys.path.insert(0, str(node_dir.parent))
    spec = importlib.util.spec_from_file_location(
        "Comfy_WaveSpeed_Fixed", node_dir / "__init__.py",
        submodule_search_locations=[str(node_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["Comfy_WaveSpeed_Fixed"] = module
    spec.loader.exec_module(module)
    ApplyFBCacheOnModel = module.NODE_CLASS_MAPPINGS["ApplyFBCacheOnModel"]

    diffusion_model = model.get_model_object("diffusion_model")
    info = describe(diffusion_model, module.first_block_cache)
    print(f"\nclass      : {info['class']}")
    print(f"attributes : {' '.join(info['present'])}")
    print(f"route      : {info['route']}\n", flush=True)

    params = getattr(diffusion_model, "params", None)
    context_dim = resolve_context_dim(diffusion_model, params)
    latent_channels = model.model.latent_format.latent_channels
    scale = model.model.latent_format.__dict__.get("scale_factor", 8) or 8
    side = max(args.size // 8, 8)
    temporal = model.model.latent_format.__class__.__name__
    frames = args.frames if args.frames == 1 else (args.frames - 1) // 4 + 1
    shape = ([1, latent_channels, frames, side, side] if args.frames > 1
             else [1, latent_channels, side, side])
    print(f"latent {shape}  (latent_format {temporal}, scale {scale})")
    print(f"context {[1, args.context_tokens, context_dim]}\n", flush=True)

    device = comfy.model_management.get_torch_device()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.zeros(shape, device="cpu")
    context = torch.randn([1, args.context_tokens, context_dim],
                          generator=generator, dtype=torch.float32) * 0.02
    # HunyuanVideo's encode_adm reads kwargs["pooled_output"] unconditionally, so a conditioning
    # dict without it raises before the model is ever called. The width is irrelevant when
    # vec_in_dim is None (vector_in does not exist), which is the case for HunyuanVideo 1.5.
    vec_in_dim = getattr(params, "vec_in_dim", None) or 768
    pooled = torch.zeros([1, vec_in_dim], dtype=torch.float32)
    if clip is not None:
        # A checkpoint brings its own encoder, so use it: SD1.5 and SDXL disagree with each other
        # about context width and about whether a pooled vector is required at all, and guessing
        # either wrong fails before the model runs.
        tokens = clip.tokenize(args.prompt)
        positive = [list(clip.encode_from_tokens_scheduled(tokens)[0])]
        positive[0][0] = positive[0][0].clone()
        negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(""))[0])]
        print(f"encoded prompt through the checkpoint's CLIP: "
              f"context {list(positive[0][0].shape)}\n", flush=True)
        del clip
        comfy.model_management.soft_empty_cache()
    else:
        positive = [[context, {"pooled_output": pooled}]]
        negative = [[torch.zeros_like(context), {"pooled_output": pooled.clone()}]]

    results = {}

    # Count how often the cache actually short-circuited. Without this a "speedup" cannot be
    # attributed: the first sampling run on a cold context pays CUDA autotuning and allocator
    # growth that the second one does not, which on the first version of this probe produced a
    # 3.85x "speedup" at a threshold low enough that no cache hit was possible.
    counters = {"forwards": 0, "misses": 0}
    fbc = module.first_block_cache
    original_remaining = fbc.CachedTransformerBlocks.call_remaining_transformer_blocks
    original_forward = fbc.CachedTransformerBlocks.forward

    def counting_forward(self, *a, **kw):
        counters["forwards"] += 1
        return original_forward(self, *a, **kw)

    def counting_remaining(self, *a, **kw):
        counters["misses"] += 1
        return original_remaining(self, *a, **kw)

    fbc.CachedTransformerBlocks.forward = counting_forward
    fbc.CachedTransformerBlocks.call_remaining_transformer_blocks = counting_remaining

    # The baseline is measured twice, first and last. Whichever config runs first pays for
    # whatever the two-step warm-up did not finish paying -- and on a model too large to stay
    # fully resident, that is a lot: the same Wan 14B baseline timed 32s in one process and 115s
    # in another, which is larger than any effect being measured. Two baselines that disagree
    # mean the numbers in between are not comparable, and the run says so instead of reporting a
    # speedup built on a penalised reference.
    runs = ([("baseline", 0.0)] + [(f"t={t:g}", t) for t in args.threshold]
            + [("baseline2", 0.0)])
    for label, threshold in runs:
        print(f"--- {label} (threshold={threshold}) ---", flush=True)
        try:
            patched = ApplyFBCacheOnModel().patch(
                model, "diffusion_model", threshold, start=0.0, end=1.0,
                max_consecutive_cache_hits=-1)[0]
        except Exception:
            print(f"{label}: PATCH FAILED")
            traceback.print_exc()
            results[label] = None
            continue
        try:
            # Warm-up pass, discarded: it absorbs kernel autotuning and allocator growth so the
            # timed pass measures the sampler rather than the first-call tax.
            run_once(patched, latent, positive, negative, 2, args.seed, args.cfg)
            counters["forwards"] = counters["misses"] = 0
            out, elapsed = run_once(patched, latent, positive, negative,
                                    args.steps, args.seed, args.cfg)
        except Exception:
            print(f"{label}: SAMPLING FAILED")
            traceback.print_exc()
            results[label] = None
            continue
        hits = counters["forwards"] - counters["misses"]
        print(f"{label}: {elapsed:.2f}s for {args.steps} steps "
              f"({elapsed / args.steps:.3f} s/step)  "
              f"block-calls {counters['forwards']}, cache hits {hits}", flush=True)
        # "0 hits" on its own is not a finding, it is a question. In ComfyUI the node prints its
        # reason breakdown from patch_get_output_data, which wraps execution.get_output_data --
        # a module this probe never loads, so the line would be lost exactly where it is most
        # needed. Ask for it directly instead.
        fbc.clear_all_cache_contexts(report=True)
        results[label] = (out.float().cpu(), elapsed, counters["forwards"], hits)
        comfy.model_management.soft_empty_cache()

    print(f"\n{'=' * 74}")
    print(f"model  : {args.model}")
    print(f"route  : {info['route']}")
    print(f"shape  : latent {shape}, {args.steps} steps\n")
    base = results.get("baseline")
    base2 = results.get("baseline2")
    if base is None:
        print("baseline FAILED -- the unpatched path itself did not run, so nothing "
              "can be concluded about FBCache")
        return 1
    # Time everything against the *slower* of the two baselines is the wrong way round: the
    # faster one is the warm one, and it is the honest reference. Report both so the reader can
    # see how much of any speedup is really just the reference being cold.
    reference = min(base[1], base2[1]) if base2 else base[1]
    drift = abs(base[1] - base2[1]) / max(base[1], base2[1]) if base2 else 0.0
    print(f"{'threshold':>10}  {'seconds':>9}  {'speedup':>8}  {'hits':>9}  {'relL2':>8}")
    print(f"{'baseline':>10}  {base[1]:9.2f}  {'1.00x':>8}  {'-':>9}  {'-':>8}")
    if base2:
        drift_note = "same run, unpatched both times"
        print(f"{'baseline2':>10}  {base2[1]:9.2f}  {base[1] / base2[1]:7.2f}x  "
              f"{'-':>9}  {'-':>8}   {drift_note}")
    failed = 0
    for label, threshold in (runs[1:-1] if base2 else runs[1:]):
        entry = results.get(label)
        if entry is None:
            print(f"{threshold:>10g}  {'FAILED -- this model does not fit the selected path':<40}")
            failed += 1
            continue
        speedup = reference / entry[1] if entry[1] else float("nan")
        diff = ((entry[0] - base[0]).norm() / base[0].norm().clamp(min=1e-12)).item()
        print(f"{threshold:>10g}  {entry[1]:9.2f}  {speedup:7.2f}x  "
              f"{entry[3]:4d}/{entry[2]:<4d}  {diff:8.4f}")
        # These counters wrap CachedTransformerBlocks, which only the generic route uses. On the
        # unet and flux routes there is nothing here to count, so a zero in this column means
        # "this probe did not instrument it", not "the cache never fired". The node itself does
        # count those routes -- the `[WaveSpeed] FBCache: N hits, M misses` line printed above
        # each run comes from the node's own counters and covers every route, including flux.
        # Read that line, not this column, when the column says 0/0.
        if entry[2] == 0:
            print(f"           (this column does not instrument the {info['route'].split()[0]} "
                  f"route -- see the [WaveSpeed] FBCache line above for the node's own hit count; "
                  f"relL2 {'> 0 also shows the cache fired' if diff > 1e-6 else '== 0 shows it did not'})")
        elif entry[3] == 0 and speedup > 1.05:
            print("           WARNING: zero cache hits, so this speedup is noise, not FBCache")
    if base2 and drift > 0.10:
        print(f"\n!! The two unpatched runs disagree by {drift * 100:.0f}%. Nothing above is a "
              f"reliable timing:\n   whichever config runs first absorbs load the later ones do "
              f"not, and on a model that\n   cannot stay fully resident that dwarfs the cache. "
              f"The relL2 column is still valid --\n   it is deterministic. Rerun smaller, or on "
              f"a model that fits, before quoting a speedup.")
    elif base2:
        print(f"\nThe two unpatched runs agree within {drift * 100:.0f}%, so the timings above "
              f"are comparable.\nSpeedups are against the faster of the two ({reference:.2f}s).")
    print("\nrelL2 is against the unpatched run at the same seed. Conditioning is random, so it "
          "measures\nhow far the cache moved the trajectory -- not visual quality.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
