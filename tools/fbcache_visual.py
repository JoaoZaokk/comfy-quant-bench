"""End-to-end FBCache A/B on HunyuanVideo 1.5 with real conditioning and decoded frames.

The threshold sweep in fbcache_probe.py drives the sampler with random conditioning, which is
enough to prove the cache fires and to time it, but its relL2 is not a quality number: with noise
for a prompt the model is not tracking any image, so the trajectory wanders more than it would on
a real generation and the divergence reads high for reasons that have nothing to do with the cache.

This runs the actual graph -- the same nodes the UI uses -- encodes a real prompt through the
Qwen2.5-VL + ByT5 pair HunyuanVideo 1.5 expects, samples with and without the cache at a fixed
seed, and decodes both to pixels. The comparison is then on the image, where "how much worse" is
a question that means something.

The text encoders are unloaded before the transformer is loaded: together they do not fit
alongside a 16.6 GB fp16 transformer on a 24 GB card.

    python tools/fbcache_visual.py --threshold 0.12 0.2 --steps 20 --frames 33
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

PROMPT = ("a red apple on a wooden table, soft daylight, shallow depth of field, "
          "slow camera push in")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unet", default="hunyuanvideo1.5_720p_t2v_fp16.safetensors")
    parser.add_argument("--clip1", default="qwen_2.5_vl_7b.safetensors")
    parser.add_argument("--clip2", default="byt5_small_glyphxl_fp16.safetensors")
    parser.add_argument("--vae", default="hunyuanvideo15_vae_fp16.safetensors")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.12, 0.2])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path,
                        default=PORTABLE_ROOT / "_fbcache_visual")
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


def save_grid(images: torch.Tensor, path: Path, columns: int = 4) -> None:
    """Write a contact sheet of evenly spaced frames, so one file shows the whole clip."""
    from PIL import Image
    import numpy as np

    count = min(images.shape[0], columns * 2)
    picks = torch.linspace(0, images.shape[0] - 1, count).round().long()
    tiles = [(images[i].clamp(0, 1).numpy() * 255).astype(np.uint8) for i in picks]
    rows = (len(tiles) + columns - 1) // columns
    h, w, _ = tiles[0].shape
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    Image.fromarray(sheet).save(path)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import asyncio

    import comfy.model_management
    import nodes

    extra_paths = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra_paths.is_file():
        from utils.extra_config import load_extra_path_config

        load_extra_path_config(str(extra_paths))

    # ModelSamplingSD3 and EmptyHunyuanLatentVideo live in comfy_extras, which main.py registers
    # at boot. Importing `nodes` alone gets only the core mappings. Custom nodes stay out: this
    # loads the WaveSpeed module by path itself, and pulling in all 60 installed packs would put
    # unrelated code between the model and the measurement.
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))

    print("encoding prompt through the real text encoders...", flush=True)
    clip = nodes.NODE_CLASS_MAPPINGS["DualCLIPLoader"]().load_clip(
        args.clip1, args.clip2, "hunyuan_video_15", "default")[0]
    encode = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    positive = encode.encode(clip, args.prompt)[0]
    negative = encode.encode(clip, "")[0]
    del clip, encode
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    print("text encoders unloaded\n", flush=True)

    model = nodes.NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(args.unet, "default")[0]
    model = nodes.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]().patch(model, args.shift)[0]
    vae = nodes.NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(args.vae)[0]
    latent = nodes.NODE_CLASS_MAPPINGS["EmptyHunyuanLatentVideo"]().generate(
        args.width, args.height, args.frames, 1)[0]

    module = load_node_module()
    ApplyFBCacheOnModel = module.NODE_CLASS_MAPPINGS["ApplyFBCacheOnModel"]
    fbc = module.first_block_cache
    counters = {"forwards": 0, "misses": 0}
    original_forward = fbc.CachedTransformerBlocks.forward
    original_remaining = fbc.CachedTransformerBlocks.call_remaining_transformer_blocks

    def counting_forward(self, *a, **kw):
        counters["forwards"] += 1
        return original_forward(self, *a, **kw)

    def counting_remaining(self, *a, **kw):
        counters["misses"] += 1
        return original_remaining(self, *a, **kw)

    fbc.CachedTransformerBlocks.forward = counting_forward
    fbc.CachedTransformerBlocks.call_remaining_transformer_blocks = counting_remaining

    sampler = nodes.NODE_CLASS_MAPPINGS["KSampler"]()
    decoder = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]()
    results = {}

    # Sample every configuration first and keep the latents on CPU, then decode. Decoding while
    # the 16.6 GB transformer still holds VRAM pushes the VAE onto ComfyUI's tiled 3D path, which
    # on this build dies with "Inplace update to inference tensor outside InferenceMode" -- an
    # unrelated bug that would otherwise mask the measurement.
    for label, threshold in [("baseline", 0.0)] + [(f"t={t:g}", t) for t in args.threshold]:
        patched = ApplyFBCacheOnModel().patch(model, "diffusion_model", threshold,
                                              start=0.0, end=1.0,
                                              max_consecutive_cache_hits=-1)[0]
        counters["forwards"] = counters["misses"] = 0
        started = time.perf_counter()
        samples = sampler.sample(patched, args.seed, args.steps, args.cfg,
                                 "euler", "simple", positive, negative, latent, 1.0)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        hits = counters["forwards"] - counters["misses"]
        results[label] = {"threshold": threshold, "seconds": elapsed, "hits": hits,
                          "calls": counters["forwards"],
                          "latent": {"samples": samples["samples"].cpu().clone()}}
        print(f"{label:>10}  sampled in {elapsed:7.1f}s  "
              f"hits {hits}/{counters['forwards']}", flush=True)
        comfy.model_management.soft_empty_cache()

    import gc

    del model, patched
    fbc.clear_all_cache_contexts()
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"\ntransformer unloaded, {free / 1024**3:.1f} of {total / 1024**3:.1f} GiB free, "
          f"decoding\n", flush=True)

    for label, entry in results.items():
        # Decode inside inference mode. If the untiled decode still does not fit, ComfyUI falls
        # back to decode_tiled_3d, whose output comes from `tiled_scale_multidim` -- itself
        # decorated @torch.inference_mode() -- and is then handed to a `process_output` that does
        # `image.add_(1.0)` in place. Outside inference mode that combination raises
        # "Inplace update to inference tensor outside InferenceMode", an upstream bug in
        # comfy/sd.py:1141 unrelated to anything measured here. Running the whole decode inside
        # inference mode makes the in-place update legal and lets the fallback work.
        with torch.inference_mode():
            images = decoder.decode(vae, entry["latent"])[0].cpu().clone()
        path = args.out / f"{Path(args.unet).stem}_{label.replace('=', '')}.png"
        save_grid(images, path)
        entry["images"] = images
        entry["path"] = path
        print(f"{label:>10}  -> {path.name}", flush=True)

    print(f"\n{'=' * 76}")
    print(f"{args.unet}   {args.width}x{args.height}, {args.frames} frames, "
          f"{args.steps} steps, cfg {args.cfg}, seed {args.seed}")
    print(f'prompt: "{args.prompt}"\n')
    base = results["baseline"]
    print(f"{'threshold':>10}  {'seconds':>9}  {'speedup':>8}  {'hits':>10}  "
          f"{'pixel relL2':>12}  {'max px diff':>12}")
    print(f"{'baseline':>10}  {base['seconds']:9.1f}  {'1.00x':>8}  {'-':>10}  "
          f"{'-':>12}  {'-':>12}")
    for label, entry in results.items():
        if label == "baseline":
            continue
        diff = entry["images"] - base["images"]
        rel = (diff.norm() / base["images"].norm().clamp(min=1e-12)).item()
        print(f"{entry['threshold']:>10g}  {entry['seconds']:9.1f}  "
              f"{base['seconds'] / entry['seconds']:7.2f}x  "
              f"{entry['hits']:>4d}/{entry['calls']:<5d}  {rel:12.4f}  "
              f"{diff.abs().max().item():12.4f}")
    print(f"\nframes written to {args.out}")
    print("pixel relL2 is measured after VAE decode, so it is a difference in the image, not in "
          "the latent.\nLook at the contact sheets before calling any of these acceptable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
