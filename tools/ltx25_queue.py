"""Queue an LTX 2.5 text-to-video job on a running ComfyUI and report what came back.

Built after three failed attempts to drive LTX 2.5 from a standalone script. The failures were
not bad luck; they were the wrong venue:

  * `comfy.sample.sample` with a plain latent is not how this model runs. LTX 2.5 needs
    `LTXVEmptyLatentAudio`, `LTXVConcatAVLatent`, `LTXVConditioning` and `SamplerCustomAdvanced`
    with explicit sigmas -- an audio-video latent, not an image one.
  * ComfyUI-MultiGPU's `distorch_2` opens with a relative import and cannot be loaded as a loose
    module. That surfaced *after* a 39 GiB checkpoint had been read into RAM.
  * comfy-aimdo's DynamicVRAM never came up, because `init_devices` returns False until
    `control.init()` has loaded the DLL, and main.py calls those two 200 lines apart.

The server already solves all three, and it validates the graph before it runs a step. So this
builds the API-format prompt and posts it, rather than reimplementing the pipeline.

    python_embeded\\python.exe -s tools/ltx25_queue.py \\
        --unet ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors \\
        --donor cuda:1 --virtual-vram 8

Start the server first; `iniciar_comfy.bat video` (port 8190; LTX 2.5 needs --disable-dynamic-vram) is the one this defaults to.

HTTP goes through `comfy_client.py` (review of 2026-09-29). The previous inline loop had its own
`get()` inside the poll with no exception handling (one network hiccup killed the run) and polled
only the submitted server -- but this graph uses DisTorch2, i.e. ComfyUI-MultiGPU, whose workers
record the result in THEIR /history, so the client could report "still running" over a finished
render. `run_and_wait` polls the workers too, with the same deadline.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_client as cc  # noqa: E402

PORTABLE_ROOT = Path(__file__).resolve().parent.parent

# From the workflow the MultiGPU pack ships: a distilled LTX takes three real steps and a
# terminal zero. Not a schedule to invent -- the distillation was trained against it.
DISTILLED_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"


def build_prompt(args) -> dict:
    """The API-format graph. Node ids are strings; links are [node_id, output_index]."""
    return {
        "1": {"class_type": "UNETLoaderDisTorch2MultiGPU", "inputs": {
            "unet_name": args.unet,
            "weight_dtype": "default",
            # default, not fp8: on a checkpoint that carries its own per-layer format the widget
            # is not inert. Measured 2026-08-19 -- it casts the tensors the profile deliberately
            # left in high precision and moves the output more than a LoRA does.
            "compute_device": args.compute_device,
            "virtual_vram_gb": args.virtual_vram,
            "donor_device": args.donor,
            "expert_mode_allocations": args.expert or "",
        }},
        "2": {"class_type": "CLIPLoaderMultiGPU", "inputs": {
            "clip_name": args.clip, "type": "ltxv", "device": args.clip_device}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": args.prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": args.negative, "clip": ["2", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": args.video_vae}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": args.audio_vae}},
        "7": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": args.width, "height": args.height, "length": args.length, "batch_size": 1}},
        "8": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": args.length, "frame_rate": args.frame_rate, "batch_size": 1,
            "audio_vae": ["6", 0]}},
        "9": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["7", 0], "audio_latent": ["8", 0]}},
        "10": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(args.frame_rate)}},
        "11": {"class_type": "CFGGuider", "inputs": {
            "model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1], "cfg": args.cfg}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}},
        "13": {"class_type": "ManualSigmas", "inputs": {"sigmas": args.sigmas}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0],
            "latent_image": ["9", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["5", 0]}},
        "18": {"class_type": "SaveImage", "inputs": {
            "images": ["17", 0], "filename_prefix": args.prefix}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8190")
    parser.add_argument("--unet", required=True)
    parser.add_argument("--clip", default="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    parser.add_argument("--clip-device", default="cpu", choices=["cpu", "cuda:0", "cuda:1"],
                        help="the LTX 2.5 encoder is a 24 GiB gemma4-12B; cpu keeps it off the "
                             "card that has to hold a 20-39 GiB transformer")
    parser.add_argument("--video-vae", default="ltx-2.5-video-vae-bf16.safetensors")
    parser.add_argument("--audio-vae", default="ltx-2.5-audio-vae-bf16.safetensors")
    parser.add_argument("--compute-device", default="cuda:0")
    parser.add_argument("--donor", default="cuda:1", choices=["cpu", "cuda:0", "cuda:1"],
                        help="where blocks that do not fit go. A second GPU's VRAM beats pageable "
                             "host RAM; cpu is the fallback when the donor card is too small")
    parser.add_argument("--virtual-vram", type=float, default=8.0)
    parser.add_argument("--expert", default=None,
                        help="explicit per-device quota, e.g. 'cuda:0,13gb;cuda:1,7gb;cpu,*'. "
                             "Overrides --donor/--virtual-vram when given")
    parser.add_argument("--prompt", default="a brass trumpet on a wooden table, morning light")
    parser.add_argument("--negative", default="blurry, out of focus, low contrast, washed out")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--length", type=int, default=49)
    parser.add_argument("--frame-rate", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--sigmas", default=DISTILLED_SIGMAS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefix", default="ltx25_accept")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    comfy = cc.Comfy(args.server)
    try:
        comfy.system_stats()
    except cc.REDE as exc:
        raise SystemExit(f"no ComfyUI at {args.server} ({exc}). Start "
                         f"iniciar_comfy.bat video first.")

    prompt = build_prompt(args)
    print(f"queueing {args.unet}\n  {args.width}x{args.height}x{args.length} @ {args.frame_rate} "
          f"fps, seed {args.seed}, sigmas {args.sigmas}")
    print(f"  DisTorch2 compute {args.compute_device}, "
          f"{args.expert or f'{args.virtual_vram} GiB from {args.donor}'}", flush=True)

    try:
        entry = cc.run_and_wait(comfy, prompt, args.timeout, poll_s=3.0)
    except urllib.error.HTTPError as exc:
        # The server validates the whole graph before running a step, and this is the reason to
        # use it: a bad link or a missing file comes back as a named node and a reason (printed
        # by comfy_client), rather than as an exception thrown after a 39 GiB read.
        print(f"the server refused the graph ({exc.code})")
        return 1
    except cc.PromptRefused as exc:
        print(f"the server refused part of the graph: {exc}")
        return 1
    except cc.PollTimeout:
        print(f"still running after {args.timeout}s; check the server console")
        return 1

    if entry.status != "success":
        print(f"finished unsuccessfully after {entry.wall:.1f}s: {entry.erro}")
        return 1
    images = [i for out in (entry.hist.get("outputs") or {}).values()
              for i in out.get("images", [])]
    print(f"completed in {entry.wall:.1f}s (server-side {entry.server_side_s}s), "
          f"{len(images)} frame(s)")
    for image in images[:4]:
        print(f"  {image.get('subfolder','')}/{image.get('filename')}")
    if len(images) > 4:
        print(f"  ... and {len(images) - 4} more")
    if entry.cache_hit:
        print("  CACHE HIT: server-side execution under the threshold -- this timed nothing")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
