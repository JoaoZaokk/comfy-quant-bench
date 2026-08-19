"""Capture the activations a diffusion model really sees, so precision can be chosen per layer.

Every quantization decision in this project so far was made from the weights alone. That answers
half the question: ConvRot W4A4 quantizes the weight *and* the activation, and the activation half
is the one that hurts. Its scheme is one scale per token across every channel, so a single outlier
channel sets the scale for the whole vector -- and which layers get outlier-heavy inputs is a
property of the model running, not of any tensor on disk.

This runs the real model through real sampling steps and keeps, per candidate Linear:

  * `sample`         a reservoir of real input rows, fp16 on CPU -- the tensors the layer was
                     actually handed, not a Gaussian stand-in
  * `channel_absmax` running per-channel max|x| over every row seen, not just the sampled ones
  * `crest`          per-token max|x| / rms(x), summarised as mean/p50/p99/max. Kept as a
                     diagnostic, NOT as a predictor: measured across 170 Z-Image layers its
                     rank correlation with the W4A4 error the converter goes on to measure is
                     +0.10, i.e. none. The mechanism is real -- one scale per token means an
                     outlier channel sets the scale for the whole vector -- but it does not
                     reach the output, because the channel that blows up the scale is usually
                     also the channel that dominates the result. `feed_forward.w2` has the
                     highest crest in the model (p50 98, against a theoretical max of
                     sqrt(10240) = 101.2) and `attention.out` has the lowest (p50 16), and
                     `attention.out` holds the second-worst layer. Choose precision by measuring
                     the kernels, not by this number.
  * `calls`/`rows`   how much traffic the layer saw, so a layer that barely ran can be told apart
                     from one that ran constantly

The reservoir is Algorithm R, not "first N rows": the first sampling step of a diffusion model
sees pure noise and looks nothing like the last, and a converter calibrated on step 0 would be
calibrated on the one distribution the model spends the least time in.

Output is a `.calib.pt` consumed by tools/quant_mixed.py. It does not decide anything itself --
deciding needs the weights and the kernels, and that belongs in the converter.

    python_embeded\\python.exe -s tools/calibrate_activations.py \\
        --model beyond-reality-zimage-v2_bf16.safetensors --profile zimage \\
        --clip qwen_3_4b.safetensors --clip-type lumina2 \\
        --prompt-file tools/benchmark_prompt.txt --steps 8 --seeds 1234 5678 \\
        --out calib/zimage_v2.calib.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

# Kept identical to the converter's, and imported by it, so a layer can never be calibrated under
# one definition and quantized under another.
PROFILE_PATTERNS = {
    # Z-Image / Lumina-NextDiT, in ComfyUI's *native* naming -- `attention.qkv` (q, k and v fused
    # into one [3*dim, dim] Linear) and `attention.out`, not the checkpoint's diffusers-style
    # `to_q`/`to_k`/`to_v`/`to_out.0`. A published Z-Image checkpoint is in diffusers naming and
    # must go through tools/to_native.py first; quantizing the diffusers names produces a file
    # ComfyUI silently loads without its scales (see that tool's docstring).
    # 34 blocks (30 `layers` + 2 of each refiner) x 5 Linears = 170.
    # Deliberately excluded: `adaLN_modulation`, every norm, `cap_embedder`, `x_embedder` and
    # `final_layer` -- the patchify/unpatchify pair and the modulation path, which is the "keep
    # the start and the end higher" rule this project settled on.
    "zimage": re.compile(
        r"^(?:layers|noise_refiner|context_refiner)\.\d+\."
        r"(?:attention\.(?:qkv|out)|feed_forward\.w[123])$"
    ),
    "ltx_2_5": re.compile(
        r"^(?:model\.diffusion_model\.)?"
        r"(?:"
        r"transformer_blocks\.\d+\."
        r"(?:"
        r"(?:audio_)?attn\d+\.(?:to_[qkv]|to_out\.\d+)"
        r"|(?:audio_to_video|video_to_audio)_attn\.(?:to_[qkv]|to_out\.\d+)"
        r"|(?:audio_)?ff\.net\.\d+(?:\.proj)?"
        r")"
        r"|(?:audio|video)_embeddings_connector\.transformer_\d+d_blocks\.\d+\."
        r"(?:attn\d+\.(?:to_[qkv]|to_out\.\d+)|ff\.net\.\d+(?:\.proj)?)"
        r")$"
    ),
    # HunyuanVideo 1.5, 54 `double_blocks` x 8 Linears = 432. Corrected 2026-08-19 on first
    # execution: this pattern read `img_attn_qkv` and `img_mlp.fc1`, and the model's modules are
    # `img_attn.qkv` and `img_mlp.0` -- a dot where it expected an underscore, and numbered
    # Sequential entries where it expected named ones. It matched **nothing**, and
    # `calibrate_activations` said so loudly ("matched no Linear in HunyuanVideo") rather than
    # calibrating an empty set, which is the only reason this was cheap to find. The profile had
    # been written from the checkpoint's key names without ever being run against a loaded model.
    # Excluded on purpose, same rule as zimage: `img_mod.lin` / `txt_mod.lin` are the modulation
    # path, and `txt_in`, `vision_in`, `byt5_in`, `time_in` and `final_layer` are the ends.
    "hunyuan_video_15": re.compile(
        r"^double_blocks\.\d+\.(?:(?:img|txt)_attn\.(?:qkv|proj)|(?:img|txt)_mlp\.[02])$"
    ),
    # WAN 2.1, 30 `blocks` x 10 Linears = 300. `vace_blocks` (the VACE control branch) is left
    # out: it only runs when a control input is present, so a calibration would record nothing for
    # it and the converter would then have to guess. Excluded as ever: patch/vace embeddings,
    # `norm_q`/`norm_k`, `time_embedding`, `text_embedding` and the head.
    "wan_2_1": re.compile(
        r"^blocks\.\d+\.(?:(?:self|cross)_attn\.[qkvo]|ffn\.[02])$"
    ),
}

# The patterns above match **module** names, because that is what this file hooks. Downstream,
# `quant_mixed.py` matches **checkpoint key** names, because that is what it rewrites. For Z-Image
# and LTX those are the same string and nothing forced the distinction into the open. For
# HunyuanVideo 1.5 they are not:
#
#     file    double_blocks.0.img_attn_qkv.weight     double_blocks.0.img_mlp.fc1.weight
#     module  double_blocks.0.img_attn.qkv            double_blocks.0.img_mlp.0
#
# ComfyUI renames on load. A calibration keyed by module name would therefore describe layers that
# `quant_mixed` cannot find, and the merge would come back empty -- or worse, partially populated.
# So the calibration is written out keyed by the **file** name, and the translation lives here,
# next to the pattern it belongs to.
MODULE_TO_FILE = {
    "hunyuan_video_15": (
        (re.compile(r"\.(img|txt)_attn\.(qkv|proj)$"), r".\1_attn_\2"),
        (re.compile(r"\.(img|txt)_mlp\.0$"), r".\1_mlp.fc1"),
        (re.compile(r"\.(img|txt)_mlp\.2$"), r".\1_mlp.fc2"),
    ),
    # WAN keeps the module names and prefixes them. Third distinct convention in this table, and
    # the third time the file and the module disagreed -- worth stating plainly: assume they
    # differ until a dump of both says otherwise.
    "wan_2_1": (
        (re.compile(r"^blocks\."), "model.diffusion_model.blocks."),
    ),
}


def to_file_name(profile: str, module_name: str) -> str:
    """Module name -> checkpoint key stem, for profiles where ComfyUI renames on load."""
    for pattern, replacement in MODULE_TO_FILE.get(profile, ()):
        module_name = pattern.sub(replacement, module_name)
    return module_name


# What `quant_mixed.py` matches against checkpoint keys. Only profiles whose file naming differs
# from their module naming need an entry; the rest fall back to PROFILE_PATTERNS.
PROFILE_FILE_PATTERNS = dict(PROFILE_PATTERNS)
PROFILE_FILE_PATTERNS["hunyuan_video_15"] = re.compile(
    r"^double_blocks\.\d+\.(?:(?:img|txt)_attn_(?:qkv|proj)|(?:img|txt)_mlp\.fc[12])$"
)
PROFILE_FILE_PATTERNS["wan_2_1"] = re.compile(
    r"^(?:model\.diffusion_model\.)?blocks\.\d+\.(?:(?:self|cross)_attn\.[qkvo]|ffn\.[02])$"
)


class Reservoir:
    """Algorithm R over token rows, so the sample represents the whole run and not its start.

    Vectorised, because the obvious per-row loop is not viable here: one Z-Image call hands a
    layer ~4100 token rows, and 238 layers over 8 steps and 4 runs is 31 million rows. Instead
    every row in a call draws its slot at once, and only the rows that survive -- after collapsing
    duplicate slots to the last writer, which is what the sequential algorithm would have left --
    cross to the CPU buffer.
    """

    def __init__(self, capacity: int, width: int, seed: int):
        self.capacity = capacity
        # bfloat16, not float16, for the same two bytes. Measured on Z-Image:
        # `layers.0.feed_forward.w2` receives channel magnitudes up to 344064, which overflows
        # fp16's 65504 to inf. Stored that way, every error metric for that layer came back nan,
        # `nan > threshold` is False, and the layer with the largest activations in the model was
        # assigned the *cheapest* format. bf16 carries fp32's exponent range, so it cannot.
        self.buffer = torch.zeros(capacity, width, dtype=torch.bfloat16)
        self.filled = 0
        self.seen = 0
        self.generator = torch.Generator().manual_seed(seed)

    def offer(self, rows: torch.Tensor) -> None:
        n = rows.shape[0]
        if n == 0:
            return
        if self.filled < self.capacity:
            take = min(self.capacity - self.filled, n)
            self.buffer[self.filled:self.filled + take] = rows[:take].to(
                device="cpu", dtype=torch.bfloat16)
            self.filled += take
            self.seen += take
            rows = rows[take:]
            n = rows.shape[0]
            if n == 0:
                return

        # Row i (1-based over the whole stream) draws a slot uniform in [0, i) and is kept only
        # when that lands inside the reservoir. float64 because `seen` passes 2**24 within a
        # single run, where float32 can no longer represent consecutive integers.
        index = torch.arange(self.seen + 1, self.seen + n + 1, dtype=torch.float64)
        draw = torch.rand(n, generator=self.generator, dtype=torch.float64)
        slots = (draw * index).floor().to(torch.long)
        self.seen += n

        keep = (slots < self.capacity).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            return
        chosen = slots[keep]
        # Stable sort by slot, then take the last entry of each run: that is the row the
        # sequential algorithm would have left in the slot, and unlike index_put_ with duplicate
        # indices it is deterministic.
        order = torch.argsort(chosen, stable=True)
        chosen, keep = chosen[order], keep[order]
        last = torch.ones_like(chosen, dtype=torch.bool)
        last[:-1] = chosen[1:] != chosen[:-1]
        self.buffer[chosen[last]] = rows[keep[last]].to(device="cpu", dtype=torch.bfloat16)


class LayerStats:
    """Per-layer accumulators. The *statistics* stay on the GPU until finish().

    Pulling a scalar to the CPU inside a forward hook forces a device sync on every one of the
    ~7600 calls a run makes, so `channel_absmax` and the crest chunks are accumulated on device
    and moved once, at the end.

    The reservoir is a different matter and this docstring used to overstate it: `offer()` writes
    into a CPU buffer, so it syncs on any call that accepts at least one row. With the real
    calibration numbers (`layers.0`, 159744 rows over 32 calls, capacity 128) the chance a given
    late call accepts nothing is `(154752/159744)**128 = 0.017`, i.e. it syncs on ~98% of calls.
    That is inherent to keeping the sample on the host; what was removed is the *avoidable* sync,
    not all of it.
    """

    def __init__(self, width: int, capacity: int, seed: int, device: torch.device):
        self.reservoir = Reservoir(capacity, width, seed)
        self.channel_absmax = torch.zeros(width, dtype=torch.float32, device=device)
        self.crest_chunks = []
        self.calls = 0
        self.rows = 0

    def observe(self, x: torch.Tensor, crest_rows: int) -> None:
        flat = x.reshape(-1, x.shape[-1])
        self.calls += 1
        self.rows += flat.shape[0]

        magnitude = flat.abs().float()
        torch.maximum(self.channel_absmax, magnitude.amax(dim=0), out=self.channel_absmax)

        # Crest factor is what the activation quantizer's one-scale-per-token design is sensitive
        # to. Capped rows per call because it is a diagnostic, not the decision.
        head = magnitude[:crest_rows]
        rms = head.pow(2).mean(dim=1).sqrt().clamp(min=1e-12)
        self.crest_chunks.append(head.amax(dim=1) / rms)

        self.reservoir.offer(flat)

    def finish(self) -> dict:
        crest = (torch.cat(self.crest_chunks).cpu() if self.crest_chunks
                 else torch.zeros(1))
        quantiles = torch.quantile(crest.float(), torch.tensor([0.5, 0.99]))
        return {
            "sample": self.reservoir.buffer[:self.reservoir.filled].clone(),
            "channel_absmax": self.channel_absmax.cpu(),
            "crest_mean": float(crest.mean()),
            "crest_p50": float(quantiles[0]),
            "crest_p99": float(quantiles[1]),
            "crest_max": float(crest.max()),
            "calls": self.calls,
            "rows": self.rows,
            "sampled_from": self.reservoir.seen,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="diffusion model name or path; must be the high-precision source")
    parser.add_argument("--profile", choices=list(PROFILE_PATTERNS), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip", nargs="+", required=True)
    parser.add_argument("--clip-type", default="lumina2")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", action="append", default=[], type=Path,
                        help="file read as one prompt; repeatable")
    parser.add_argument("--negative", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234],
                        help="one full sampling run per seed")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=1,
                        help="frames for a video model; ignored when the latent format is 2-D")
    parser.add_argument("--rows", type=int, default=128,
                        help="reservoir rows kept per layer")
    parser.add_argument("--crest-rows", type=int, default=64,
                        help="rows per call used for the crest-factor diagnostic")
    parser.add_argument("--attention", choices=["default", "sage"], default="default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing calibration: {args.out}")
    prompts = list(args.prompt) + [p.read_text(encoding="utf-8").strip()
                                   for p in args.prompt_file]
    if not prompts:
        raise SystemExit("Give at least one --prompt or --prompt-file")

    # Same ordering constraint as tools/nunchaku_compare.py: the attention backend is chosen when
    # comfy.ldm.modules.attention is first imported, and every block binds the chosen function
    # into its own namespace at that moment. Setting the flag later patches a name nobody reads.
    import comfy.cli_args
    if args.attention == "sage":
        comfy.cli_args.args.use_sage_attention = True

    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import folder_paths

    extra = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra.is_file():
        import utils.extra_config
        utils.extra_config.load_extra_path_config(str(extra))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; calibration must run where the model runs")

    # HIGH_VRAM keeps the whole transformer resident, which is what makes the hooks cheap -- but
    # it is a promise the card cannot always keep. LTX 2.5 is 39 GiB of BF16 against 24 GiB of
    # RTX 3090, and forcing HIGH_VRAM there turns "slow" into "out of memory". Decided from the
    # file size against the device, not assumed.
    candidate_size = Path(args.model)
    if not candidate_size.is_file():
        import folder_paths as _fp
        candidate_size = Path(_fp.get_full_path_or_raise("diffusion_models", args.model))
    model_gib = candidate_size.stat().st_size / 2 ** 30
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if model_gib < 0.7 * total_gib:
        comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
        comfy.model_management.set_vram_to = comfy.model_management.VRAMState.HIGH_VRAM
    else:
        print(f"{model_gib:.1f} GiB model against {total_gib:.1f} GiB of VRAM: leaving ComfyUI's "
              f"own VRAM policy alone. Calibration will offload and be slow; the numbers are the "
              f"same, the wall clock is not.", flush=True)

    # Encode every prompt and drop the encoder before the transformer is loaded. Qwen3-4B is
    # ~8 GiB and a 12 GiB BF16 transformer alongside it does not fit on a 24 GiB card.
    clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", c) for c in args.clip]
    print(f"encoding {len(prompts)} prompt(s) with {[Path(c).name for c in clip_paths]}",
          flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=clip_paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, args.clip_type.upper()))
    conditioning = []
    for text in prompts:
        positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(text))[0])]
        negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))[0])]
        positive[0][0] = positive[0][0].clone().cpu()
        negative[0][0] = negative[0][0].clone().cpu()
        conditioning.append((positive, negative))
        print(f"  {list(positive[0][0].shape)}  |cond| {float(positive[0][0].float().norm()):.4f}",
              flush=True)
    del clip
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()

    candidate = Path(args.model)
    path = str(candidate) if candidate.is_file() else \
        folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    print(f"loading {path}", flush=True)
    model = comfy.sd.load_diffusion_model(path)
    diffusion_model = model.get_model_object("diffusion_model")

    # A quantized checkpoint would calibrate the damage instead of the signal. Refuse rather than
    # produce a plausible-looking file that describes the wrong thing.
    #
    # This used to test `named_buffers()` for a `comfy_quant` suffix, and that check could never
    # fire. Measured 2026-08-19 on the mixed Z-Image checkpoint: 170 `comfy_quant` keys in
    # `state_dict()`, 170 modules whose `.weight` is a QuantizedTensor, **0** in `named_buffers()`
    # and 0 in `named_parameters()`. The marker is materialised by the quantized Linear, not
    # registered as a buffer. So the guard that was supposed to stop a calibration from running on
    # an already-quantized file had never stopped anything -- it had simply never been given one.
    from comfy_kitchen.tensor.base import QuantizedTensor
    already = [name for name, module in diffusion_model.named_modules()
               if isinstance(getattr(module, "weight", None), QuantizedTensor)]
    if already:
        raise SystemExit(f"{path} has {len(already)} QuantizedTensor weights; calibration needs "
                         "the high-precision source")

    pattern = PROFILE_PATTERNS[args.profile]
    targets = {name: module for name, module in diffusion_model.named_modules()
               if isinstance(module, torch.nn.Linear) and pattern.match(name)}
    if not targets:
        raise SystemExit(f"Profile {args.profile!r} matched no Linear in "
                         f"{type(diffusion_model).__name__}")
    print(f"hooking {len(targets)} Linear layers matching profile {args.profile!r}", flush=True)

    stats: dict[str, LayerStats] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs):
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            entry = stats.get(name)
            if entry is None:
                # Seeded per layer name so a rerun with the same arguments samples the same rows,
                # which is what makes two calibrations comparable. crc32 rather than hash():
                # Python randomises string hashing per process, so hash() would silently make
                # every run pick different rows while this comment claimed otherwise.
                entry = LayerStats(x.shape[-1], args.rows,
                                   seed=zlib.crc32(name.encode("utf-8")),
                                   device=x.device)
                stats[name] = entry
            with torch.no_grad():
                entry.observe(x.detach(), args.crest_rows)
        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    latent_format = model.model.latent_format
    side = max(args.size // 8, 8)
    # `latent_dimensions` is 2 for image models and 3 for video ones, and a video model handed a
    # 4-D latent does not fail cleanly -- it fails somewhere inside the transformer with a shape
    # error that says nothing about the latent. Read the format rather than assuming images.
    dimensions = getattr(latent_format, "latent_dimensions", 2)
    if dimensions == 3:
        ratio = getattr(latent_format, "temporal_downscale_ratio", 4)
        frames = max(1, (args.frames - 1) // ratio + 1)
        shape = [1, latent_format.latent_channels, frames, side, side]
        print(f"video latent {shape} ({args.frames} frames / temporal ratio {ratio})", flush=True)
    else:
        shape = [1, latent_format.latent_channels, side, side]
    started = time.perf_counter()
    runs = 0
    try:
        for prompt_index, (positive, negative) in enumerate(conditioning):
            for seed in args.seeds:
                latent = torch.zeros(shape, device="cpu")
                noise = comfy.sample.prepare_noise(latent, seed, None)
                print(f"run {runs + 1}/{len(conditioning) * len(args.seeds)} "
                      f"prompt {prompt_index} seed {seed} steps {args.steps}", flush=True)
                comfy.sample.sample(
                    model, noise, args.steps, args.cfg, args.sampler, args.scheduler,
                    positive, negative, latent,
                    denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, callback=None,
                    disable_pbar=True, seed=seed)
                torch.cuda.synchronize()
                runs += 1
    finally:
        for handle in handles:
            handle.remove()

    elapsed = time.perf_counter() - started
    missing = sorted(set(targets) - set(stats))
    if missing:
        # A hooked layer that never fired is a layer the converter must not calibrate from. Say
        # so loudly: it usually means the profile matched something outside the sampled path.
        print(f"warning: {len(missing)} hooked layer(s) never ran, e.g. {missing[:3]}")

    # Keyed by checkpoint name, not module name -- see MODULE_TO_FILE. Checked rather than
    # trusted: a translation that collides would silently drop layers from the calibration.
    payload = {to_file_name(args.profile, name): entry.finish() for name, entry in stats.items()}
    if len(payload) != len(stats):
        raise SystemExit(f"the module->file translation collapsed {len(stats)} layers into "
                         f"{len(payload)}; MODULE_TO_FILE for profile {args.profile!r} is wrong")
    meta = {
        "source": str(path),
        "profile": args.profile,
        "prompts": len(prompts),
        "seeds": args.seeds,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "size": args.size,
        "rows": args.rows,
        "runs": runs,
        "layers": len(payload),
        "never_ran": missing,
        "seconds": round(elapsed, 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    torch.save({"meta": meta, "layers": payload}, partial)
    partial.replace(args.out)

    print(f"\n{json.dumps(meta, indent=2)}")
    worst = sorted(payload.items(), key=lambda kv: -kv[1]["crest_p99"])[:10]
    print(f"\n{'layer':<44}{'rows':>10}{'crest p50':>12}{'crest p99':>12}{'crest max':>12}")
    for name, entry in worst:
        print(f"{name:<44}{entry['rows']:>10}{entry['crest_p50']:>12.2f}"
              f"{entry['crest_p99']:>12.2f}{entry['crest_max']:>12.2f}")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB) in {elapsed:.1f}s")
    print("crest factor is a diagnostic. The promotion decision is measured against the real "
          "kernels in tools/quant_mixed.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
