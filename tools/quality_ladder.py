"""How far does each quantized checkpoint move the picture away from the BF16 one?

`--promote-error 0.15` was picked, not derived. It decides how many layers get the expensive
format, and on this model the curve through it is steep -- 0.10 promotes 119 layers of 170, 0.15
promotes 55, 0.20 promotes 10 -- so the choice is not a rounding detail. Nothing so far has
measured what those layers buy.

This runs the same prompts and the same seeds through a reference checkpoint and through each
candidate, and reports, per candidate:

    divergence   relative L2 of the sampled latent against the reference's latent, same seed
    s/step       wall time per sampling step
    GiB          file size

Two things it deliberately does **not** do:

  * It does not call the result "better" or "worse". Latent divergence is a distance, not a
    quality score: a checkpoint can sit closer to BF16 and still produce a picture someone
    prefers. The images are written out so a person can look.
  * It does not average over one seed. Divergence at a single seed is one sample of a noisy
    quantity, and the earlier m_crossover work on this bench showed what one sample is worth.
    Several seeds per model, and the spread is printed next to the mean.

The reference must be the high-precision checkpoint. Comparing two quantized files to each other
answers a different and less useful question.

    python_embeded\\python.exe -s tools/quality_ladder.py ^
        --reference beyond-reality-zimage-v2_native.safetensors ^
        --models zimage-v2-w4a4.safetensors zimage-v2-mixed.safetensors ^
        --clip qwen_3_4b.safetensors --seeds 1 2 3 --steps 8 --size 1024
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent divergence ladder over quantized models")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--clip", nargs="+", required=True)
    parser.add_argument("--clip-type", default="lumina2")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--negative", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--vae", default=None,
                        help="file in models/vae. Without it no images are written and the run "
                             "reports latents only.")
    parser.add_argument("--out", type=Path, default=PORTABLE_ROOT / "bench" / "quality_ladder")
    return parser.parse_args()


def encode(args, folder_paths, comfy_sd, comfy_mm, prompts):
    """Encode every prompt once, then drop the encoder before any transformer is loaded.

    Qwen3-4B is ~8 GiB and a 12 GiB BF16 transformer alongside it does not fit on a 24 GiB card.
    Same constraint calibrate_activations.py hit; same solution.
    """
    paths = [folder_paths.get_full_path_or_raise("text_encoders", c) for c in args.clip]
    clip = comfy_sd.load_clip(
        ckpt_paths=paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy_sd.CLIPType, args.clip_type.upper()))
    out = []
    for text in prompts:
        positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(text))[0])]
        negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))[0])]
        positive[0][0] = positive[0][0].clone().cpu()
        negative[0][0] = negative[0][0].clone().cpu()
        out.append((positive, negative))
    del clip
    comfy_mm.soft_empty_cache()
    torch.cuda.empty_cache()
    return out


def sample_all(args, name, conditioning, comfy_sample, comfy_sd, comfy_mm, folder_paths):
    """Every (prompt, seed) for one checkpoint, returning latents and per-step wall time."""
    candidate = Path(name)
    path = str(candidate) if candidate.is_file() else \
        folder_paths.get_full_path_or_raise("diffusion_models", name)
    model = comfy_sd.load_diffusion_model(path)
    latent_format = model.model.latent_format
    side = max(args.size // 8, 8)

    results, per_step = {}, []
    for prompt_index, (positive, negative) in enumerate(conditioning):
        for seed in args.seeds:
            latent = torch.zeros([1, latent_format.latent_channels, side, side], device="cpu")
            noise = comfy_sample.prepare_noise(latent, seed, None)
            torch.cuda.synchronize()
            started = time.perf_counter()
            samples = comfy_sample.sample(
                model, noise, args.steps, args.cfg, args.sampler, args.scheduler,
                positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None,
                last_step=None, force_full_denoise=False, noise_mask=None, callback=None,
                disable_pbar=True, seed=seed)
            torch.cuda.synchronize()
            per_step.append((time.perf_counter() - started) / args.steps)
            results[(prompt_index, seed)] = samples.detach().float().cpu()
    size_gib = Path(path).stat().st_size / 2 ** 30
    del model
    comfy_mm.soft_empty_cache()
    torch.cuda.empty_cache()
    return results, per_step, size_gib


def main() -> int:
    args = parse_args()
    prompts = list(args.prompt)
    if args.prompt_file:
        prompts.append(args.prompt_file.read_text(encoding="utf-8").strip())
    if not prompts:
        prompts = ["a still life with brass instruments on a wooden table, morning light"]

    import comfy.model_management as comfy_mm
    import comfy.sample as comfy_sample
    import comfy.sd as comfy_sd
    import folder_paths

    extra = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra.is_file():
        import utils.extra_config
        utils.extra_config.load_extra_path_config(str(extra))
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    comfy_mm.vram_state = comfy_mm.VRAMState.HIGH_VRAM
    comfy_mm.set_vram_to = comfy_mm.VRAMState.HIGH_VRAM

    conditioning = encode(args, folder_paths, comfy_sd, comfy_mm, prompts)
    runs = len(prompts) * len(args.seeds)
    print(f"{len(prompts)} prompt(s) x {len(args.seeds)} seed(s) = {runs} run(s) per checkpoint, "
          f"{args.steps} steps at {args.size}px", flush=True)

    print(f"\nreference: {args.reference}", flush=True)
    ref_latents, ref_steps, ref_gib = sample_all(
        args, args.reference, conditioning, comfy_sample, comfy_sd, comfy_mm, folder_paths)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"reference": args.reference, "steps": args.steps, "size": args.size,
              "seeds": args.seeds, "prompts": prompts, "models": {}}
    all_latents = {args.reference: ref_latents}

    rows = [(args.reference, 0.0, 0.0, statistics.median(ref_steps), ref_gib, len(ref_steps))]
    for name in args.models:
        print(f"\n{name}", flush=True)
        latents, steps, gib = sample_all(
            args, name, conditioning, comfy_sample, comfy_sd, comfy_mm, folder_paths)
        all_latents[name] = latents
        divergences = []
        for key, reference in ref_latents.items():
            got = latents[key]
            divergences.append(float((got - reference).norm() / reference.norm()))
        rows.append((name, statistics.mean(divergences),
                     max(divergences) - min(divergences),
                     statistics.median(steps), gib, len(steps)))
        report["models"][name] = {
            "divergence_mean": statistics.mean(divergences),
            "divergence_min": min(divergences),
            "divergence_max": max(divergences),
            "per_run": {f"prompt{p}_seed{s}": d
                        for (p, s), d in zip(ref_latents.keys(), divergences)},
            "s_per_step_median": statistics.median(steps),
            "gib": gib,
        }

    print(f"\n{'checkpoint':<44}{'divergence':>12}{'spread':>9}{'s/step':>9}{'GiB':>7}{'runs':>6}")
    for name, mean, spread, sec, gib, n in rows:
        shown = "-" if mean == 0.0 else f"{mean:.4f}"
        spread_shown = "-" if mean == 0.0 else f"{spread:.4f}"
        print(f"{Path(name).name:<44}{shown:>12}{spread_shown:>9}{sec:>9.3f}{gib:>7.2f}{n:>6}")

    # The unpaired means above are the wrong statistic and are kept only because they are what a
    # reader expects to see. Divergence varies more between seeds than between checkpoints -- the
    # first run of this file had one checkpoint spread 0.2301 across three seeds while the entire
    # gap between best and worst checkpoint was 0.1065 -- so a table of means invites an ordering
    # the data does not support. The runs are paired by construction (same seed, same noise, same
    # conditioning), so the comparison that survives is per-run.
    baseline = args.models[0]
    print(f"\nPaired against {Path(baseline).name}, run by run:")
    print(f"{'checkpoint':<44}{'mean delta':>12}{'worst':>9}{'best':>9}{'wins':>10}")
    total = len(ref_latents)
    for name in args.models:
        deltas = [report["models"][name]["per_run"][k] - report["models"][baseline]["per_run"][k]
                  for k in report["models"][baseline]["per_run"]]
        wins = sum(1 for d in deltas if d < 0)
        marker = "  (baseline)" if name == baseline else ""
        print(f"{Path(name).name:<44}{statistics.mean(deltas):>+12.4f}{max(deltas):>+9.4f}"
              f"{min(deltas):>+9.4f}{f'{wins}/{total}':>10}{marker}")
        report["models"][name]["paired_vs_baseline"] = {
            "baseline": baseline, "mean": statistics.mean(deltas),
            "worst": max(deltas), "best": min(deltas), "wins": wins, "runs": total}

    # A checkpoint that wins on some runs and loses on others has not been shown to differ. Say
    # that out loud rather than leaving a reader to infer it from a sign.
    undecided = [Path(n).name for n in args.models if n != baseline
                 and 0 < report["models"][n]["paired_vs_baseline"]["wins"] < total]
    if undecided:
        print(f"\nSplit decisions, i.e. not separated at {total} run(s): {', '.join(undecided)}")
    if total < 8:
        print(f"{total} runs is a small sample for a quantity this noisy. Treat any ordering here "
              "as provisional until it survives more seeds.")

    if args.vae:
        _write_images(args, all_latents, folder_paths, comfy_sd)
    else:
        print("\nNo --vae given, so no images were written. Divergence is a distance from the "
              "reference latent, not a verdict on the picture: nothing here says which checkpoint "
              "looks better, and on this model the images have not visibly separated before.")

    (args.out / "ladder.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'ladder.json'}")
    return 0


def _write_images(args, all_latents, folder_paths, comfy_sd):
    """Decode every latent with one VAE so the only difference between images is the transformer."""
    import numpy as np
    from PIL import Image

    vae_path = folder_paths.get_full_path_or_raise("vae", args.vae)
    import comfy.utils
    vae = comfy_sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
    written = 0
    for name, latents in all_latents.items():
        for (prompt_index, seed), samples in latents.items():
            with torch.no_grad():
                image = vae.decode(samples.cuda())
            if image.ndim == 4:
                # .detach() is not decoration: comfy's VAE.decode returns a tensor that still
                # carries grad, and .numpy() on it raises. The whole ladder had already run when
                # this fired, which is the argument for writing images before, not after.
                array = (image[0].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                out = args.out / f"{Path(name).stem}__p{prompt_index}_s{seed}.png"
                Image.fromarray(array).save(out)
                written += 1
    print(f"\nwrote {written} image(s) to {args.out}")
    print("Look at them. Divergence orders the checkpoints by distance from BF16; it does not "
          "order them by how the picture reads, and those are not the same question.")


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    with BenchGuard("quality_ladder") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
