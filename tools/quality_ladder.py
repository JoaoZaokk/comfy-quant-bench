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

# O bootstrap que o ComfyUI espera de quem o importa como biblioteca, e que esta ferramenta nao
# fazia: sem ele o `comfy.cli_args.args` nao existe com os defaults, e sob a 0.33 o loader vai
# parar em caminhos que so o `main.py` prepara. E o mesmo preambulo de
# `tools/probe_int4_vs_int8_visual.py`, que decodifica imagem com sucesso nesta mesma versao.
# Duas armadilhas, uma dentro da outra:
#   1. o parser do ComfyUI le `sys.argv` e reclama dos argumentos DESTA ferramenta
#      ("argument --models-directory: The path 'zimage-v2-w4a4.safetensors' does not exist");
#   2. `enable_args_parsing()` so LIGA o parse -- ele acontece de verdade no primeiro
#      `import comfy.cli_args`, que e tarde. Neutralizar o argv sem forcar o parse ali nao
#      adianta, e nao restaurar depois deixa o `parse_args()` daqui sem nenhum argumento.
# Entao: neutraliza, liga, FORCA o parse, restaura.
#      E o argv neutro e o unico lugar onde da para escolher o dtype de calculo do difusor: o
#      ComfyUI decide isso no parse e nao expoe depois. `--bf16-unet` existe aqui porque o
#      `capybara_v0.1` tem pesos BF16 enquanto o `hunyuanvideo1.5` da MESMA arquitetura tem F16 --
#      o ComfyUI escolhe fp16 para essa familia e o forward morre com
#      `mat1 and mat2 must have the same dtype, but got Half and BFloat16` na `time_in`, que nem e
#      uma camada quantizada. Derruba os DOIS bracos, referencia inclusive, entao nao e defeito de
#      quantizacao nenhum: e o dtype do arquivo.
_ARGV = sys.argv[:]
_DTYPE = [a for a in _ARGV if a in ("--bf16-unet", "--fp16-unet", "--fp32-unet")]
sys.argv = ["main.py", *_DTYPE]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401  -- e aqui que o parse acontece
sys.argv = [a for a in _ARGV if a not in _DTYPE]

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
    parser.add_argument("--frames", type=int, default=1,
                        help="frames for a video model; ignored when the latent format is 2-D")
    parser.add_argument("--vace-strength", type=float, default=None,
                        help="forca do ramo VACE, aplicada igual em todos os bracos. Use 0.0 para "
                             "amostrar um checkpoint VACE como T2V comum: o default do ComfyUI e "
                             "1.0 sobre quadros zerados, e isso destroi a saida -- inclusive a da "
                             "referencia nao quantizada.")
    parser.add_argument("--shift", type=float, default=None,
                        help="shift do ModelSamplingSD3, aplicado igual em todos os bracos. "
                             "Obrigatorio para Wan (fabrica usa 8.0); sem ele a referencia "
                             "nao-quantizada tambem sai destruida.")
    parser.add_argument("--clip-device", choices=["default", "cpu"], default="default",
                        help="'cpu' keeps the text encoder off the card entirely. LTX 2.5's is a "
                             "24 GiB gemma4-12B, which does not share a 24 GiB card with anything")
    parser.add_argument("--distorch", default=None,
                        help="distribute the model's blocks instead of offloading it whole, using "
                             "ComfyUI-MultiGPU's DisTorch2. Two forms:\n"
                             "  simple  'cuda:0;20;cpu'  -- compute on cuda:0, borrow 20 GiB from "
                             "one donor\n"
                             "  expert  'cuda:0,15gb;cuda:1,8gb;cpu,*'  -- explicit quota per "
                             "device in priority order, '*' takes the remainder\n"
                             "Prefer the expert form with a second GPU ahead of cpu: donating to "
                             "another card's VRAM beats donating to pageable host RAM. Applies "
                             "only to checkpoints that do not fit; see the note printed at the "
                             "end about what it does to the s/step column.")
    parser.add_argument("--distorch-compute", default="cuda:0",
                        help="the device that runs the maths when --distorch uses the expert form")
    return parser.parse_args()


def encode(args, folder_paths, comfy_sd, comfy_mm, prompts):
    """Encode every prompt once, then drop the encoder before any transformer is loaded.

    Qwen3-4B is ~8 GiB and a 12 GiB BF16 transformer alongside it does not fit on a 24 GiB card.
    Same constraint calibrate_activations.py hit; same solution.
    """
    paths = [folder_paths.get_full_path_or_raise("text_encoders", c) for c in args.clip]
    options = {}
    if args.clip_device == "cpu":
        options["load_device"] = options["offload_device"] = torch.device("cpu")
    # `disable_dynamic=True` NAO e uma otimizacao: e o que impede o ComfyUI 0.33 de construir um
    # `CoreModelPatcher`, cujo `register_load_device` aloca um HostBuffer do comfy-aimdo. Num
    # script que importa `comfy.sd` direto -- sem o `comfy_aimdo.control.init()` que `main.py:63`
    # faz -- isso morre. Se o aimdo for inicializado pela metade, e pior: o processo cai sem
    # traceback nenhum, deixando o lock preso. Esta ferramenta foi escrita sob a 0.29, onde esse
    # caminho nao existia. Para um benchmark, dynamic VRAM tambem so adicionaria variancia.
    clip = comfy_sd.load_clip(
        ckpt_paths=paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy_sd.CLIPType, args.clip_type.upper()),
        model_options=options,
        disable_dynamic=True)
    out = []
    for text in prompts:
        positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(text))[0])]
        negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))[0])]
        positive[0][0] = positive[0][0].clone().cpu()
        negative[0][0] = negative[0][0].clone().cpu()
        # Um checkpoint VACE amostrado SEM no VACE nao fica "sem controle": `WAN21_Vace.extra_conds`
        # (comfy/model_base.py:1710-1737) preenche `vace_frames` com zeros, passa cada bloco por
        # `process_latent_in` -- que subtrai a media do formato latente e portanto transforma zero
        # em valor NAO nulo --, concatena uma mascara toda de UNS e aplica com `vace_strength=1.0`.
        # Medido em 2026-09-01 no wan2.1_vace_1.3B_fp16, um eixo variado, mesma semente:
        #     forca 1.0 -> trama tecida, |latente| 607,6      forca 0.0 -> oficina, |latente| 1543,2
        # Sem isto a referencia NAO quantizada tambem sai destruida, e a comparacao nao mede nada.
        if args.vace_strength is not None:
            for cond in (positive, negative):
                cond[0][1]["vace_strength"] = [args.vace_strength]
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
    # Same size-aware rule as calibrate_activations: forcing HIGH_VRAM on a 39 GiB checkpoint
    # against a 24 GiB card turns slow into out-of-memory. Decided per model, inside this
    # function, because a ladder can mix a BF16 reference that must offload with quantized
    # candidates that fit.
    gib = Path(path).stat().st_size / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if args.distorch and gib >= 0.7 * total:
        # DisTorch2 assigns each block a device and moves the parameters there, then patches
        # ModelPatcher.load_models_gpu so the loader stops trying to place the whole model. That
        # is a different thing from ComfyUI's own offload, which keeps the model whole and swaps
        # it against whatever else wants the card.
        comfy_mm.vram_state = comfy_mm.VRAMState.NORMAL_VRAM
        comfy_mm.set_vram_to = comfy_mm.VRAMState.NORMAL_VRAM
        model = comfy_sd.load_diffusion_model(path, disable_dynamic=True)
        # Loaded as a package, not as a loose module. `distorch_2.py` opens with
        # `from .device_utils import get_device_list`, and a bare `import distorch_2` off sys.path
        # dies with "attempted relative import with no known parent package" -- after the model has
        # already been read into RAM, which is the expensive way to find out. The directory name
        # carries a hyphen, so it cannot be imported by name either; ComfyUI itself loads custom
        # nodes through importlib for the same reason.
        import importlib.util
        pack = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-MultiGPU"
        spec = importlib.util.spec_from_file_location(
            "comfyui_multigpu", pack / "__init__.py", submodule_search_locations=[str(pack)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_multigpu"] = package
        spec.loader.exec_module(package)
        distorch_2 = importlib.import_module("comfyui_multigpu.distorch_2")
        distorch_2.register_patched_safetensor_modelpatcher()
        # Two syntaxes into one argument. The expert form carries a comma -- 'cuda:1,8gb;cpu,*' --
        # and goes before the '#'; the simple form is 'compute;gb;donor' and goes after it. Told
        # apart by the comma rather than by a second flag, because the pack's own strings are
        # written this way and a reader copying one from its README should have it work.
        allocation = (f"{args.distorch}#{args.distorch_compute}" if "," in args.distorch
                      else f"#{args.distorch}")
        distorch_2.analyze_safetensor_loading(model, allocation)
        print(f"  DisTorch2 {allocation!r} on {gib:.1f} GiB", flush=True)
    else:
        model = None
        if gib < 0.7 * total:
            comfy_mm.vram_state = comfy_mm.VRAMState.HIGH_VRAM
            comfy_mm.set_vram_to = comfy_mm.VRAMState.HIGH_VRAM
        else:
            comfy_mm.vram_state = comfy_mm.VRAMState.NORMAL_VRAM
            comfy_mm.set_vram_to = comfy_mm.VRAMState.NORMAL_VRAM
            print(f"  {gib:.1f} GiB against {total:.1f} GiB of VRAM: offloading whole. Slower, "
                  f"same numbers. --distorch distributes blocks instead.", flush=True)

    if model is None:
        model = comfy_sd.load_diffusion_model(path, disable_dynamic=True)
    # O `ModelSamplingSD3` e o no que os workflows de fabrica de modelo de flow inserem, e existe
    # aqui para que este ladder possa reproduzir um workflow real.
    #
    # CUIDADO com o que ele NAO faz. Entrou nesta ferramenta em 2026-09-01 como hipotese para o
    # Wan 2.1 sair destruido, e a hipotese estava errada: com shift 8.0 o latente veio
    # BIT-IDENTICO ao sem shift. O patch aplica -- `get_model_object("model_sampling").shift`
    # vira 8.0 -- mas `comfy.samplers.calculate_sigmas(ms, "simple", 25)` devolve exatamente os
    # mesmos sigmas, porque o scheduler `simple` nao consulta o shift. A causa real era
    # `vace_strength`; ver `--vace-strength` abaixo. Aplicado igual em todos os bracos, que e o
    # unico jeito de a comparacao continuar casada.
    if args.shift is not None:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        model = ModelSamplingSD3().patch(model, args.shift)[0]
        print(f"  ModelSamplingSD3 shift={args.shift}", flush=True)
    latent_format = model.model.latent_format
    side = max(args.size // 8, 8)
    if getattr(latent_format, "latent_dimensions", 2) == 3:
        ratio = getattr(latent_format, "temporal_downscale_ratio", 4)
        frames = max(1, (args.frames - 1) // ratio + 1)
        shape = [1, latent_format.latent_channels, frames, side, side]
    else:
        shape = [1, latent_format.latent_channels, side, side]

    results, per_step = {}, []
    for prompt_index, (positive, negative) in enumerate(conditioning):
        for seed in args.seeds:
            latent = torch.zeros(shape, device="cpu")
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

    # Deliberately not set here any more: sample_all decides per checkpoint, because a ladder can
    # hold a 39 GiB BF16 reference that must offload next to quantized candidates that fit.

    # Before anything is loaded. Without it comfy/ops.py builds every Linear eagerly and the whole
    # checkpoint is committed in host RAM -- 44.5 GiB working set for a 39 GiB model on this box,
    # measured, with 4 GiB of system memory left over. See tools/_dynamic_vram.py.
    from _dynamic_vram import enable as enable_dynamic_vram
    if enable_dynamic_vram():
        print("DynamicVRAM enabled: weights load lazily rather than all at once", flush=True)
    else:
        print("DynamicVRAM NOT available. Every weight will be materialised in host RAM at load; "
              "a large BF16 reference may not fit.", flush=True)

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
    if args.distorch:
        # The divergence column is unaffected -- the same arithmetic runs, on parameters that live
        # somewhere else -- but s/step is not. A distributed model pays a transfer per block per
        # step, so its seconds are a property of the placement, not of the format, and putting
        # them in the same column as a resident model's invites exactly the wrong comparison.
        print("\n--distorch was used on at least one checkpoint. Its s/step measures the block "
              "placement, not the format: do not compare that row against a resident model's. "
              "Divergence is unaffected, and that is checkable -- the same checkpoint run with "
              "and without --distorch must give the same latent.")

    # O relatorio e gravado ANTES das imagens. Numa execucao anterior o decode do VAE morreu e
    # levou junto vinte minutos de amostragem ja medidos: as imagens sao um extra, a medicao e o
    # produto, e um extra nao pode derrubar o produto.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ladder.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'ladder.json'}")

    # Os latentes vao para o disco antes do decode, e nao so o relatorio. Uma amostragem custa
    # dezenas de minutos de GPU e o decode ja morreu duas vezes por motivos que nao tem nada a ver
    # com a medicao; com os latentes salvos, decodificar de novo custa segundos num processo a
    # parte, em vez de repetir tudo.
    latdir = args.out / "latents"
    latdir.mkdir(exist_ok=True)
    for name, latents in all_latents.items():
        for (prompt_index, seed), samples in latents.items():
            torch.save(samples.cpu(),
                       latdir / f"{Path(name).stem}__p{prompt_index}_s{seed}.pt")
    print(f"wrote {sum(len(v) for v in all_latents.values())} latentes em {latdir}")

    if args.vae:
        try:
            _write_images(args, all_latents, folder_paths, comfy_sd)
        except Exception as exc:   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"\nAS IMAGENS FALHARAM ({type(exc).__name__}), mas a medicao acima esta "
                  "gravada e continua valida. Os numeros nao dependem do decode.")
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
    # `comfy.sd.VAE.__init__` escolhe `CoreModelPatcher` a menos que o proprio VAE peca
    # `disable_offload`, e nao aceita a escolha por parametro -- diferente de `load_clip` e
    # `load_diffusion_model`, que aceitam `disable_dynamic`. Entao aqui o comfy-aimdo precisa
    # existir de verdade. Inicializado SO NESTE PONTO: fazer isso no inicio do programa, com o
    # CLIP ainda passando pelo caminho dinamico, derrubou o processo sem traceback nenhum.
    # NAO tentar inicializar o comfy-aimdo aqui e religar `host_buffer.lib` na mao. Foi tentado:
    # `comfy_aimdo/host_buffer.py:6` faz `lib = control.lib` no import, e o
    # `comfy/memory_management.py:7` importa esse modulo no import do ComfyUI, entao a `lib` fica
    # `None` congelada. Religar depois do `init()` faz o processo morrer com SIGSEGV (exit 139),
    # sem traceback e deixando o lock da GPU preso. A biblioteca nativa nao esta preparada para
    # ser adotada no meio do caminho. O bootstrap correto e o do topo deste arquivo.
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
