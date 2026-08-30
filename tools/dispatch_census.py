"""During a real generation, count which dispatch branch every quantized Linear actually took.

The audit of 2026-08-18 flagged `comfy_kitchen/tensor/convrot_w4a4.py:237` twice, independently,
and it is the one finding whose consequence is exactly what this project forbids:

    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)

A weight whose `transposed` flag is set does not reach the ConvRot kernel at all. It is
dequantized to BF16 and multiplied in BF16 -- INT4 weight-only storage with 16-bit compute, which
is precisely the outcome CLAUDE.md says defeats the project. `_handle_convrot_w4a4_t` only flips
the flag, so any `aten.t` reaching the tensor before the linear lands here.

**The mechanism is proved by reading. What was never established is whether anything in this
checkout actually transposes.** The two auditors split on exactly that: one rated it medium
because no caller was found, the other rated it high from the mechanism alone. Reading cannot
settle it, because the caller could be anywhere -- ComfyUI's ops, a custom node, or nothing.

So this counts. It wraps every registered layout handler for the ConvRot layout plus the generic
fallback, runs a real generation through the real nodes, and reports how many calls took each
branch. A zero in the transposed column is as much a result as a nonzero one: it demotes a
high-severity audit finding to "unreachable on this path", which is a thing you can only learn by
running it.

    python_embeded\\python.exe -s tools/dispatch_census.py ^
        --model zimage-v2-mixed.safetensors --clip qwen_3_4b.safetensors ^
        --prompt "a still life with brass instruments" --steps 4 --size 512

Caveat that belongs with any number this prints: it covers the sampler path of one model at one
resolution. A branch that never fires here can still fire under a LoRA, under torch.compile, or in
a node that calls `mm` on a transposed weight.

And one blindness that is structural rather than incidental, because it makes the counter read
zero while the kernel runs: **a Python-side counter counts nothing during CUDA graph replay.**
The wrapper below is Python; a captured graph replays the recorded device work without
re-entering it. MEASURED in the sibling LLM project on 2026-08-16 -- with its per-call counter
forced on, generating 48 tokens moved the count by **zero**, because the custom op's Python body
does not re-execute on replay. So a count like "680 of 680 native" proves the right branch was
taken while it was being *built*; it does not prove how many times it ran in steady state.
Whether anything on ComfyUI's sampler path captures a graph was NOT checked here -- so treat
every number above as valid for eager execution and unestablished for anything captured. The
fixes are the same two the sibling names: force eager, or count inside the kernel.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

COUNTS = Counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count ConvRot dispatch branches in a real run")
    parser.add_argument("--model", required=True, help="file in models/diffusion_models, or a path")
    parser.add_argument("--clip", nargs="+", required=True)
    parser.add_argument("--clip-type", default="lumina2")
    parser.add_argument("--prompt", default="a still life with brass instruments")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora", default=None,
                        help="file in models/loras to apply before sampling. Tests the second "
                             "audit finding: comfy/ops.py:1377 requires len(self.weight_function) "
                             "== 0 to take the quantized path, and a LoRA patch installs one.")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=1,
                        help="frames for a video model; ignored when the latent format is 2-D")
    parser.add_argument("--weight-dtype",
                        choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                        default="default",
                        help="the UNETLoader widget, passed through exactly as nodes.py:993 does. "
                             "Tests comfy/sd.py:2303, which assigns the widget value to unet_dtype "
                             "even when quant_config is set, while :2306 protects "
                             "manual_cast_dtype from the same thing.")
    return parser.parse_args()


# Which positional argument holds the quantized weight, per op. `addmm` is the odd one --
# (bias, a, b) -- and reading args[1] there would test the *activation* for a flag it never has,
# reporting every addmm as native regardless of what it did.
WEIGHT_ARG = {"linear": 1, "mm": 1, "addmm": 2, "t": 0}

# `transposed` does not mean the same thing in every handler, and getting this backwards makes the
# tool label the correct path as the dangerous one. Read from convrot_w4a4.py:
#
#   linear + transposed=True   -> F.linear(x, weight.dequantize()) -- BF16 compute, the finding
#   linear + transposed=False  -> the ConvRot kernel
#   mm/addmm + transposed=True -> _resolve_convrot_w4a4_rhs accepts, kernel runs. This is the
#                                 *required* state: mm(x, W.t()) is how a Linear reaches the op.
#   mm/addmm + transposed=False-> _resolve_convrot_w4a4_rhs raises RuntimeError
DEQUANTIZES_WHEN_TRANSPOSED = {"linear": True, "mm": False, "addmm": False}


def instrument() -> list:
    """Wrap the ConvRot layout handlers so every dispatch is counted.

    Patched inside the registry dict, not on the module globals: `register_layout_op` stores the
    function object at import time, so rebinding the module-level name would leave the registry
    pointing at the original -- counting nothing while appearing to work.
    """
    import comfy.quant_ops  # noqa: F401  (registers the backends and the layouts)
    from comfy_kitchen.tensor import base as tensor_base
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout

    # Every quantized layout this project produces. Instrumenting a subset gives a census that
    # looks complete and is not: an int8 checkpoint run against the first two would report zero
    # dispatches and read as an answer. All three have the same shape of bug in the same place --
    # `int8.py:275`, `w4a8_int8.py:330` and `convrot_w4a4.py:237` each dequantize to BF16 when the
    # weight carries a transposed flag.
    layouts = {"w4a4": TensorCoreConvRotW4A4Layout, "w4a8": AsymW4A8Int8Layout,
               "int8": TensorWiseINT8Layout}
    table = getattr(tensor_base, "_LAYOUT_DISPATCH_TABLE", None)
    if table is None:
        raise SystemExit("comfy_kitchen.tensor.base has no _LAYOUT_DISPATCH_TABLE; the registry "
                         "moved and this probe would silently count nothing")

    undo = []

    def wrap(op, layout, name):
        original = table[op][layout]

        def counting(qt, args, kwargs):
            COUNTS[name] += 1
            # `name` is "<layout>.<op>"; the lookup tables are keyed by the op alone.
            op_name = name.split(".", 1)[1]
            index = WEIGHT_ARG.get(op_name)
            weight = args[index] if index is not None and len(args) > index else None
            params = getattr(weight, "_params", None)
            transposed = getattr(params, "transposed", None)
            if params is None:
                COUNTS[f"{name}:not-quantized"] += 1
            elif op_name == "t":
                # aten.t computes nothing; it flips the flag that decides which branch the *next*
                # op takes. Counted separately on purpose -- this is the op the audit was asking
                # about when it asked who transposes.
                COUNTS[f"{name}:flips {transposed} -> {not transposed}"] += 1
            elif transposed is DEQUANTIZES_WHEN_TRANSPOSED.get(op_name):
                COUNTS[f"{name}:DEQUANTIZED to BF16"] += 1
            else:
                COUNTS[f"{name}:native kernel"] += 1
            return original(qt, args, kwargs)

        table[op][layout] = counting
        undo.append((op, layout, original))

    for tag, layout in layouts.items():
        for op, by_layout in list(table.items()):
            if layout not in by_layout:
                continue
            # torch.ops.aten.linear.default -> "linear"
            parts = str(op).split(".")
            wrap(op, layout, f"{tag}.{parts[-2] if len(parts) >= 2 else op}")
    return undo


def main() -> int:
    args = parse_args()

    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import folder_paths

    extra = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra.is_file():
        import utils.extra_config
        utils.extra_config.load_extra_path_config(str(extra))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this must run where the model runs")

    comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
    comfy.model_management.set_vram_to = comfy.model_management.VRAMState.HIGH_VRAM

    clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", c) for c in args.clip]
    print(f"encoding prompt with {[Path(c).name for c in clip_paths]}", flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=clip_paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, args.clip_type.upper()))
    positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))[0])]
    negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))[0])]
    positive[0][0] = positive[0][0].clone().cpu()
    negative[0][0] = negative[0][0].clone().cpu()
    del clip
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()

    candidate = Path(args.model)
    path = str(candidate) if candidate.is_file() else \
        folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    # Built the way nodes.py:993 builds it, so this exercises the real widget rather than a
    # plausible imitation of it.
    model_options = {}
    if args.weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif args.weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif args.weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    print(f"loading {path} (weight_dtype={args.weight_dtype})", flush=True)
    model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    print(f"unet_dtype={model.model.get_dtype()}  "
          f"manual_cast={getattr(model.model, 'manual_cast_dtype', None)}", flush=True)
    diffusion_model = model.get_model_object("diffusion_model")
    # Detected on the weight object, not on `named_buffers()`. Measured on the mixed Z-Image
    # checkpoint: 170 `comfy_quant` keys in `state_dict()`, 170 modules whose `.weight` is a
    # QuantizedTensor, and **zero** in `named_buffers()` or `named_parameters()`. A check written
    # against named_buffers cannot fire, whichever way it is pointed.
    from comfy_kitchen.tensor.base import QuantizedTensor
    quantized = [n for n, mod in diffusion_model.named_modules()
                 if isinstance(getattr(mod, "weight", None), QuantizedTensor)]
    if not quantized:
        raise SystemExit(f"{path} loaded with no QuantizedTensor weights; there is nothing to "
                         "dispatch and this census would report an honest zero for the wrong "
                         "reason")
    print(f"{len(quantized)} quantized layer(s) in the loaded graph", flush=True)

    # The dtype histogram of everything that is *not* quantized. `unet_dtype` reaching fp8 is only
    # half a finding; what makes it a defect is which tensors land there. The quantized layers are
    # already 4-bit and cannot be affected, so any damage falls on the norms, embeddings and
    # modulation that the profile deliberately left in high precision.
    from collections import Counter as _Counter
    hist = _Counter()
    for name, p in list(diffusion_model.named_parameters()) + list(diffusion_model.named_buffers()):
        if isinstance(p, QuantizedTensor):
            hist["QuantizedTensor"] += 1
        else:
            hist[str(p.dtype)] += 1
    print("unquantized tensor dtypes: "
          + ", ".join(f"{k} x{v}" for k, v in sorted(hist.items())), flush=True)

    if args.lora:
        import comfy.utils
        lora_path = folder_paths.get_full_path_or_raise("loras", args.lora)
        print(f"applying LoRA {Path(lora_path).name} at strength {args.lora_strength}", flush=True)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, args.lora_strength, 0)
        keys = list(model.patches)
        print(f"{len(keys)} patched key(s)", flush=True)
        # Patched-count alone proves nothing about this question. The finding is that a weight
        # function on a *quantized* Linear forces the dequantized path, so what matters is the
        # overlap between the patched keys and the quantized layers. A LoRA that patches 150 keys,
        # none of them quantized, would produce an unchanged census that reads as "LoRA is fine".
        from comfy_kitchen.tensor.base import QuantizedTensor as _QT
        qnames = {n for n, mod in diffusion_model.named_modules()
                  if isinstance(getattr(mod, "weight", None), _QT)}
        overlap = sorted(k for k in keys
                         if k.removeprefix("diffusion_model.").rsplit(".", 1)[0] in qnames)
        print(f"{len(overlap)} of them land on quantized layers"
              f"{': e.g. ' + overlap[0] if overlap else ''}", flush=True)
        if not overlap:
            raise SystemExit("this LoRA patches no quantized layer, so a census under it would "
                             "measure the unpatched path and be reported as a LoRA result. Pick "
                             "a LoRA that targets the quantized modules.")

    undo = instrument()
    print(f"wrapped {len(undo)} layout handler(s) across ConvRot W4A4 and AsymW4A8Int8",
          flush=True)

    latent_format = model.model.latent_format
    side = max(args.size // 8, 8)
    # Same rule as calibrate_activations: a video model handed a 4-D latent fails inside the
    # transformer with a shape error that never mentions latents.
    if getattr(latent_format, "latent_dimensions", 2) == 3:
        ratio = getattr(latent_format, "temporal_downscale_ratio", 4)
        frames = max(1, (args.frames - 1) // ratio + 1)
        shape = [1, latent_format.latent_channels, frames, side, side]
        print(f"video latent {shape} ({args.frames} frames / temporal ratio {ratio})", flush=True)
    else:
        shape = [1, latent_format.latent_channels, side, side]
    latent = torch.zeros(shape, device="cpu")
    noise = comfy.sample.prepare_noise(latent, args.seed, None)
    started = time.perf_counter()
    samples = comfy.sample.sample(
        model, noise, args.steps, args.cfg, args.sampler, args.scheduler,
        positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None,
        last_step=None, force_full_denoise=False, noise_mask=None, callback=None,
        disable_pbar=True, seed=args.seed)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    # The control for the LoRA question, and it is not optional. An unchanged dispatch census
    # under a LoRA has two possible causes -- the LoRA ran and the kernel still ran, or the LoRA
    # was silently dropped -- and the call counts cannot tell them apart. The latent can: run
    # twice with the same seed, once with --lora and once without, and compare these numbers.
    # Identical means the LoRA changed nothing, which is a worse finding than the one being
    # tested, not a passing result.
    flat = samples.detach().float().reshape(-1)
    print(f"\nlatent fingerprint  norm {float(flat.norm()):.6f}  mean {float(flat.mean()):+.6f}  "
          f"std {float(flat.std()):.6f}  first {float(flat[0]):+.6f}")

    from comfy_kitchen.tensor import base as tensor_base
    for op, layout, original in undo:
        tensor_base._LAYOUT_DISPATCH_TABLE[op][layout] = original

    print(f"\n{args.steps} steps at {args.size}px in {elapsed:.1f}s\n")
    print(f"{'branch':<48}{'calls':>10}")
    for name, count in sorted(COUNTS.items()):
        print(f"{name:<48}{count:>10}")

    dequantized = sum(v for k, v in COUNTS.items() if k.endswith(":DEQUANTIZED to BF16"))
    native = sum(v for k, v in COUNTS.items() if k.endswith(":native kernel"))
    print(f"\nnative kernel: {native}    dequantized to BF16: {dequantized}")
    if dequantized:
        print("\nThe dequantizing branch fired. That is INT4 storage with BF16 compute on those "
              "calls, which is the outcome this project exists to avoid. Find the caller.")
    else:
        print("\nThe dequantizing branch never fired on this path. That demotes the audit finding "
              "for this model, this sampler and this resolution -- and for nothing else. LoRA, "
              "torch.compile and any node calling linear() on an already-transposed weight are "
              "all untested here.")
    print("\nCovered: TensorCoreConvRotW4A4Layout and AsymW4A8Int8Layout. Any other quantized "
          "layout in the checkpoint dispatches through handlers that are not wrapped here and is "
          "absent from every number above.")
    print("NOT covered: CUDA graph replay. This counter is Python; a replayed graph re-runs the "
          "device work without re-entering it, so the count would read zero while the kernel "
          "runs. Measured in the sibling LLM project (48 tokens moved its counter by zero). "
          "Whether this path captures a graph at all was not checked -- read these numbers as "
          "eager-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
