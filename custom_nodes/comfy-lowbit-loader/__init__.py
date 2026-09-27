"""Native ComfyUI loader for ternary / binary diffusion models (Bonsai Image and any BF16 ternary checkpoint).

One node reads the gemlite pack (`state_dict.pt`), the MLX pack, the "unpacked" BF16 release, or a file
saved from this format, and returns an ordinary MODEL whose linear layers keep their 1- or 2-bit codes in
memory. Because the result is a normal ModelPatcher, everything ComfyUI does with models applies:
automatic offload to RAM when VRAM is short (dynamic VRAM), load on any GPU or the CPU, parking the
offloaded copy on a second GPU, model saving and the core multi-GPU nodes.

Matmuls run in the compute dtype after a fused dequantization (Triton on CUDA, torch elsewhere), so speed
is at best that of the BF16 model; the gain is memory and offload bandwidth.
"""

import json
import logging
import os

import torch

import comfy.model_management
import comfy.sd
import folder_paths

from . import formats, kernel, layout

layout.register()

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _devices():
    names = ["auto", "cpu"]
    if torch.cuda.is_available():
        names += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return names


class LowBitDiffusionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "device": (_devices(), {"tooltip": "GPU (or CPU) that runs the model."}),
                "offload_device": (_devices(), {"tooltip": "Where weights wait when not on the run device. auto = ComfyUI default (RAM). A second GPU keeps them in its VRAM."}),
                "compute_dtype": (["auto", "bfloat16", "float16", "float32"],),
            },
            "optional": {
                "path": ("STRING", {"default": "", "tooltip": "Absolute path to a file outside the model folders (e.g. a gemlite state_dict.pt). Overrides unet_name when set."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load"
    CATEGORY = "model/loaders"
    DESCRIPTION = (
        "Loads ternary/binary diffusion models (Bonsai Image gemlite, MLX or unpacked, or any BF16 checkpoint "
        "whose weights are ternary/binary per group) keeping the packed codes in memory. Format is detected "
        "from the file contents."
    )

    def load(self, unet_name, device, offload_device, compute_dtype, path=""):
        source = path.strip().strip('"') or folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"lowbit loader: {source} is not a file")

        model_options = {}
        if device != "auto":
            model_options["load_device"] = torch.device(device)
        if offload_device != "auto":
            model_options["offload_device"] = torch.device(offload_device)
        if compute_dtype != "auto":
            model_options["dtype"] = DTYPES[compute_dtype]

        run_device = model_options.get("load_device", comfy.model_management.get_torch_device())
        pack_device = run_device if run_device.type == "cuda" else "cpu"
        kind, sd, report = formats.load(source, pack_device=pack_device)
        if kind == "dense" and not report["packed_layers_by_bits"]:
            logging.warning("lowbit loader: %s has no ternary/binary layers; loading it as a regular model", source)

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)
        if model is None:
            raise RuntimeError(f"lowbit loader: ComfyUI could not detect the model type of {source}")

        report.update({
            "file": source,
            "run_device": str(run_device),
            "offload_device": str(model.offload_device),
            "kernel": kernel.kernel_for(run_device),
            "lowbit_layers_in_model": sum(
                1 for m in model.model.modules() if getattr(m, "quant_format", None) == layout.FORMAT
            ),
        })
        text = json.dumps(report, indent=1)
        logging.info("lowbit loader: %s", text)
        return (model, text)


NODE_CLASS_MAPPINGS = {"LowBitDiffusionLoader": LowBitDiffusionLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"LowBitDiffusionLoader": "Load Diffusion Model (ternary / binary)"}
