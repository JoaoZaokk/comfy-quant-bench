"""Turn on ComfyUI's DynamicVRAM the way `main.py` does, for scripts that are not `main.py`.

Every tool in this directory loads models by calling `comfy.sd.load_diffusion_model` directly. That
skips a block near `main.py:265-292` which the server always runs, and the difference is not
cosmetic: without it, `comfy.memory_management.aimdo_enabled` stays False, and `comfy/ops.py:521`
then declines to build its lazy `Linear` --

    if (not comfy.memory_management.aimdo_enabled
        or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
        super().__init__(in_features, out_features, bias, device, dtype)
        return

    self.weight = None          # otherwise: allocated, committed, and filled

-- so every weight is materialised in host memory at construction and again at load. Upstream's own
comment says why that matters here specifically: "Windows doesn't over-commit memory so without
this, we are momentarily commit charged for the weight even though we might zero-copy it when we
load the state dict."

Measured on this machine, 2026-08-19: loading `ltx-2.5-22b-distilled-transformer-bf16` (39.1 GiB)
without DynamicVRAM took the process to a 44.5 GiB working set with 4.0 GiB of system RAM left, on
a 63.1 GiB box, before any block had reached a GPU. The peak is the whole model regardless of where
the weights are meant to end up, which makes block-distribution schemes like DisTorch2 fix the
wrong half of the problem: they decide where weights live *after* everything has been paid for.

    from _dynamic_vram import enable
    if not enable():
        print("DynamicVRAM unavailable; expect the full model in host RAM")

Returns True only when comfy-aimdo initialised the devices. A False is worth printing rather than
swallowing: it changes the memory profile of everything that follows.
"""

from __future__ import annotations


def enable(headroom_gib: float = 1.0) -> bool:
    """Enable DynamicVRAM. Returns whether it actually came up.

    `headroom_gib` mirrors `--vram-headroom`: memory per device that aimdo leaves alone. The
    default matches ComfyUI's own.
    """
    try:
        import comfy_aimdo.control
        import comfy.memory_management
        import comfy.model_patcher
        import comfy.model_management
    except Exception:
        return False

    # `init()` first, and this is the whole trap. It is what loads `aimdo.dll` through ctypes and
    # assigns the module-level `lib`; until it runs, `init_devices` opens with `if lib is None:
    # return False` and reports a clean, silent, entirely misleading failure. main.py calls it far
    # earlier (main.py:63-70) than the device setup at :265, so copying only the visible half of
    # the block gives a False that looks like "your hardware is unsupported".
    try:
        comfy_aimdo.control.init()
    except Exception:
        return False

    devices = comfy.model_management.get_all_torch_devices()
    try:
        initialised = comfy_aimdo.control.init_devices(
            (d.index, int(headroom_gib * 1024 ** 3)) for d in devices)
    except TypeError:
        # comfy-aimdo 0.4.9 protocol, kept because main.py keeps it.
        initialised = comfy_aimdo.control.init_devices(d.index for d in devices)

    if not initialised:
        return False

    # main.py sets a log level here; this build's `control` has no set_log_* at all, so calling
    # one the way main.py does raises AttributeError. Skipped rather than guessed at.
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    return True
