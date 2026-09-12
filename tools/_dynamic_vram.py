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

import json
import pathlib
import struct


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


def dtypes_float_2d(caminho) -> set[str]:
    """Os dtypes de ponto flutuante dos tensores 2-D de um safetensors, so pelo header.

    2-D porque e o que vira peso de `F.linear`. Escalas e normas sao 1-D e podem ser F32 sem
    que ninguem se importe; um PESO em F32 no meio de um modelo BF16 e outra coisa.
    """
    FLUTUANTES = {"F64", "F32", "F16", "BF16"}
    with pathlib.Path(caminho).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return {v["dtype"] for v in hdr.values()
            if v["dtype"] in FLUTUANTES and len(v.get("shape", ())) == 2}


def perigoso_para_lazy(caminho) -> set[str] | None:
    """Os dtypes em conflito, ou `None` se o arquivo pode passar pelo caminho preguicoso.

    MEDIDO 2026-09-12, e custou uma corrida inteira do `quality_ladder`. Com DynamicVRAM ligado,
    `comfy/ops.py:552` carrega o peso DIRETO DO ARQUIVO e **nao converte para o dtype do modulo**.
    Isso e invisivel enquanto o checkpoint tem um dtype so. O `krea2_turbo_bf16` nao tem: os 256
    pesos de `blocks` e `txtfusion` sao BF16 e as pontas -- `first`, `last.linear`, `tmlp`,
    `tproj`, `txtmlp` -- sao **F32**, que e a politica de precisao do modelo, nao um defeito.
    O resultado foi

        RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float

    em `self.first(img)`, depois de carregar 24,5 GiB, com a mensagem apontando para a primeira
    camada do modelo e nao para o mecanismo que a quebrou. Sem DynamicVRAM o mesmo arquivo carrega
    com `first.weight` em bfloat16 e amostra normalmente -- verificado variando um eixo so.
    """
    d = dtypes_float_2d(caminho)
    return d if len(d) > 1 else None
