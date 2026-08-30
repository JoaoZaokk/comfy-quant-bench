"""Does --reserve-vram protect a GPU tenant this process cannot see?

CONDITION THIS NEEDS: another process holding VRAM on cuda:0 (the glm-w4 container in
WSL2). Expires when that training ends.

WHAT IS VARIED: the value of --reserve-vram, nothing else. Same device, same tenant, same
interpreter. The comparison ComfyUI actually makes is recomputed at each value.

WHY THE QUESTION IS NOT OBVIOUS: --reserve-vram reads like "leave N GB alone for other
applications" -- that is close to the help text's own wording. But reserve does not enter
get_free_memory(); it is added to the DEMAND side in model_management (the
`minimum_memory_required + extra_reserved_memory()` and
`memory_required + extra_reserved_memory()` expressions), and then compared against
get_free_memory(), which was measured on 2026-08-30 to include ~15 GiB belonging to a
neighbour that WDDM will evict. So the reserve is subtracted from an inflated number.

This probe does NOT load a model and does NOT start ComfyUI. It recomputes ComfyUI's own
decision arithmetic with its own functions and the real numbers on this card, which is
enough to answer whether the guard can bind. What it therefore CANNOT show is a real
load being allowed or refused end to end -- that is traced, not executed.
"""
import importlib
import subprocess
import sys

RESERVES_GIB = [None, 4, 8, 16, 20]


def smi(idx=0):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits", "-i", str(idx)],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    used, total = (int(x.strip()) for x in out.split(","))
    return total - used, used


sys.path.insert(0, "ComfyUI")
MIB = 1024 * 1024

print(f"{'--reserve-vram':>15} {'comfy free':>11} {'reserve':>9} {'demand@1GiB':>12} "
      f"{'loads?':>7} {'smi free':>9}")

for res in RESERVES_GIB:
    argv = ["main.py"]
    if res is not None:
        argv += ["--reserve-vram", str(res)]
    sys.argv = argv

    for mod in [m for m in list(sys.modules) if m.startswith("comfy")]:
        del sys.modules[mod]
    import comfy.options
    comfy.options.enable_args_parsing()
    mm = importlib.import_module("comfy.model_management")

    import torch
    dev = torch.device("cuda:0")
    free = mm.get_free_memory(dev)
    extra = int(mm.extra_reserved_memory())
    minimum = int(mm.minimum_inference_memory())

    # The shape of the guard at model_management.py:918 -- a 1 GiB model wanting to load.
    model = 1 * 1024**3
    required = int(max(minimum, model + extra))
    loads = required < free

    sfree, _ = smi(0)
    label = "(default)" if res is None else f"{res} GiB"
    print(f"{label:>15} {free // MIB:8d} MiB {extra // MIB:6d} MiB {required // MIB:9d} MiB "
          f"{('YES' if loads else 'no'):>7} {sfree:6d} MiB")

print()
print("Read the last two columns together. 'loads?' is ComfyUI's own verdict; 'smi free' is")
print("what the card actually has. Where they disagree, the reserve did not bind on the")
print("thing it reads like it protects.")
print()
print("NOT executed: no model was loaded, and ComfyUI was not started. This is ComfyUI's",
      file=sys.stderr)
print("arithmetic recomputed with its own functions, not an observed load or refusal.",
      file=sys.stderr)
