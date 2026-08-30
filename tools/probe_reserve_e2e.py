"""End to end: does ComfyUI actually put a 15.5 GiB model on a card with ~10.7 GiB free?

This is the run the earlier arithmetic probe could not do. It calls ComfyUI's real loader
(comfy.sd.load_diffusion_model) and ComfyUI's real placement decision
(comfy.model_management.load_models_gpu), with a real checkpoint, while another process
holds the card.

CONDITION THIS NEEDS: the glm-w4 container holding VRAM on cuda:0. Expires with it.

WHAT IS VARIED: --reserve-vram, nothing else. Same model file, same device, one arm per
process (model_management reads args at import, so each arm is a fresh subprocess).

WHAT IS HELD: the model, the card, the neighbour. Measured before and after each arm.

THE PREDICTION BEING TESTED, from reading the code: with the default reserve the guard
compares 15.5 GiB + 0.7 GiB against a free number that includes the neighbour's memory,
so it says yes and the load succeeds by evicting. If it instead fails or falls back to
CPU, the prediction was wrong and that is the finding.

NOT COVERED even by this: no sampling is run, so this measures placement, not whether a
generation would then complete. The neighbour's throughput is not measured, only its
resident VRAM.
"""
import os
import subprocess
import sys

MODEL = os.path.join("ComfyUI", "models", "diffusion_models", "capybara_v0.1.safetensors")
MIB = 1024 * 1024


def smi_used(idx=0):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    return int(out)


def arm(reserve):
    """Run one arm in a fresh interpreter; returns its stdout."""
    code = (
        "import sys, os, subprocess, json\n"
        "sys.path.insert(0, 'ComfyUI')\n"
        f"sys.argv = ['main.py'] + ({repr(['--reserve-vram', str(reserve)]) if reserve is not None else '[]'})\n"
        "import comfy.options; comfy.options.enable_args_parsing()\n"
        "import torch, comfy.sd, comfy.model_management as mm\n"
        "def used():\n"
        "    o = subprocess.run(['nvidia-smi','--query-gpu=memory.used',"
        "'--format=csv,noheader,nounits','-i','0'], capture_output=True, text=True).stdout.strip()\n"
        "    return int(o)\n"
        "dev = torch.device('cuda:0')\n"
        "before_smi = used()\n"
        "before_torch = mm.get_free_memory(dev) // (1024*1024)\n"
        f"m = comfy.sd.load_diffusion_model({MODEL!r})\n"
        "size = m.model_size() // (1024*1024)\n"
        "after_load_smi = used()\n"
        "err = ''\n"
        "try:\n"
        "    mm.load_models_gpu([m], memory_required=m.model_size())\n"
        "    placed = str(m.model.device) + ' resident=' + "
        "str(getattr(m, 'model_loaded_weight_memory', 0)//(1024*1024)) + 'MiB'\n"
        "except Exception as e:\n"
        "    placed = 'EXCEPTION'; err = type(e).__name__ + ': ' + str(e).splitlines()[0][:140]\n"
        "after_smi = used()\n"
        "print(json.dumps(dict(reserve_gib=%s, model_mib=size, free_torch=before_torch,"
        " smi_used_before=before_smi, smi_used_after_readfile=after_load_smi,"
        " smi_used_after=after_smi, placed=placed, err=err,"
        " extra_reserved_mib=int(mm.extra_reserved_memory())//(1024*1024))))\n"
        % (repr(reserve),)
    )
    r = subprocess.run(
        [os.path.join("python_embeded", "python.exe"), "-s", "-c", code],
        capture_output=True, text=True, timeout=1800,
    )
    return r


if not os.path.exists(MODEL):
    sys.exit(f"model not found: {MODEL}")

print(f"model: {MODEL}  ({os.path.getsize(MODEL) / 2**30:.1f} GiB)")
print(f"neighbour (glm-w4) resident on cuda:0 at start: {smi_used(0)} MiB")
print()

import json

for reserve in (None, 8, 20):
    label = "default" if reserve is None else f"{reserve} GiB"
    print(f"--- arm: --reserve-vram {label} ---", flush=True)
    r = arm(reserve)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        print("  NO RESULT. stdout tail:", r.stdout.strip()[-500:])
        print("  stderr tail:", r.stderr.strip()[-800:])
        continue
    d = json.loads(line[-1])
    print(f"  model size ComfyUI computed : {d['model_mib']} MiB")
    print(f"  reserve in effect           : {d['extra_reserved_mib']} MiB")
    print(f"  free per ComfyUI (before)   : {d['free_torch']} MiB")
    print(f"  nvidia-smi used before      : {d['smi_used_before']} MiB"
          f"  (-> {24576 - d['smi_used_before']} MiB really free)")
    print(f"  nvidia-smi used after load  : {d['smi_used_after']} MiB")
    print(f"  model placed on             : {d['placed']}  {d['err']}")
    delta = d["smi_used_after"] - d["smi_used_before"]
    print(f"  delta on the card           : {delta:+d} MiB")
    print()

print(f"neighbour resident on cuda:0 at end: {smi_used(0)} MiB")
print()
print("NOT covered: no sampling was run, so this is placement and not a completed",
      "generation; the neighbour's throughput was never measured, only its resident VRAM.",
      file=sys.stderr)
