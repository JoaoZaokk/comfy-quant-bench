"""Does the 3090's 'free' VRAM that torch reports actually exist, with a WSL tenant on the card?

CONDITION THIS NEEDS, and why it expires: another process (the glm-w4 container, inside
WSL2/docker-desktop) must be holding VRAM on cuda:0. Once it stops, the discrepancy this
probe exists to explain is gone and the question cannot be asked again.

WHAT IS BEING VARIED: how much this process has allocated on cuda:0. Everything else is
held -- same device, same interpreter, no other work started here.

THE TWO OUTCOMES, and they mean opposite things:
  allocation succeeds past the driver's reported free  -> torch is RIGHT. WDDM evicts the
      other tenant's pages to system RAM on demand. The number is true but expensive, and
      the hazard is a silent collapse of the neighbour's throughput, not an OOM.
  allocation fails at or near the driver's reported free -> torch is WRONG. The number is
      phantom, and ComfyUI's own fit test (model_management.get_free_memory) will approve a
      model that cannot land.

NOT COVERED: this says nothing about Linux, nothing about TCC-mode cards, and nothing about
what happens when the neighbour is a native Windows process rather than a WSL one.

The owner explicitly authorised breaking the training to get this ("se parar ou quebrar o
treino, fodase, ele volta no checkpoint"). Everything is freed at the end regardless.
"""
import subprocess
import sys

import torch

DEV = 0
STEP_MIB = 1024
CAP_MIB = 22528  # hard stop, ~22 GiB on a 24 GiB card


def smi_free_mib(idx):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits", "-i", str(idx)],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    used, total = (int(x.strip()) for x in out.split(","))
    return total - used, used


def torch_free_mib(idx):
    free, total = torch.cuda.mem_get_info(idx)
    return free // 2**20, (total - free) // 2**20


props = torch.cuda.get_device_properties(DEV)
print(f"device cuda:{DEV} = {props.name}, total {props.total_memory // 2**20} MiB")

t_free0, t_used0 = torch_free_mib(DEV)
s_free0, s_used0 = smi_free_mib(DEV)
print(f"BASELINE  torch free={t_free0:6d} MiB  used={t_used0:6d}")
print(f"BASELINE  smi   free={s_free0:6d} MiB  used={s_used0:6d}")
print(f"BASELINE  gap (torch claims free but driver does not) = {t_free0 - s_free0} MiB")
print()
print(f"{'held MiB':>9} {'alloc':>6} {'torch free':>11} {'smi free':>9} {'smi used':>9}")

blocks = []
held = 0
verdict = "cap reached without failure"
try:
    while held < CAP_MIB:
        try:
            blocks.append(torch.empty(STEP_MIB * 2**20, dtype=torch.uint8, device=f"cuda:{DEV}"))
            torch.cuda.synchronize(DEV)
            ok = "OK"
        except Exception as exc:  # noqa: BLE001 - want the exact type/message
            print(f"{held:9d} {'FAIL':>6}  {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
            verdict = f"FAILED after holding {held} MiB (driver said {s_free0} MiB free at start)"
            break
        held += STEP_MIB
        tf, _ = torch_free_mib(DEV)
        sf, su = smi_free_mib(DEV)
        print(f"{held:9d} {ok:>6} {tf:11d} {sf:9d} {su:9d}")
finally:
    n = len(blocks)
    blocks.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEV)
    tf, _ = torch_free_mib(DEV)
    sf, su = smi_free_mib(DEV)
    print()
    print(f"freed {n} blocks. torch free={tf} MiB, smi free={sf} MiB, smi used={su} MiB")

print()
print("VERDICT:", verdict)
print(f"driver-reported free at start was {s_free0} MiB; torch claimed {t_free0} MiB")
if held > s_free0 + STEP_MIB:
    print(f"-> allocated {held - s_free0} MiB PAST what the driver called free."
          " WDDM evicted the neighbour. torch's number is real but costs the other tenant.")
else:
    print("-> did NOT get meaningfully past the driver's number."
          " torch's free is phantom on a shared card.")
print()
print("NOT covered: neighbour's throughput was not measured during this, so 'evicted' here"
      " means 'my allocation succeeded', not 'the training slowed by X'. Linux and TCC-mode"
      " cards untested. A native-Windows neighbour untested.", file=sys.stderr)
