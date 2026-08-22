# 04 - Three launchers bind ComfyUI to 0.0.0.0 with no authentication

Type: task
Status: ready-for-human
Blocked by: -
Severity: medium
Provenance: EXECUTED 2026-08-22 (grepped all nine .bat)

## Problem

Nine `.bat` files at the repo root. Grepped every one for `--listen`:

```
run_8190_limpo.bat                  --listen 0.0.0.0
run_nvidia_gpu_8190.bat             --listen 0.0.0.0
run_nvidia_gpu_8190_flash.bat       --listen 0.0.0.0
run_nvidia_gpu_8190_loopback.bat    (127.0.0.1 -- the 0.0.0.0 hit is inside an echoed comment)
run_cpu.bat / run_nvidia_gpu.bat / run_nvidia_gpu_fast_fp16_accumulation.bat   (no flag -> loopback)
baixar_vae22.bat / run_deepcompressor.bat   (not launchers)
```

ComfyUI's `/prompt` and `/upload/image` have no authentication. Bound to `0.0.0.0`, anything that can
reach this host on 8190 can queue a workflow on the 3090 and upload files into `ComfyUI/input`.

This is `ready-for-human` and not `ready-for-agent` because the answer is a policy call, not a code
one: this host runs the owner's ERP and sits on a LAN whose exposure only the owner knows. Deleting
the flag from three launchers is trivial; deciding whether remote access to ComfyUI is wanted is not.

## Second, independent finding in the same files

**Zero of the nine launchers passes `--disable-dynamic-vram`** -- confirmed by grep, 0 hits.
CLAUDE.md already says this and it is still true. Both the Nunchaku SVDQuant loaders and the LTX 2.5
workflow need it, and without it the LTX render dies with 35 GB staged on a 24 GB card, with an error
message that names neither the flag nor the subsystem. Following "use the loopback launcher"
reproduces that OOM.

## Closing criterion (written before the fix)

Closed when the owner has decided, and the decision is written into the launcher file itself as a
comment, for each of the three:

- **keep 0.0.0.0** -> the comment says who needs remote access and from where, and the launcher is
  renamed to carry `_lan` so the bind is visible at the call site;
- **drop it** -> `--listen 127.0.0.1`, and a note that ComfyUI has no auth is left in place so nobody
  re-adds it.

Separately and independently of that decision: at least one launcher passes `--disable-dynamic-vram`,
and its filename says so.
