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

## What `--listen 0.0.0.0` actually reaches on this host -- MEASURED 2026-08-22

`0.0.0.0` is not "the LAN". It is every IPv4 interface the machine has, and this one has six:

    172.21.160.1   vEthernet (WSL (Hyper-V firewall))   Manual     <- the ERP containers' path to the host
    172.18.192.1   vEthernet (Default Switch)           Manual
    10.5.0.2       NordLynx                             Manual     Private, NoTraffic
    172.16.0.2     CloudflareWARP                       Manual     **Public, Internet**
    192.168.3.133  Ethernet                             Dhcp       Private, Internet  <- the LAN with the NAS at .40
    169.254.94.107 OpenVPN Data Channel Offload         WellKnown

Two of those are worth naming specifically.

**`CloudflareWARP` is on the `Public` profile with `Internet` connectivity.** A Public-profile
interface is the one Windows treats as hostile by default, and it is the one a "just the LAN"
mental model does not include.

**`vEthernet (WSL)` is how the eleven ERP containers reach the host.** So a ComfyUI on `0.0.0.0`
is reachable from inside the ERP's own network namespace. Nothing suggests that matters today; it
is listed because "the LAN" is not the whole answer and this interface is the one nobody pictures.

### What I could NOT measure, and will not

Whether inbound traffic actually arrives depends on the firewall rules, and reading them needs
elevation:

    Get-NetFirewallPortFilter: Access is denied.

What IS readable: all three firewall profiles are **Enabled**, with `DefaultInboundAction:
NotConfigured` -- which means the built-in default applies rather than an explicit policy. On
Windows that default is block, so an unsolicited inbound connection is probably refused **unless a
rule exists for the listening process**, and applications that bind a port routinely acquire one
from the first-run prompt. I cannot tell which of those is true here.

**So the honest statement is: the bind is wide, and whether it is reachable is unmeasured.** Do
not read this section as "it is exposed" and do not read it as "it is fine". The one command that
would settle it needs an elevated shell, and elevating is the owner's call, not an agent's:

    Get-NetFirewallRule -Direction Inbound -Enabled True |
      Get-NetFirewallPortFilter | Where-Object LocalPort -in 8188,8190

### What does not depend on the firewall at all

ComfyUI's `/prompt` and `/upload/image` have **no authentication**, so anything that can reach the
port can queue work on the 3090 and write files into `ComfyUI/input`. That is a property of the
application, not of the network, and it is why the bind is worth a decision rather than a shrug.

And `run_nvidia_gpu_8190_loopback.bat:6-11` already records a second, non-security reason to
prefer loopback on this host: bound to `0.0.0.0`, the asyncio accept loop dies with
`OSError(22, 'The specified network name is no longer available', 64)` whenever NordLynx/NordVPN
reconnects -- the process stays alive and the port stops accepting. Given `NordLynx` and
`OpenVPN Data Channel Offload for NordVPN` both hold addresses above, that is not hypothetical.
