# Triage labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the label
strings used in this repo's tracker. Defaults kept as-is — the local tracker had no prior label
vocabulary to collide with.

Because the tracker is markdown files rather than a label API, a "label" here is the value of the
`Status:` line near the top of a ticket file.

| Role in mattpocock/skills | String in our tracker | Meaning                                            |
| ------------------------- | --------------------- | -------------------------------------------------- |
| `needs-triage`            | `needs-triage`        | Not yet evaluated                                   |
| `needs-info`              | `needs-info`          | Waiting on the bench owner for more information     |
| `ready-for-agent`         | `ready-for-agent`     | Fully specified, an AFK agent can take it           |
| `ready-for-human`         | `ready-for-human`     | Needs the owner: GPU time, a package change, a call |
| `wontfix`                 | `wontfix`             | Will not be actioned                                |

## What `ready-for-human` means specifically on this bench

It is not a politeness. Three whole categories of work here cannot be delegated to an agent,
because the repo's hard rules put them on the owner:

- **anything that installs or upgrades a package** — Torch, CUDA, ComfyUI, comfy-kitchen, pytest,
  ruff. `_pip_freeze_before_w4a4_cu130_20260816.txt` is the known-good baseline.
- **anything that needs the GPU** — the 3090 is shared and arbitrated by `tools/gpu_lock.ps1`. A
  ticket that needs a measurement needs a lock window, and a window is asked for, not taken.
- **anything that stops, restarts or reconfigures WSL** — `docker-desktop` runs the owner's ERP.

A ticket needing any of those is `ready-for-human` even when the code change itself is trivial.
