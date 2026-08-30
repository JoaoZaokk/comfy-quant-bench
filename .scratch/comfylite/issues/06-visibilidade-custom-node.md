# 06 - `visible_to_comfyui` is core-only visibility, and it under-reports by 110 GiB

Type: grilling
Status: ready-for-human
Blocked by: -
Provenance: EXECUTED end to end on 2026-08-22 (real scan, real folder_paths import, real grep of the
custom node)

## What was measured

The first real ComfyLite scan finished in **4.52 s** over **505 files / 1.01 TiB** and reported
**144 files visible to ComfyUI**. Digging into the 55 rows that are `role=model` but
`visible_to_comfyui=0` — 197.4 GiB — split them cleanly in two:

| set | files | size | is the FALSE correct? |
|---|---|---|---|
| `.gguf` in directories `folder_paths` **does** map | 11 | 110.3 GiB | **No** |
| custom-node-owned trees reached by hardcoded path | 44 | 87.1 GiB | Yes |

The second set (SEEDVR2, FlashVSR, DiffuEraser, `viral_d/diffusers/...`) is genuinely outside
`folder_paths`. The first set is the defect.

## Why the `.gguf` rows are wrong

EXECUTED, embedded interpreter, no custom node loaded:

```
supported_pt_extensions: ['.bin','.ckpt','.pkl','.pt','.pt2','.pth','.safetensors','.sft']
.gguf in it: False
get_filename_list('diffusion_models'): 26 entries; 0 are .gguf
unet_gguf category exists: False
```

TRACED, `ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py:22-33`:

```python
folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])
```

**Custom nodes mutate `folder_names_and_paths` at their own import time.** The worker imports
`folder_paths` and nothing else — deliberately, because importing arbitrary custom-node code is a far
larger hazard than a missing category — so it sees the table as it exists *before* any node runs.
`ComfyUI-GGUF` is present and **not** disabled (EXECUTED: it is not among the `.disabled` entries), so
a running ComfyUI lists all 11 of those files and ComfyLite says it cannot.

This is not a bug in the seam decision from ticket 01. The seam is right; its *reach* is narrower than
the column name promises.

## Already done

The column's meaning is now stated where it surfaces, with the measurement inline, in both
`discovery.py`'s and `scanner.py`'s `NOT COVERED` blocks. A `FALSE` no longer reads as "ComfyUI cannot
use this file" to anyone who reads the tool's own output.

## The decision, and it is the owner's

How should ComfyLite learn the *effective* table?

**Option A — ask a running ComfyUI.** `GET /object_info` returns every node's `INPUT_TYPES`, which is
where the loader dropdowns' file lists actually live, custom nodes included. Honest and never stale.
Cost: it only works when ComfyUI is up, so the column becomes two columns (`visible_core`, and
`visible_effective` which is NULL until a live server has been asked). Fits the handoff, which already
treats ComfyUI as a runtime ComfyLite talks to.

**Option B — a static table of known custom-node categories.** Cheap and offline. Also *exactly* the
failure ticket 01 rejected: a hand-copied table that goes stale one custom-node update later, silently.

**Option C — leave it, label only.** The label is already there. The number stays wrong in the UI.

Recommendation: **A**, with the column split, and `visible_effective` shown in the UI only once a live
ComfyUI has been asked at least once. Do not build B.

## Closing criterion

Closed when the owner picks A, B or C. If A: additionally when a scan against a running ComfyUI marks
those 11 `.gguf` files effective-visible, and against a stopped one leaves the field NULL rather than
guessing.
