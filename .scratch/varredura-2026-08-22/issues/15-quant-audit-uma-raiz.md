# 15 - quant_audit covers one of the two model roots, so the checked-in inventory is missing 409 GiB

Type: task
Status: ready-for-agent
Blocked by: -
Severity: medium
Provenance: OBSERVED 2026-08-22

## Problem

`tools/quant_audit.py:40` defaults `--models-root` to `ComfyUI/models`, and `:478-479` does
`path.relative_to(models_root)` -- so it is structurally single-root, not merely configured that way.

The bench has **two** roots. `ComfyUI/extra_model_paths.yaml` maps `D:/ComfyUI-Models/`, which exists
and holds **409 GiB across 81 files** (34 of them `.safetensors`, one `.gguf`), including a 326 GiB
`diffusion_models` and a 54 GiB `text_encoders`.

Measured: `grep -c "ComfyUI-Models" quantization_inventory.json` returns **0**.

So the inventory that CLAUDE.md instructs every session to regenerate and trust describes 623 GiB of a
1.03 TiB installation, and nothing in its output says so.

Two smaller gaps in the same tool, both from the same single-root shape:

- **No hashes.** Grep for `hashlib|sha256|md5` over the file: 0 hits. `tools/model_audit.py:111-122`
  already has the right ladder (size -> `blake2b` of first+last 1 MiB -> full `sha256` only on
  collision) and is a separate tool.
- **72 top-level directories under `ComfyUI/models`**, of which roughly 45 are custom-node-owned
  categories (`BiRefNet`, `sam2`, `sam3`, `matanyone`, `liveportrait`, `reactor`, ...) that
  `folder_paths` does not know and the YAML does not map. An inventory built on
  `folder_paths.get_filename_list()` is blind to them; one built on a disk walk sees them with no
  category semantics.

## Why this is `medium` and not `low`

It is the one finding where the tool is *silently incomplete* rather than wrong. Everything downstream
that asks "do we already have a compatible variant" -- which is the whole of ComfyLite phase 1 -- gets
a "no" that means "not in the half I looked at".

## Closing criterion (written before the fix)

Closed when:

1. `quant_audit.py` accepts multiple roots, and by default reads them from
   `ComfyUI/extra_model_paths.yaml` rather than requiring them on the command line;
2. every record carries which root it came from, so `relative_to` is unambiguous;
3. the output states the roots it covered and the count per root, on every run;
4. `grep -c "ComfyUI-Models" quantization_inventory.json` is greater than zero after a regeneration;
5. the tool states in its output that it did not hash, or it hashes using `model_audit.py`'s ladder.

Item 3 alone is what stops this recurring: an inventory that names its own scope cannot be misread as
complete.
